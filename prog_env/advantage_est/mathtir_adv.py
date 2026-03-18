import torch
import numpy as np
from collections import defaultdict
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("mathtir")
def compute_grpo_mathtir_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    **kwargs,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1)

    group_adv = scores
    seq_len = token_level_rewards.size(1)
    
    lam = 0.9
    turns_list = []          
    turn_rewards_list = []   
    for i in range(bsz):
        mask = response_mask[i]
        rewards = token_level_rewards[i]

        turns = []
        start = None
        for t in range(seq_len):
            if mask[t] == 1 and start is None:
                start = t
            if mask[t] == 0 and start is not None:
                turns.append((start, t - 1))
                start = None
        if start is not None:
            turns.append((start, seq_len - 1))

        turns_list.append(turns)

        turn_rewards = torch.stack([rewards[e] for (_, e) in turns])#定位每个turn的reward
        turn_rewards_list.append(turn_rewards)

    discounted_rewards_list = []
    for turn_rewards in turn_rewards_list:
        N = len(turn_rewards)
        discounted = torch.zeros_like(turn_rewards)
        acc = 0.0
        for t in reversed(range(N)):
            acc = turn_rewards[t] + lam * acc
            discounted[t] = acc
        discounted_rewards_list.append(discounted)
    
    step_key2score = defaultdict(list)

    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)
        for j in range(T):
            key = f"{index[i]}-{j}"     
            step_key2score[key].append(disc[j])

    step_mean = {}
    step_std = {}

    for key, vals in step_key2score.items():
        vals = torch.stack(vals)
        if len(vals) == 1:
            step_mean[key] = torch.tensor(0.0)
            step_std[key] = torch.tensor(1.0)
        else:
            step_mean[key] = vals.mean()
            step_std[key] = vals.std()

    discounted_adv_list = []
    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)

        adv_turn = torch.zeros_like(disc)
        for j in range(T):
            key = f"{index[i]}-{j}"
            adv_turn[j] = (disc[j] - step_mean[key]) / (step_std[key] + epsilon)
        discounted_adv_list.append(adv_turn)
    
    final_scores = torch.zeros(bsz, seq_len)

    for i in range(bsz):
        turns = turns_list[i]
        step_adv = discounted_adv_list[i]         
        fused_adv = (group_adv[i] + step_adv) / 2 

        for j, (s, e) in enumerate(turns):
            final_scores[i, s:e+1] = fused_adv[j]
            
    final_scores = final_scores * response_mask

    return final_scores, final_scores