import numpy as np
import torch
from collections import defaultdict
import os
from verl.trainer.ppo.core_algos import register_adv_est


def _compute_group_advantages(
    scores: torch.Tensor,
    index: np.ndarray,
    epsilon: float,
    norm_adv_by_std_in_grpo: bool,
) -> torch.Tensor:
    id2score = defaultdict(list)
    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])

    id2mean = {}
    id2std = {}
    for idx, vals in id2score.items():
        if len(vals) == 1:
            id2mean[idx] = scores.new_tensor(0.0)
            id2std[idx] = scores.new_tensor(1.0)
        else:
            stacked = torch.stack(vals)
            id2mean[idx] = stacked.mean()
            id2std[idx] = stacked.std()

    group_adv = scores.clone()
    for i in range(scores.shape[0]):
        if norm_adv_by_std_in_grpo:
            group_adv[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            group_adv[i] = scores[i] - id2mean[index[i]]
    return group_adv.unsqueeze(-1)


def _extract_turns_and_rewards(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    prompts: torch.Tensor,
):
    bsz = token_level_rewards.shape[0]
    turns_list = []
    turn_rewards_list = []

    for i in range(bsz):
        mask = response_mask[i]
        rewards = token_level_rewards[i]
        prompt_length = prompts[i].shape[-1]
        valid_length = int(attention_mask[i][prompt_length:].sum(-1).item())

        if int(mask.sum().item()) == 0 or valid_length <= 0:
            turns_list.append([])
            turn_rewards_list.append(rewards.new_empty(0))
            continue

        turns = []
        start = None
        for t in range(valid_length):
            if mask[t] == 1 and start is None:
                start = t
            if mask[t] == 0 and start is not None:
                turns.append((start, t - 1))
                start = None
        if start is not None:
            turns.append((start, valid_length - 1))

        turns_list.append(turns)
        if len(turns) == 0:
            turn_rewards_list.append(rewards.new_empty(0))
        else:
            turn_rewards_list.append(torch.stack([rewards[e] for (_, e) in turns]))

    return turns_list, turn_rewards_list


def _discount_turn_rewards(turn_rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    discounted = torch.zeros_like(turn_rewards)
    acc = turn_rewards.new_tensor(0.0)
    for t in reversed(range(len(turn_rewards))):
        acc = turn_rewards[t] + gamma * acc
        discounted[t] = acc
    return discounted


def _make_turn_key(prompt_id, j: int, total_turns: int, mode: str = "forward") -> str:
    """Match the grouping-key style used in mathtir_fast_reverse."""
    if mode == "reverse":
        return f"{prompt_id}-mid_rev{total_turns - 1 - j}"
    if mode == "forward":
        return f"{prompt_id}-mid_fwd{j}"
    if mode == "group":
        return f"{prompt_id}"

    # hybrid-like behavior: keep first/final explicit, align middle turns.
    if j == total_turns - 1:
        return f"{prompt_id}-final"
    if j == 0:
        return f"{prompt_id}-first"
    if "reverse" in mode:
        return f"{prompt_id}-mid_rev{total_turns - 1 - j}"
    return f"{prompt_id}-mid_fwd{j}"


@register_adv_est("gagpo_fast")
def compute_gagpo_fast_turn_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    attention_mask: torch.Tensor = None,
    prompts: torch.Tensor = None,
    **kwargs,
):
    if attention_mask is None or prompts is None:
        raise ValueError("gagpo_fast requires attention_mask and prompts")
    if "5" in os.environ.get("RAY_DEBUG_MODE","0"):
        breakpoint()
    config = kwargs.get("config", {})
    #FIXME:通过config获取gamma和lam
    # gamma = float(config.get("gamma", 0.9))
    # lam = float(config.get("lam", 0.8))
    gamma = 0.9
    lam = 0.8
    step_adv_mode = config.get("step_adv_mode", "forward")

    scores = token_level_rewards.sum(dim=-1)
    group_adv = _compute_group_advantages(
        scores=scores,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )

    turns_list, turn_rewards_list = _extract_turns_and_rewards(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        attention_mask=attention_mask,
        prompts=prompts,
    )
    discounted_rewards_list = [_discount_turn_rewards(turn_rewards, gamma) for turn_rewards in turn_rewards_list]

    value_by_key = defaultdict(list)
    for i in range(len(discounted_rewards_list)):
        disc = discounted_rewards_list[i]
        total_turns = len(disc)
        for j in range(total_turns):
            key = _make_turn_key(index[i], j, total_turns, mode=step_adv_mode)
            value_by_key[key].append(disc[j])
    value_by_key = {key: torch.stack(vals).mean() for key, vals in value_by_key.items()}

    bsz, seq_len = token_level_rewards.shape
    final_scores = torch.zeros(bsz, seq_len, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
    for i in range(bsz):
        turns = turns_list[i]
        if len(turns) == 0:
            continue

        turn_rewards = turn_rewards_list[i]
        num_turns = len(turn_rewards)
        turn_values = torch.stack(
            [
                value_by_key.get(
                    _make_turn_key(index[i], j, num_turns, mode=step_adv_mode),
                    token_level_rewards.new_tensor(0.0),
                )
                for j in range(num_turns)
            ]
        )

        deltas = torch.zeros_like(turn_rewards)
        for j in range(num_turns):
            if j + 1 < num_turns:
                next_key = _make_turn_key(index[i], j + 1, num_turns, mode=step_adv_mode)
                next_v = value_by_key.get(next_key, turn_rewards.new_tensor(0.0))
            else:
                next_v = turn_rewards.new_tensor(0.0)
            deltas[j] = turn_rewards[j] + gamma * next_v - turn_values[j]

        turn_adv = torch.zeros_like(turn_rewards)
        gae = turn_rewards.new_tensor(0.0)
        for j in reversed(range(num_turns)):
            gae = deltas[j] + gamma * lam * gae
            turn_adv[j] = gae

        fused_adv = (group_adv[i] + turn_adv) / 2
        for j, (s, e) in enumerate(turns):
            final_scores[i, s : e + 1] = fused_adv[j]

    final_scores = final_scores * response_mask
    return final_scores, final_scores


def smoke_test_gagpo_fast(bsz: int = 4, seq_len: int = 24, rollout_n: int = 2):
    if bsz < 2 or bsz % rollout_n != 0:
        raise ValueError(f"bsz must be >= 2 and divisible by rollout_n; got bsz={bsz}, rollout_n={rollout_n}")
    if seq_len < 9:
        raise ValueError(f"seq_len must be >= 9 (template length); got {seq_len}")

    num_groups = bsz // rollout_n
    index = np.array([f"p{g}" for g in range(num_groups) for _ in range(rollout_n)], dtype=object)
    patterns = [
        [1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0],
    ]
    prompt_len = 1024
    response_mask = torch.zeros(bsz, seq_len, dtype=torch.float32)
    prompts = torch.zeros(bsz, prompt_len, dtype=torch.float32)
    attention_mask = torch.zeros(bsz, seq_len + prompt_len, dtype=torch.float32)
    token_level_rewards = torch.zeros(bsz, seq_len, dtype=torch.float32)

    for i in range(bsz):
        prefix = patterns[i % len(patterns)]
        row = prefix + [0] * (seq_len - len(prefix))
        response_mask[i] = torch.tensor(row, dtype=torch.float32)
        attention_mask[i] = torch.cat([torch.ones(prompt_len + len(prefix)), torch.zeros(seq_len - len(prefix))], dim=-1)

        start = None
        for t in range(seq_len):
            if int(response_mask[i, t].item()) == 1 and start is None:
                start = t
            if int(response_mask[i, t].item()) == 0 and start is not None:
                token_level_rewards[i, t - 1] = float(i + 1)
                start = None
        if start is not None:
            token_level_rewards[i, seq_len - 1] = float(i + 1)

    adv, ret = compute_gagpo_fast_turn_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        attention_mask=attention_mask,
        prompts=prompts,
        norm_adv_by_std_in_grpo=True,
    )
    assert adv.shape == (bsz, seq_len)
    assert ret.shape == (bsz, seq_len)
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv * (1 - response_mask), torch.zeros_like(adv))

    for i in range(bsz):
        start = None
        for t in range(seq_len):
            cur = int(response_mask[i, t].item())
            if cur == 1 and start is None:
                start = t
            if cur == 0 and start is not None:
                assert torch.allclose(adv[i, start:t], adv[i, start].expand(t - start))
                start = None
        if start is not None:
            assert torch.allclose(adv[i, start:seq_len], adv[i, start].expand(seq_len - start))

    print("gagpo_fast smoke test OK:", adv.shape)


if __name__ == "__main__":
    smoke_test_gagpo_fast()
