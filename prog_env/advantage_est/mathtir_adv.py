import os
import torch
import numpy as np
from collections import defaultdict
from verl.trainer.ppo.core_algos import register_adv_est
#--------THREEGOLDCHANGE--------#
import logging
logger = logging.getLogger(__name__)
#--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    '''
    1.新增打印部分:logger.error打印compute_grpo_mathtir_outcome_advantage的开始时间
    '''
    import time
    t0 = time.time()
    logger.error(f"input shape: {token_level_rewards.shape}")
    #--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    t1 = time.time()
    logger.error(f"id2mean and id2std time: {t1 - t0}")
    #--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    '''
    2.新增打印部分:logger.error打印compute_grpo_mathtir_outcome_advantage的结束时间
    '''
    t2 = time.time()
    logger.error(f"turns and turn_rewards time: {t2 - t1}")
    #--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    '''
    3.新增打印部分:logger.error打印step_key2score的开始时间
    '''
    t3 = time.time()
    logger.error(f"step_key2score time: {t3 - t2}")
    #--------THREEGOLDCHANGE--------#
    #TODO:这里改成batch-level的std/mean会不会好一些，或者去除std?
    #TODO:或者看看其他step-level的adv estimator是怎么做的?
    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)
        for j in range(T):
            key = f"{index[i]}-{j}"     
            step_key2score[key].append(disc[j])

    step_mean = {}
    step_std = {}
    #--------THREEGOLDCHANGE--------#
    '''
    4.新增打印部分:logger.error打印step_mean和step_std的开始时间
    '''
    t4 = time.time()
    logger.error(f"step_mean and step_std time: {t4 - t3}")
    #--------THREEGOLDCHANGE--------#
    for key, vals in step_key2score.items():
        vals = torch.stack(vals)
        if len(vals) == 1:
            step_mean[key] = torch.tensor(0.0)
            step_std[key] = torch.tensor(1.0)
        else:
            step_mean[key] = vals.mean()
            step_std[key] = vals.std()

    #--------THREEGOLDCHANGE--------#
    '''
    5.新增打印部分:logger.error打印discounted_adv_list的开始时间
    '''
    t5 = time.time()
    logger.error(f"discounted_adv_list time: {t5 - t4}")
    #--------THREEGOLDCHANGE--------#
    discounted_adv_list = []
    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)

        adv_turn = torch.zeros_like(disc)
        for j in range(T):
            key = f"{index[i]}-{j}"
            adv_turn[j] = (disc[j] - step_mean[key]) / (step_std[key] + epsilon)
        discounted_adv_list.append(adv_turn)
    #--------THREEGOLDCHANGE--------#
    '''
    6.新增打印部分:logger.error打印discounted_adv_list的结束时间
    '''
    t6 = time.time()
    logger.error(f"discounted_adv_list time: {t6 - t5}")
    #--------THREEGOLDCHANGE--------#
    final_scores = torch.zeros(bsz, seq_len)

    for i in range(bsz):
        turns = turns_list[i]
        step_adv = discounted_adv_list[i]         
        fused_adv = (group_adv[i] + step_adv) / 2 

        for j, (s, e) in enumerate(turns):
            final_scores[i, s:e+1] = fused_adv[j]
    #--------THREEGOLDCHANGE--------#
    '''
    7.新增打印部分:logger.error打印final_scores的结束时间
    '''
    t7 = time.time()
    logger.error(f"final_scores time: {t7 - t6}")
    #--------THREEGOLDCHANGE--------#
    final_scores = final_scores * response_mask
    
    return final_scores, final_scores
@register_adv_est("mathtir_fast")
def compute_grpo_mathtir_fast_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    attention_mask: torch.Tensor = None,
    prompts: torch.Tensor = None,
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
        attention_mask: `(torch.Tensor)`
            shape: (bs, response_length)
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
    #--------THREEGOLDCHANGE--------#
    '''
    1.新增打印部分:logger.error打印compute_grpo_mathtir_outcome_advantage的开始时间
    '''
    import time
    t0 = time.time()
    logger.error(f"input shape: {token_level_rewards.shape}")
    #--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    t1 = time.time()
    logger.error(f"id2mean and id2std time: {t1 - t0}")
    #--------THREEGOLDCHANGE--------#
    group_adv = scores
    seq_len = token_level_rewards.size(1)
    
    lam = 0.9
    turns_list = []          
    turn_rewards_list = []   
    for i in range(bsz):
        mask = response_mask[i]
        #--------THREEGOLDCHANGE--------#
        '''
        新增:计算vaild_length
        '''
        prompt_length = prompts[i].shape[-1]
        vaild_length = attention_mask[i][prompt_length:].sum(-1)
        #--------THREEGOLDCHANGE--------#
        rewards = token_level_rewards[i]

        turns = []
        start = None
        #--------THREEGOLDCHANGE--------#
        # for t in range(seq_len):
        for t in range(int(vaild_length)):
            if mask[t] == 1 and start is None:
                start = t
            if mask[t] == 0 and start is not None:
                turns.append((start, t - 1))
                start = None
        if start is not None:
            # turns.append((start, seq_len - 1))
            turns.append((start, vaild_length - 1))
        #--------THREEGOLDCHANGE--------#

        turns_list.append(turns)

        turn_rewards = torch.stack([rewards[e] for (_, e) in turns])#定位每个turn的reward
        turn_rewards_list.append(turn_rewards)
    #--------THREEGOLDCHANGE--------#
    '''
    2.新增打印部分:logger.error打印compute_grpo_mathtir_outcome_advantage的结束时间
    '''
    t2 = time.time()
    logger.error(f"turns and turn_rewards time: {t2 - t1}")
    #--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    '''
    3.新增打印部分:logger.error打印step_key2score的开始时间
    '''
    t3 = time.time()
    logger.error(f"step_key2score time: {t3 - t2}")
    #--------THREEGOLDCHANGE--------#
    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)
        for j in range(T):
            key = f"{index[i]}-{j}"     
            step_key2score[key].append(disc[j])

    step_mean = {}
    step_std = {}
    #--------THREEGOLDCHANGE--------#
    '''
    4.新增打印部分:logger.error打印step_mean和step_std的开始时间
    '''
    t4 = time.time()
    logger.error(f"step_mean and step_std time: {t4 - t3}")
    #--------THREEGOLDCHANGE--------#
    for key, vals in step_key2score.items():
        vals = torch.stack(vals)
        if len(vals) == 1:
            step_mean[key] = torch.tensor(0.0)
            step_std[key] = torch.tensor(1.0)
        else:
            step_mean[key] = vals.mean()
            step_std[key] = vals.std()

    #--------THREEGOLDCHANGE--------#
    '''
    5.新增打印部分:logger.error打印discounted_adv_list的开始时间
    '''
    t5 = time.time()
    logger.error(f"discounted_adv_list time: {t5 - t4}")
    #--------THREEGOLDCHANGE--------#
    discounted_adv_list = []
    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)

        adv_turn = torch.zeros_like(disc)
        for j in range(T):
            key = f"{index[i]}-{j}"
            adv_turn[j] = (disc[j] - step_mean[key]) / (step_std[key] + epsilon)
        discounted_adv_list.append(adv_turn)
    #--------THREEGOLDCHANGE--------#
    '''
    6.新增打印部分:logger.error打印discounted_adv_list的结束时间
    '''
    t6 = time.time()
    logger.error(f"discounted_adv_list time: {t6 - t5}")
    #--------THREEGOLDCHANGE--------#
    final_scores = torch.zeros(bsz, seq_len)

    for i in range(bsz):
        turns = turns_list[i]
        step_adv = discounted_adv_list[i]         
        fused_adv = (group_adv[i] + step_adv) / 2 

        for j, (s, e) in enumerate(turns):
            final_scores[i, s:e+1] = fused_adv[j]
    #--------THREEGOLDCHANGE--------#
    '''
    7.新增打印部分:logger.error打印final_scores的结束时间
    '''
    t7 = time.time()
    logger.error(f"final_scores time: {t7 - t6}")
    #--------THREEGOLDCHANGE--------#
    final_scores = final_scores * response_mask
    if "4" in os.environ.get("RAY_DEBUG_MODE","0"):
        breakpoint()
        fast_final_scores,_ = compute_grpo_mathtir_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=True,
        )
        assert torch.allclose(fast_final_scores, final_scores), f"max diff: {(fast_final_scores - final_scores).abs().max()}"
    return final_scores, final_scores

def _smoke_test_mathtir_adv(bsz: int = 4, seq_len: int = 24, rollout_n: int = 2):
    """Deterministic multi-turn-style masks. Run: python -m prog_env.advantage_est.mathtir_adv --bsz 8 --seq-len 64

    Args:
        bsz: batch size (must be >= 2 and divisible by rollout_n).
        seq_len: padded response length (must be >= 9, the template prefix length).
        rollout_n: rollouts per prompt id (matches training ``rollout.n`` for grouping).
    """
    if bsz < 2 or bsz % rollout_n != 0:
        raise ValueError(f"bsz must be >= 2 and divisible by rollout_n; got bsz={bsz}, rollout_n={rollout_n}")
    if seq_len < 9:
        raise ValueError(f"seq_len must be >= 9 (template length); got {seq_len}")

    num_groups = bsz // rollout_n
    index = np.array([f"p{g}" for g in range(num_groups) for _ in range(rollout_n)], dtype=object)

    # Four fixed turn patterns (length 9); cycle by row index
    _pat = [
        [1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0],
    ]
    prompt_len = 1024
    response_mask = torch.zeros(bsz, seq_len, dtype=torch.float32)
    prompts = torch.zeros(bsz, prompt_len, dtype=torch.float32)
    attention_mask = torch.zeros(bsz, seq_len+prompt_len, dtype=torch.float32)
    for i in range(bsz):
        prefix = _pat[i % len(_pat)]
        row = prefix + [0] * (seq_len - len(prefix))
        response_mask[i] = torch.tensor(row, dtype=torch.float32)
        attention_mask[i] = torch.cat([torch.ones(prompt_len+len(prefix)), torch.zeros(seq_len-len(prefix))], dim=-1)
    token_level_rewards = torch.zeros(bsz, seq_len, dtype=torch.float32)
    # Put scalar reward on last token of each assistant segment (end index e per turn)
    for i in range(bsz):
        m = response_mask[i].int()
        starts_ends = []
        start = None
        for t in range(seq_len):
            if m[t] == 1 and start is None:
                start = t
            if m[t] == 0 and start is not None:
                starts_ends.append((start, t - 1))
                start = None
        if start is not None:
            starts_ends.append((start, seq_len - 1))
        for _, e in starts_ends:
            token_level_rewards[i, e] = float(i + 1)
    adv, returns = compute_grpo_mathtir_outcome_advantage(
        token_level_rewards,
        response_mask,
        index,
        norm_adv_by_std_in_grpo=True,
    )
    assert adv.shape == (bsz, seq_len)
    assert returns.shape == (bsz, seq_len)
    assert torch.isfinite(adv).all(), "advantages must be finite"
    assert torch.allclose(adv * (1 - response_mask), torch.zeros_like(adv))
    assert torch.allclose(returns * (1 - response_mask), torch.zeros_like(returns))
    
    print("mathtir_adv smoke test OK:", adv.shape)
    adv2, returns2 = compute_grpo_mathtir_fast_outcome_advantage(
        token_level_rewards,
        response_mask,
        index,
        norm_adv_by_std_in_grpo=True,
        attention_mask=attention_mask,
        prompts=prompts,
    )
    print("mathtir_fast_adv shape:", adv2.shape)
    assert torch.allclose(adv, adv2), f"max diff: {(adv - adv2).abs().max()}"
    assert ((returns-returns2)==torch.zeros_like(returns)).all()
    print("mathtir_fast_adv smoke test OK:", adv.shape)
@register_adv_est("mathtir_fast_reverse")
def compute_grpo_mathtir_fast_reverse_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    attention_mask: torch.Tensor = None,
    prompts: torch.Tensor = None,
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
        attention_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    Description:
        除了vaild_length的计算方法，还引入了advantage normalization mode    
    """
    if "4" in os.environ.get("RAY_DEBUG_MODE","0"):
        breakpoint()
    scores = token_level_rewards.sum(dim=-1)
    config = kwargs.get("config", {})
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    #--------THREEGOLDCHANGE--------#
    '''
    1.新增打印部分:logger.error打印compute_grpo_mathtir_outcome_advantage的开始时间
    '''
    import time
    t0 = time.time()
    logger.error(f"input shape: {token_level_rewards.shape}")
    #--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    t1 = time.time()
    logger.error(f"id2mean and id2std time: {t1 - t0}")
    #--------THREEGOLDCHANGE--------#
    group_adv = scores
    seq_len = token_level_rewards.size(1)
    #TODO:是否要从config中传入lam参数
    # lam = config.get("lam", 0.9)
    
    lam = 0.9
    turns_list = []          
    turn_rewards_list = []   
    for i in range(bsz):
        mask = response_mask[i]
        #--------THREEGOLDCHANGE--------#
        '''
        新增:计算vaild_length
        '''
        prompt_length = prompts[i].shape[-1]
        vaild_length = attention_mask[i][prompt_length:].sum(-1)
        #--------THREEGOLDCHANGE--------#
        rewards = token_level_rewards[i]

        turns = []
        start = None
        #--------THREEGOLDCHANGE--------#
        # for t in range(seq_len):
        for t in range(int(vaild_length)):
            if mask[t] == 1 and start is None:
                start = t
            if mask[t] == 0 and start is not None:
                turns.append((start, t - 1))
                start = None
        if start is not None:
            # turns.append((start, seq_len - 1))
            turns.append((start, vaild_length - 1))
        #--------THREEGOLDCHANGE--------#

        turns_list.append(turns)

        turn_rewards = torch.stack([rewards[e] for (_, e) in turns])#定位每个turn的reward
        turn_rewards_list.append(turn_rewards)
    #--------THREEGOLDCHANGE--------#
    '''
    2.新增打印部分:logger.error打印compute_grpo_mathtir_outcome_advantage的结束时间
    '''
    t2 = time.time()
    logger.error(f"turns and turn_rewards time: {t2 - t1}")
    #--------THREEGOLDCHANGE--------#
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
    #--------THREEGOLDCHANGE--------#
    '''
    3.新增打印部分:logger.error打印step_key2score的开始时间
    #TODO:是否要有-single分组
    '''
    t3 = time.time()
    logger.error(f"step_key2score time: {t3 - t2}")
    mode = config.get("step_adv_mode", "reverse")
    def make_turn_key(prompt_id, j, T, mode="reverse"):
        if mode == "reverse":
            return f"{prompt_id}-mid_rev{T - 1 - j}"
        if mode == "forward":
            return f"{prompt_id}-mid_fwd{j}"
        #否则就是hybird:first和final保持原样
        # if T == 1:
        #     return f"{prompt_id}-single"
        if j == T - 1:
            return f"{prompt_id}-final"
        if j == 0:
            return f"{prompt_id}-first"

        # 中间 turn：按距离 final 的倒序位置对齐
        if "reverse" in mode:
            rev_mid = T - 1 - j   # 1 表示 final 前一 turn，2 表示 final 前二 turn
            return f"{prompt_id}-mid_rev{rev_mid}"
        fwd_mid = j
        return f"{prompt_id}-mid_fwd{fwd_mid}"
    # def make_turn_key(prompt_id, j, T, mode="hybrid"):
    #     if T == 1:
    #         return f"{prompt_id}-single"

    #     if mode == "forward":
    #         return f"{prompt_id}-fwd{j}"

    #     if mode == "reverse":
    #         return f"{prompt_id}-rev{T - 1 - j}"

    #     if mode == "hybrid":
    #         if j == 0:
    #             return f"{prompt_id}-first"
    #         if j == T - 1:
    #             return f"{prompt_id}-final"

    #         rev_mid = T - 1 - j
    #         return f"{prompt_id}-mid_rev{rev_mid}"

    #     raise ValueError(f"unknown mode: {mode}")
    
    #--------THREEGOLDCHANGE--------#
    '''
    #TODO:传入mode参数
    '''
    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)
        for j in range(T):
            key = make_turn_key(index[i], j, T, mode=mode)     
            step_key2score[key].append(disc[j])

    step_mean = {}
    step_std = {}
    #--------THREEGOLDCHANGE--------#
    '''
    4.新增打印部分:logger.error打印step_mean和step_std的开始时间
    '''
    t4 = time.time()
    logger.error(f"step_mean and step_std time: {t4 - t3}")
    #--------THREEGOLDCHANGE--------#
    for key, vals in step_key2score.items():
        vals = torch.stack(vals)
        if len(vals) == 1:
            step_mean[key] = torch.tensor(0.0)
            step_std[key] = torch.tensor(1.0)
        else:
            step_mean[key] = vals.mean()
            step_std[key] = vals.std()

    #--------THREEGOLDCHANGE--------#
    '''
    5.新增打印部分:logger.error打印discounted_adv_list的开始时间
    '''
    t5 = time.time()
    logger.error(f"discounted_adv_list time: {t5 - t4}")
    #--------THREEGOLDCHANGE--------#
    discounted_adv_list = []
    for i in range(bsz):
        disc = discounted_rewards_list[i]
        T = len(disc)

        adv_turn = torch.zeros_like(disc)
        for j in range(T):
            key = make_turn_key(index[i], j, T, mode=mode)
            if norm_adv_by_std_in_grpo:
                adv_turn[j] = (disc[j] - step_mean[key]) / (step_std[key] + epsilon)
            else:
                adv_turn[j] = disc[j] - step_mean[key]
        discounted_adv_list.append(adv_turn)
    num2count = defaultdict(int)
    for key in step_key2score:
        num2count[len(step_key2score[key])] += 1
    logger.error(f"step group num2count: {sorted(num2count.items(),key=lambda x: x[1])}")
    #--------THREEGOLDCHANGE--------#
    '''
    6.新增打印部分:logger.error打印discounted_adv_list的结束时间
    '''
    t6 = time.time()
    logger.error(f"discounted_adv_list time: {t6 - t5}")
    #--------THREEGOLDCHANGE--------#
    final_scores = torch.zeros(bsz, seq_len)

    for i in range(bsz):
        turns = turns_list[i]
        step_adv = discounted_adv_list[i]         
        fused_adv = (group_adv[i] + step_adv) / 2 

        for j, (s, e) in enumerate(turns):
            final_scores[i, s:e+1] = fused_adv[j]
    #--------THREEGOLDCHANGE--------#
    '''
    7.新增打印部分:logger.error打印final_scores的结束时间
    '''
    t7 = time.time()
    logger.error(f"final_scores time: {t7 - t6}")
    #--------THREEGOLDCHANGE--------#
    final_scores = final_scores * response_mask
    if "4" in os.environ.get("RAY_DEBUG_MODE","0"):
        breakpoint()
        fast_final_scores,_ = compute_grpo_mathtir_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=True,
        )
        assert torch.allclose(fast_final_scores, final_scores), f"max diff: {(fast_final_scores - final_scores).abs().max()}"
    return final_scores, final_scores

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Smoke test for mathtir advantage.")
    p.add_argument("--bsz", type=int, default=4, help="batch size (default: 4)")
    p.add_argument("--seq-len", type=int, default=24, help="padded response length (default: 24)")
    p.add_argument(
        "--rollout-n",
        type=int,
        default=2,
        help="rollouts per prompt, bsz must be divisible by this (default: 2)",
    )
    args = p.parse_args()
    _smoke_test_mathtir_adv(bsz=args.bsz, seq_len=args.seq_len, rollout_n=args.rollout_n)