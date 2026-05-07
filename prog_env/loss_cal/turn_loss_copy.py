import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
import os

@register_policy_loss("gtpo_test")
def compute_policy_loss_gtpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config=None,
    rollout_is_weights=None,
    attention_mask: torch.Tensor = None,
    prompts: torch.Tensor = None,
    **kwargs,
):
    """Compute GTPO policy loss (turn-level importance ratio).

    Descriptions:
        先基于 turn 计算 turn-level importance ratio，
        再把这个 turn ratio 以 stop-gradient 的方式回注到 token-level loss。
        turn 切分规则与 ``mathtir_fast`` 对齐：
        - 先根据 ``attention_mask`` 与 ``prompt`` 长度得到 response 的 ``valid_length``；
        - 再在 ``response_mask`` 上把连续 1 段视作一个 turn。

        与其他保持一致，本函数返回：
        - `pg_loss`
        - `pg_clipfrac`
        - `ppo_kl`
        - `pg_clipfrac_lower`

    Args:
        old_log_prob: `(torch.Tensor)`
            旧策略在已采样 token 上的对数概率，shape 为 `(bs, response_length)`。
        log_prob: `(torch.Tensor)`
            新策略在已采样 token 上的对数概率，shape 为 `(bs, response_length)`。
        advantages: `(torch.Tensor)`
            token 级 advantage，shape 为 `(bs, response_length)`。
        response_mask: `(torch.Tensor)`
            response 有效 token mask，shape 为 `(bs, response_length)`。
        loss_agg_mode: `(str)`
            loss 聚合模式。GSPO 通常推荐使用 `"seq-mean-token-mean"`。
        config:
            训练算法配置对象。这里会读取 `clip_ratio`、`clip_ratio_low`、`clip_ratio_high`。
        rollout_is_weights:
            可选的 rollout importance sampling 权重。如果提供，则对 token loss 再乘一层校正权重。
        **kwargs:
            预留参数。

    Returns:
        pg_loss: `torch.Tensor`
            聚合后的 policy gradient loss 标量。
        pg_clipfrac: `torch.Tensor`
            GTPO 中 sequence ratio 被 clipping 的比例。
        ppo_kl: `torch.Tensor`
            基于 token log-prob 差计算的近似 KL 监控量。
        pg_clipfrac_lower: `torch.Tensor`
            为了兼容当前 trainer 保留的指标；GTPO 中这里返回 0。
    """
    if "5" in os.environ.get("RAY_DEBUG_MODE","0"):
        breakpoint()
    assert config is not None, "GTPO loss requires config to provide clipping hyperparameters."

    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)

    # GTPO turn-level ratio:
    # 1) 使用与 mathtir_fast 一致的 valid_length 规则切分 turn；
    # 2) 每个 turn 上用 mean(delta log-prob) 构造几何均值 importance ratio；
    # 3) 再广播回 token 以复用现有 token-level PPO clipping。
    bsz, response_len = response_mask.shape
    token_turn_mean_kl = torch.zeros_like(negative_approx_kl)
    # 兜底读取 input_ids：当 prompts 缺失时，用 input_ids 长度反推 prompt 长度。
    input_ids = kwargs.get("input_ids", None)
    if prompts is not None:
        prompt_lengths = [int(prompts[i].shape[-1]) for i in range(bsz)]
    elif input_ids is not None:
        prompt_len = max(int(input_ids.shape[-1] - response_len), 0)
        prompt_lengths = [prompt_len for _ in range(bsz)]
    else:
        prompt_lengths = [0 for _ in range(bsz)]

    for i in range(bsz):
        if attention_mask is not None:
            prompt_len = prompt_lengths[i]
            attn = attention_mask[i]
            # 与 mathtir_fast 对齐：在 [prompt_len, prompt_len + response_len) 上统计有效 response 长度。
            if prompt_len < attn.shape[-1]:
                valid_length = int(attn[prompt_len : prompt_len + response_len].sum().item())
            else:
                valid_length = response_len
        else:
            valid_length = response_len
        valid_length = max(0, min(valid_length, response_len))

        turns = []
        start = None
        # 连续 1 段 = 一个 turn；0 代表 observation/tool/padding，不参与 turn ratio。
        for t in range(valid_length):
            if response_mask[i, t] == 1 and start is None:
                start = t
            if response_mask[i, t] == 0 and start is not None:
                turns.append((start, t - 1))
                start = None
        if start is not None:
            turns.append((start, valid_length - 1))

        # 在每个 turn 内计算 mean(delta log-prob)，并广播回 turn 内 token。
        for s, e in turns:
            turn_kl_mean = negative_approx_kl[i, s : e + 1].mean()
            token_turn_mean_kl[i, s : e + 1] = turn_kl_mean

    # stop-gradient 设计：
    # - 数值上使用 turn 聚合后的 log-ratio；
    # - 梯度路径仍来自 token-level log_prob，保持与现有 token loss 兼容。
    log_turn_importance_ratio = log_prob - log_prob.detach() + token_turn_mean_kl.detach()
    log_turn_importance_ratio = torch.clamp(log_turn_importance_ratio, min=-20.0, max=20.0)
    turn_importance_ratio = torch.exp(log_turn_importance_ratio)

    pg_losses1 = -advantages * turn_importance_ratio
    pg_losses2 = -advantages * torch.clamp(turn_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # 是否再乘一个矫正权重， 是多少？怎么设置？ TODO
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


class _SmokeGTPOConfig:
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2


def _build_smoke_test_inputs():
    prompt_len = 3
    response_len = 10

    old_log_prob = torch.tensor(
        [
            [-0.90, -0.70, -1.15, -1.20, -0.80, -0.60, -0.95, -1.05, -0.75, -1.10],
            [-0.50, -1.00, -0.75, -0.65, -0.70, -1.10, -0.80, -1.00, -0.90, -1.20],
        ],
        dtype=torch.float32,
    )
    log_prob = torch.tensor(
        [
            [-0.82, -0.64, -1.15, -1.20, -0.72, -0.58, -0.88, -1.05, -0.70, -1.10],
            [-0.55, -1.00, -0.68, -0.60, -0.62, -1.10, -0.88, -1.00, -0.90, -1.20],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    advantages = torch.tensor(
        [
            [0.4, 0.4, 0.0, 0.0, 0.4, 0.4, 0.4, 0.0, 0.4, 0.0],
            [-0.4, 0.0, -0.4, -0.4, -0.4, 0.0, -0.4, 0.0, -0.4, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    prompts = torch.tensor(
        [
            [101, 102, 103],
            [201, 202, 203],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    input_ids = torch.cat([prompts, torch.zeros((2, response_len), dtype=torch.long)], dim=-1)

    return {
        "old_log_prob": old_log_prob,
        "log_prob": log_prob,
        "advantages": advantages,
        "response_mask": response_mask,
        "attention_mask": attention_mask,
        "prompts": prompts,
        "input_ids": input_ids,
        "config": _SmokeGTPOConfig(),
    }


def _run_smoke_test_compute_policy_loss_gtpo():
    torch.manual_seed(0)
    inputs = _build_smoke_test_inputs()
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_gtpo(
        old_log_prob=inputs["old_log_prob"],
        log_prob=inputs["log_prob"],
        advantages=inputs["advantages"],
        response_mask=inputs["response_mask"],
        config=inputs["config"],
        attention_mask=inputs["attention_mask"],
        prompts=inputs["prompts"],
        input_ids=inputs["input_ids"],
    )
    pg_loss.backward()

    print("GTPO smoke test")
    print(f"pg_loss={pg_loss.item():.6f}")
    print(f"pg_clipfrac={pg_clipfrac.item():.6f}")
    print(f"ppo_kl={ppo_kl.item():.6f}")
    print(f"pg_clipfrac_lower={pg_clipfrac_lower.item():.6f}")
    print(f"log_prob_grad_norm={inputs['log_prob'].grad.norm().item():.6f}")


if __name__ == "__main__":
    _run_smoke_test_compute_policy_loss_gtpo()
