# zhzhu
# 先copy一份过来再调整
# TODO: 

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config=None,
    rollout_is_weights=None,
    **kwargs,
):
    """Compute the GSPO policy loss in a format compatible with the current LHTIR trainer.

    Descriptions:
        先基于整条 response 计算 sequence-level importance ratio，
        再把这个 sequence ratio 以 stop-gradient 的方式回注到 token-level loss 

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
            GSPO 中 sequence ratio 被 clipping 的比例。
        ppo_kl: `torch.Tensor`
            基于 token log-prob 差计算的近似 KL 监控量。
        pg_clipfrac_lower: `torch.Tensor`
            为了兼容当前 trainer 保留的指标；GSPO 中这里返回 0。
    """
    if "5" in os.environ.get("RAY_DEBUG_MODE","0"):
        breakpoint()
    assert config is not None, "GSPO loss requires config to provide clipping hyperparameters."

    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)

    # GSPO 使用 sequence-level importance ratio:
    # s_i(theta) = exp((1 / |y_i|) * sum_t(log pi_theta - log pi_old))
    seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
    negative_approx_kl_seq = (negative_approx_kl * response_mask).sum(dim=-1) / seq_lengths

    # 这里是GSPO-token 版本，在理论上等价于GSPO
    # 保留 token log_prob 上的梯度路径，相当于把 sequence-level ratio 广播到 token-level来计算 loss
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, min=-20.0, max=20.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # 是否再乘一个矫正权重， 是多少？怎么设置？ TODO
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
