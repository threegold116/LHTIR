import torch
import os
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


# def _high_entropy_mask_from_nll(old_log_prob: torch.Tensor, response_mask: torch.Tensor, quantile: float) -> torch.Tensor:
#     """用 -log pi(a) 作 NLL 代理：高于有效 token 分位数的视为高熵（更不确定）。"""
#     nll = -old_log_prob
#     valid = response_mask > 0
#     flat = nll.masked_select(valid)
#     if flat.numel() == 0:
#         return torch.zeros_like(response_mask, dtype=torch.bool)
#     q = torch.quantile(flat.float(), quantile)
#     return (nll > q) & valid


@register_policy_loss("aspo")
def compute_policy_loss_aspo(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    entropy,
    loss_agg_mode: str = "token-mean", 
    config=None,
    **kwargs
):
    if "5" in os.environ.get("RAY_DEBUG_MODE","0"):
        breakpoint()
    negative_clip_ratio_c = config.get("negative_clip_ratio_c", 3.0)
    positive_clip_ratio_c = config.get("positive_clip_ratio_c", 3.0)
    use_dynamic_clip = config.get("use_dynamic_clip", False)
    assert negative_clip_ratio_c > 1.0, "The negative_clip_ratio_c should be greater than 1.0," + f" but get the value: {negative_clip_ratio_c}."
    assert positive_clip_ratio_c > 1.0, "The positive_clip_ratio_c should be greater than 1.0," + f" but get the value: {positive_clip_ratio_c}."
    
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # 根据 use_dynamic_clip 决定是否使用熵自适应的 clip 范围
    if use_dynamic_clip:
        # 动态 clip：高熵用宽松范围，低熵用严格范围
        negative_low_entropy_clip_ratio_low = config.get("negative_low_entropy_clip_ratio_low", 0.2)
        negative_high_entropy_clip_ratio_low = config.get("negative_high_entropy_clip_ratio_low", 0.4)
        positive_low_entropy_clip_ratio_high = config.get("positive_low_entropy_clip_ratio_high", 0.2)
        positive_high_entropy_clip_ratio_high = config.get("positive_high_entropy_clip_ratio_high", 0.4)
        # 计算高熵mask
        with torch.no_grad():
            token_entropy_quantile = config.get("token_entropy_quantile", 0.8)
            masked_entropy = torch.where(response_mask.bool(), entropy.detach(), torch.nan)  # (bsz, response_length)
            q80 = torch.nanquantile(masked_entropy, q=token_entropy_quantile, dim=-1, keepdim=True)  # (bsz, 1)
            high_entropy_mask = (masked_entropy <= q80) & response_mask # only low entropy token is True
            low_entropy_mask = (masked_entropy > q80) & response_mask #  only high entropy token is True

        negative_clip_ratio = torch.where(
            high_entropy_mask,
            torch.clamp(ratio, min=1-negative_low_entropy_clip_ratio_low, max=None),
            torch.clamp(ratio, min=1-negative_high_entropy_clip_ratio_low, max=None)
        )
        positive_clip_ratio = torch.where(
            high_entropy_mask,
            torch.clamp(ratio, min=None, max=1+positive_low_entropy_clip_ratio_high),
            torch.clamp(ratio, min=None, max=1+positive_high_entropy_clip_ratio_high)
        )
    else:
        # 静态 clip：直接用 low_entropy 对应的参数作为固定 ε
        clip_ratio_low = config.get("clip_ratio_low", 0.2)
        clip_ratio_high = config.get("clip_ratio_high", 0.2)
        negative_clip_ratio = torch.clamp(ratio, min=1-clip_ratio_low, max=None)
        positive_clip_ratio = torch.clamp(ratio, min=None, max=1+clip_ratio_high)

    # 根据 advantage 符号选择对应的 clip ratio，用于统计监控
    # A < 0 → 用 negative_clip_ratio（有下界限制）
    # A > 0 → 用 positive_clip_ratio（有上界限制）
    clip_ratio = torch.where(advantages < 0, negative_clip_ratio, positive_clip_ratio)

    # 统计 ratio 被上界截断的比例（ratio > clip_ratio，说明新策略比旧策略激进太多）
    pg_clipfrac_upper = verl_F.masked_mean(torch.gt(ratio, clip_ratio).float(), response_mask)
    # 统计 ratio 被下界截断的比例（ratio < clip_ratio，说明新策略比旧策略退步太多）
    pg_clipfrac_lower = verl_F.masked_mean(torch.lt(ratio, clip_ratio).float(), response_mask)

    # ============================================================
    # A < 0 的 loss 计算
    # ============================================================

    # 标准 clip loss：直接用 clip 后的 ratio 计算惩罚
    # A < 0，negative_clip_ratio 有下界，防止 ratio 过小时惩罚不足
    negative_pg_losses_clip = -advantages * negative_clip_ratio

    # --- Dual Clip ---
    # 对 negative_clip_ratio 再加一个上界 c
    # 防止 ratio 虽然被下界夹住了，但值仍然偏大导致惩罚依然不足
    negative_dual_clip_ratio = torch.clamp(negative_clip_ratio, min=None, max=negative_clip_ratio_c)

    # 触发 dual clip 的位置：negative_clip_ratio > c（说明第一次 clip 后值还是太大）
    negative_clipped_mask = torch.gt(negative_clip_ratio, negative_dual_clip_ratio)

    # 统计 A<0 的样本中触发 dual clip 的比例（分母只算 A<0 的位置）
    negative_pg_clipfrac_dual = verl_F.masked_mean(
        negative_clipped_mask.float(), 
        response_mask & (advantages < 0)
    )

    # dual clip 触发时的 loss：用 c 替换 clip ratio，梯度只通过 log_prob 传播
    # detach() 让 c 作为常数，不参与梯度计算
    negative_pg_losses_dual = -advantages * negative_dual_clip_ratio.detach() * log_prob

    # 最终 loss：触发 dual clip 用更强惩罚，否则用标准 clip loss
    negative_pg_losses = torch.where(
        negative_clipped_mask,
        negative_pg_losses_dual,  # 触发 dual clip：强制更大惩罚
        negative_pg_losses_clip   # 未触发：标准 clip loss
    )

    # ============================================================
    # A > 0 的 loss 计算
    # ============================================================

    # 标准 clip loss：x / x.detach() 保留梯度通路但数值为 1
    # 整体等价于 -A / positive_clip_ratio，对 ratio 过大的奖励做缩减
    positive_pg_losses_clip = -advantages * (positive_clip_ratio / positive_clip_ratio.detach()) / positive_clip_ratio.detach()

    # --- Dual Clip ---
    # 对 1/positive_clip_ratio 加上界 c（与 negative 对称）
    # 防止 ratio 过大时 1/clip_ratio 过小，奖励被过度压制
    positive_dual_clip_ratio = torch.clamp(1/positive_clip_ratio, min=None, max=positive_clip_ratio_c)

    # 触发 dual clip 的位置：1/positive_clip_ratio > c（ratio 过大）
    positive_clipped_mask = torch.gt(1/positive_clip_ratio, positive_dual_clip_ratio)

    # 统计 A>0 的样本中触发 dual clip 的比例（分母只算 A>0 的位置）
    positive_pg_clipfrac_dual = verl_F.masked_mean(
        positive_clipped_mask.float(), 
        response_mask & (advantages > 0)
    )

    # dual clip 触发时的 loss：用 c 替换，梯度只通过 log_prob 传播
    positive_pg_losses_dual = -advantages * positive_dual_clip_ratio.detach() * log_prob

    # 最终 loss：触发 dual clip 用受限奖励，否则用标准 clip loss
    positive_pg_losses = torch.where(
        positive_clipped_mask,
        positive_pg_losses_dual,  # 触发 dual clip：限制过度奖励
        positive_pg_losses_clip   # 未触发：标准 clip loss
    )

    # ============================================================
    # 汇总最终 loss
    # ============================================================

    # 根据 advantage 符号合并两部分 loss
    pg_losses = torch.where(advantages < 0, negative_pg_losses, positive_pg_losses)

    # 对整个 batch 做聚合（token-mean 或其他模式）
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac_upper, ppo_kl, pg_clipfrac_lower

