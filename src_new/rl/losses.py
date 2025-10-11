"""Loss utilities for manual GRPO training."""

from __future__ import annotations

from typing import Optional

import torch


def compute_kl(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """Return TRL-style per-token KL proxy: exp(ref - policy) - (ref - policy) - 1."""

    if ref_logps.shape != policy_logps.shape:
        raise ValueError("policy_logps and ref_logps must share the same shape")
    diff = ref_logps - policy_logps
    return torch.exp(diff) - diff - 1.0


def compute_grpo_loss(
    logps: torch.Tensor,
    old_logps: Optional[torch.Tensor],
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    epsilon_low: float,
    epsilon_high: float,
    loss_type: str = "grpo",
    beta: float = 0.0,
    per_token_kl: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the GRPO objective with clipping and optional KL penalty."""

    if completion_mask.dim() != 2:
        raise ValueError("completion_mask must be 2D")
    if logps.shape != completion_mask.shape:
        raise ValueError("logps and completion_mask must have identical shapes")
    if old_logps is not None and old_logps.shape != logps.shape:
        raise ValueError("old_logps must match logps shape")

    batch_size = logps.size(0)
    if advantages.dim() == 1:
        adv = advantages.view(batch_size, 1)
    else:
        adv = advantages

    # Compute in float32 for stability
    logps_f = logps.float()
    adv_f = adv.float().expand_as(logps_f)
    mask = completion_mask.to(dtype=logps_f.dtype)
    denom = mask.sum().clamp_min(1.0)

    if mask.sum() == 0:
        raise ValueError("completion_mask contains no valid tokens")

    if old_logps is None:
        diff = logps_f
    else:
        diff = logps_f - old_logps.float()

    # Clamp difference to avoid overflow in exp
    diff = diff.clamp(min=-50.0, max=50.0)
    ratio = torch.exp(diff)

    clipped_ratio = torch.clamp(
        ratio, 1.0 - float(epsilon_low), 1.0 + float(epsilon_high)
    )

    if str(loss_type).lower() != "grpo":
        raise ValueError(f"Only 'grpo' loss_type is supported; got '{loss_type}'")

    surrogate_a = ratio * adv_f
    surrogate_b = clipped_ratio * adv_f
    policy_loss = -torch.minimum(surrogate_a, surrogate_b)

    policy_loss = (policy_loss * mask).sum() / denom

    if beta > 0.0 and per_token_kl is not None:
        if per_token_kl.shape != logps.shape:
            raise ValueError("per_token_kl must match logps shape")
        kl_term = (per_token_kl.float() * mask).sum() / denom
        policy_loss = policy_loss + float(beta) * kl_term

    return policy_loss


__all__ = ["compute_kl", "compute_grpo_loss"]
