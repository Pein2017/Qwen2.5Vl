#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from src_post.tf.teacher_forcing import compute_logprobs


def _fail_if_nan_inf(t: Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} contains non-finite values (NaN/Inf). Shape={tuple(t.shape)}")


def build_reply_mask(prompt_len: int, logits: Tensor) -> Tensor:
    """Build a boolean mask over per-step logits to select reply tokens only.

    Args:
        prompt_len: Number of prompt tokens (including BOS), as returned by teacher forcing.
        logits: [B, Lm1, V] next-token logits (without the last position).

    Returns:
        mask: [B, Lm1] boolean mask where True marks reply positions.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3-D [B,L,V], got shape={tuple(logits.shape)}")
    B, Lm1, _ = logits.shape
    if prompt_len <= 0 or prompt_len > (Lm1 + 1):
        raise ValueError(f"prompt_len={prompt_len} invalid for logits length (Lm1+1)={Lm1+1}")
    mask = torch.zeros((B, Lm1), dtype=torch.bool, device=logits.device)
    # reply positions correspond to indices >= prompt_len-1
    mask[:, prompt_len - 1 :] = True
    return mask


def per_token_logps_from_logits(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Compute per-token log-probabilities for the given logits and targets.

    Shapes:
        logits: [B, Lm1, V]
        target_ids: [B, Lm1]
    Returns:
        logps: [B, Lm1]
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3-D [B,L,V], got shape={tuple(logits.shape)}")
    if target_ids.ndim != 2:
        raise ValueError(f"target_ids must be 2-D [B,L], got shape={tuple(target_ids.shape)}")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError(f"shape mismatch: logits[:2]={tuple(logits.shape[:2])} vs target_ids.shape={tuple(target_ids.shape)}")
    logps = compute_logprobs(logits, target_ids)
    _fail_if_nan_inf(logps, "per_token_logps")
    return logps


def compute_ratio_and_clip(cur_logps: Tensor, old_logps: Optional[Tensor], eps_low: float, eps_high: float) -> Tuple[Tensor, Tensor]:
    """Compute importance ratios and clipped ratios.

    Args:
        cur_logps: [B, L] current per-token log-probs
        old_logps: [B, L] old (baseline) per-token log-probs (if None, uses detached cur_logps)
        eps_low: lower clip epsilon in (0,1]
        eps_high: upper slack >= 0 (0 for symmetric clipping)
    Returns:
        coef_1: exp(cur - old)
        coef_2: clipped coef_1 in [1-eps_low, 1+eps_high]
    """
    base_old = old_logps if old_logps is not None else cur_logps.detach()
    if cur_logps.shape != base_old.shape:
        raise ValueError(f"shape mismatch: cur_logps={tuple(cur_logps.shape)} vs old_logps={tuple(base_old.shape)}")
    if not (0.0 < float(eps_low) <= 1.0):
        raise ValueError(f"eps_low must be in (0,1], got {eps_low}")
    if float(eps_high) < 0.0:
        raise ValueError(f"eps_high must be >= 0, got {eps_high}")
    ratio = torch.exp(cur_logps - base_old)
    _fail_if_nan_inf(ratio, "ratio")
    low = 1.0 - float(eps_low)
    high = 1.0 + float(eps_high)
    ratio_clipped = torch.clamp(ratio, min=low, max=high)
    return ratio, ratio_clipped


def apply_entropy_mask_from_logits(
    logits: Tensor,
    reply_mask: Tensor,
    *,
    top_quantile: Optional[float] = None,
    min_threshold: Optional[float] = None,
) -> Tensor:
    """Build an entropy-based mask for reply tokens.

    Args:
        logits: [B, Lm1, V]
        reply_mask: [B, Lm1] boolean
        top_quantile: if provided in (0,1], keep tokens with entropy >= quantile threshold
        min_threshold: if provided (>0), keep tokens with entropy >= min_threshold
    Returns:
        ent_mask: [B, Lm1] boolean
    """
    if logits.ndim != 3 or reply_mask.ndim != 2 or logits.shape[:2] != reply_mask.shape:
        raise ValueError("logits and reply_mask shape mismatch for entropy masking")
    if (top_quantile is None) == (min_threshold is None):
        raise ValueError("Specify exactly one of top_quantile or min_threshold")

    logp = F.log_softmax(logits.float(), dim=-1)
    p = torch.exp(logp)
    ent = -(p * logp).sum(dim=-1)  # [B, Lm1]
    _fail_if_nan_inf(ent, "entropy")

    if top_quantile is not None:
        if not (0.0 < float(top_quantile) <= 1.0):
            raise ValueError(f"top_quantile must be in (0,1], got {top_quantile}")
        # compute threshold across reply positions only to avoid prompt tokens bias
        vals = ent[reply_mask]
        if vals.numel() == 0:
            return torch.zeros_like(reply_mask)
        thresh = torch.quantile(vals, 1.0 - float(top_quantile))
        ent_mask = ent >= thresh
    else:
        if not (float(min_threshold) > 0.0):
            raise ValueError(f"min_threshold must be > 0, got {min_threshold}")
        ent_mask = ent >= float(min_threshold)

    return ent_mask & reply_mask


def reduce_loss(
    per_token_loss: Tensor,
    reply_mask: Tensor,
    loss_type: str = "grpo",
    *,
    keep_batch: bool = False,
) -> Tensor:
    """Reduce per-token loss to scalar according to loss_type.

    - grpo: mean over sentences of (sum over reply tokens / num reply tokens)
    - bnpo: sum over reply tokens / total reply tokens
    - dr_grpo: same as bnpo (fallback) to avoid requiring global caps
    """
    if per_token_loss.ndim != 2 or reply_mask.ndim != 2 or per_token_loss.shape != reply_mask.shape:
        raise ValueError("per_token_loss and reply_mask must be [B,L] with identical shapes")
    B, L = per_token_loss.shape
    denom = reply_mask.sum(dim=1).clamp(min=1)
    if loss_type == "grpo":
        per_sample = (per_token_loss * reply_mask).sum(dim=1) / denom
        return per_sample if keep_batch else per_sample.mean()
    if loss_type in {"bnpo", "dr_grpo"}:
        total = reply_mask.sum().clamp(min=1)
        if keep_batch:
            return (per_token_loss * reply_mask).sum(dim=1)
        loss = (per_token_loss * reply_mask).sum() / total
        return loss
    raise ValueError(f"Unknown loss_type: {loss_type}")
