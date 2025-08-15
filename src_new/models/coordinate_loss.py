"""
Auxiliary Coordinate Loss Functions

This module implements auxiliary coordinate loss functions for the coordinate token system:
- Kernelized KL divergence loss with sparse windows
- Unlikelihood loss for non-coordinate tokens at coordinate positions

These functions are used when coord_aux_enabled=true in the configuration.
"""

from typing import Tuple

import torch

# Configure rank-aware logger
from ..utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)


def build_kernel_indices_and_q(
    y: torch.Tensor, K: int, sigma: float, window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-sample sparse kernel window around ground-truth bin indices.

    Args:
            y: Tensor of shape [N] with integer ground-truth in [0, K]
            K: Maximum bin value (inclusive). Total bins = K+1
            sigma: Kernel width in bins (Gaussian). Must be > 0
            window: Half-window size (radius). Output width = 2*window + 1

    Returns:
            idxs: Long tensor [N, W] with clamped indices in [0, K]
            q_vals: Float tensor [N, W] with unnormalized kernel values per index
    """
    if y is None or y.numel() == 0:
        width = 2 * int(window) + 1
        return (
            torch.empty(0, width, dtype=torch.long),
            torch.empty(0, width, dtype=torch.float32),
        )

    if not torch.is_tensor(y):
        raise TypeError("y must be a torch.Tensor")

    y_long = y.to(dtype=torch.long)
    width = 2 * int(window) + 1

    # Offsets [-window, ..., +window]
    offsets = torch.arange(-window, window + 1, dtype=torch.long, device=y.device)
    # Broadcast to [N, W]
    idxs = y_long.unsqueeze(1) + offsets.unsqueeze(0)
    # Clamp to [0, K]
    idxs = torch.clamp(idxs, min=0, max=int(K))

    # Distances for kernel (in bins)
    d = (idxs - y_long.unsqueeze(1)).to(dtype=torch.float32)
    sigma_val = float(max(sigma, 1e-6))
    q_vals = torch.exp(-(d * d) / (2.0 * (sigma_val**2)))

    return idxs, q_vals


def kernelized_kl_sparse(
    coord_logits: torch.Tensor,  # [N, K+1]
    idxs: torch.Tensor,  # [N, W]
    q_vals: torch.Tensor,  # [N, W]
    tau: float,
    eps: float,  # Required parameter, no default
) -> torch.Tensor:
    """
    Compute KL(q||p) where q is a sparse kernel distribution on a window and p
    is the model distribution over full coordinate bins.

    If inputs are empty (N==0), returns 0.0 on the correct device.
    """
    # Handle empty batch
    if (
        coord_logits is None
        or coord_logits.numel() == 0
        or idxs is None
        or idxs.numel() == 0
    ):
        device = (
            coord_logits.device
            if isinstance(coord_logits, torch.Tensor) and coord_logits.numel() > 0
            else None
        )
        return coord_logits.new_tensor(0.0) if device is not None else torch.tensor(0.0)

    # Ensure float32 and clamp for numerical stability
    logits = (coord_logits.float() / float(max(tau, eps))).clamp(-50.0, 50.0)
    p_full = torch.softmax(logits, dim=-1)

    # Gather probabilities on the sparse window
    if idxs.dtype != torch.long:
        idxs = idxs.to(dtype=torch.long)
    p_w = p_full.gather(dim=-1, index=idxs)

    # Normalize q over the window EXACTLY (avoid epsilon-in-denominator shrinkage)
    q_raw = q_vals.to(dtype=torch.float32)
    q_sum = q_raw.sum(dim=-1, keepdim=True)
    # If any q_sum is zero (should not happen with Gaussian), safely return 0
    if torch.any(q_sum <= 0):
        return logits.new_tensor(0.0)
    q_w = q_raw / q_sum

    # Clamp probabilities before log to avoid log(0) while preserving KL structure
    p_w = torch.clamp(p_w, min=1e-12)
    q_w = torch.clamp(q_w, min=1e-12)

    # KL(q||p_window-unnormalized) = KL(q||p_window_normalized) - log S, S=sum p_w
    # Computing it directly as sum q * (log q - log p) is non-negative in exact math.
    kl_vec = (q_w * (torch.log(q_w) - torch.log(p_w))).sum(dim=-1)

    out = torch.nan_to_num(kl_vec.mean(), nan=0.0, posinf=1e6, neginf=1e6)
    # Small negative values can occur from floating error; clip to zero floor
    if out < 0:
        out = out.clamp_min(0.0)
    return out


def unlikelihood_topk_text(
    logits_all: torch.Tensor,  # [B, T, V]
    coord_mask: torch.Tensor,  # [B, T]
    noncoord_vocab_mask: torch.BoolTensor,  # [V]
    topk: int,  # Required parameter, no default
    eps: float,  # Required parameter, no default
) -> torch.Tensor:
    """
    Unlikelihood loss on non-coordinate tokens at coordinate positions.

    Select top-k probabilities within the non-coordinate sub-vocab and penalize
    them via -log(1 - p). Returns 0.0 when coord_mask has no true positions.
    """
    if coord_mask is None:
        return logits_all.new_tensor(0.0)

    # Strict shape checks (fail-fast)
    if coord_mask.dim() != 2:
        raise ValueError(
            f"coord_mask must be 2D [B,T], got shape={tuple(coord_mask.shape)}"
        )
    if logits_all.dim() != 3:
        raise ValueError(
            f"logits_all must be 3D [B,T,V], got shape={tuple(logits_all.shape)}"
        )
    B, T, V = logits_all.shape
    if coord_mask.shape[0] != B or coord_mask.shape[1] != T:
        raise ValueError(
            f"Shape mismatch: logits_all[0:2]={B, T} vs coord_mask={tuple(coord_mask.shape)}"
        )
    if noncoord_vocab_mask is None or noncoord_vocab_mask.numel() != V:
        raise ValueError(
            f"noncoord_vocab_mask must have length V={V}, got {None if noncoord_vocab_mask is None else noncoord_vocab_mask.numel()}"
        )

    # Slice logits to non-coordinate vocab
    logits_text = logits_all[..., noncoord_vocab_mask].float().clamp(-50.0, 50.0)
    probs_text = torch.softmax(logits_text, dim=-1)

    k = int(min(int(topk), probs_text.size(-1)))
    if k <= 0:
        return logits_all.new_tensor(0.0)

    # Top-k over non-coordinate probabilities
    top_vals, _ = torch.topk(probs_text, k=k, dim=-1)
    loss = -torch.log(1.0 - top_vals + eps)  # [B, T, k]

    denom = coord_mask.sum() * k + eps
    out = (loss * coord_mask.unsqueeze(-1).float()).sum() / denom
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=1e6)
