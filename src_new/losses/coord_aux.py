from typing import Tuple

import torch

from ..utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)


def build_kernel_indices_and_q(
    y: torch.Tensor, K: int, sigma: float, window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    offsets = torch.arange(-window, window + 1, dtype=torch.long, device=y.device)
    idxs = y_long.unsqueeze(1) + offsets.unsqueeze(0)
    idxs = torch.clamp(idxs, min=0, max=int(K))
    d = (idxs - y_long.unsqueeze(1)).to(dtype=torch.float32)
    sigma_val = float(max(sigma, 1e-6))
    q_vals = torch.exp(-(d * d) / (2.0 * (sigma_val**2)))
    return idxs, q_vals


def kernelized_kl_sparse(
    coord_logits: torch.Tensor,
    idxs: torch.Tensor,
    q_vals: torch.Tensor,
    tau: float,
    eps: float,
) -> torch.Tensor:
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
    logits = (coord_logits.float() / float(max(tau, eps))).clamp(-50.0, 50.0)
    p_full = torch.softmax(logits, dim=-1)
    if idxs.dtype != torch.long:
        idxs = idxs.to(dtype=torch.long)
    p_w = p_full.gather(dim=-1, index=idxs)
    q_raw = q_vals.to(dtype=torch.float32)
    q_sum = q_raw.sum(dim=-1, keepdim=True)
    if torch.any(q_sum <= 0):
        return logits.new_tensor(0.0)
    q_w = q_raw / q_sum
    p_w = torch.clamp(p_w, min=1e-12)
    q_w = torch.clamp(q_w, min=1e-12)
    kl_vec = (q_w * (torch.log(q_w) - torch.log(p_w))).sum(dim=-1)
    out = torch.nan_to_num(kl_vec.mean(), nan=0.0, posinf=1e6, neginf=1e6)
    if out < 0:
        out = out.clamp_min(0.0)
    return out


def unlikelihood_topk_text(
    logits_all: torch.Tensor,
    coord_mask: torch.Tensor,
    noncoord_vocab_mask: torch.BoolTensor,
    topk: int,
    eps: float,
) -> torch.Tensor:
    if coord_mask is None:
        return logits_all.new_tensor(0.0)
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
    logits_text = logits_all[..., noncoord_vocab_mask].float().clamp(-50.0, 50.0)
    probs_text = torch.softmax(logits_text, dim=-1)
    k = int(min(int(topk), probs_text.size(-1)))
    if k <= 0:
        return logits_all.new_tensor(0.0)
    top_vals, _ = torch.topk(probs_text, k=k, dim=-1)
    loss = -torch.log(1.0 - top_vals + eps)
    denom = coord_mask.sum() * k + eps
    out = (loss * coord_mask.unsqueeze(-1).float()).sum() / denom
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=1e6)


def build_noncoord_vocab_mask(
    vocab_size: int, coord_start: int, coord_end_exclusive: int
) -> torch.Tensor:
    """
    Build boolean mask [V] selecting all NON-coordinate tokens given an exclusive coord range.
    """
    if not (0 <= coord_start <= coord_end_exclusive <= vocab_size):
        raise ValueError(
            f"Invalid coord range ({coord_start},{coord_end_exclusive}) for vocab_size={vocab_size}"
        )
    noncoord = torch.ones(vocab_size, dtype=torch.bool)
    noncoord[coord_start:coord_end_exclusive] = False
    return noncoord


__all__ = [
    "build_kernel_indices_and_q",
    "kernelized_kl_sparse",
    "unlikelihood_topk_text",
    "build_noncoord_vocab_mask",
]
