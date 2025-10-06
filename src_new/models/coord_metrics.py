'''
Legacy feature. Totally deprecated already.
'''




from typing import Dict

import torch


# Shared base names for coordinate diagnostics (without teacher_/student_ prefixes)
DIAGNOSTIC_METRIC_NAMES = [
    "window_mass",
    "outside_window_mass",
    "window_entropy",
    "gt_prob",
    "expected_mae_bins",
    "mean_bin_offset",
    "top1_acc",
    "top5_acc",
    "margin_top1_top2",
    "coord_slice_mass",
    "noncoord_topk_mass",
    "coord_pos_count",
]


def compute_coord_diagnostics(
    *,
    shifted_logits: torch.Tensor,  # [B,S,V]
    coord_logits: torch.Tensor,  # [N, K+1]
    y_bins: torch.Tensor,  # [N]
    idxs: torch.Tensor,  # [N, W]
    noncoord_mask: torch.Tensor,  # [V]
    tau: float,
    topk_noncoord: int,
) -> Dict[str, torch.Tensor]:
    """Compute coordinate diagnostics for positions with coordinate labels.

    Returns a dict of scalar tensors (averaged over N positions):
    - window_mass, outside_window_mass, window_entropy
    - gt_prob
    - expected_mae_bins, mean_bin_offset
    - top1_acc, top5_acc, margin_top1_top2
    - coord_slice_mass
    - noncoord_topk_mass
    - coord_pos_count
    """
    if coord_logits is None or coord_logits.numel() == 0:
        zero = shifted_logits.new_tensor(0.0)
        return {
            "window_mass": zero,
            "outside_window_mass": zero,
            "window_entropy": zero,
            "gt_prob": zero,
            "expected_mae_bins": zero,
            "mean_bin_offset": zero,
            "top1_acc": zero,
            "top5_acc": zero,
            "margin_top1_top2": zero,
            "coord_slice_mass": zero,
            "noncoord_topk_mass": zero,
            "coord_pos_count": zero,
        }

    # Softmax over coordinate slice
    logits_coord_scaled = (coord_logits.float() / float(max(tau, 1e-6))).clamp(
        -50.0, 50.0
    )
    p_coord = torch.softmax(logits_coord_scaled, dim=-1)

    # Window mass in coordinate slice (deduplicated; no edge double-counting)
    K = p_coord.size(-1) - 1
    width = idxs.size(1)
    radius = int(max((width - 1) // 2, 0))
    low = torch.clamp(y_bins - radius, min=0)
    high = torch.clamp(y_bins + radius, max=K)

    # Use cumulative sum for fast inclusive range sums
    cdf = torch.cumsum(p_coord, dim=-1)
    mass_high = cdf.gather(dim=-1, index=high.view(-1, 1)).squeeze(1)
    # Avoid negative indices in gather: clamp (low - 1) then zero out where low==0
    safe_lowm1 = torch.clamp(low - 1, min=0)
    mass_lowm1 = cdf.gather(dim=-1, index=safe_lowm1.view(-1, 1)).squeeze(1)
    mass_lowm1 = mass_lowm1.masked_fill(low <= 0, 0.0)
    window_mass_vec = (mass_high - mass_lowm1).clamp(0.0, 1.0)
    window_mass = window_mass_vec.mean()
    outside_window_mass = (1.0 - window_mass_vec).mean()

    # Window entropy within [low, high] using a boolean slice mask (no duplicates)
    cols = torch.arange(K + 1, device=p_coord.device).view(1, -1)
    mask = (cols >= low.view(-1, 1)) & (cols <= high.view(-1, 1))
    p_w = p_coord * mask.to(dtype=p_coord.dtype)
    p_w_sum = p_w.sum(dim=-1, keepdim=True)
    p_w_sum = torch.clamp(p_w_sum, min=torch.finfo(p_w_sum.dtype).eps)
    p_w_norm = p_w / p_w_sum
    # Use float32 for stable entropy when in bf16
    p_w_norm_f32 = p_w_norm.to(dtype=torch.float32)
    window_entropy = (
        -(p_w_norm_f32 * torch.log(p_w_norm_f32 + 1e-12)).sum(dim=-1).mean()
    )

    # GT probability within coordinate slice
    # Gather with safe dtype and indices
    y_bins = y_bins.to(dtype=torch.long)
    gt_prob = p_coord.gather(dim=-1, index=y_bins.view(-1, 1)).squeeze(1).mean()

    # Expected bin MAE (in bins) and signed offset
    K = p_coord.size(-1) - 1
    bin_idx = torch.arange(0, K + 1, device=p_coord.device, dtype=torch.float32)
    # Accumulate in float32 for numerical stability when running bf16
    expected_bin = (p_coord.to(dtype=torch.float32) * bin_idx.unsqueeze(0)).sum(dim=-1)
    diff = expected_bin - y_bins.to(dtype=torch.float32)
    expected_mae = diff.abs().mean()
    mean_bin_offset = diff.mean()

    # Top-1 / Top-5 accuracy within coordinate slice, and margin
    topk2_vals, topk2_idx = torch.topk(p_coord.to(dtype=torch.float32), k=2, dim=-1)
    top1_acc = (topk2_idx[:, 0] == y_bins).float().mean()
    k5 = int(min(5, p_coord.size(-1)))
    if k5 > 1:
        topk_vals, topk_idx = torch.topk(p_coord, k=k5, dim=-1)
        y_expanded = y_bins.view(-1, 1).expand_as(topk_idx)
        top5_acc = (topk_idx == y_expanded).any(dim=-1).float().mean()
    else:
        top5_acc = top1_acc
    margin_top1_top2 = (topk2_vals[:, 0] - topk2_vals[:, 1]).mean()

    # Fraction of full-vocab mass on coordinate slice: exp(lse_coord - lse_full)
    # We need the matching full logits rows; caller should have indexed already.
    # Here we infer them based on coord_logits device; we cannot compute this without full logits per-position,
    # so we return zero unless the caller provides them sliced. The caller (loss_manager) computes it directly.

    # Non-coordinate top-k mass at those positions: caller provides per-position full logits slice
    # so we compute from shifted_logits and mask at the same positions; to keep this helper generic,
    # compute using the provided noncoord_mask and topk with a dummy selection of rows if available.

    # Caller must pass the exact rows of shifted_logits that match coord_logits; validate strictly.
    if shifted_logits.dim() != 2:
        raise ValueError(
            f"shifted_logits must be 2D [N,V] rows matching coord positions; got shape={tuple(shifted_logits.shape)}"
        )
    if shifted_logits.size(0) != coord_logits.size(0):
        raise ValueError(
            f"Row count mismatch for diagnostics: shifted_logits N={shifted_logits.size(0)} vs coord_logits N={coord_logits.size(0)}"
        )
    V = int(shifted_logits.size(1))
    # Validate noncoord mask
    if noncoord_mask is None or int(noncoord_mask.numel()) != V:
        raise ValueError(
            f"noncoord_mask must have length V={V}, got {None if noncoord_mask is None else noncoord_mask.numel()}"
        )
    if noncoord_mask.dtype != torch.bool:
        raise TypeError(
            f"noncoord_mask must be boolean; got dtype={noncoord_mask.dtype}"
        )
    full_logits_pos = shifted_logits.to(dtype=torch.float32)
    lse_coord = torch.logsumexp(coord_logits.to(dtype=torch.float32), dim=-1)
    lse_full = torch.logsumexp(full_logits_pos, dim=-1)
    coord_slice_mass = torch.exp(lse_coord - lse_full).mean()

    # Ensure boolean mask is on the same device as logits
    noncoord_mask = noncoord_mask.to(device=full_logits_pos.device)
    logits_text = full_logits_pos[..., noncoord_mask].clamp(-50.0, 50.0)
    probs_text = torch.softmax(logits_text, dim=-1)
    k = int(min(int(topk_noncoord), probs_text.size(-1)))
    if k > 0:
        noncoord_topk_mass = torch.topk(probs_text, k=k, dim=-1)[0].sum(dim=-1).mean()
    else:
        # Fail-fast contract: topk must be > 0 to produce a meaningful value
        raise ValueError(
            f"topk_noncoord must be > 0 for diagnostics; got {topk_noncoord}"
        )

    return {
        "window_mass": window_mass,
        "outside_window_mass": outside_window_mass,
        "window_entropy": window_entropy,
        "gt_prob": gt_prob,
        "expected_mae_bins": expected_mae,
        "mean_bin_offset": mean_bin_offset,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "margin_top1_top2": margin_top1_top2,
        "coord_slice_mass": coord_slice_mass,
        "noncoord_topk_mass": noncoord_topk_mass,
        "coord_pos_count": shifted_logits.new_tensor(float(coord_logits.size(0))),
    }
