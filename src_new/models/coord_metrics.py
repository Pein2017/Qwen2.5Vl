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
    noncoord_mask: torch.BoolTensor,  # [V]
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

    # Window mass in coordinate slice
    p_w = p_coord.gather(dim=-1, index=idxs)
    window_mass_vec = p_w.sum(dim=-1)
    window_mass = window_mass_vec.mean()
    outside_window_mass = (1.0 - window_mass_vec).mean()

    # Window entropy (normalized within window)
    p_w_norm = p_w / (p_w.sum(dim=-1, keepdim=True) + 1e-12)
    window_entropy = -(p_w_norm * (torch.log(p_w_norm + 1e-12))).sum(dim=-1).mean()

    # GT probability within coordinate slice
    gt_prob = p_coord.gather(dim=-1, index=y_bins.view(-1, 1)).squeeze(1).mean()

    # Expected bin MAE (in bins) and signed offset
    K = p_coord.size(-1) - 1
    bin_idx = torch.arange(0, K + 1, device=p_coord.device, dtype=torch.float32)
    expected_bin = (p_coord * bin_idx.unsqueeze(0)).sum(dim=-1)
    diff = expected_bin - y_bins.to(dtype=torch.float32)
    expected_mae = diff.abs().mean()
    mean_bin_offset = diff.mean()

    # Top-1 / Top-5 accuracy within coordinate slice, and margin
    topk2_vals, topk2_idx = torch.topk(p_coord, k=2, dim=-1)
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

    # Caller must pass the exact rows of shifted_logits that match coord_logits; otherwise skip.
    try:
        # Attempt to gather matching rows from shifted_logits via a side channel
        # If shapes mismatch, fall back to zero for these two metrics
        # Expected shape: shifted_logits_selected: [N, V]
        # Using a heuristic: if shifted_logits.dim()==2 and shifted_logits.size(0)==coord_logits.size(0)
        if shifted_logits.dim() == 2 and shifted_logits.size(0) == coord_logits.size(0):
            full_logits_pos = shifted_logits
            lse_coord = torch.logsumexp(coord_logits, dim=-1)
            lse_full = torch.logsumexp(full_logits_pos, dim=-1)
            coord_slice_mass = torch.exp(lse_coord - lse_full).mean()

            logits_text = full_logits_pos[..., noncoord_mask].float().clamp(-50.0, 50.0)
            probs_text = torch.softmax(logits_text, dim=-1)
            k = int(min(int(topk_noncoord), probs_text.size(-1)))
            if k > 0:
                noncoord_topk_mass = (
                    torch.topk(probs_text, k=k, dim=-1)[0].sum(dim=-1).mean()
                )
            else:
                noncoord_topk_mass = shifted_logits.new_tensor(0.0)
        else:
            coord_slice_mass = shifted_logits.new_tensor(0.0)
            noncoord_topk_mass = shifted_logits.new_tensor(0.0)
    except Exception:
        coord_slice_mass = shifted_logits.new_tensor(0.0)
        noncoord_topk_mass = shifted_logits.new_tensor(0.0)

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
