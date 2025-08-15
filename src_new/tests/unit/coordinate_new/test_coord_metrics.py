import torch

from src_new.models.coord_metrics import compute_coord_diagnostics
from src_new.models.coordinate_loss import build_kernel_indices_and_q


def test_compute_coord_diagnostics_simple_peaked_case():
    # Coordinate bins 0..K
    K = 20
    V = 100  # full vocab
    y = torch.tensor([5, 12], dtype=torch.long)
    # Build coord logits with strong peaks at GT
    coord_logits = torch.full((y.shape[0], K + 1), -6.0)
    for i, yi in enumerate(y.tolist()):
        coord_logits[i, yi] = 8.0
        if yi - 1 >= 0:
            coord_logits[i, yi - 1] = 4.0
        if yi + 1 <= K:
            coord_logits[i, yi + 1] = 4.0

    # Full-vocab logits rows corresponding to the same positions
    full_logits_rows = torch.full((y.shape[0], V), -6.0)
    # Place the same peaks in the coordinate slice segment [0:K+1] for a consistent relation
    full_logits_rows[:, : K + 1] = coord_logits

    # Non-coordinate mask (exclude [0:K+1])
    noncoord_mask = torch.ones(V, dtype=torch.bool)
    noncoord_mask[: K + 1] = False

    # Build window indices around GT
    idxs, _ = build_kernel_indices_and_q(y=y, K=K, sigma=2.0, window=4)

    metrics = compute_coord_diagnostics(
        shifted_logits=full_logits_rows,
        coord_logits=coord_logits,
        y_bins=y,
        idxs=idxs,
        noncoord_mask=noncoord_mask,
        tau=1.2,
        topk_noncoord=5,
    )

    # Basic presence and finiteness
    for key in [
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
    ]:
        assert key in metrics, f"Missing metric: {key}"
        assert torch.isfinite(metrics[key])

    # Behavioral checks in a peaked scenario
    assert metrics["window_mass"].item() > 0.5
    assert metrics["gt_prob"].item() > 0.3
    assert metrics["top1_acc"].item() >= 0.5
    # Count check
    assert abs(metrics["coord_pos_count"].item() - y.numel()) < 1e-6
