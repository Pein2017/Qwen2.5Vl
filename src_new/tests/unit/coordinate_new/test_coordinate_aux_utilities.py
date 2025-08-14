import torch

from src_new.models.coordinate_loss import (
    build_kernel_indices_and_q,
    kernelized_kl_sparse,
    unlikelihood_topk_text,
)


def test_kernelized_kl_sparse_decreases_when_logits_peak_near_target():
    # Coordinate bins 0..K
    K = 20
    W = 5  # window radius => actual width = 2*W+1
    y = torch.tensor([5, 12], dtype=torch.long)
    idxs, q_vals = build_kernel_indices_and_q(y=y, K=K, sigma=2.0, window=W)
    assert idxs.shape == (y.shape[0], 2 * W + 1)
    assert q_vals.shape == (y.shape[0], 2 * W + 1)

    # Build logits: close vs far
    coord_logits_close = torch.full((y.shape[0], K + 1), -4.0)
    coord_logits_far = torch.full((y.shape[0], K + 1), -4.0)
    # Concentrate mass near targets for "close"
    for i, yi in enumerate(y.tolist()):
        coord_logits_close[i, yi] = 6.0
        if yi - 1 >= 0:
            coord_logits_close[i, yi - 1] = 4.0
        if yi + 1 <= K:
            coord_logits_close[i, yi + 1] = 4.0
    # Concentrate mass far away for "far"
    coord_logits_far[:, 0] = 6.0
    coord_logits_far[:, 1] = 4.0

    kl_close = kernelized_kl_sparse(coord_logits_close, idxs, q_vals, tau=1.0, eps=1e-6)
    kl_far = kernelized_kl_sparse(coord_logits_far, idxs, q_vals, tau=1.0, eps=1e-6)
    assert torch.isfinite(kl_close)
    assert torch.isfinite(kl_far)
    assert kl_close.item() < kl_far.item()


def test_kernelized_kl_sparse_handles_empty_inputs():
    # Empty batch (no positions)
    K = 10
    idxs = torch.empty(0, 3, dtype=torch.long)
    q_vals = torch.empty(0, 3, dtype=torch.float32)
    coord_logits = torch.empty(0, K + 1, dtype=torch.float32)
    kl = kernelized_kl_sparse(coord_logits, idxs, q_vals, tau=1.0, eps=1e-6)
    assert torch.isfinite(kl)
    assert kl.item() == 0.0


def test_unlikelihood_topk_text_empty_mask_returns_zero():
    B, T, V = 1, 4, 32
    logits_all = torch.randn(B, T, V)
    coord_mask = torch.zeros(B, T, dtype=torch.bool)
    # Coordinate range [10, 20)
    noncoord_vocab_mask = torch.ones(V, dtype=torch.bool)
    noncoord_vocab_mask[10:20] = False
    loss = unlikelihood_topk_text(
        logits_all=logits_all,
        coord_mask=coord_mask,
        noncoord_vocab_mask=noncoord_vocab_mask,
        topk=5,
        eps=1e-6,
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_unlikelihood_topk_text_decreases_when_noncoord_mass_is_suppressed():
    B, T, V = 1, 3, 30
    coord_start, coord_end = 10, 20
    noncoord_vocab_mask = torch.ones(V, dtype=torch.bool)
    noncoord_vocab_mask[coord_start:coord_end] = False
    coord_mask = torch.zeros(B, T, dtype=torch.bool)
    coord_mask[0, 1] = True  # one coordinate position

    # Case A: make a peaked distribution over non-coordinate tokens
    logits_a = torch.zeros(B, T, V)
    # Choose a couple of non-coordinate ids to be high
    noncoord_ids = torch.nonzero(noncoord_vocab_mask).squeeze(-1).tolist()
    peak_ids = noncoord_ids[:2]
    for pid in peak_ids:
        logits_a[..., pid] = 8.0

    # Case B: flat (uniform) over non-coordinate tokens
    logits_b = torch.zeros(B, T, V)
    # keep zeros so softmax over the subset is uniform

    loss_a = unlikelihood_topk_text(
        logits_all=logits_a,
        coord_mask=coord_mask,
        noncoord_vocab_mask=noncoord_vocab_mask,
        topk=5,
        eps=1e-6,
    )
    loss_b = unlikelihood_topk_text(
        logits_all=logits_b,
        coord_mask=coord_mask,
        noncoord_vocab_mask=noncoord_vocab_mask,
        topk=5,
        eps=1e-6,
    )
    assert torch.isfinite(loss_a) and torch.isfinite(loss_b)
    assert loss_b.item() < loss_a.item()
