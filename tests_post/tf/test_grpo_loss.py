import math
import pytest
import torch
from torch import tensor

from src_post.tf.grpo_loss import (
    build_reply_mask,
    per_token_logps_from_logits,
    compute_ratio_and_clip,
    apply_entropy_mask_from_logits,
    reduce_loss,
)


def test_build_reply_mask_basic():
    logits = torch.zeros((1, 5, 4))
    mask = build_reply_mask(prompt_len=3, logits=logits)
    assert mask.shape == (1, 5)
    # reply from indices >= prompt_len-1 == 2
    assert mask[0, :2].sum().item() == 0
    assert mask[0, 2:].all().item() == 1


def test_per_token_logps_from_logits_shape_and_finite():
    # logits favor class 1 consistently
    logits = torch.zeros((1, 4, 3))
    logits[..., 1] = 4.0
    # targets are arbitrary ids in 0..2
    targets = torch.tensor([[0, 1, 2, 1]])
    logps = per_token_logps_from_logits(logits, targets)
    assert logps.shape == (1, 4)
    assert torch.isfinite(logps).all()


def test_compute_ratio_and_clip_symmetric():
    cur = torch.log(torch.tensor([[0.2, 0.8]]))
    old = torch.log(torch.tensor([[0.25, 0.75]]))
    ratio, ratio_clip = compute_ratio_and_clip(cur, old, eps_low=0.2, eps_high=0.0)
    assert ratio.shape == ratio_clip.shape == (1, 2)
    # lower bound 1 - eps_low = 0.8
    assert (ratio_clip >= 0.8).all()


def test_entropy_mask_quantile_and_threshold():
    # logits with strong peak at last positions
    logits = torch.randn(1, 6, 5)
    reply_mask = torch.zeros((1, 6), dtype=torch.bool)
    reply_mask[:, 2:] = True
    m1 = apply_entropy_mask_from_logits(logits, reply_mask, top_quantile=0.5)
    assert m1.shape == reply_mask.shape
    assert (m1 <= reply_mask).all()
    m2 = apply_entropy_mask_from_logits(logits, reply_mask, min_threshold=0.1)
    assert m2.shape == reply_mask.shape
    assert (m2 <= reply_mask).all()
    with pytest.raises(ValueError):
        _ = apply_entropy_mask_from_logits(logits, reply_mask, top_quantile=0.5, min_threshold=0.1)


def test_reduce_loss_modes():
    per_tok = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    # grpo: mean over sentences of (sum over reply / num reply)
    loss_grpo = reduce_loss(per_tok, mask, 'grpo')
    assert math.isclose(loss_grpo.item(), (3.0 + 4.0) / 2.0, rel_tol=1e-6)
    # bnpo: sum over reply / total reply
    loss_bnpo = reduce_loss(per_tok, mask, 'bnpo')
    assert math.isclose(loss_bnpo.item(), (3.0 + 4.0) / 2.0, rel_tol=1e-6)
    with pytest.raises(ValueError):
        _ = reduce_loss(per_tok, mask, 'unknown')
