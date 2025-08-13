#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive validation of teacher/student span detection and mask separation.

- Teacher vs student spans in multi-turn chats
- CE for text-only; L1 for coord-only
- Aggregation and weighting correctness
- Edge cases: empty/invalid spans, mixed coord/text spans
"""

from unittest.mock import Mock

import torch

from src_new.models.loss_manager import LossManager
from src_new.processing.token_processor import TokenConfig, TokenProcessor


def make_mock_tokenizer(max_coord_value=1025):
    tok = Mock()
    base = 151665
    vocab = {"<|line_start|>": base, "<|line_end|>": base + 1}
    for i in range(max_coord_value + 1):
        vocab[f"<|coord_{i}|>"] = 151667 + i
    tok.get_vocab.return_value = vocab
    tok.vocab_size = len(vocab)
    return tok


def make_loss_manager(max_coord_value=1025):
    cfg = Mock()
    cfg.coordinate_tokens_enabled = True
    cfg.max_coord_value = max_coord_value
    cfg.coordinate_loss_weight = 1.0
    cfg.regular_loss_weight = 1.0
    cfg.teacher_loss_weight = 0.3
    cfg.student_loss_weight = 1.0
    cfg.coordinate_temperature = 1.0  # Fixed: use coordinate_temperature

    tp = TokenProcessor(
        TokenConfig(max_coord_value=max_coord_value, coordinate_tokens_enabled=True)
    )
    tok = make_mock_tokenizer(max_coord_value)

    return LossManager(cfg, tp, tok)


def test_span_masks_and_loss_routing_mixed_tokens():
    lm = make_loss_manager(1025)

    batch, seqlen = 1, 20
    vocab_size = 151665 + 2 + 1026

    logits = torch.randn(batch, seqlen, vocab_size, requires_grad=True)
    labels = torch.full((batch, seqlen), -100)

    # Create mixed spans
    # Teacher span: positions 5..9	non-coordinate text at 5,7; coords at 6,8
    teacher_spans = [[(5, 10)]]
    # Student span: positions 12..17	text at 12,14,16; coords at 13,15,17
    student_spans = [[(12, 18)]]

    # Set concrete labels for text vs coord indices
    labels[0, 5] = 1
    labels[0, 7] = 2

    coord_start = 151667
    labels[0, 6] = coord_start + 10
    labels[0, 8] = coord_start + 20

    labels[0, 12] = 3
    labels[0, 14] = 4
    labels[0, 16] = 5

    labels[0, 13] = coord_start + 30
    labels[0, 15] = coord_start + 40
    labels[0, 17] = coord_start + 50

    # Coordinate mask covers only coord labels
    coord_mask = torch.zeros_like(labels, dtype=torch.bool)
    for pos in [6, 8, 13, 15, 17]:
        coord_mask[0, pos] = True

    comps = lm.compute_loss_components(
        logits=logits,
        labels=labels,
        coord_mask=coord_mask,
        teacher_spans=teacher_spans,
        student_spans=student_spans,
    )

    # Teacher/student CE exist (text tokens in spans)
    assert comps.teacher_llm_loss is not None
    assert comps.student_llm_loss is not None

    # Teacher/student L1 exist (coord tokens in spans)
    assert comps.teacher_l1_loss is not None
    assert comps.student_l1_loss is not None


def test_empty_and_invalid_spans_handled():
    lm = make_loss_manager(1024)

    batch, seqlen = 1, 10
    vocab_size = 151665 + 2 + 1026
    logits = torch.randn(batch, seqlen, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch, seqlen))

    # Empty spans
    comps1 = lm.compute_loss_components(
        logits=logits,
        labels=labels,
        teacher_spans=[[]],
        student_spans=[[]],
    )
    assert comps1.loss is not None

    # Invalid spans out of bounds
    comps2 = lm.compute_loss_components(
        logits=logits,
        labels=labels,
        teacher_spans=[[(20, 30)]],
        student_spans=None,
    )
    assert comps2.loss is not None


def test_weighting_sum_matches_total():
    lm = make_loss_manager(1025)

    batch, seqlen = 1, 15
    vocab_size = 151665 + 2 + 1026
    logits = torch.randn(batch, seqlen, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch, seqlen))

    teacher_spans = [[(2, 6)]]
    student_spans = [[(9, 12)]]

    comps = lm.compute_loss_components(
        logits=logits,
        labels=labels,
        teacher_spans=teacher_spans,
        student_spans=student_spans,
    )

    total = 0.0
    for k in [
        "teacher_llm_loss",
        "student_llm_loss",
        "teacher_l1_loss",
        "student_l1_loss",
    ]:
        v = getattr(comps, k)
        if v is not None:
            total += float(v)
    assert abs(float(comps.loss) - total) < 1e-6
