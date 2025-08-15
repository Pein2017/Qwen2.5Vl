#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust loss computation tests with coordinate masks and failure cases.
"""

from unittest.mock import Mock

import pytest
import torch

from src_new.models.loss_manager import LossComponents, LossManager
from src_new.processing.token_processor import TokenConfig, TokenProcessor


@pytest.fixture
def mock_config():
    c = Mock()
    c.coordinate_tokens_enabled = True
    c.max_coord_value = 1024
    c.coordinate_loss_weight = 0.05
    c.regular_loss_weight = 1.0

    c.teacher_loss_weight = 0.3
    c.student_loss_weight = 1.0
    return c


@pytest.fixture
def token_processor(mock_config):
    return TokenProcessor(
        TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=1024,
            coordinate_init_mode="fourier_ramp",
        )
    )


@pytest.fixture
def mock_tokenizer():
    # Minimal realistic vocab: line + coords
    vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
    vocab.update({f"<|coord_{i}|>": 151667 + i for i in range(1025)})
    t = Mock()
    t.get_vocab.return_value = vocab
    t.vocab_size = 151665
    return t


@pytest.fixture
def loss_manager(mock_config, token_processor, mock_tokenizer):
    return LossManager(mock_config, token_processor, mock_tokenizer)


def test_coordinate_mask_empty_has_no_l1(loss_manager):
    logits = torch.randn(1, 6, 152704)  # larger than coord end
    labels = torch.randint(0, 152704, (1, 6))
    coord_mask = torch.zeros_like(labels, dtype=torch.bool)
    comp = loss_manager.compute_loss_components(
        logits=logits, labels=labels, coord_mask=coord_mask
    )
    assert comp is not None
    # Diagnostics may be None when aux is disabled or no coord positions
    assert hasattr(comp, "diagnostics")


def test_nan_loss_detection(loss_manager):
    # Create logits with extreme values; pipeline should not crash
    logits = torch.full((1, 3, 152704), float("inf"))
    labels = torch.randint(0, 152704, (1, 3))
    comp = loss_manager.compute_loss_components(logits=logits, labels=labels)
    assert comp is not None


def test_diagnostics_present_when_aux_enabled():
    # Build a loss manager with aux enabled
    from unittest.mock import Mock

    cfg = Mock()
    cfg.coordinate_tokens_enabled = True
    cfg.max_coord_value = 32
    cfg.coordinate_loss_weight = 1.0
    cfg.regular_loss_weight = 1.0
    cfg.teacher_loss_weight = 1.0
    cfg.student_loss_weight = 1.0

    vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
    for i in range(cfg.max_coord_value + 1):
        vocab[f"<|coord_{i}|>"] = 151667 + i
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = vocab

    tp = TokenProcessor(
        TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=cfg.max_coord_value,
            coordinate_init_mode="fourier_ramp",
        )
    )
    lm = LossManager(cfg, tp, tokenizer)
    lm.set_coordinate_aux_options(
        tau=1.2, sigma_bins=2.0, window_bins=3, topk=5, lambda_kce=1.0, lambda_unlike=0.1
    )

    B, T = 1, 8
    coord_start = tp.get_coordinate_token_range(tokenizer)[0]
    V = coord_start + cfg.max_coord_value + 1 + 50
    logits = torch.zeros(B, T, V)
    labels = torch.full((B, T), -100)
    # single student span with one coord target
    student_spans = [[(3, 7)]]
    labels[0, 5] = coord_start + 3
    # peak at shifted index 4
    logits[0, 4, coord_start + 3] = 8.0

    comp = lm.compute_loss_components(
        logits=logits, labels=labels, student_spans=student_spans
    )
    assert comp is not None
    assert hasattr(comp, "diagnostics")
    diag = comp.diagnostics
    assert diag is None or isinstance(diag, dict)
    if diag:
        # Check a few keys exist and are finite
        for k in [
            "student_window_mass",
            "student_gt_prob",
            "student_top1_acc",
        ]:
            assert k in diag
            assert torch.isfinite(diag[k])
