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
    c.coordinate_temperature = 1.0  # Fixed: use coordinate_temperature
    c.teacher_loss_weight = 0.3
    c.student_loss_weight = 1.0
    return c


@pytest.fixture
def token_processor(mock_config):
    return TokenProcessor(
        TokenConfig(coordinate_tokens_enabled=True, max_coord_value=1024)
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
    assert isinstance(comp, LossComponents)
    assert comp.student_l1_loss is None


def test_coordinate_loss_valid_spans(loss_manager):
    logits = torch.randn(1, 10, 152704, requires_grad=True)
    labels = torch.randint(0, 152704, (1, 10))

    # Put coordinate labels in known positions
    coord_start = 151667
    labels[0, 3] = coord_start + 5
    labels[0, 4] = coord_start + 10

    coord_mask = torch.zeros_like(labels, dtype=torch.bool)
    coord_mask[0, 3] = True
    coord_mask[0, 4] = True

    comp = loss_manager.compute_loss_components(
        logits=logits, labels=labels, coord_mask=coord_mask
    )
    assert comp.student_l1_loss is not None
    assert torch.isfinite(comp.student_l1_loss)


def test_nan_loss_detection(loss_manager):
    # Create logits with extreme values; pipeline should not crash
    logits = torch.full((1, 3, 152704), float("inf"))
    labels = torch.randint(0, 152704, (1, 3))
    comp = loss_manager.compute_loss_components(logits=logits, labels=labels)
    assert comp is not None


def test_student_span_coord_only_excludes_ce(loss_manager):
    logits = torch.randn(1, 7, 152704)
    labels = torch.full((1, 7), -100)
    coord_start = 151667
    # Student span with only coordinate targets at positions 4..6 (exclusive end at 6)
    labels[0, 4] = coord_start + 1
    labels[0, 5] = coord_start + 2
    student_spans = [[(4, 6)]]

    comp = loss_manager.compute_loss_components(
        logits=logits, labels=labels, student_spans=student_spans
    )
    assert comp.student_llm_loss is None
    assert comp.student_l1_loss is not None
    assert torch.isfinite(comp.student_l1_loss)
