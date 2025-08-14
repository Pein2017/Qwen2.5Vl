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
    assert isinstance(comp, LossComponents)
    assert comp.student_l1_loss is None


def test_nan_loss_detection(loss_manager):
    # Create logits with extreme values; pipeline should not crash
    logits = torch.full((1, 3, 152704), float("inf"))
    labels = torch.randint(0, 152704, (1, 3))
    comp = loss_manager.compute_loss_components(logits=logits, labels=labels)
    assert comp is not None
