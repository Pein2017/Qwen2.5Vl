#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Loss Manager Tests

This module tests the dual-loss system (LLM + coordinate) with proper
teacher-student span handling and mathematical correctness validation.

Key Features:
- Tests all four loss components (teacher_llm, student_llm, teacher_l1, student_l1)
- Validates soft expectation coordinate loss computation
- Tests span-based loss masking
- Verifies loss aggregation and weighting
- Tests edge cases and error handling
"""

from unittest.mock import Mock

import pytest
import torch

from src_new.models.loss_manager import LossComponents, LossManager
from src_new.processing.token_processor import TokenConfig, TokenProcessor


class TestLossManager:
    """Test suite for LossManager with comprehensive coverage."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock()
        config.coordinate_tokens_enabled = True
        config.max_coord_value = 1024
        config.coordinate_loss_weight = 0.05
        config.regular_loss_weight = 1.0
        config.coordinate_temperature = 1.0  # Updated to match LossManager expectation
        config.teacher_loss_weight = 0.3
        config.student_loss_weight = 1.0
        return config

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.vocab_size = 151665
        vocab = {}
        # Add line tokens (these come before coordinate tokens)
        vocab["<|line_start|>"] = 151665
        vocab["<|line_end|>"] = 151666
        # Add coordinate tokens
        for i in range(1025):  # 0 to 1024
            vocab[f"<|coord_{i}|>"] = 151667 + i
        tokenizer.get_vocab.return_value = vocab
        return tokenizer

    @pytest.fixture
    def token_processor(self, mock_config):
        """Create token processor for testing."""
        token_config = TokenConfig(
            coordinate_tokens_enabled=mock_config.coordinate_tokens_enabled,
            max_coord_value=mock_config.max_coord_value,
        )
        return TokenProcessor(token_config)

    @pytest.fixture
    def loss_manager(self, mock_config, mock_tokenizer, token_processor):
        """Create loss manager for testing."""
        return LossManager(mock_config, token_processor, mock_tokenizer)

    def test_loss_manager_initialization(self, loss_manager, mock_config):
        """Test loss manager initialization."""
        # Check that loss weights are properly extracted from config
        assert loss_manager.coordinate_loss_weight == mock_config.coordinate_loss_weight
        assert loss_manager.regular_loss_weight == mock_config.regular_loss_weight
        assert loss_manager.teacher_loss_weight == mock_config.teacher_loss_weight
        assert loss_manager.student_loss_weight == mock_config.student_loss_weight

        # Check that coordinate loss function is initialized
        assert hasattr(loss_manager, "coordinate_loss_fn")
        assert loss_manager.coordinate_loss_fn is not None

    def test_compute_loss_components_basic(self, loss_manager):
        """Test basic loss computation without spans."""
        batch_size, seq_len, vocab_size = (
            2,
            10,
            151665 + 2 + 1025,
        )  # base + line_tokens + coord_tokens

        # Create test data
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mask some positions
        labels[:, :3] = -100  # Mask first 3 positions

        # Compute loss
        loss_components = loss_manager.compute_loss_components(
            logits=logits,
            labels=labels,
            teacher_spans=None,
            student_spans=None,
        )

        # Validate results
        assert isinstance(loss_components, LossComponents)
        assert torch.isfinite(loss_components.loss)
        assert loss_components.loss.item() >= 0

        # Should have student LLM loss (no spans provided)
        assert loss_components.student_llm_loss is not None
        assert torch.isfinite(loss_components.student_llm_loss)

    def test_compute_loss_components_with_spans(self, loss_manager):
        """Test loss computation with teacher-student spans."""
        batch_size, seq_len, vocab_size = (
            2,
            20,
            151665 + 2 + 1025,
        )  # base + line_tokens + coord_tokens

        # Create test data
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mask some positions
        labels[:, :5] = -100  # Mask first 5 positions

        # Define spans
        teacher_spans = [
            [(6, 10), (12, 15)],  # Teacher spans for batch item 0
            [(7, 11)],  # Teacher spans for batch item 1
        ]

        student_spans = [
            [(16, 19)],  # Student spans for batch item 0
            [(13, 17)],  # Student spans for batch item 1
        ]

        # Compute loss
        loss_components = loss_manager.compute_loss_components(
            logits=logits,
            labels=labels,
            teacher_spans=teacher_spans,
            student_spans=student_spans,
        )

        # Validate results
        assert isinstance(loss_components, LossComponents)
        assert torch.isfinite(loss_components.loss)

        # Should have both teacher and student losses
        assert loss_components.teacher_llm_loss is not None
        assert loss_components.student_llm_loss is not None
        assert torch.isfinite(loss_components.teacher_llm_loss)
        assert torch.isfinite(loss_components.student_llm_loss)

    def test_coordinate_loss_computation(self, loss_manager):
        """Test coordinate token loss computation with soft expectation."""
        batch_size, seq_len, vocab_size = (
            1,
            10,
            151665 + 2 + 1025,
        )  # base + line_tokens + coord_tokens

        # Create test data with coordinate tokens
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Set some coordinate token labels
        coord_start = 151667
        labels[0, 5] = coord_start + 100  # <|coord_100|>
        labels[0, 6] = coord_start + 200  # <|coord_200|>

        # Create coordinate mask
        coord_mask = torch.zeros_like(labels, dtype=torch.bool)
        coord_mask[0, 5] = True
        coord_mask[0, 6] = True

        # Compute loss
        loss_components = loss_manager.compute_loss_components(
            logits=logits,
            labels=labels,
            coord_mask=coord_mask,
            teacher_spans=None,
            student_spans=None,
        )

        # Should have coordinate loss
        assert loss_components.student_l1_loss is not None
        assert torch.isfinite(loss_components.student_l1_loss)

    def test_loss_aggregation_weights(self, loss_manager, mock_config):
        """Test that loss components are properly weighted."""
        batch_size, seq_len, vocab_size = (
            1,
            15,
            151665 + 2 + 1026,
        )  # base + line_tokens + coord_tokens

        # Create test data
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Define spans
        teacher_spans = [[(2, 5)]]
        student_spans = [[(8, 12)]]

        # Compute loss
        loss_components = loss_manager.compute_loss_components(
            logits=logits,
            labels=labels,
            teacher_spans=teacher_spans,
            student_spans=student_spans,
        )

        # Verify total loss is sum of weighted components
        # Note: LossManager applies both specific weights AND regular/coordinate weights
        expected_total = 0.0
        if loss_components.teacher_llm_loss is not None:
            expected_total += loss_components.teacher_llm_loss
        if loss_components.student_llm_loss is not None:
            expected_total += loss_components.student_llm_loss
        if loss_components.teacher_l1_loss is not None:
            expected_total += loss_components.teacher_l1_loss
        if loss_components.student_l1_loss is not None:
            expected_total += loss_components.student_l1_loss

        # Allow for small floating point differences
        # Avoid wrapping a tensor in torch.tensor() which raises a warning
        assert torch.allclose(
            loss_components.loss, expected_total.clone().detach(), atol=1e-6
        )

    def test_empty_spans_handling(self, loss_manager):
        """Test handling of empty spans."""
        batch_size, seq_len, vocab_size = (
            1,
            10,
            151665 + 2 + 1027,
        )  # base + line_tokens + coord_tokens

        # Create test data
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test with empty spans
        loss_components = loss_manager.compute_loss_components(
            logits=logits,
            labels=labels,
            teacher_spans=[[]],  # Empty spans
            student_spans=[[]],
        )

        # Should still compute a valid total loss
        assert isinstance(loss_components, LossComponents)
        assert torch.isfinite(loss_components.loss)

    def test_invalid_spans_handling(self, loss_manager):
        """Test handling of invalid span boundaries."""
        batch_size, seq_len, vocab_size = (
            1,
            10,
            151665 + 2 + 1026,
        )  # base + line_tokens + coord_tokens

        # Create test data
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test with invalid spans (out of bounds)
        invalid_spans = [[(15, 20)]]  # Beyond sequence length

        loss_components = loss_manager.compute_loss_components(
            logits=logits,
            labels=labels,
            teacher_spans=invalid_spans,
            student_spans=None,
        )

        # Should handle gracefully
        assert isinstance(loss_components, LossComponents)
        assert torch.isfinite(loss_components.loss)

    def test_next_token_alignment_ce_min_loss_when_logits_match_next_label(
        self, loss_manager
    ):
        """CE should be near-zero when logits[t] put mass on labels[t+1] (next-token alignment)."""
        batch_size, seq_len = 1, 5
        vocab_size = 151665 + 2 + 1025
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100)

        # Choose non-coordinate token ids well below coord_start
        tok1, tok2, tok3 = 42, 43, 44
        # Labels at t=1..3 (so next-token targets exist)
        labels[0, 1] = tok1
        labels[0, 2] = tok2
        labels[0, 3] = tok3
        # Make logits[t] peak at labels[t+1]
        logits[0, 0, tok1] = 50.0
        logits[0, 1, tok2] = 50.0
        logits[0, 2, tok3] = 50.0
        # The rest remain 0

        comp = loss_manager.compute_loss_components(logits=logits, labels=labels)
        assert comp.student_llm_loss is not None
        # With strong peaks at the correct next labels, CE should be very small
        assert comp.student_llm_loss.item() < 1e-3

    def test_coordinate_alignment_with_spans_shifted(self, loss_manager):
        """Coordinate L1 should align to next-token positions inside spans (label-based, shifted)."""
        batch_size, seq_len = 1, 8
        vocab_size = 151665 + 2 + 1025
        coord_start = 151667
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100)

        # Put coordinate targets at positions 3 and 4 inside a teacher span [3,5)
        labels[0, 3] = coord_start + 100
        labels[0, 4] = coord_start + 200
        teacher_spans = [[(3, 5)]]
        student_spans = None

        # Since the model predicts next token, set logits at t=2 and t=3 to peak at those coords
        logits[0, 2, coord_start + 100] = 50.0  # predicts labels[3]
        logits[0, 3, coord_start + 200] = 50.0  # predicts labels[4]

        comp = loss_manager.compute_loss_components(
            logits=logits,
            labels=labels,
            teacher_spans=teacher_spans,
            student_spans=student_spans,
        )
        # Teacher coordinate loss should be very small (correct predictions)
        assert comp.teacher_l1_loss is not None
        assert comp.teacher_l1_loss.item() < 1e-3
        # CE now includes coordinate targets as well
        assert comp.teacher_llm_loss is not None
        assert torch.isfinite(comp.teacher_llm_loss)

    def test_llm_includes_coordinate_targets_with_spans(self, loss_manager):
        """When span contains only coordinate targets, CE component should still be present (included)."""
        batch_size, seq_len = 1, 6
        vocab_size = 151665 + 2 + 1025
        coord_start = 151667
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100)
        # Coordinates in span
        labels[0, 2] = coord_start + 5
        labels[0, 3] = coord_start + 6
        teacher_spans = [[(2, 4)]]

        comp = loss_manager.compute_loss_components(
            logits=logits, labels=labels, teacher_spans=teacher_spans
        )
        # LLM loss included for coordinate-only targets
        assert comp.teacher_llm_loss is not None
        assert torch.isfinite(comp.teacher_llm_loss)
        # Coordinate loss present
        assert comp.teacher_l1_loss is not None
        assert torch.isfinite(comp.teacher_l1_loss)
