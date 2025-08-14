#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Coordinate Token System Tests

This module tests the complete coordinate token system including conversion,
tokenizer extension, and coordinate processing functionality.

Key Features:
- Tests coordinate token conversion and validation
- Tests tokenizer vocabulary extension
- Tests coordinate mask creation
- Tests coordinate extraction from tokens
- Tests integration between components
"""

from unittest.mock import Mock

import pytest
import torch

from src_new.models.wrapper import CoordinateProcessor
from src_new.processing.coordinate_converter import CoordinateTokenConverter
from src_new.processing.token_processor import TokenConfig, TokenProcessor


class TestCoordinateTokenSystem:
    """Test suite for the complete coordinate token system."""

    @pytest.fixture
    def coordinate_converter(self):
        """Create coordinate token converter."""
        return CoordinateTokenConverter(
            max_coord_value=1024, coordinate_tokens_enabled=True
        )

    @pytest.fixture
    def token_config(self):
        """Create token processor configuration."""
        return TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=1024,
            new_geometry_tokens=[],
            coordinate_init_mode="fourier_ramp",
        )

    @pytest.fixture
    def token_processor(self, token_config):
        """Create token processor."""
        return TokenProcessor(token_config)

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.coordinate_tokens_enabled = True
        config.max_coord_value = 1024
        return config

    @pytest.fixture
    def coordinate_processor(self, mock_config):
        """Create coordinate processor."""
        return CoordinateProcessor(mock_config)

    def test_coordinate_converter_bbox_2d(self, coordinate_converter):
        """Test bbox_2d coordinate conversion."""
        objects = [{"bbox_2d": [100, 200, 300, 400], "desc": "Test device"}]

        result = coordinate_converter.convert_objects_to_tokens(objects)

        expected = "<|object_ref_start|>Test device<|object_ref_end|><|box_start|>[<|coord_100|>, <|coord_200|>, <|coord_300|>, <|coord_400|>]<|box_end|>"
        assert result == expected

    def test_coordinate_converter_quad(self, coordinate_converter):
        """Test quad coordinate conversion."""
        objects = [{"quad": [50, 60, 70, 80, 90, 100, 110, 120], "desc": "Test quad"}]

        result = coordinate_converter.convert_objects_to_tokens(objects)

        assert "<|quad_start|>" in result
        assert "<|quad_end|>" in result
        assert "Test quad" in result
        assert "<|coord_50|>" in result

    def test_coordinate_converter_line(self, coordinate_converter):
        """Test line coordinate conversion."""
        objects = [{"line": [10, 20, 30, 40, 50, 60], "desc": "Test line"}]

        result = coordinate_converter.convert_objects_to_tokens(objects)

        assert "<|line_start|>" in result
        assert "<|line_end|>" in result
        assert "Test line" in result
        assert "<|coord_10|>" in result

    def test_coordinate_converter_clamping(self, coordinate_converter):
        """Test coordinate clamping to valid range."""
        objects = [
            {
                "bbox_2d": [-10, 2000, 100, 200],  # Out of range coordinates
                "desc": "Test clamping",
            }
        ]

        result = coordinate_converter.convert_objects_to_tokens(objects)

        # -10 should be clamped to 0, 2000 should be clamped to 1024
        assert "<|coord_0|>" in result
        assert "<|coord_1024|>" in result
        assert "<|coord_-10|>" not in result
        assert "<|coord_2000|>" not in result

    def test_coordinate_converter_multiple_objects(self, coordinate_converter):
        """Test conversion of multiple objects."""
        objects = [
            {"bbox_2d": [100, 200, 150, 250], "desc": "Device 1"},
            {"quad": [50, 60, 70, 80, 90, 100, 110, 120], "desc": "Device 2"},
        ]

        result = coordinate_converter.convert_objects_to_tokens(objects)
        lines = result.split("\n")

        assert len(lines) == 2
        assert "Device 1" in lines[0]
        assert "Device 2" in lines[1]
        assert "<|box_start|>" in lines[0]
        assert "<|quad_start|>" in lines[1]

    def test_coordinate_converter_empty_objects(self, coordinate_converter):
        """Test handling of empty objects list."""
        with pytest.raises(ValueError, match="Empty objects list encountered"):
            coordinate_converter.convert_objects_to_tokens([])

    def test_coordinate_converter_invalid_geometry(self, coordinate_converter):
        """Test handling of invalid geometry types."""
        objects = [{"invalid_geom": [100, 200], "desc": "Invalid"}]

        with pytest.raises(ValueError, match="unsupported geometry type"):
            coordinate_converter.convert_objects_to_tokens(objects)

    def test_token_processor_vocabulary_extension(
        self, token_processor, real_base_tokenizer
    ):
        """Test tokenizer vocabulary is pre-expanded and usable (no runtime extension)."""
        # The model_path in bbu_v2_use_coord.yaml points to a pre-expanded cache with exact size 152692
        final_vocab_size = len(real_base_tokenizer)
        vocab = real_base_tokenizer.get_vocab()

        # Validate required tokens and ranges for max_coord_1024 cache
        assert vocab["<|line_start|>"] == 151665
        assert vocab["<|line_end|>"] == 151666
        ids = [vocab[f"<|coord_{i}|>"] for i in range(1025)]
        assert min(ids) == 151667 and max(ids) == 152691
        assert final_vocab_size == 152692

    def test_token_processor_coordinate_extraction(
        self, token_processor, real_extended_tokenizer
    ):
        """Test coordinate extraction from tokenized input with real tokenizer."""
        # Get vocabulary from extended tokenizer
        vocab = real_extended_tokenizer.get_vocab()

        # Find coordinate token IDs
        coord_token_ids = []
        for i in range(4):  # Test with first 4 coordinate tokens
            coord_token = f"<|coord_{i}|>"
            if coord_token in vocab:
                coord_token_ids.append(vocab[coord_token])

        # Skip test if coordinate tokens not found
        if len(coord_token_ids) < 4:
            pytest.skip("Not enough coordinate tokens found in extended tokenizer")

        # Create input with coordinate tokens
        input_ids = torch.tensor(coord_token_ids)

        # Extract coordinates
        sequences = token_processor.extract_coordinates_from_tokens(
            input_ids, real_extended_tokenizer
        )

        # Should find coordinate sequence
        assert len(sequences) > 0, "No coordinate sequences found"

    def test_coordinate_processor_initialization(
        self, coordinate_processor, mock_config
    ):
        """Test coordinate processor initialization."""
        assert coordinate_processor.config == mock_config
        assert (
            coordinate_processor.coordinate_tokens_enabled
            == mock_config.coordinate_tokens_enabled
        )
        assert coordinate_processor.max_coord_value == mock_config.max_coord_value

    def test_coordinate_processor_tokenizer_setting(self, coordinate_processor):
        """Test setting tokenizer in coordinate processor."""
        # Create mock tokenizer
        mock_tokenizer = Mock()
        vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
        for i in range(1025):
            vocab[f"<|coord_{i}|>"] = 151667 + i
        mock_tokenizer.get_vocab.return_value = vocab

        # Set tokenizer
        coordinate_processor.set_tokenizer(mock_tokenizer)

        # Should update coordinate token range
        assert coordinate_processor.coordinate_token_range is not None
        assert coordinate_processor.original_vocab_size is not None

    def test_coordinate_processor_mask_creation(self, coordinate_processor):
        """Test coordinate mask creation."""
        # Set up coordinate processor with tokenizer
        mock_tokenizer = Mock()
        vocab = {}
        for i in range(1025):
            vocab[f"<|coord_{i}|>"] = 151667 + i
        mock_tokenizer.get_vocab.return_value = vocab
        coordinate_processor.set_tokenizer(mock_tokenizer)

        # Create input with coordinate tokens
        input_ids = torch.tensor(
            [[151667, 151668, 100, 151669]]
        )  # coord tokens mixed with regular

        # Get coordinate mask
        mask = coordinate_processor.get_coordinate_mask(input_ids)

        # Should identify coordinate tokens
        expected_mask = torch.tensor([[True, True, False, True]])
        assert torch.equal(mask, expected_mask)

    def test_coordinate_processor_logit_masking(self, coordinate_processor):
        """Test coordinate logit masking."""
        # Set up coordinate processor
        mock_tokenizer = Mock()
        vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
        for i in range(1025):
            vocab[f"<|coord_{i}|>"] = 151667 + i
        mock_tokenizer.get_vocab.return_value = vocab
        coordinate_processor.set_tokenizer(mock_tokenizer)

        # Create test logits and input
        batch_size, seq_len, vocab_size = 1, 5, 152692  # Include coordinate tokens
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.tensor([[100, 151667, 200, 151668, 300]])  # Mixed tokens

        # Apply masking
        masked_logits = coordinate_processor.mask_coordinate_logits(
            logits, input_ids, coordinate_processor.coordinate_token_range
        )

        # Should return masked logits
        assert masked_logits.shape == logits.shape
        assert not torch.equal(
            masked_logits, logits
        )  # Should be different due to masking

    def test_coordinate_system_integration(self, coordinate_converter, token_processor):
        """Test integration between coordinate converter and token processor."""
        # Convert objects to tokens
        objects = [{"bbox_2d": [100, 200, 300, 400], "desc": "Test device"}]
        token_string = coordinate_converter.convert_objects_to_tokens(objects)

        # Verify token string contains coordinate tokens
        assert "<|coord_100|>" in token_string
        assert "<|coord_200|>" in token_string
        assert "<|coord_300|>" in token_string
        assert "<|coord_400|>" in token_string

        # Create mock tokenizer
        class SimpleTok:
            def __init__(self):
                self._v = {f"tok_{i}": i for i in range(151665)}

            def get_vocab(self):
                return dict(self._v)

            def add_special_tokens(self, d):
                toks = d.get("additional_special_tokens", [])
                for t in toks:
                    if t not in self._v:
                        self._v[t] = len(self._v)
                return len(toks)

        mock_tokenizer = SimpleTok()

        # Token processor should extend and then include coordinate tokens
        extended_tokenizer = token_processor.extend_tokenizer_vocabulary(mock_tokenizer)
        vocab = extended_tokenizer.get_vocab()
        assert vocab["<|line_start|>"] == 151665
        assert vocab["<|line_end|>"] == 151666
        ids = [vocab[f"<|coord_{i}|>"] for i in range(1025)]
        assert min(ids) == 151667 and max(ids) == 152691
        assert extended_tokenizer is not None

    def test_coordinate_system_disabled(self, real_base_tokenizer):
        """Test coordinate system when disabled."""
        # Create disabled configuration
        disabled_config = TokenConfig(
            coordinate_tokens_enabled=False,
            max_coord_value=1024,
            new_geometry_tokens=[],
            coordinate_init_mode="fourier_ramp",
        )

        disabled_processor = TokenProcessor(disabled_config)

        # Get current vocabulary size (already-expanded tokenizer)
        initial_vocab_size = len(real_base_tokenizer)

        # Should handle disabled state gracefully: return tokenizer unchanged
        extended_tokenizer = disabled_processor.extend_tokenizer_vocabulary(
            real_base_tokenizer
        )

        # Should return tokenizer without coordinate token modification
        final_vocab_size = len(extended_tokenizer)
        assert final_vocab_size == initial_vocab_size
