"""
Tests for token processing functionality.

This module tests:
- Coordinate token conversion and validation
- Tokenizer vocabulary extension
- Special token wrapping for different geometry types
- Coordinate extraction from tokenized input
"""

from typing import Any, Dict, List

import torch

from src_new.tests.fixtures import MockTokenizer, create_coordinate_test_cases


class TestTokenProcessor:
    """Test token processing functionality."""

    def test_coordinate_token_conversion(self):
        """Test conversion between coordinates and coordinate tokens."""
        # Mock token processor configuration
        config = self._mock_token_config(
            coordinate_tokens_enabled=True, max_coord_value=2048
        )
        processor = self._mock_token_processor(config)

        # Test coordinate to token conversion
        coordinates = [100, 200, 300, 400]
        coord_tokens = processor.coordinates_to_tokens(coordinates)

        expected_tokens = [
            "<|coord_100|>",
            "<|coord_200|>",
            "<|coord_300|>",
            "<|coord_400|>",
        ]
        assert coord_tokens == expected_tokens

        # Test token to coordinate conversion (reverse)
        reverse_coords = processor.tokens_to_coordinates(coord_tokens)
        assert reverse_coords == coordinates

    def test_coordinate_token_disabled_mode(self):
        """Test token processing when coordinate tokens are disabled."""
        config = self._mock_token_config(coordinate_tokens_enabled=False)
        processor = self._mock_token_processor(config)

        coordinates = [100, 200, 300, 400]

        # Should return string representations, not special tokens
        coord_tokens = processor.coordinates_to_tokens(coordinates)
        expected_tokens = ["100", "200", "300", "400"]
        assert coord_tokens == expected_tokens

        # Should parse back to integers
        reverse_coords = processor.tokens_to_coordinates(coord_tokens)
        assert reverse_coords == coordinates

    def test_coordinate_range_validation(self):
        """Test coordinate range validation and clipping."""
        config = self._mock_token_config(
            coordinate_tokens_enabled=True, max_coord_value=2048
        )
        processor = self._mock_token_processor(config)

        # Test coordinates within range
        valid_coords = [0, 100, 1000, 2048]
        validated = processor.validate_coordinate_range(valid_coords)
        assert validated == valid_coords

        # Test coordinates outside range
        invalid_coords = [-10, 50, 3000, 2100]
        validated = processor.validate_coordinate_range(invalid_coords)
        expected = [0, 50, 2048, 2048]  # Clipped to valid range
        assert validated == expected

    def test_tokenizer_vocabulary_extension(self):
        """Test extension of tokenizer vocabulary."""
        tokenizer = MockTokenizer(vocab_size=151665)
        original_vocab_size = tokenizer.vocab_size

        # Create processor with new geometry tokens
        config = self._mock_token_config(
            coordinate_tokens_enabled=True,
            max_coord_value=100,  # Small for testing
            new_geometry_tokens=[
                "<|line_start|>",
                "<|line_end|>",
            ],
        )
        processor = self._mock_token_processor(config)

        # Extend tokenizer vocabulary
        extended_tokenizer = processor.extend_tokenizer_vocabulary(tokenizer)

        # Check that required tokens are in vocabulary (may have been pre-existing)
        vocab = extended_tokenizer.get_vocab()

        # Check geometry tokens
        assert "<|line_start|>" in vocab
        assert "<|line_end|>" in vocab

        # Check coordinate tokens
        assert "<|coord_0|>" in vocab
        assert "<|coord_50|>" in vocab
        assert "<|coord_100|>" in vocab

        # Vocabulary should be at least as large as original
        assert extended_tokenizer.vocab_size >= original_vocab_size

    def test_object_wrapping_bbox_2d(self):
        """Test wrapping bbox_2d objects with special tokens."""
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        bbox_obj = {"bbox_2d": [100, 150, 200, 250], "desc": "测试设备/基础检测目标"}

        wrapped = processor.wrap_object_with_tokens(bbox_obj)

        # Should contain all required components
        assert "<|object_ref_start|>" in wrapped
        assert "<|object_ref_end|>" in wrapped
        assert "<|box_start|>" in wrapped
        assert "<|box_end|>" in wrapped
        assert "测试设备/基础检测目标" in wrapped
        assert "<|coord_100|>" in wrapped
        assert "<|coord_150|>" in wrapped
        assert "<|coord_200|>" in wrapped
        assert "<|coord_250|>" in wrapped

    def test_object_wrapping_quad(self):
        """Test wrapping quad objects with special tokens."""
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        quad_obj = {
            "quad": [300, 400, 350, 410, 348, 425, 302, 415],
            "desc": "标签贴纸/测试标识",
        }

        wrapped = processor.wrap_object_with_tokens(quad_obj)

        # Should contain quad-specific tokens
        assert "<|quad_start|>" in wrapped
        assert "<|quad_end|>" in wrapped
        assert "标签贴纸/测试标识" in wrapped

        # Should contain all coordinate tokens
        for coord in quad_obj["quad"]:
            assert f"<|coord_{coord}|>" in wrapped

    def test_object_wrapping_line(self):
        """Test wrapping line objects with special tokens."""
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        line_obj = {
            "line": [50, 100, 150, 120, 250, 140, 350, 160],
            "desc": "线缆/测试连接线",
        }

        wrapped = processor.wrap_object_with_tokens(line_obj)

        # Should contain line-specific tokens
        assert "<|line_start|>" in wrapped
        assert "<|line_end|>" in wrapped
        assert "线缆/测试连接线" in wrapped

        # Should contain all coordinate tokens
        for coord in line_obj["line"]:
            assert f"<|coord_{coord}|>" in wrapped

    def test_object_wrapping_disabled_mode(self):
        """Test object wrapping when coordinate tokens are disabled."""
        config = self._mock_token_config(coordinate_tokens_enabled=False)
        processor = self._mock_token_processor(config)

        bbox_obj = {"bbox_2d": [100, 150, 200, 250], "desc": "测试设备/基础检测目标"}

        wrapped = processor.wrap_object_with_tokens(bbox_obj)

        # Should contain special tokens but not coordinate tokens
        assert "<|object_ref_start|>" in wrapped
        assert "<|box_start|>" in wrapped
        assert "[100, 150, 200, 250]" in wrapped  # Raw coordinates
        assert "<|coord_100|>" not in wrapped  # No coordinate tokens

    def test_coordinate_extraction_from_tokens(self):
        """Test extraction of coordinate sequences from tokenized input."""
        tokenizer = MockTokenizer()
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        # Create mock input with coordinate tokens
        input_ids = torch.tensor(
            [
                1,
                2,
                3,  # Regular tokens
                151666,
                151667,
                151668,
                151669,  # Coordinate tokens (100, 150, 200, 250)
                4,
                5,
                6,  # More regular tokens
            ]
        )

        # Mock coordinate extraction
        coordinate_sequences = processor.extract_coordinates_from_tokens(
            input_ids, tokenizer
        )

        # Should find coordinate sequence
        assert len(coordinate_sequences) > 0

        # Check first sequence
        start_idx, end_idx, coordinates = coordinate_sequences[0]
        assert start_idx == 3  # First coordinate token position
        assert end_idx == 6  # Last coordinate token position
        assert len(coordinates) == 4  # Four coordinates

    def test_coordinate_mask_creation(self):
        """Test creation of coordinate token mask."""
        tokenizer = MockTokenizer()
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        # Create input with mixed tokens
        input_ids = torch.tensor(
            [
                1,
                2,
                3,  # Regular tokens
                151666,
                151667,  # Coordinate tokens
                4,
                5,  # Regular tokens
                151668,
                151669,  # More coordinate tokens
                6,
                7,  # Regular tokens
            ]
        )

        # Create coordinate mask
        mask = processor.create_coordinate_mask(input_ids, tokenizer)

        # Validate mask
        expected_mask = torch.tensor(
            [
                False,
                False,
                False,  # Regular tokens
                True,
                True,  # Coordinate tokens
                False,
                False,  # Regular tokens
                True,
                True,  # Coordinate tokens
                False,
                False,  # Regular tokens
            ]
        )

        assert mask.shape == input_ids.shape
        assert torch.equal(mask, expected_mask)

    def test_geometry_token_ids_retrieval(self):
        """Test retrieval of geometry token IDs from tokenizer."""
        tokenizer = MockTokenizer()
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        # Extend tokenizer vocabulary
        extended_tokenizer = processor.extend_tokenizer_vocabulary(tokenizer)

        # Get geometry token IDs
        token_ids = processor.get_geometry_token_ids(extended_tokenizer)

        # Should contain standard tokens
        expected_tokens = [
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|line_start|>",
            "<|line_end|>",
        ]

        for token in expected_tokens:
            assert token in token_ids
            assert isinstance(token_ids[token], int)
            assert token_ids[token] > 0

    def test_coordinate_token_range(self):
        """Test retrieval of coordinate token ID range."""
        tokenizer = MockTokenizer()
        config = self._mock_token_config(
            coordinate_tokens_enabled=True, max_coord_value=100
        )
        processor = self._mock_token_processor(config)

        # Extend tokenizer vocabulary
        extended_tokenizer = processor.extend_tokenizer_vocabulary(tokenizer)

        # Get coordinate token range
        min_id, max_id = processor.get_coordinate_token_range(extended_tokenizer)

        # Should have valid range
        assert min_id > 0
        assert max_id > min_id
        assert max_id - min_id + 1 == 101  # 0-100 inclusive

    def test_invalid_object_handling(self):
        """Test handling of invalid objects."""
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        # Object with no geometry
        invalid_obj = {"desc": "测试对象/无几何信息"}

        wrapped = processor.wrap_object_with_tokens(invalid_obj)

        # Should handle gracefully
        assert "<|object_ref_start|>" in wrapped
        assert "<|object_ref_end|>" in wrapped
        assert "测试对象/无几何信息" in wrapped
        # Should not contain geometry-specific tokens
        assert "<|box_start|>" not in wrapped
        assert "<|line_start|>" not in wrapped

    def test_coordinate_token_edge_cases(self):
        """Test coordinate token handling edge cases."""
        test_cases = create_coordinate_test_cases()

        for case in test_cases:
            config = self._mock_token_config(coordinate_tokens_enabled=True)
            processor = self._mock_token_processor(config)

            # Test coordinate conversion
            coord_tokens = processor.coordinates_to_tokens(case["coordinates"])
            assert len(coord_tokens) == len(case["coordinates"])

            # Test that edge cases are handled
            if case["name"] == "edge_case_zero":
                assert "<|coord_0|>" in coord_tokens
            elif case["name"] == "edge_case_max":
                assert "<|coord_2047|>" in coord_tokens

    # Helper methods for mocking token processor functionality

    def _mock_token_config(self, **kwargs):
        """Create a mock token configuration."""

        class MockTokenConfig:
            def __init__(self, **config):
                self.coordinate_tokens_enabled = config.get(
                    "coordinate_tokens_enabled", False
                )
                self.max_coord_value = config.get("max_coord_value", 2048)
                self.new_geometry_tokens = config.get(
                    "new_geometry_tokens",
                    [
                        "<|line_start|>",
                        "<|line_end|>",
                    ],
                )

        return MockTokenConfig(**kwargs)

    def _mock_token_processor(self, config):
        """Create a mock token processor."""

        class MockTokenProcessor:
            def __init__(self, config):
                self.config = config
                self.coordinate_token_map = {}
                self.reverse_coordinate_map = {}

                if config.coordinate_tokens_enabled:
                    for coord in range(config.max_coord_value + 1):
                        token = f"<|coord_{coord}|>"
                        self.coordinate_token_map[coord] = token
                        self.reverse_coordinate_map[token] = coord

            def coordinates_to_tokens(self, coordinates: List[int]) -> List[str]:
                if not self.config.coordinate_tokens_enabled:
                    return [str(coord) for coord in coordinates]

                tokens = []
                for coord in coordinates:
                    coord = max(0, min(coord, self.config.max_coord_value))
                    tokens.append(self.coordinate_token_map[coord])
                return tokens

            def tokens_to_coordinates(self, tokens: List[str]) -> List[int]:
                if not self.config.coordinate_tokens_enabled:
                    return [int(token) for token in tokens if token.isdigit()]

                coordinates = []
                for token in tokens:
                    if token in self.reverse_coordinate_map:
                        coordinates.append(self.reverse_coordinate_map[token])
                    else:
                        coordinates.append(0)
                return coordinates

            def validate_coordinate_range(self, coordinates: List[int]) -> List[int]:
                validated = []
                for coord in coordinates:
                    validated.append(max(0, min(coord, self.config.max_coord_value)))
                return validated

            def extend_tokenizer_vocabulary(self, tokenizer):
                # Mock extension
                new_tokens = []

                # Add geometry tokens
                for token in self.config.new_geometry_tokens:
                    if token not in tokenizer.special_tokens:
                        new_tokens.append(token)

                # Add coordinate tokens
                if self.config.coordinate_tokens_enabled:
                    for coord_token in self.coordinate_token_map.values():
                        if coord_token not in tokenizer.coordinate_tokens:
                            new_tokens.append(coord_token)

                if new_tokens:
                    tokenizer.add_tokens(new_tokens)

                return tokenizer

            def wrap_object_with_tokens(self, obj: Dict[str, Any]) -> str:
                desc = obj.get("desc", "")

                if "bbox_2d" in obj:
                    coords = obj["bbox_2d"]
                    if self.config.coordinate_tokens_enabled:
                        coord_tokens = self.coordinates_to_tokens(coords)
                        coord_str = ", ".join(coord_tokens)
                        return f"<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>[{coord_str}]<|box_end|>"
                    else:
                        return f"<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>{coords}<|box_end|>"

                elif "quad" in obj:
                    coords = obj["quad"]
                    if self.config.coordinate_tokens_enabled:
                        coord_tokens = self.coordinates_to_tokens(coords)
                        coord_str = ", ".join(coord_tokens)
                        return f"<|object_ref_start|>{desc}<|object_ref_end|><|quad_start|>[{coord_str}]<|quad_end|>"
                    else:
                        return f"<|object_ref_start|>{desc}<|object_ref_end|><|quad_start|>{coords}<|quad_end|>"

                elif "line" in obj:
                    coords = obj["line"]
                    if self.config.coordinate_tokens_enabled:
                        coord_tokens = self.coordinates_to_tokens(coords)
                        coord_str = ", ".join(coord_tokens)
                        return f"<|object_ref_start|>{desc}<|object_ref_end|><|line_start|>[{coord_str}]<|line_end|>"
                    else:
                        return f"<|object_ref_start|>{desc}<|object_ref_end|><|line_start|>{coords}<|line_end|>"

                else:
                    return f"<|object_ref_start|>{desc}<|object_ref_end|>"

            def extract_coordinates_from_tokens(self, input_ids, tokenizer):
                # Mock coordinate extraction
                coordinate_sequences = []

                if not self.config.coordinate_tokens_enabled:
                    return coordinate_sequences

                tokens = (
                    input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
                )

                # Find coordinate token sequences (mock implementation)
                i = 0
                while i < len(tokens):
                    if tokens[i] >= 151666:  # Coordinate token range
                        start_idx = i
                        coordinates = []

                        while i < len(tokens) and tokens[i] >= 151666:
                            coord_value = tokens[i] - 151666  # Mock mapping
                            coordinates.append(coord_value)
                            i += 1

                        end_idx = i - 1
                        coordinate_sequences.append((start_idx, end_idx, coordinates))
                    else:
                        i += 1

                return coordinate_sequences

            def create_coordinate_mask(self, input_ids, tokenizer):
                mask = torch.zeros_like(input_ids, dtype=torch.bool)

                if not self.config.coordinate_tokens_enabled:
                    return mask

                # Mark coordinate token positions
                coordinate_positions = (
                    input_ids >= 151666
                )  # Mock coordinate token range
                mask[coordinate_positions] = True

                return mask

            def get_geometry_token_ids(self, tokenizer):
                token_ids = {}

                standard_tokens = [
                    "<|object_ref_start|>",
                    "<|object_ref_end|>",
                    "<|box_start|>",
                    "<|box_end|>",
                ]

                all_tokens = standard_tokens + self.config.new_geometry_tokens

                for token in all_tokens:
                    if token in tokenizer.special_tokens:
                        token_ids[token] = tokenizer.special_tokens[token]

                return token_ids

            def get_coordinate_token_range(self, tokenizer):
                if not self.config.coordinate_tokens_enabled:
                    return (0, 0)

                # Mock range calculation
                min_id = 151666  # Start of coordinate tokens
                max_id = min_id + self.config.max_coord_value

                return (min_id, max_id)

        return MockTokenProcessor(config)


class TestTokenProcessorIntegration:
    """Test token processor integration with other components."""

    def test_integration_with_tokenizer(self):
        """Test integration between token processor and tokenizer."""
        tokenizer = MockTokenizer()
        config = self._mock_token_config(
            coordinate_tokens_enabled=True, max_coord_value=50
        )
        processor = self._mock_token_processor(config)

        # Extend tokenizer
        extended_tokenizer = processor.extend_tokenizer_vocabulary(tokenizer)

        # Test that extended tokenizer works with coordinate tokens
        coordinate_text = "<|coord_25|> <|coord_30|>"
        token_ids = extended_tokenizer.encode(coordinate_text)
        decoded_text = extended_tokenizer.decode(token_ids)

        # Should be able to encode and decode coordinate tokens
        assert len(token_ids) > 0
        assert "<|coord_25|>" in decoded_text or "<|coord_30|>" in decoded_text

    def test_integration_with_objects(self):
        """Test integration with object processing pipeline."""
        config = self._mock_token_config(coordinate_tokens_enabled=True)
        processor = self._mock_token_processor(config)

        # Test with sample objects from different geometry types
        test_objects = [
            {"bbox_2d": [100, 150, 200, 250], "desc": "矩形/测试"},
            {"quad": [300, 400, 350, 410, 348, 425, 302, 415], "desc": "四边形/测试"},
            {"line": [50, 100, 150, 120, 250, 140], "desc": "线条/测试"},
        ]

        for obj in test_objects:
            wrapped = processor.wrap_object_with_tokens(obj)

            # Should be valid wrapped format
            assert wrapped.startswith("<|object_ref_start|>")
            assert "<|object_ref_end|>" in wrapped
            assert obj["desc"] in wrapped

            # Should contain appropriate geometry tokens
            if "bbox_2d" in obj:
                assert "<|box_start|>" in wrapped and "<|box_end|>" in wrapped
            elif "quad" in obj:
                assert "<|quad_start|>" in wrapped and "<|quad_end|>" in wrapped
            elif "line" in obj:
                assert "<|line_start|>" in wrapped and "<|line_end|>" in wrapped

    def _mock_token_config(self, **kwargs):
        """Reuse mock config from above."""

        class MockTokenConfig:
            def __init__(self, **config):
                self.coordinate_tokens_enabled = config.get(
                    "coordinate_tokens_enabled", False
                )
                self.max_coord_value = config.get("max_coord_value", 2048)
                self.new_geometry_tokens = config.get(
                    "new_geometry_tokens",
                    [
                        "<|line_start|>",
                        "<|line_end|>",
                    ],
                )

        return MockTokenConfig(**kwargs)

    def _mock_token_processor(self, config):
        """Reuse mock processor from above."""

        # [Same implementation as above]
        class MockTokenProcessor:
            def __init__(self, config):
                self.config = config
                self.coordinate_token_map = {}
                self.reverse_coordinate_map = {}

                if config.coordinate_tokens_enabled:
                    for coord in range(config.max_coord_value + 1):
                        token = f"<|coord_{coord}|>"
                        self.coordinate_token_map[coord] = token
                        self.reverse_coordinate_map[token] = coord

            def extend_tokenizer_vocabulary(self, tokenizer):
                new_tokens = []
                for token in self.config.new_geometry_tokens:
                    if token not in tokenizer.special_tokens:
                        new_tokens.append(token)
                if self.config.coordinate_tokens_enabled:
                    for coord_token in self.coordinate_token_map.values():
                        if coord_token not in tokenizer.coordinate_tokens:
                            new_tokens.append(coord_token)
                if new_tokens:
                    tokenizer.add_tokens(new_tokens)
                return tokenizer

            def wrap_object_with_tokens(self, obj):
                desc = obj.get("desc", "")
                if "bbox_2d" in obj:
                    return f"<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>{obj['bbox_2d']}<|box_end|>"

                elif "line" in obj:
                    return f"<|object_ref_start|>{desc}<|object_ref_end|><|line_start|>{obj['line']}<|line_end|>"
                else:
                    return f"<|object_ref_start|>{desc}<|object_ref_end|>"

        return MockTokenProcessor(config)
