"""
Edge Case Tests for Multi-Geometry Token Parser

Tests edge cases, boundary conditions, and unusual input patterns
for the multi-geometry token parsing functionality.
"""

import sys
from pathlib import Path
from unittest.mock import patch


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger_utils import configure_global_logging, get_logger
from src.utils.response_parser import ResponseParser


logger = get_logger("test_multi_geometry_edge_cases")


class TestMultiGeometryEdgeCases:
    """Edge case tests for multi-geometry token parsing."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        configure_global_logging(rank=0, world_size=1)
        cls.parser = ResponseParser()

    def test_empty_content_between_tokens(self):
        """Test handling of empty content between geometry tokens."""
        test_cases = [
            "<obj_ref_start><bbox_2d_start><bbox_2d_end><obj_ref_end>",
            "<obj_ref_start><line_start><line_end><obj_ref_end>",
            "<obj_ref_start><square_start><square_end><obj_ref_end>",
        ]

        for text in test_cases:
            objects = self.parser._parse_multi_geometry_tokens(
                text, coordinate_tokens_enabled=False
            )
            assert len(objects) == 0, f"Expected no objects for empty content: {text}"

    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content between tokens."""
        test_cases = [
            "<obj_ref_start><bbox_2d_start>   <bbox_2d_end><obj_ref_end>",
            "<obj_ref_start><bbox_2d_start>\t\n<bbox_2d_end><obj_ref_end>",
            "<obj_ref_start><bbox_2d_start>     \t  \n  <bbox_2d_end><obj_ref_end>",
        ]

        for text in test_cases:
            objects = self.parser._parse_multi_geometry_tokens(
                text, coordinate_tokens_enabled=False
            )
            assert len(objects) == 0, (
                f"Expected no objects for whitespace-only content: {text}"
            )

    def test_multiple_geometry_objects_single_response(self):
        """Test parsing multiple geometry objects in a single response."""
        text = (
            "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"
            "<obj_ref_start><line_start>光纤<|coord_100|><|coord_200|><|coord_150|><|coord_250|><line_end><obj_ref_end>"
            "<obj_ref_start><square_start>标签<|coord_50|><|coord_60|><|coord_80|><|coord_65|><|coord_85|><|coord_95|><|coord_55|><|coord_90|><square_end><obj_ref_end>"
        )

        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=True
        )

        assert len(objects) == 3

        # Verify first object (bbox_2d)
        assert objects[0]["geometry_type"] == "bbox_2d"
        assert objects[0]["caption"] == "BBU设备"
        assert objects[0]["coordinates"] == [150, 10, 211, 35]

        # Verify second object (line)
        assert objects[1]["geometry_type"] == "line"
        assert objects[1]["caption"] == "光纤"
        assert objects[1]["coordinates"] == [100, 200, 150, 250]

        # Verify third object (square)
        assert objects[2]["geometry_type"] == "square"
        assert objects[2]["caption"] == "标签"
        assert objects[2]["coordinates"] == [50, 60, 80, 65, 85, 95, 55, 90]

    def test_mixed_coordinate_and_caption_only_objects(self):
        """Test parsing mixed objects with and without coordinates."""
        text = (
            "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"
            "<obj_ref_start><line_start>光纤<line_end><obj_ref_end>"  # No coordinates
            "<obj_ref_start><square_start>标签<|coord_50|><|coord_60|><square_end><obj_ref_end>"  # Partial coordinates
        )

        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=True
        )

        assert len(objects) == 3

        # First object should have coordinates
        assert objects[0]["coordinates"] == [150, 10, 211, 35]
        assert objects[0]["formatted_output"] == "bbox_2d:BBU设备[150,10,211,35]"

        # Second object should have no coordinates
        assert objects[1]["coordinates"] == []
        assert objects[1]["formatted_output"] == "line:光纤"

        # Third object should have partial coordinates
        assert objects[2]["coordinates"] == [50, 60]
        assert objects[2]["formatted_output"] == "square:标签[50,60]"

    def test_large_coordinate_values_boundary_conditions(self):
        """Test handling of large coordinate values at boundary conditions."""
        test_cases = [
            ("<|coord_0|>", [0]),  # Minimum value
            ("<|coord_2047|>", [2047]),  # Typical maximum
            ("<|coord_4096|>", [4096]),  # Warning threshold
            ("<|coord_9999|>", [9999]),  # Very large value
        ]

        for coord_token, expected_coords in test_cases:
            content = f"测试{coord_token}"

            with patch.object(self.parser.parser_logger, "warning") as mock_warning:
                result = self.parser._parse_geometry_content(
                    content, coordinate_tokens_enabled=True
                )

                assert result is not None
                assert result["coordinates"] == expected_coords

                # Should warn for values > 4096
                if expected_coords[0] > 4096:
                    mock_warning.assert_called()
                else:
                    mock_warning.assert_not_called()

    def test_many_coordinate_tokens(self):
        """Test handling of many coordinate tokens (boundary condition)."""
        # Create 50 coordinate tokens
        coords = "".join(f"<|coord_{i}|>" for i in range(50))
        content = f"测试{coords}"

        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=True
        )

        assert result is not None
        assert result["caption"] == "测试"
        assert len(result["coordinates"]) == 50
        assert result["coordinates"] == list(range(50))

    def test_extremely_many_coordinate_tokens(self):
        """Test handling of extremely many coordinate tokens (>100)."""
        # Create 150 coordinate tokens (should trigger warning)
        coords = "".join(f"<|coord_{i}|>" for i in range(150))
        content = f"测试{coords}"

        with patch.object(self.parser.parser_logger, "warning") as mock_warning:
            result = self.parser._parse_geometry_content(
                content, coordinate_tokens_enabled=True
            )

            assert result is not None
            assert len(result["coordinates"]) == 150
            mock_warning.assert_called()  # Should warn about large number of coordinates

    def test_alternative_token_patterns_spacing(self):
        """Test alternative token patterns with various spacing."""
        test_cases = [
            (
                "< obj_ref_start >< bbox_2d_start >BBU设备< bbox_2d_end >< obj_ref_end >",
                True,
            ),
            (
                "<obj_ref_start> <bbox_2d_start> BBU设备 <bbox_2d_end> <obj_ref_end>",
                False,
            ),  # This pattern doesn't work
            ("<obj_ref_start><bbox_2d_start> BBU设备 <bbox_2d_end><obj_ref_end>", True),
            (
                "<\tobj_ref_start\t><\tbbox_2d_start\t>BBU设备<\tbbox_2d_end\t><\tobj_ref_end\t>",
                True,
            ),
        ]

        for text, should_parse in test_cases:
            objects = self.parser._parse_multi_geometry_tokens(
                text, coordinate_tokens_enabled=False
            )

            if should_parse:
                assert len(objects) == 1, f"Failed to parse: {text}"
                assert objects[0]["geometry_type"] == "bbox_2d"
                assert objects[0]["caption"].strip() == "BBU设备"
            else:
                # Some spacing patterns may not be supported
                assert len(objects) == 0, f"Unexpectedly parsed: {text}"

    def test_case_insensitive_alternative_patterns(self):
        """Test case-insensitive alternative patterns."""
        test_cases = [
            "<OBJ_REF_START><BBOX_2D_START>BBU设备<BBOX_2D_END><OBJ_REF_END>",
            "<Obj_Ref_Start><Bbox_2d_Start>BBU设备<Bbox_2d_End><Obj_Ref_End>",
        ]

        for text in test_cases:
            objects = self.parser._parse_multi_geometry_tokens(
                text, coordinate_tokens_enabled=False
            )

            assert len(objects) == 1, f"Failed to parse case variant: {text}"
            assert objects[0]["geometry_type"] == "bbox_2d"
            assert objects[0]["caption"] == "BBU设备"

    def test_malformed_token_sequences(self):
        """Test handling of malformed token sequences."""
        malformed_cases = [
            "<obj_ref_start><bbox_2d_start>BBU设备<line_end><obj_ref_end>",  # Mismatched geometry tokens
            "<obj_ref_start><bbox_2d_start>BBU设备<obj_ref_end>",  # Missing geometry end
            "<bbox_2d_start>BBU设备<bbox_2d_end><obj_ref_end>",  # Missing obj_ref_start
            "<obj_ref_start>BBU设备<bbox_2d_end><obj_ref_end>",  # Missing geometry start
            "<obj_ref_start><bbox_2d_start>BBU设备<bbox_2d_end>",  # Missing obj_ref_end
        ]

        for text in malformed_cases:
            objects = self.parser._parse_multi_geometry_tokens(
                text, coordinate_tokens_enabled=False
            )
            # Should handle gracefully (return empty list or skip malformed parts)
            assert isinstance(objects, list), (
                f"Should return list for malformed input: {text}"
            )

    def test_nested_token_like_content(self):
        """Test handling of content that looks like tokens but isn't."""
        text = "<obj_ref_start><bbox_2d_start>设备<fake_token>内容<|not_coord_123|>更多文本<bbox_2d_end><obj_ref_end>"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=True
        )

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "bbox_2d"
        # Should include fake tokens in caption since they're not real coordinate tokens
        assert "<fake_token>" in obj["caption"]
        assert "<|not_coord_123|>" in obj["caption"]
        assert obj["coordinates"] == []  # No valid coordinate tokens

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in captions."""
        test_cases = [
            "BBU设备/华为,显示完整",
            "螺丝、光纤插头/BBU安装螺丝",
            "标签/5G-BBU",
            "设备@#$%^&*()测试",
            "emoji测试🔧🎯📊",
            "换行\n测试\t制表符",
        ]

        for caption in test_cases:
            text = f"<obj_ref_start><bbox_2d_start>{caption}<bbox_2d_end><obj_ref_end>"
            objects = self.parser._parse_multi_geometry_tokens(
                text, coordinate_tokens_enabled=False
            )

            assert len(objects) == 1
            assert objects[0]["caption"] == caption
            assert objects[0]["formatted_output"] == f"bbox_2d:{caption}"

    def test_very_long_captions(self):
        """Test handling of very long captions."""
        long_caption = "很长的设备描述" * 100  # 500+ characters
        text = f"<obj_ref_start><bbox_2d_start>{long_caption}<bbox_2d_end><obj_ref_end>"

        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=False
        )

        assert len(objects) == 1
        assert objects[0]["caption"] == long_caption
        assert len(objects[0]["caption"]) > 500

    def test_coordinate_tokens_with_leading_zeros(self):
        """Test coordinate tokens with leading zeros."""
        content = "测试<|coord_007|><|coord_0123|><|coord_00|>"
        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=True
        )

        assert result is not None
        assert result["caption"] == "测试"
        # Should parse as integers (leading zeros ignored)
        assert result["coordinates"] == [7, 123, 0]

    def test_mixed_valid_invalid_coordinate_tokens(self):
        """Test mixed valid and invalid coordinate tokens."""
        content = (
            "测试<|coord_100|><|coord_abc|><|coord_200|><|coord_-50|><|coord_300|>"
        )

        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=True
        )

        assert result is not None
        # Invalid tokens remain in caption, valid ones are removed
        assert result["caption"] == "测试<|coord_abc|><|coord_-50|>"
        # Should only include valid coordinates
        assert result["coordinates"] == [100, 200, 300]

    def test_coordinate_tokens_in_caption_text(self):
        """Test coordinate tokens mixed within caption text."""
        content = "设备<|coord_100|>型号BBU<|coord_200|>版本V1<|coord_300|>"
        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=True
        )

        assert result is not None
        assert result["caption"] == "设备型号BBU版本V1"  # Coordinate tokens removed
        assert result["coordinates"] == [100, 200, 300]

    def test_empty_caption_after_coordinate_removal(self):
        """Test handling when caption becomes empty after coordinate token removal."""
        content = "<|coord_100|><|coord_200|><|coord_300|>"  # Only coordinate tokens

        with patch.object(self.parser.parser_logger, "warning") as mock_warning:
            result = self.parser._parse_geometry_content(
                content, coordinate_tokens_enabled=True
            )

            assert result is not None
            assert result["caption"] == ""
            assert result["coordinates"] == [100, 200, 300]
            mock_warning.assert_called()  # Should warn about empty caption
