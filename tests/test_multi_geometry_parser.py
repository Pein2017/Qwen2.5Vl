"""
Unit Tests for Multi-Geometry Token Parser

Tests the _parse_multi_geometry_tokens() and related methods in ResponseParser
with comprehensive coverage of coordinate token parsing functionality.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger_utils import configure_global_logging, get_logger
from src.utils.response_parser import ResponseParser


logger = get_logger("test_multi_geometry_parser")


class TestMultiGeometryParser:
    """Unit tests for multi-geometry token parsing functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        configure_global_logging(rank=0, world_size=1)
        cls.parser = ResponseParser()

    def test_parse_geometry_content_caption_only(self):
        """Test parsing content with caption only (coordinate_tokens_enabled=False)."""
        content = "BBU设备"
        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=False
        )

        assert result is not None
        assert result["caption"] == "BBU设备"
        assert result["coordinates"] == []

    def test_parse_geometry_content_with_coordinates(self):
        """Test parsing content with coordinate tokens (coordinate_tokens_enabled=True)."""
        content = "BBU设备<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=True
        )

        assert result is not None
        assert result["caption"] == "BBU设备"
        assert result["coordinates"] == [100, 200, 300, 400]

    def test_parse_geometry_content_mixed_content(self):
        """Test parsing content with coordinates mixed within caption."""
        content = "设备<|coord_150|>型号<|coord_10|>BBU<|coord_211|><|coord_35|>"
        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=True
        )

        assert result is not None
        assert result["caption"] == "设备型号BBU"  # Coordinates removed
        assert result["coordinates"] == [150, 10, 211, 35]

    @pytest.mark.parametrize(
        "invalid_coord,expected_caption",
        [
            ("<|coord_abc|>", "BBU设备<|coord_abc|>"),  # Non-numeric (stays in caption)
            ("<|coord_-100|>", "BBU设备<|coord_-100|>"),  # Negative (stays in caption)
            ("<coord_100>", "BBU设备<coord_100>"),  # Missing pipes (stays in caption)
            ("<|coord_|>", "BBU设备<|coord_|>"),  # Empty value (stays in caption)
        ],
    )
    def test_parse_geometry_content_invalid_coordinates(
        self, invalid_coord, expected_caption
    ):
        """Test handling of invalid coordinate tokens."""
        content = f"BBU设备{invalid_coord}<|coord_200|>"
        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=True
        )

        assert result is not None
        assert result["caption"] == expected_caption
        # Should only contain valid coordinate (200), invalid ones filtered out
        assert 200 in result["coordinates"]
        assert len(result["coordinates"]) == 1

    def test_parse_geometry_content_large_coordinates(self):
        """Test handling of unusually large coordinate values."""
        content = "设备<|coord_5000|><|coord_100|>"  # 5000 > 4096 threshold

        with patch.object(self.parser.parser_logger, "warning") as mock_warning:
            result = self.parser._parse_geometry_content(
                content, coordinate_tokens_enabled=True
            )

            assert result is not None
            assert result["coordinates"] == [
                5000,
                100,
            ]  # Should still include large value
            mock_warning.assert_called()  # Should log warning

    def test_parse_multi_geometry_tokens_bbox_2d_caption_only(self):
        """Test parsing bbox_2d with caption only."""
        text = "<obj_ref_start><bbox_2d_start>BBU设备<bbox_2d_end><obj_ref_end>"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=False
        )

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "bbox_2d"
        assert obj["caption"] == "BBU设备"
        assert obj["coordinates"] == []
        assert obj["formatted_output"] == "bbox_2d:BBU设备"

    def test_parse_multi_geometry_tokens_bbox_2d_with_coordinates(self):
        """Test parsing bbox_2d with coordinate tokens."""
        text = "<obj_ref_start><bbox_2d_start>BBU设备<|coord_150|><|coord_10|><|coord_211|><|coord_35|><bbox_2d_end><obj_ref_end>"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=True
        )

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "bbox_2d"
        assert obj["caption"] == "BBU设备"
        assert obj["coordinates"] == [150, 10, 211, 35]
        assert obj["formatted_output"] == "bbox_2d:BBU设备[150,10,211,35]"

    @pytest.mark.parametrize(
        "geometry_type,coord_count,expected_warning",
        [
            ("bbox_2d", 3, True),  # bbox_2d should have 4 coordinates
            ("bbox_2d", 4, False),  # bbox_2d with correct count
            ("square", 6, True),  # square should have 8 coordinates
            ("square", 8, False),  # square with correct count
            ("line", 2, True),  # line should have at least 4 coordinates
            ("line", 6, False),  # line with valid count
        ],
    )
    def test_coordinate_count_validation(
        self, geometry_type, coord_count, expected_warning
    ):
        """Test validation of coordinate counts for different geometry types."""
        coords = "".join(f"<|coord_{i * 10}|>" for i in range(coord_count))
        text = f"<obj_ref_start><{geometry_type}_start>测试{coords}<{geometry_type}_end><obj_ref_end>"

        with patch.object(self.parser.parser_logger, "warning") as mock_warning:
            objects = self.parser._parse_multi_geometry_tokens(
                text, coordinate_tokens_enabled=True
            )

            assert len(objects) == 1
            obj = objects[0]
            assert obj["geometry_type"] == geometry_type
            assert len(obj["coordinates"]) == coord_count

            if expected_warning:
                mock_warning.assert_called()
            else:
                mock_warning.assert_not_called()

    def test_parse_multi_geometry_tokens_line_geometry(self):
        """Test parsing line geometry with coordinates."""
        text = "<obj_ref_start><line_start>光纤<|coord_100|><|coord_200|><|coord_150|><|coord_250|><|coord_200|><|coord_300|><line_end><obj_ref_end>"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=True
        )

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "line"
        assert obj["caption"] == "光纤"
        assert obj["coordinates"] == [100, 200, 150, 250, 200, 300]
        assert obj["formatted_output"] == "line:光纤[100,200,150,250,200,300]"

    def test_parse_multi_geometry_tokens_square_geometry(self):
        """Test parsing square geometry with coordinates."""
        text = "<obj_ref_start><square_start>标签<|coord_50|><|coord_60|><|coord_80|><|coord_65|><|coord_85|><|coord_95|><|coord_55|><|coord_90|><square_end><obj_ref_end>"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=True
        )

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "square"
        assert obj["caption"] == "标签"
        assert obj["coordinates"] == [50, 60, 80, 65, 85, 95, 55, 90]
        assert obj["formatted_output"] == "square:标签[50,60,80,65,85,95,55,90]"

    def test_parse_multi_geometry_tokens_invalid_geometry_type(self):
        """Test handling of invalid geometry types."""
        text = "<obj_ref_start><invalid_start>测试<invalid_end><obj_ref_end>"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=False
        )

        # Should return empty list or skip invalid geometry
        assert len(objects) == 0

    def test_parse_multi_geometry_tokens_empty_content(self):
        """Test handling of empty content between geometry tokens."""
        text = "<obj_ref_start><bbox_2d_start><bbox_2d_end><obj_ref_end>"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=False
        )

        # Should return empty list due to empty content
        assert len(objects) == 0

    def test_parse_multi_geometry_tokens_alternative_patterns(self):
        """Test alternative parsing patterns with spacing variations."""
        text = "< obj_ref_start >< bbox_2d_start >BBU设备< bbox_2d_end >< obj_ref_end >"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=False
        )

        assert len(objects) == 1
        obj = objects[0]
        assert obj["geometry_type"] == "bbox_2d"
        assert obj["caption"] == "BBU设备"
        assert obj["formatted_output"] == "bbox_2d:BBU设备"

    def test_parse_multi_geometry_tokens_no_matches(self):
        """Test behavior when no multi-geometry tokens are found."""
        text = "This is just regular text without any special tokens"
        objects = self.parser._parse_multi_geometry_tokens(
            text, coordinate_tokens_enabled=False
        )

        assert len(objects) == 0

    def test_parse_geometry_content_error_handling(self):
        """Test error handling in _parse_geometry_content."""
        # Test with None content
        result = self.parser._parse_geometry_content(
            None, coordinate_tokens_enabled=True
        )
        assert result is None

        # Test with empty string
        result = self.parser._parse_geometry_content("", coordinate_tokens_enabled=True)
        assert result is not None
        assert result["caption"] == ""
        assert result["coordinates"] == []

    def test_coordinate_tokens_disabled_ignores_coord_tokens(self):
        """Test that coordinate tokens are ignored when coordinate_tokens_enabled=False."""
        content = "BBU设备<|coord_100|><|coord_200|>"
        result = self.parser._parse_geometry_content(
            content, coordinate_tokens_enabled=False
        )

        assert result is not None
        assert (
            result["caption"] == "BBU设备<|coord_100|><|coord_200|>"
        )  # Tokens not removed
        assert result["coordinates"] == []
