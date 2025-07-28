#!/usr/bin/env python3
"""
Comprehensive test suite for coordinate normalization in multi-geometry support.

Tests cover:
- Coordinate ordering consistency for lines, squares, and bboxes
- Degenerate case handling (horizontal/vertical lines, zero-area boxes)
- Edge cases (boundary coordinates, single-pixel objects)
- Real-world data validation
- Semantic preservation of geometry types
"""

import sys
from pathlib import Path

import pytest


# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_conversion.coordinate_manager import CoordinateManager


class TestCoordinateNormalization:
    """Test suite for coordinate normalization algorithms."""

    def test_problematic_line_case(self):
        """Test the specific problematic case that causes training errors."""
        # The exact case from the training error
        problematic_line = [174, 304, 10, 304]

        # Normalize coordinates
        normalized = CoordinateManager.normalize_line_coordinates(
            problematic_line, width=420, height=896
        )

        # Should produce consistent ordering without degenerate bbox
        assert len(normalized) == 4
        x1, y1, x2, y2 = normalized

        # Verify canonical ordering (left-to-right for horizontal lines)
        assert x1 <= x2, f"x1 ({x1}) should be <= x2 ({x2})"

        # For horizontal lines, y-coordinates should have minimal padding to avoid degeneracy
        if y1 == y2:  # Original horizontal line
            # Should add padding to make it non-degenerate
            assert abs(y2 - y1) >= 1, (
                f"Horizontal line should have padding: y1={y1}, y2={y2}"
            )

    def test_line_coordinate_ordering_consistency(self):
        """Test that different orderings of the same line produce identical results."""
        # Test case 1: Same line with endpoints reversed
        line_variants_1 = [
            [10, 20, 100, 200],  # left-to-right, top-to-bottom
            [100, 200, 10, 20],  # right-to-left, bottom-to-top (same line, reversed)
        ]

        normalized_results_1 = []
        for line_coords in line_variants_1:
            normalized = CoordinateManager.normalize_line_coordinates(
                line_coords, width=500, height=500
            )
            normalized_results_1.append(normalized)

        # Both variants should produce the same normalized result
        assert normalized_results_1[0] == normalized_results_1[1], (
            f"Same line with reversed endpoints should normalize identically: {normalized_results_1[0]} vs {normalized_results_1[1]}"
        )

        # Test case 2: Different line with endpoints reversed
        line_variants_2 = [
            [10, 200, 100, 20],  # different line
            [100, 20, 10, 200],  # same different line, reversed
        ]

        normalized_results_2 = []
        for line_coords in line_variants_2:
            normalized = CoordinateManager.normalize_line_coordinates(
                line_coords, width=500, height=500
            )
            normalized_results_2.append(normalized)

        # Both variants should produce the same normalized result
        assert normalized_results_2[0] == normalized_results_2[1], (
            f"Same line with reversed endpoints should normalize identically: {normalized_results_2[0]} vs {normalized_results_2[1]}"
        )

        # The two different lines should produce different results
        assert normalized_results_1[0] != normalized_results_2[0], (
            f"Different lines should produce different normalized results: {normalized_results_1[0]} vs {normalized_results_2[0]}"
        )

    def test_horizontal_line_degenerate_handling(self):
        """Test handling of horizontal lines that would create degenerate bboxes."""
        horizontal_lines = [
            [0, 100, 200, 100],  # Horizontal line at y=100
            [50, 50, 150, 50],  # Horizontal line at y=50
            [174, 304, 10, 304],  # The problematic case
        ]

        for line_coords in horizontal_lines:
            normalized = CoordinateManager.normalize_line_coordinates(
                line_coords, width=420, height=896
            )

            x1, y1, x2, y2 = normalized

            # Should maintain horizontal nature but add padding
            assert x1 <= x2, "X-coordinates should be ordered"
            assert abs(y2 - y1) >= 1, (
                f"Should add padding to horizontal line: {normalized}"
            )

    def test_vertical_line_degenerate_handling(self):
        """Test handling of vertical lines that would create degenerate bboxes."""
        vertical_lines = [
            [100, 0, 100, 200],  # Vertical line at x=100
            [50, 25, 50, 175],  # Vertical line at x=50
        ]

        for line_coords in vertical_lines:
            normalized = CoordinateManager.normalize_line_coordinates(
                line_coords, width=420, height=896
            )

            x1, y1, x2, y2 = normalized

            # Should maintain vertical nature but add padding
            assert y1 <= y2, "Y-coordinates should be ordered"
            assert abs(x2 - x1) >= 1, (
                f"Should add padding to vertical line: {normalized}"
            )

    def test_square_coordinate_ordering_consistency(self):
        """Test that different orderings of square vertices produce identical results."""
        # Same square represented with different starting vertices
        square_variants = [
            [10, 10, 100, 10, 100, 100, 10, 100],  # Start top-left, clockwise
            [100, 10, 100, 100, 10, 100, 10, 10],  # Start top-right, clockwise
            [100, 100, 10, 100, 10, 10, 100, 10],  # Start bottom-right, clockwise
            [10, 100, 10, 10, 100, 10, 100, 100],  # Start bottom-left, clockwise
        ]

        normalized_results = []
        for square_coords in square_variants:
            normalized = CoordinateManager.normalize_square_coordinates(
                square_coords, width=500, height=500
            )
            normalized_results.append(normalized)

        # All variants should produce the same normalized result
        first_result = normalized_results[0]
        for i, result in enumerate(normalized_results[1:], 1):
            assert result == first_result, (
                f"Square variant {i} produced different result: {result} vs {first_result}"
            )

    def test_bbox_coordinate_normalization(self):
        """Test bounding box coordinate normalization."""
        bbox_variants = [
            [10, 20, 100, 200],  # Normal order
            [100, 200, 10, 20],  # Reversed order
            [100, 20, 10, 200],  # Mixed order
            [10, 200, 100, 20],  # Mixed order
        ]

        for bbox_coords in bbox_variants:
            normalized = CoordinateManager.normalize_bbox_coordinates(
                bbox_coords, width=500, height=500
            )

            x1, y1, x2, y2 = normalized

            # Should always be properly ordered
            assert x1 < x2, f"x1 ({x1}) should be < x2 ({x2})"
            assert y1 < y2, f"y1 ({y1}) should be < y2 ({y2})"

    def test_boundary_coordinates(self):
        """Test handling of coordinates at image boundaries."""
        width, height = 420, 896

        # Test line at boundaries
        boundary_line = [0, 0, width - 1, height - 1]
        normalized = CoordinateManager.normalize_line_coordinates(
            boundary_line, width=width, height=height
        )

        x1, y1, x2, y2 = normalized
        assert 0 <= x1 < width and 0 <= x2 < width
        assert 0 <= y1 < height and 0 <= y2 < height

    def test_out_of_bounds_coordinates(self):
        """Test handling of coordinates outside image boundaries."""
        width, height = 420, 896

        # Line with out-of-bounds coordinates
        oob_line = [-10, -5, width + 10, height + 5]
        normalized = CoordinateManager.normalize_line_coordinates(
            oob_line, width=width, height=height
        )

        x1, y1, x2, y2 = normalized

        # Should be clamped to image bounds
        assert 0 <= x1 < width and 0 <= x2 < width
        assert 0 <= y1 < height and 0 <= y2 < height

    def test_complex_polyline_normalization(self):
        """Test normalization of complex polylines with canonical direction."""
        # Complex polyline (6 points) - represents a path like a cable/wire
        # Points: (10,10), (50,30), (80,20), (120,60), (100,90), (70,80)
        # Canonical start should be (80,20) - topmost point (y=20)
        polyline = [10, 10, 50, 30, 80, 20, 120, 60, 100, 90, 70, 80]

        normalized = CoordinateManager.normalize_line_coordinates(
            polyline, width=500, height=500
        )

        # Should maintain all points
        assert len(normalized) == len(polyline)

        # Should establish canonical direction starting from topmost point
        # The topmost point is (80,20), which is at index 2 in original path
        # Since it's in the middle, compare endpoints: (10,10) vs (70,80)
        # (10,10) is more canonical (lower y, lower x), so keep original direction
        expected_points = [(10, 10), (50, 30), (80, 20), (120, 60), (100, 90), (70, 80)]
        actual_points = [
            (normalized[i], normalized[i + 1]) for i in range(0, len(normalized), 2)
        ]

        assert actual_points == expected_points, (
            f"Multi-point line should have canonical direction: expected {expected_points}, got {actual_points}"
        )

    def test_polyline_directional_normalization(self):
        """Test that polylines with same path but opposite directions normalize to same result."""
        # Test case 1: Path that starts from canonical point
        forward_path = [10, 20, 30, 40, 50, 60]  # (10,20) -> (30,40) -> (50,60)

        # Test case 2: Same path traced in reverse direction
        reverse_path = [50, 60, 30, 40, 10, 20]  # (50,60) -> (30,40) -> (10,20)

        # Both should normalize to start from topmost-leftmost point (10,20)
        forward_normalized = CoordinateManager.normalize_line_coordinates(
            forward_path, width=100, height=100
        )
        reverse_normalized = CoordinateManager.normalize_line_coordinates(
            reverse_path, width=100, height=100
        )

        # Both should produce the same canonical result
        expected_canonical = [10, 20, 30, 40, 50, 60]  # Start from (10,20)

        assert forward_normalized == expected_canonical, (
            f"Forward path normalization failed: expected {expected_canonical}, got {forward_normalized}"
        )
        assert reverse_normalized == expected_canonical, (
            f"Reverse path normalization failed: expected {expected_canonical}, got {reverse_normalized}"
        )
        assert forward_normalized == reverse_normalized, (
            "Forward and reverse paths should normalize to same result"
        )

    def test_polyline_canonical_start_in_middle(self):
        """Test polyline where canonical start point is in the middle of the path."""
        # Path: (30,50) -> (10,20) -> (40,60)
        # Canonical start should be (10,20) - topmost point
        # Since it's in middle, compare endpoints: (30,50) vs (40,60)
        # (30,50) is more canonical, so keep original direction
        path = [30, 50, 10, 20, 40, 60]

        normalized = CoordinateManager.normalize_line_coordinates(
            path, width=100, height=100
        )

        # Should keep original direction since start (30,50) is more canonical than end (40,60)
        expected = [30, 50, 10, 20, 40, 60]

        assert normalized == expected, (
            f"Path with canonical point in middle: expected {expected}, got {normalized}"
        )

    def test_polyline_canonical_start_at_end(self):
        """Test polyline where canonical start point is at the end."""
        # Path: (40,60) -> (30,50) -> (10,20)
        # Canonical start should be (10,20) - topmost point at end
        # Should reverse entire path
        path = [40, 60, 30, 50, 10, 20]

        normalized = CoordinateManager.normalize_line_coordinates(
            path, width=100, height=100
        )

        # Should reverse to start from canonical point (10,20)
        expected = [10, 20, 30, 50, 40, 60]

        assert normalized == expected, (
            f"Path should be reversed to start from canonical point: expected {expected}, got {normalized}"
        )

    def test_single_pixel_objects(self):
        """Test handling of single-pixel or very small objects."""
        # Single pixel line
        single_pixel_line = [100, 100, 100, 100]

        normalized = CoordinateManager.normalize_line_coordinates(
            single_pixel_line, width=500, height=500
        )

        x1, y1, x2, y2 = normalized

        # Should add minimal padding to make it non-degenerate
        assert abs(x2 - x1) >= 1 or abs(y2 - y1) >= 1, (
            f"Single pixel should be expanded: {normalized}"
        )

    def test_geometry_type_preservation(self):
        """Test that geometry types are preserved during normalization."""
        test_objects = [
            {"bbox_2d": [10, 20, 100, 200], "desc": "test bbox"},
            {"line": [10, 20, 100, 200], "desc": "test line"},
            {"square": [10, 10, 100, 10, 100, 100, 10, 100], "desc": "test square"},
        ]

        for obj in test_objects:
            normalized_obj = CoordinateManager.normalize_object_coordinates(
                obj, width=500, height=500
            )

            # Should preserve the original geometry type
            if "bbox_2d" in obj:
                assert "bbox_2d" in normalized_obj
                assert "line" not in normalized_obj
                assert "square" not in normalized_obj
            elif "line" in obj:
                assert "line" in normalized_obj
                assert "bbox_2d" not in normalized_obj
                assert "square" not in normalized_obj
            elif "square" in obj:
                assert "square" in normalized_obj
                assert "bbox_2d" not in normalized_obj
                assert "line" not in normalized_obj

            # Description should be preserved
            assert normalized_obj["desc"] == obj["desc"]


class TestRealWorldData:
    """Test coordinate normalization with real training data."""

    def test_problematic_training_sample(self):
        """Test the exact sample that caused the training error."""
        # Sample from line 36 of all_samples.jsonl
        problematic_sample = {
            "images": ["images/QC-20240827-0032694_4524378.jpeg"],
            "objects": [
                {
                    "bbox_2d": [178, 282, 200, 309],
                    "desc": "螺丝、光纤插头/地排处接地螺丝,显示完整,符合要求",
                },
                {
                    "line": [174, 304, 10, 304],
                    "desc": "电线/有遮挡,捆扎整齐",
                },  # Problematic line
                {
                    "square": [144, 316, 144, 296, 77, 289, 76, 312],
                    "desc": "标签/5G-DCDU-接地线",
                },
            ],
            "width": 420,
            "height": 896,
        }

        # Normalize all objects
        normalized_objects = []
        for obj in problematic_sample["objects"]:
            normalized_obj = CoordinateManager.normalize_object_coordinates(
                obj, width=420, height=896
            )
            normalized_objects.append(normalized_obj)

        # Find the problematic line object
        line_obj = next(obj for obj in normalized_objects if "line" in obj)
        line_coords = line_obj["line"]

        # Should not create degenerate bbox
        x1, y1, x2, y2 = line_coords
        assert x1 <= x2, f"Line x-coordinates should be ordered: {line_coords}"

        # For horizontal line, should have padding
        if abs(y2 - y1) < 2:  # Nearly horizontal
            assert abs(y2 - y1) >= 1, (
                f"Horizontal line should have padding: {line_coords}"
            )

    def test_batch_normalization_consistency(self):
        """Test that batch normalization produces consistent results."""
        # Multiple objects that should normalize consistently
        test_batch = [
            {"line": [174, 304, 10, 304], "desc": "horizontal line"},
            {
                "line": [10, 304, 174, 304],
                "desc": "same horizontal line, different order",
            },
            {"square": [10, 10, 100, 10, 100, 100, 10, 100], "desc": "square"},
            {"bbox_2d": [100, 200, 10, 20], "desc": "reversed bbox"},
        ]

        # Normalize each object
        normalized_batch = []
        for obj in test_batch:
            normalized_obj = CoordinateManager.normalize_object_coordinates(
                obj, width=420, height=896
            )
            normalized_batch.append(normalized_obj)

        # First two line objects should produce identical results
        line1_coords = normalized_batch[0]["line"]
        line2_coords = normalized_batch[1]["line"]
        assert line1_coords == line2_coords, (
            f"Same line with different ordering should normalize identically: {line1_coords} vs {line2_coords}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
