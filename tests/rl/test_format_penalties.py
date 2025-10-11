"""Unit tests for duplicate_penalty and pattern_penalty format rewards.

These tests verify the line geometry quality penalties introduced for RL training.
"""

import importlib

import pytest


@pytest.fixture
def format_rewards():
    """Import format_rewards module."""
    return importlib.import_module("src_new.rl.rewards.format_rewards")


# ============================================================================
# duplicate_penalty tests
# ============================================================================


class TestDuplicatePenalty:
    """Test suite for duplicate_penalty function."""

    def test_no_line_geometry_returns_one(self, format_rewards):
        """When no line geometry is present, should return 1.0 (no penalty)."""
        # Empty text
        assert format_rewards.duplicate_penalty("") == 1.0

        # Text with no geometry
        assert format_rewards.duplicate_penalty("Some random text") == 1.0

        # Text with bbox/quad but no line
        text_bbox = (
            "<|object_ref_start|>Box<|object_ref_end|>"
            "<|box_start|>[100, 100, 200, 200]<|box_end|>"
        )
        assert format_rewards.duplicate_penalty(text_bbox) == 1.0

        text_quad = (
            "<|object_ref_start|>Quad<|object_ref_end|>"
            "<|quad_start|>[10, 10, 20, 10, 20, 20, 10, 20]<|quad_end|>"
        )
        assert format_rewards.duplicate_penalty(text_quad) == 1.0

    def test_perfect_line_no_penalty(self, format_rewards):
        """Well-formed line with no duplicates or zero-length segments."""
        # Simple horizontal line with good separation
        text = (
            "<|object_ref_start|>Line<|object_ref_end|>"
            "<|line_start|>[0, 100, 50, 100, 100, 100, 150, 100]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)
        assert result == pytest.approx(1.0, abs=1e-6)

        # Diagonal line with good vertex separation
        text2 = (
            "<|object_ref_start|>Diagonal<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 10, 20, 20, 30, 30]<|line_end|>"
        )
        result2 = format_rewards.duplicate_penalty(text2)
        assert result2 == pytest.approx(1.0, abs=1e-6)

    def test_zero_length_segment_penalty(self, format_rewards):
        """Line with consecutive identical points (zero-length segment)."""
        # One zero-length segment (dx=0, dy=0)
        text = (
            "<|object_ref_start|>Line<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 10, 10, 10, 20, 20]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)

        # Default zero_length_seg_penalty=0.02, so penalty = 1 * 0.02 = 0.02
        expected = 1.0 - 0.02
        assert result == pytest.approx(expected, abs=1e-6)

    def test_multiple_zero_length_segments(self, format_rewards):
        """Multiple consecutive duplicate points."""
        # Two zero-length segments
        text = (
            "<|object_ref_start|>Line<|object_ref_end|>"
            "<|line_start|>[0, 0, 5, 5, 5, 5, 10, 10, 10, 10, 20, 20]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)

        # 2 zero-length segments: penalty = 2 * 0.02 = 0.04
        expected = 1.0 - 0.04
        assert result == pytest.approx(expected, abs=1e-6)

    def test_near_duplicate_vertex_penalty(self, format_rewards):
        """Adjacent vertices with separation < min_vertex_separation."""
        # dx + dy = 1 < 2 (min_vertex_separation default)
        text = (
            "<|object_ref_start|>Line<|object_ref_end|>"
            "<|line_start|>[0, 0, 1, 0, 10, 10]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)

        # One near-duplicate: penalty = 1 * 0.01 = 0.01
        expected = 1.0 - 0.01
        assert result == pytest.approx(expected, abs=1e-6)

    def test_combined_penalties(self, format_rewards):
        """Line with both zero-length and near-duplicate segments."""
        # Segment 1->2: zero-length (0,0)
        # Segment 2->3: near-duplicate (dx+dy=1)
        text = (
            "<|object_ref_start|>Line<|object_ref_end|>"
            "<|line_start|>[0, 0, 0, 0, 1, 0, 20, 20]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)

        # Penalty = 1*0.02 (zero) + 1*0.01 (near-dup) = 0.03
        expected = 1.0 - 0.03
        assert result == pytest.approx(expected, abs=1e-6)

    def test_max_penalty_capping(self, format_rewards):
        """Penalty should be capped at max_penalty (default 0.20)."""
        # Create line with many zero-length segments to exceed max
        # 11 zero-length segments would give 11*0.02 = 0.22 > 0.20
        coords = [0, 0]  # Start
        for _ in range(11):
            coords.extend([0, 0])  # Duplicate point
        coords.extend([100, 100])  # End normally

        text = (
            "<|object_ref_start|>Bad<|object_ref_end|>"
            f"<|line_start|>{coords}<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)

        # Penalty capped at 0.20
        expected = 1.0 - 0.20
        assert result == pytest.approx(expected, abs=1e-6)

    def test_multiple_lines_averaged(self, format_rewards):
        """Multiple line geometries are averaged."""
        # Line 1: perfect (score=1.0)
        # Line 2: one zero-length (score=0.98)
        text = (
            "<|object_ref_start|>Line1<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 10, 20, 20]<|line_end|>"
            "<|object_ref_start|>Line2<|object_ref_end|>"
            "<|line_start|>[0, 0, 5, 5, 5, 5, 10, 10]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)

        # Average: (1.0 + 0.98) / 2 = 0.99
        expected = (1.0 + (1.0 - 0.02)) / 2.0
        assert result == pytest.approx(expected, abs=1e-6)

    def test_malformed_line_returns_zero(self, format_rewards):
        """Invalid line geometry (odd number of coords) should return 0.0 for that line."""
        # Odd number of coordinates
        text = (
            "<|object_ref_start|>Bad<|object_ref_end|>"
            "<|line_start|>[0, 0, 10]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)
        assert result == 0.0

        # Too few coordinates (< 4)
        text2 = (
            "<|object_ref_start|>Bad<|object_ref_end|><|line_start|>[0, 0]<|line_end|>"
        )
        result2 = format_rewards.duplicate_penalty(text2)
        assert result2 == 0.0

    def test_mixed_valid_invalid_lines(self, format_rewards):
        """Mix of valid and invalid lines - only valid ones count."""
        # Line 1: invalid (odd coords) -> score=0.0
        # Line 2: valid, perfect -> score=1.0
        text = (
            "<|object_ref_start|>Bad<|object_ref_end|>"
            "<|line_start|>[0, 0, 10]<|line_end|>"
            "<|object_ref_start|>Good<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 10, 20, 20]<|line_end|>"
        )
        result = format_rewards.duplicate_penalty(text)

        # Average: (0.0 + 1.0) / 2 = 0.5
        assert result == pytest.approx(0.5, abs=1e-6)


# ============================================================================
# pattern_penalty tests
# ============================================================================


class TestPatternPenalty:
    """Test suite for pattern_penalty function."""

    def test_no_line_geometry_returns_one(self, format_rewards):
        """When no line geometry is present, should return 1.0 (no penalty)."""
        # Empty text
        assert format_rewards.pattern_penalty("") == 1.0

        # Text with no geometry
        assert format_rewards.pattern_penalty("Some random text") == 1.0

        # Text with bbox but no line
        text = (
            "<|object_ref_start|>Box<|object_ref_end|>"
            "<|box_start|>[100, 100, 200, 200]<|box_end|>"
        )
        assert format_rewards.pattern_penalty(text) == 1.0

    def test_diverse_line_no_penalty(self, format_rewards):
        """Well-formed line with diverse deltas (no axis runs or repeats)."""
        # Zigzag pattern with diverse deltas
        text = (
            "<|object_ref_start|>Line<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 5, 15, 12, 25, 18, 30, 30]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # No axis runs, no repeated deltas -> should be close to 1.0
        assert result == pytest.approx(1.0, abs=0.05)

    def test_horizontal_axis_run_penalty(self, format_rewards):
        """Line with only horizontal segments (all dy=0)."""
        # Pure horizontal line: all segments have dy=0
        text = (
            "<|object_ref_start|>Horizontal<|object_ref_end|>"
            "<|line_start|>[0, 100, 50, 100, 100, 100, 150, 100]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # axis_ratio = 3/3 = 1.0 (100% axis-aligned)
        # step_ratio = 3/3 = 1.0 (all same delta (50,0))
        # overshoot = (1.0 - 0.4) + (1.0 - 0.4) = 1.2
        # penalty = min(0.20, 0.05 * 1.2) = 0.06
        expected = 1.0 - 0.06
        assert result == pytest.approx(expected, abs=1e-6)

    def test_vertical_axis_run_penalty(self, format_rewards):
        """Line with only vertical segments (all dx=0)."""
        # Pure vertical line: all segments have dx=0
        text = (
            "<|object_ref_start|>Vertical<|object_ref_end|>"
            "<|line_start|>[100, 0, 100, 50, 100, 100, 100, 150]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # axis_ratio = 3/3 = 1.0
        # step_ratio = 3/3 = 1.0 (all same delta (0,50))
        # overshoot = (1.0 - 0.4) + (1.0 - 0.4) = 1.2
        # penalty = min(0.20, 0.05 * 1.2) = 0.06
        expected = 1.0 - 0.06
        assert result == pytest.approx(expected, abs=1e-6)

    def test_repeated_step_penalty(self, format_rewards):
        """Line with identical repeated delta pattern."""
        # All segments have same delta (10, 10)
        text = (
            "<|object_ref_start|>Repeat<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 10, 20, 20, 30, 30, 40, 40]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # step_ratio = 4/4 = 1.0 (all deltas identical)
        # overshoot = 1.0 - 0.4 = 0.6
        # penalty = 0.03
        expected = 1.0 - 0.03
        assert result == pytest.approx(expected, abs=1e-6)

    def test_combined_axis_and_repeat_penalty(self, format_rewards):
        """Line with both high axis-run and high step-repeat ratios."""
        # Horizontal line with repeated delta (50, 0)
        text = (
            "<|object_ref_start|>Both<|object_ref_end|>"
            "<|line_start|>[0, 100, 50, 100, 100, 100, 150, 100]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # axis_ratio = 3/3 = 1.0 (all horizontal)
        # step_ratio = 3/3 = 1.0 (all same delta)
        # overshoot = (1.0 - 0.4) + (1.0 - 0.4) = 1.2
        # penalty = min(0.20, 0.05 * 1.2) = 0.06
        expected = 1.0 - 0.06
        assert result == pytest.approx(expected, abs=1e-6)

    def test_partial_axis_run_no_penalty(self, format_rewards):
        """Line with axis-run ratio below threshold."""
        # 1 axis-aligned segment out of 3 = 33% < 40% threshold
        text = (
            "<|object_ref_start|>Mixed<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 0, 15, 5, 20, 12]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # axis_ratio = 1/3 ≈ 0.33 < 0.40 -> no axis overshoot
        # step_ratio might vary but should be < 0.40 -> no penalty
        assert result == pytest.approx(1.0, abs=0.05)

    def test_max_penalty_capping(self, format_rewards):
        """Penalty should be capped at max_penalty (default 0.20)."""
        # Create line that would exceed max penalty
        # Both axis_ratio=1.0 and step_ratio=1.0 with extreme overshoot
        text = (
            "<|object_ref_start|>Extreme<|object_ref_end|>"
            "<|line_start|>[0, 100, 100, 100, 200, 100, 300, 100, 400, 100, 500, 100]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # overshoot = (1.0 - 0.4) + (1.0 - 0.4) = 1.2
        # penalty_uncapped = 0.05 * 1.2 = 0.06
        # But with more segments, could push higher
        # In any case, penalty capped at 0.20
        assert result >= 1.0 - 0.20

    def test_multiple_lines_averaged(self, format_rewards):
        """Multiple line geometries are averaged."""
        # Line 1: diverse (score ≈ 1.0)
        # Line 2: all horizontal (score ≈ 0.97)
        text = (
            "<|object_ref_start|>Good<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 5, 15, 12, 25, 18]<|line_end|>"
            "<|object_ref_start|>Horizontal<|object_ref_end|>"
            "<|line_start|>[0, 100, 50, 100, 100, 100]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # Should be average of two scores
        assert 0.9 < result < 1.0

    def test_malformed_line_returns_fallback(self, format_rewards):
        """Invalid line geometry should return (1.0, 1.0) for ratios."""
        # Odd number of coordinates
        text = (
            "<|object_ref_start|>Bad<|object_ref_end|>"
            "<|line_start|>[0, 0, 10]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # Malformed line contributes (1.0, 1.0) ratios
        # This triggers overshoot and some penalty
        assert result < 1.0

    def test_custom_thresholds(self, format_rewards):
        """Test with custom threshold parameters."""
        # Horizontal line with 50% axis ratio
        text = (
            "<|object_ref_start|>Line<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 0, 15, 5, 20, 10]<|line_end|>"
        )

        # Default thresholds (axis_run_max_ratio=0.40)
        result_default = format_rewards.pattern_penalty(text)

        # Custom higher threshold (axis_run_max_ratio=0.60)
        result_high = format_rewards.pattern_penalty(text, axis_run_max_ratio=0.60)

        # With higher threshold, same line should have less/no penalty
        assert result_high >= result_default

    def test_diagonal_staircase_pattern(self, format_rewards):
        """Diagonal line with alternating horizontal/vertical segments."""
        # Staircase pattern: all segments are axis-aligned
        text = (
            "<|object_ref_start|>Staircase<|object_ref_end|>"
            "<|line_start|>[0, 0, 10, 0, 10, 10, 20, 10, 20, 20]<|line_end|>"
        )
        result = format_rewards.pattern_penalty(text)

        # axis_ratio = 4/4 = 1.0
        # step_ratio varies (two (10,0), two (0,10)) = 2/4 = 0.5
        # axis overshoot = 0.6
        # step overshoot = 0.1
        # total_overshoot = 0.7
        # penalty = min(0.20, 0.05 * 0.7) = 0.035
        expected = 1.0 - 0.035
        assert result == pytest.approx(expected, abs=1e-6)


# ============================================================================
# Integration tests
# ============================================================================


class TestPenaltiesIntegration:
    """Integration tests for both penalties together."""

    def test_good_quality_line(self, format_rewards):
        """High-quality line should score well on both penalties."""
        text = (
            "<|object_ref_start|>Quality<|object_ref_end|>"
            "<|line_start|>[0, 0, 15, 8, 28, 20, 45, 35, 60, 55]<|line_end|>"
        )

        dup_score = format_rewards.duplicate_penalty(text)
        pattern_score = format_rewards.pattern_penalty(text)

        # Both should be high (close to 1.0)
        assert dup_score > 0.95
        assert pattern_score > 0.95

    def test_poor_quality_line(self, format_rewards):
        """Poor-quality line with duplicates and patterns should score low."""
        # Line with zero-length segment AND all horizontal
        text = (
            "<|object_ref_start|>Poor<|object_ref_end|>"
            "<|line_start|>[0, 100, 50, 100, 50, 100, 100, 100]<|line_end|>"
        )

        dup_score = format_rewards.duplicate_penalty(text)
        pattern_score = format_rewards.pattern_penalty(text)

        # Both should detect issues
        assert dup_score < 1.0
        assert pattern_score < 1.0

    def test_empty_and_none_cases(self, format_rewards):
        """Edge cases with empty strings and various inputs."""
        # Empty string
        assert format_rewards.duplicate_penalty("") == 1.0
        assert format_rewards.pattern_penalty("") == 1.0

        # Just wrappers, no coords
        text = "<|line_start|>[]<|line_end|>"
        assert format_rewards.duplicate_penalty(text) == 0.0
        # pattern_penalty returns (1.0, 1.0) ratios for malformed -> triggers penalty
        result = format_rewards.pattern_penalty(text)
        assert result < 1.0  # Will have some penalty from fallback ratios


# ============================================================================
# Real-world prediction tests
# ============================================================================


class TestRealWorldPredictions:
    """Tests based on actual model predictions to verify penalties work on real data."""

    def test_prediction_horizontal_line_with_near_duplicates(self, format_rewards):
        """Sample 1: Real prediction with horizontal line and many near-duplicate vertices."""
        # Extracted from prediction: mostly horizontal with tiny y variations
        text = (
            "<|object_ref_start|>电线/捆扎整齐<|object_ref_end|>"
            "<|line_start|>[29, 221, 31, 221, 33, 221, 35, 221, 37, 221, 37, 222, "
            "35, 222, 34, 222, 32, 222, 30, 222, 28, 222]<|line_end|>"
        )

        dup_score = format_rewards.duplicate_penalty(text)
        pattern_score = format_rewards.pattern_penalty(text)

        # Should detect near-duplicates (dx=2, dy=0 or 1 < min_separation=2)
        assert dup_score < 1.0, f"Expected duplicate penalty, got {dup_score}"

        # Should detect high axis-run ratio (all horizontal except one y-step)
        assert pattern_score < 1.0, f"Expected pattern penalty, got {pattern_score}"

        # Both penalties should be significant but not zero
        assert dup_score > 0.5
        assert pattern_score > 0.5

    def test_prediction_malformed_truncated_lines(self, format_rewards):
        """Sample 2: Truncated lines from prediction."""
        # Real truncated prediction - the regex won't match because missing closing bracket
        text_truncated = (
            "<|object_ref_start|>光纤/有保护措施,弯曲半径合理/蛇形管<|object_ref_end|>"
            "<|line_start|>[1, 0<|line_end|>"  # Truncated, missing ]
        )

        # Regex won't match, so no line section extracted -> returns 1.0
        assert format_rewards.duplicate_penalty(text_truncated) == 1.0
        assert format_rewards.pattern_penalty(text_truncated) == 1.0

        # Test with properly closed but too few coords
        text_too_few = (
            "<|object_ref_start|>Test<|object_ref_end|>"
            "<|line_start|>[1, 0]<|line_end|>"  # Only 2 coords
        )

        dup_score = format_rewards.duplicate_penalty(text_too_few)
        pattern_score = format_rewards.pattern_penalty(text_too_few)

        # Too few coords -> duplicate_penalty returns 0.0
        assert dup_score == 0.0

        # pattern_penalty returns fallback for malformed
        assert pattern_score < 1.0

    def test_prediction_repeated_coordinate_pattern(self, format_rewards):
        """Sample 3: Massive coordinate repetition (1, 1, 1, 1, ...)."""
        # Simulate the runaway 1,1,1,1 pattern
        coords = [180, 25] + [1, 1] * 20  # 40 more coords, all (1,1)
        text = (
            "<|object_ref_start|>Test<|object_ref_end|>"
            f"<|line_start|>{coords}<|line_end|>"
        )

        dup_score = format_rewards.duplicate_penalty(text)
        pattern_score = format_rewards.pattern_penalty(text)

        # Should detect many zero-length segments and near-duplicates
        # With 20 segments of (1,1), most will be near-duplicates or zero-length
        assert dup_score < 0.85, f"Expected strong duplicate penalty, got {dup_score}"

        # Should detect very high step-repeat ratio (all delta=(1,1))
        assert pattern_score < 0.98, (
            f"Expected pattern penalty for repetition, got {pattern_score}"
        )

    def test_prediction_quad_with_duplicate_vertices(self, format_rewards):
        """Sample 5: Quad geometries with repeated coordinates."""
        # Real prediction quads with duplicate vertices
        text1 = (
            "<|object_ref_start|>Test1<|object_ref_end|>"
            "<|quad_start|>[1, 14, 1, 1, 1, 1, 1, 1]<|quad_end|>"
        )
        text2 = (
            "<|object_ref_start|>Test2<|object_ref_end|>"
            "<|quad_start|>[419, 14, 419, 1, 419, 1, 419, 1]<|quad_end|>"
        )

        # These are quads, not lines, so penalties should return 1.0
        assert format_rewards.duplicate_penalty(text1) == 1.0
        assert format_rewards.duplicate_penalty(text2) == 1.0
        assert format_rewards.pattern_penalty(text1) == 1.0
        assert format_rewards.pattern_penalty(text2) == 1.0

    def test_good_real_ground_truth_line(self, format_rewards):
        """Ground truth line from sample 1 - should score well."""
        # Real ground truth: diverse, smooth curve
        text = (
            "<|object_ref_start|>电线/捆扎整齐<|object_ref_end|>"
            "<|line_start|>[314, 333, 325, 406, 347, 448, 366, 476, 423, 516]<|line_end|>"
        )

        dup_score = format_rewards.duplicate_penalty(text)
        pattern_score = format_rewards.pattern_penalty(text)

        # Good quality line should score high
        assert dup_score > 0.95, f"GT line should have no duplicates, got {dup_score}"
        assert pattern_score > 0.95, (
            f"GT line should have diverse pattern, got {pattern_score}"
        )

    def test_good_real_ground_truth_curved_line(self, format_rewards):
        """Ground truth curved line from sample 2 - complex winding path."""
        # Real GT: complex curve with diverse deltas
        text = (
            "<|object_ref_start|>光纤/有保护措施,弯曲半径合理/蛇形管<|object_ref_end|>"
            "<|line_start|>[317, 0, 330, 39, 308, 55, 301, 58, 285, 61, 269, 66, "
            "267, 78, 300, 96, 348, 106]<|line_end|>"
        )

        dup_score = format_rewards.duplicate_penalty(text)
        pattern_score = format_rewards.pattern_penalty(text)

        # Complex curve should score well
        assert dup_score > 0.95
        assert pattern_score > 0.90  # Some segments might be close but overall diverse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
