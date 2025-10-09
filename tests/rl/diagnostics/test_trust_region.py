"""
Unit tests for Trust Region Validation diagnostics.

Feature: 004-grpo-post-training (User Story 1)
Constitution: v4.1.1
"""

import pytest
import torch

from src_new.rl.diagnostics.trust_region import (
    TrustRegionDiagnostic,
    compute_trust_region_diagnostic,
)


class TestTrustRegionDiagnostic:
    """Tests for TrustRegionDiagnostic dataclass."""

    def test_trust_region_diagnostic_creation(self, mock_ratio_tensor):
        """T011: Test creating TrustRegionDiagnostic with mock ratio tensors."""
        # Create diagnostic from ratio statistics
        diagnostic = TrustRegionDiagnostic(
            step=100,
            ratio_mean=mock_ratio_tensor.mean().item(),
            ratio_std=mock_ratio_tensor.std().item(),
            ratio_min=mock_ratio_tensor.min().item(),
            ratio_max=mock_ratio_tensor.max().item(),
            ratio_p25=mock_ratio_tensor.quantile(0.25).item(),
            ratio_p75=mock_ratio_tensor.quantile(0.75).item(),
            is_degenerate=mock_ratio_tensor.std().item() < 0.1,
            fallback_count=0,
            clip_fraction=0.12,
            generation_logps_present=True,
            generation_logps_shape=(64, 128),
        )

        # Verify all fields are populated correctly
        assert diagnostic.step == 100
        assert 0.8 <= diagnostic.ratio_mean <= 1.5  # Healthy range
        assert diagnostic.ratio_std > 0.1  # Not degenerate
        assert diagnostic.is_degenerate is False
        assert diagnostic.fallback_count == 0
        assert diagnostic.generation_logps_present is True
        assert diagnostic.generation_logps_shape == (64, 128)

    def test_trust_region_ratio_computation(self):
        """T012: Test ratio computation from generation_logps and current_logps."""
        # Mock generation log-probs (stored from generation phase)
        generation_logps = torch.randn(64) * 0.5 - 2.0  # Simulated log-probs

        # Mock current log-probs (from current policy)
        current_logps = generation_logps + torch.randn(64) * 0.1  # Small drift

        # Compute ratios: exp(current - generation)
        ratios = torch.exp(current_logps - generation_logps)

        # Verify mean/std/percentiles match expected
        diagnostic = TrustRegionDiagnostic(
            step=50,
            ratio_mean=ratios.mean().item(),
            ratio_std=ratios.std().item(),
            ratio_min=ratios.min().item(),
            ratio_max=ratios.max().item(),
            ratio_p25=ratios.quantile(0.25).item(),
            ratio_p75=ratios.quantile(0.75).item(),
            is_degenerate=ratios.std().item() < 0.1,
            fallback_count=0,
            clip_fraction=(torch.abs(ratios - 1.0) > 0.2).float().mean().item(),
            generation_logps_present=True,
            generation_logps_shape=tuple(generation_logps.shape),
        )

        # Ratios should be close to 1.0 with some variance
        assert 0.5 < diagnostic.ratio_mean < 1.5
        assert diagnostic.ratio_std > 0.05  # Has variance
        assert diagnostic.is_degenerate is False

    def test_trust_region_degeneracy_detection(self):
        """T013: Test degeneracy detection when all ratios ≈ 1.0."""
        # Create degenerate ratios (all exactly 1.0)
        degenerate_ratios = torch.ones(100)

        diagnostic = TrustRegionDiagnostic(
            step=25,
            ratio_mean=degenerate_ratios.mean().item(),
            ratio_std=degenerate_ratios.std().item(),
            ratio_min=degenerate_ratios.min().item(),
            ratio_max=degenerate_ratios.max().item(),
            ratio_p25=1.0,
            ratio_p75=1.0,
            is_degenerate=degenerate_ratios.std().item() < 0.1,
            fallback_count=0,
            clip_fraction=0.0,
            generation_logps_present=True,
            generation_logps_shape=(100,),
        )

        # Verify degeneracy flag is set
        assert diagnostic.is_degenerate is True
        assert diagnostic.ratio_std < 0.1
        assert diagnostic.ratio_mean == 1.0

    def test_trust_region_fallback_warning(self):
        """T014: Test fallback detection when generation_logps are missing."""
        # Simulate fallback scenario (generation_logps not stored)
        diagnostic = TrustRegionDiagnostic(
            step=10,
            ratio_mean=1.0,  # Degenerate - all ratios collapse to 1.0
            ratio_std=0.0,  # Zero variance
            ratio_min=1.0,
            ratio_max=1.0,
            ratio_p25=1.0,
            ratio_p75=1.0,
            is_degenerate=True,
            fallback_count=64,  # Non-zero indicates fallback used
            clip_fraction=0.0,
            generation_logps_present=False,  # Missing!
            generation_logps_shape=None,
        )

        # Verify fallback is detected
        assert diagnostic.fallback_count > 0
        assert diagnostic.generation_logps_present is False
        assert diagnostic.generation_logps_shape is None
        assert diagnostic.is_degenerate is True  # Should trigger warning


class TestComputeTrustRegionDiagnostic:
    """Tests for compute_trust_region_diagnostic helper function."""

    def test_compute_diagnostic_from_ratios(self, mock_ratio_tensor):
        """Test computing diagnostic from ratio tensor."""
        diagnostic = compute_trust_region_diagnostic(
            step=200,
            ratios=mock_ratio_tensor,
            generation_logps_present=True,
            fallback_count=0,
        )

        # Verify computed statistics
        assert diagnostic.step == 200
        assert isinstance(diagnostic.ratio_mean, float)
        assert isinstance(diagnostic.ratio_std, float)
        assert isinstance(diagnostic.is_degenerate, bool)
        assert diagnostic.generation_logps_present is True

    def test_compute_diagnostic_degeneracy_threshold(self):
        """Test that degeneracy threshold (std < 0.1) is correctly applied."""
        # Create near-degenerate ratios (std = 0.05)
        near_degenerate = torch.ones(50) + torch.randn(50) * 0.025

        diagnostic = compute_trust_region_diagnostic(
            step=1,
            ratios=near_degenerate,
            generation_logps_present=True,
            fallback_count=0,
        )

        # Should be flagged as degenerate
        assert diagnostic.is_degenerate is True
        assert diagnostic.ratio_std < 0.1

    def test_compute_diagnostic_clip_fraction(self):
        """Test clip fraction computation."""
        # Create ratios with some outside [0.8, 1.2]
        ratios = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 1.8])

        diagnostic = compute_trust_region_diagnostic(
            step=5,
            ratios=ratios,
            generation_logps_present=True,
            fallback_count=0,
            clip_eps=0.2,  # Clip range: [0.8, 1.2]
        )

        # 3 out of 6 are outside range → clip_fraction = 0.5
        assert diagnostic.clip_fraction == pytest.approx(0.5, abs=0.01)


class TestTrustRegionTensorBoard:
    """Tests for TensorBoard export functionality."""

    def test_to_tensorboard_returns_dict(self, mock_ratio_tensor):
        """Test that to_tensorboard() returns valid scalar dict."""
        diagnostic = compute_trust_region_diagnostic(
            step=10,
            ratios=mock_ratio_tensor,
            generation_logps_present=True,
            fallback_count=0,
        )

        tb_dict = diagnostic.to_tensorboard()

        # Verify structure
        assert isinstance(tb_dict, dict)
        assert "trust_region/ratio_mean" in tb_dict
        assert "trust_region/ratio_std" in tb_dict
        assert "trust_region/is_degenerate" in tb_dict
        assert "trust_region/fallback_count" in tb_dict

        # Verify all values are floats
        for value in tb_dict.values():
            assert isinstance(value, (float, int))

    def test_export_histogram_creates_file(self, mock_ratio_tensor, tmp_output_dir):
        """Test that export_histogram() creates PNG file."""
        diagnostic = compute_trust_region_diagnostic(
            step=50,
            ratios=mock_ratio_tensor,
            generation_logps_present=True,
            fallback_count=0,
        )

        output_path = tmp_output_dir / "ratio_histogram.png"
        diagnostic.export_histogram(
            ratios=mock_ratio_tensor,
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.suffix == ".png"
