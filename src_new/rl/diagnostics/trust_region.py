"""
Trust Region Validation Diagnostics

Tracks GRPO ratio health (π_current / π_generation) to detect degeneracy.

Feature: 004-grpo-post-training (User Story 1)
Constitution: v4.1.1
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src_new.rl.diagnostics.exporters import StandardDiagnosticExporter


logger = logging.getLogger(__name__)


@dataclass
class TrustRegionDiagnostic:
    """Per-step trust region statistics for GRPO ratio π_current / π_generation."""

    step: int

    # Ratio statistics (computed from exp(cur_logps - generation_logps))
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    ratio_p25: float  # 25th percentile
    ratio_p75: float  # 75th percentile

    # Health indicators
    is_degenerate: bool  # True if std < 0.1
    fallback_count: int  # Number of completions using current policy fallback
    clip_fraction: float  # Fraction of ratios clipped by [1-eps, 1+eps]

    # Generation log-probs verification
    generation_logps_present: bool
    generation_logps_shape: Optional[tuple]

    # Optional cross-rank statistics (if cross_rank_advantages enabled)
    global_ratio_mean: Optional[float] = None
    global_ratio_std: Optional[float] = None

    def to_tensorboard(self) -> dict[str, float]:
        """Export scalars for TensorBoard logging."""
        return {
            "trust_region/ratio_mean": self.ratio_mean,
            "trust_region/ratio_std": self.ratio_std,
            "trust_region/ratio_min": self.ratio_min,
            "trust_region/ratio_max": self.ratio_max,
            "trust_region/clip_fraction": self.clip_fraction,
            "trust_region/is_degenerate": float(self.is_degenerate),
            "trust_region/fallback_count": float(self.fallback_count),
        }

    def export_histogram(
        self,
        ratios: torch.Tensor,
        output_path: Path,
    ) -> None:
        """Export ratio distribution as PNG histogram (rank-0 only)."""
        exporter = StandardDiagnosticExporter(enabled=True)
        exporter.export_histogram(
            values=ratios,
            filepath=output_path,
            title=f"Step {self.step}: Trust Region Ratio Distribution",
            xlabel="Policy Ratio (π_current / π_generation)",
            ylabel="Count",
            bins=50,
            range_limits=(0.5, 1.5),
            reference_line=1.0,
            reference_label="Identity (no drift)",
        )
        logger.debug(f"Exported trust region histogram to {output_path}")

    def log_warnings(self) -> None:
        """Log warnings if trust region health is poor."""
        if self.is_degenerate:
            logger.warning(
                f"[Step {self.step}] Trust region DEGENERATE: "
                f"ratio_std={self.ratio_std:.4f} < 0.1. "
                f"Policy ratios collapsed → no learning signal. "
                f"Check generation_logps storage."
            )

        if self.fallback_count > 0:
            logger.warning(
                f"[Step {self.step}] Trust region FALLBACK detected: "
                f"{self.fallback_count} completions fell back to current policy. "
                f"generation_logps not stored correctly → ratios degenerate to 1.0."
            )

        if not self.generation_logps_present:
            logger.error(
                f"[Step {self.step}] generation_logps NOT PRESENT! "
                f"Trust region is compromised. GRPO will not work correctly."
            )


def compute_trust_region_diagnostic(
    step: int,
    ratios: torch.Tensor,
    generation_logps_present: bool,
    fallback_count: int = 0,
    clip_eps: float = 0.2,
    generation_logps_shape: Optional[tuple] = None,
) -> TrustRegionDiagnostic:
    """
    Compute trust region diagnostic from policy ratios.

    Args:
        step: Training step number
        ratios: Policy ratio tensor [num_completions] = exp(cur_logps - gen_logps)
        generation_logps_present: Whether generation_logps were stored in buffer
        fallback_count: Number of completions that fell back to current policy
        clip_eps: Clipping epsilon for GRPO (default 0.2 → range [0.8, 1.2])
        generation_logps_shape: Shape of generation_logps tensor if present

    Returns:
        TrustRegionDiagnostic instance with computed statistics

    Logic:
        - Compute ratio statistics (mean, std, percentiles)
        - Check degeneracy: std < 0.1 → flag as degenerate
        - Compute clip fraction: fraction of ratios outside [1-eps, 1+eps]
    """
    # Compute statistics
    ratio_mean = ratios.mean().item()
    ratio_std = ratios.std().item()
    ratio_min = ratios.min().item()
    ratio_max = ratios.max().item()
    ratio_p25 = ratios.quantile(0.25).item()
    ratio_p75 = ratios.quantile(0.75).item()

    # Degeneracy check (std < 0.1)
    is_degenerate = ratio_std < 0.1

    # Clip fraction (fraction outside [1-eps, 1+eps])
    clip_lower = 1.0 - clip_eps
    clip_upper = 1.0 + clip_eps
    clipped_mask = (ratios < clip_lower) | (ratios > clip_upper)
    clip_fraction = clipped_mask.float().mean().item()

    diagnostic = TrustRegionDiagnostic(
        step=step,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        ratio_p25=ratio_p25,
        ratio_p75=ratio_p75,
        is_degenerate=is_degenerate,
        fallback_count=fallback_count,
        clip_fraction=clip_fraction,
        generation_logps_present=generation_logps_present,
        generation_logps_shape=generation_logps_shape,
    )

    # Log warnings if needed
    diagnostic.log_warnings()

    return diagnostic
