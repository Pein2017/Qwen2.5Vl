"""
GRPO Diagnostics Module

Feature: 004-grpo-post-training
Constitution: v4.1.1
"""

from src_new.rl.diagnostics.exporters import (
    StandardDiagnosticExporter,
    TensorBoardExporter,
)
from src_new.rl.diagnostics.multimodal import (
    MultimodalAlignmentCheck,
    compute_multimodal_alignment,
)
from src_new.rl.diagnostics.rewards import (
    RewardProfile,
    compute_reward_profile,
)
from src_new.rl.diagnostics.trust_region import (
    TrustRegionDiagnostic,
    compute_trust_region_diagnostic,
)


__all__ = [
    "StandardDiagnosticExporter",
    "TensorBoardExporter",
    "TrustRegionDiagnostic",
    "compute_trust_region_diagnostic",
    "MultimodalAlignmentCheck",
    "compute_multimodal_alignment",
    "RewardProfile",
    "compute_reward_profile",
]
