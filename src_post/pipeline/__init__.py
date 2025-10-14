#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline helpers for the refactored src_post runtime.

This package progressively encapsulates Stage-A/Stage-B sampling,
training loops, and structured data transfer objects (DTOs) so that
`runner.py` can shrink to a thin orchestration layer.
"""

from .dto import (
    GroupUpdate,
    StageADiagnostics,
    StageAResult,
    StageBCandidate,
    StageBResult,
    StageBSamplingDiagnostics,
)
from .metrics import MetricsAggregator
from .stage_a import StageASampler, StageASamplerConfig
from .stage_b import StageBSampler, StageBSamplerConfig


__all__ = [
    "StageAResult",
    "StageADiagnostics",
    "StageBCandidate",
    "StageBResult",
    "StageBSamplingDiagnostics",
    "GroupUpdate",
    "StageASampler",
    "StageASamplerConfig",
    "StageBSampler",
    "StageBSamplerConfig",
    "MetricsAggregator",
]
