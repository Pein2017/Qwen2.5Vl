#!/usr/bin/env python3
"""
Reward registry for RL runs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .assignment import assignment_f1
from .detection_rewards import (
    reward_geometry_sanity,
    reward_ordering,
)
from .format_rewards import (
    check_banned_vocab,
    check_coords_counts,
    check_wrappers,
    duplicate_penalty,
    length_vs_gt,
    pattern_penalty,
    separators_score,
)


REGISTRY: Dict[str, Callable[[str], float]] = {
    "duplicate_penalty": duplicate_penalty,
    "pattern_penalty": pattern_penalty,
    "wrappers": check_wrappers,
    "coords": check_coords_counts,
    "separators": separators_score,
    "vocab": check_banned_vocab,
    "geometry_sanity": reward_geometry_sanity,
    "ordering": reward_ordering,
    "length_vs_gt": length_vs_gt,
    # Assignment-based combined reward (primary)
    "assignment_f1": assignment_f1,
}


# Metadata for reward display and categorization
REGISTRY_METADATA: Dict[str, Dict[str, Any]] = {
    # Format rewards
    "duplicate_penalty": {
        "category": "format",
        "description": "Penalty for duplicate geometry lists",
    },
    "wrappers": {"category": "format", "description": "Geometry wrapper correctness"},
    "coords": {"category": "format", "description": "Coordinate count correctness"},
    "separators": {"category": "format", "description": "Separator formatting quality"},
    "vocab": {"category": "format", "description": "Vocabulary compliance"},
    # Detection rewards
    "geometry_sanity": {
        "category": "detection",
        "description": "Geometry sanity checks",
    },
    "ordering": {
        "category": "detection",
        "description": "Geometry ordering correctness",
    },
    "length_vs_gt": {"category": "format", "description": "Completion length vs GT"},
    "pattern_penalty": {
        "category": "format",
        "description": "Low-diversity line pattern penalty",
    },
    "assignment_f1": {
        "category": "detection",
        "description": "Hungarian assignment over geom+caption with FP/FN",
    },
}


def combine(text: str, weights: Dict[str, float], meta: dict | None = None) -> float:
    total_w = 0.0
    score = 0.0
    for k, v in (weights or {}).items():
        try:
            w = float(v)
        except Exception:
            continue
        if w == 0.0:
            continue
        fn = REGISTRY.get(k)
        if fn is None:
            continue
        try:
            # Try with meta and per-reward params injected by outer wrapper (runner wraps with cfg)
            val = float(fn(text, meta=meta))
        except TypeError:
            try:
                val = float(fn(text))
            except Exception:
                val = 0.0
        except Exception:
            val = 0.0
        score += w * val
        total_w += w
    return float(score / total_w) if total_w > 0.0 else 0.0


__all__ = ["REGISTRY", "REGISTRY_METADATA", "combine"]
