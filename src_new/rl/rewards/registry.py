#!/usr/bin/env python3
"""
Reward registry for RL runs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .detection_rewards import (
    caption_f1,
    grounding_acc,
    reward_bbox_giou,
    reward_coverage,
    reward_geometry_sanity,
    reward_line_giou,
    reward_line_l1,
    reward_ordering,
    reward_quad_giou,
    reward_quad_l1,
)
from .format_rewards import (
    check_banned_vocab,
    check_coords_counts,
    check_wrappers,
    duplicate_penalty,
    length_vs_gt,
    pairing_ratio,
    separators_score,
)


REGISTRY: Dict[str, Callable[[str], float]] = {
    "pairing_ratio": pairing_ratio,
    "duplicate_penalty": duplicate_penalty,
    "wrappers": check_wrappers,
    "coords": check_coords_counts,
    "separators": separators_score,
    "vocab": check_banned_vocab,
    "coverage": reward_coverage,
    "geometry_sanity": reward_geometry_sanity,
    # Only expose GIoU-based reward
    "bbox_giou": reward_bbox_giou,
    # New proximity rewards
    "quad_l1": reward_quad_l1,
    "line_l1": reward_line_l1,
    "quad_giou": reward_quad_giou,
    "line_giou": reward_line_giou,
    # Ordering constraint reward
    "ordering": reward_ordering,
    # New accuracy-style rewards
    "caption_f1": caption_f1,
    "grounding_acc": grounding_acc,
    "length_vs_gt": length_vs_gt,
}


# Metadata for reward display and categorization
REGISTRY_METADATA: Dict[str, Dict[str, Any]] = {
    # Format rewards
    "pairing_ratio": {
        "category": "format",
        "description": "Parsed objects per object-ref",
    },
    "duplicate_penalty": {
        "category": "format",
        "description": "Penalty for duplicate geometry lists",
    },
    "wrappers": {"category": "format", "description": "Geometry wrapper correctness"},
    "coords": {"category": "format", "description": "Coordinate count correctness"},
    "separators": {"category": "format", "description": "Separator formatting quality"},
    "vocab": {"category": "format", "description": "Vocabulary compliance"},
    # Detection rewards
    "coverage": {"category": "detection", "description": "Object count coverage"},
    "geometry_sanity": {
        "category": "detection",
        "description": "Geometry sanity checks",
    },
    "bbox_giou": {"category": "detection", "description": "Bbox GIoU accuracy"},
    "quad_l1": {"category": "detection", "description": "Quad L1 proximity"},
    "line_l1": {"category": "detection", "description": "Line L1 proximity"},
    "quad_giou": {"category": "detection", "description": "Quad polygon GIoU"},
    "line_giou": {"category": "detection", "description": "Polyline buffered GIoU"},
    "ordering": {
        "category": "detection",
        "description": "Geometry ordering correctness",
    },
    "caption_f1": {"category": "detection", "description": "Caption token F1"},
    "grounding_acc": {
        "category": "detection",
        "description": "Grounding threshold accuracy",
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
            val = float(fn(text, meta=meta))  # detection rewards accept meta
        except TypeError:
            val = float(fn(text))  # formatting rewards without meta
        score += w * val
        total_w += w
    return float(score / total_w) if total_w > 0.0 else 0.0


__all__ = ["REGISTRY", "REGISTRY_METADATA", "combine"]
