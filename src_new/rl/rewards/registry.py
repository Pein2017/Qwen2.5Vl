#!/usr/bin/env python3
"""
Reward registry for RL runs.
"""

from __future__ import annotations

from typing import Callable, Dict

from .detection_rewards import (
    reward_bbox_giou,
    reward_coverage,
    reward_geometry_sanity,
    reward_line_l1,
    reward_ordering,
    reward_quad_l1,
)
from .format_rewards import (
    check_ascii_separators,
    check_banned_vocab,
    check_coords_counts,
    check_wrappers,
    length_score,
    length_window,
    parse_reward,
)


REGISTRY: Dict[str, Callable[[str], float]] = {
    "parse": parse_reward,
    "wrappers": check_wrappers,
    "coords": check_coords_counts,
    "separators": check_ascii_separators,
    "vocab": check_banned_vocab,
    "length": length_score,
    "length_window": length_window,
    "coverage": reward_coverage,
    "geometry_sanity": reward_geometry_sanity,
    # Only expose GIoU-based reward
    "bbox_giou": reward_bbox_giou,
    # New proximity rewards
    "quad_l1": reward_quad_l1,
    "line_l1": reward_line_l1,
    # Ordering constraint reward
    "ordering": reward_ordering,
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


__all__ = ["REGISTRY", "combine"]
