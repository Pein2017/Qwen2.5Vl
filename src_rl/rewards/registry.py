#!/usr/bin/env python3
"""
Reward registry for RL runs.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .format_rewards import (
    check_wrappers,
    check_ascii_separators,
    check_coords_counts,
    check_banned_vocab,
    parse_reward,
    length_score,
    compute_reward,
)
from .detection_rewards import (
    reward_coverage,
    reward_geometry_sanity,
    reward_bbox_giou,
    reward_quad_l1,
    reward_line_l1,
    reward_ordering,
)


REGISTRY: Dict[str, Callable[[str], float]] = {
    "parse": parse_reward,
    "wrappers": check_wrappers,
    "coords": check_coords_counts,
    "separators": check_ascii_separators,
    "vocab": check_banned_vocab,
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


def combine(text: str, weights: Dict[str, float]) -> float:
    return compute_reward(text, weights)


__all__ = ["REGISTRY", "combine"]
