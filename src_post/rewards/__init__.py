#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Callable, Any, List

from .basic import exact_label_match_reward
from .formatting import formatting_reward
from .cleanliness import cleanliness_reward
from .taxonomy import taxonomy_reward
from .consistency import mission_consistency_reward
from .repetition import repetition_penalty, quote_penalty, special_token_penalty
from .coverage import coverage_reward
from .decision_prob import decision_prob_reward
from .violations import violation_alignment_reward


RewardFn = Callable[[Dict[str, Any]], float]


REGISTRY: Dict[str, RewardFn] = {
    "label_match": exact_label_match_reward,
    "formatting": formatting_reward,
    "cleanliness": cleanliness_reward,
    "coverage": coverage_reward,
    "taxonomy": taxonomy_reward,
    "consistency": mission_consistency_reward,
    # Dense decision probability aligned to GT
    "decision_prob": decision_prob_reward,
    # Violation alignment (positive for GT=fail mentions, negative for GT=pass mentions)
    "violations": violation_alignment_reward,
    # Penalties (use negative weights in config)
    "rep_penalty": repetition_penalty,
    "quote_penalty": quote_penalty,
    "special_penalty": special_token_penalty,
}


def build_reward_fns(names: List[str]) -> List[RewardFn]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown reward names: {unknown}. Known rewards: {sorted(list(REGISTRY.keys()))}.")
    return [REGISTRY[n] for n in names]
