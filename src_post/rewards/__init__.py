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
from .group_margin import group_margin
from .strict import strict_decision_format, pass_prior
from .lexicon import NEGATIVE_TOKENS, FORBIDDEN_DECISION_WORDS, CANONICAL_SLOTS


RewardFn = Callable[[Dict[str, Any]], float]


# Note: mission-aware rewards (coverage, taxonomy) expect `mission` and
# `checklist_lines` when available. The runner passes these via compose_reward.
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
    # Pair for margin-only dense signal
    "group_margin": group_margin,
    # Penalties (use negative weights in config)
    "rep_penalty": repetition_penalty,
    "quote_penalty": quote_penalty,
    "special_penalty": special_token_penalty,
    # New shaping
    "decision_strict": strict_decision_format,
    "pass_prior": pass_prior,
}


def build_reward_fns(names: List[str]) -> List[RewardFn]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown reward names: {unknown}. Known rewards: {sorted(list(REGISTRY.keys()))}.")
    return [REGISTRY[n] for n in names]
