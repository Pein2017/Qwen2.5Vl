#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .basic import exact_label_match_reward
from .coverage import coverage_reward
from .group_margin import group_margin
from .lexicon import CANONICAL_SLOTS, FORBIDDEN_DECISION_WORDS, NEGATIVE_TOKENS
from .lexicon_shaping import soft_lexicon_alignment, table_lexicon_conformity
from .mission_outcome import negative_alignment_reward, positive_alignment_reward
from .repetition import quote_penalty, repetition_penalty, special_token_penalty
from .soft_overlong import soft_overlong_penalty


RewardFn = Callable[[Dict[str, Any]], float]


# Minimal registry: only rewards used by the current pipeline/config
REGISTRY: Dict[str, RewardFn] = {
    "group_margin": group_margin,
    "coverage": coverage_reward,
    "label_match": exact_label_match_reward,
    # Penalties
    "rep_penalty": repetition_penalty,
    "quote_penalty": quote_penalty,
    "special_penalty": special_token_penalty,
    # Mission-aware alignment
    "neg_alignment": negative_alignment_reward,
    "pos_alignment": positive_alignment_reward,
    # Optional shaping
    "soft_overlong_penalty": soft_overlong_penalty,
    # Unified lexicon shaping
    "table_lexicon": table_lexicon_conformity,
    "soft_lexicon": soft_lexicon_alignment,
}


def build_reward_fns(names: List[str]) -> List[RewardFn]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown reward names: {unknown}. Known rewards: {sorted(list(REGISTRY.keys()))}.")
    return [REGISTRY[n] for n in names]
