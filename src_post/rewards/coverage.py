#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple, Optional
from functools import lru_cache

from src_post.utils.text import _extract_keywords
from src_post.prompting.schema import get_mission_checks
from src_post.rewards.lexicon import CANONICAL_SLOTS

# Fallback canonical slots
_SLOTS: Dict[str, Set[str]] = CANONICAL_SLOTS


def _mission_specific_coverage(mission: Optional[str], lines: List[str]) -> Optional[float]:
    if not mission:
        return None
    checks = get_mission_checks(mission)
    if not checks:
        return None
    text = "\n".join([str(x) for x in lines if isinstance(x, str)]).strip()
    if not text:
        return 0.0

    denom = 0.0
    hits = 0.0
    has_no_install = ("无需安装" in text)

    for status, toks, check_name in checks:
        if ("符合性" in check_name) and has_no_install:
            denom += 1.0
            hits += 1.0
            continue
        present = any((t in text) for t in toks)
        if status == "partial":
            if present:
                denom += 1.0
                hits += 1.0
        else:
            denom += 1.0
            if present:
                hits += 1.0
    if denom <= 0.0:
        return 0.0
    return float(hits / denom)


def coverage_reward(sample: Dict[str, Any]) -> float:
    """Compute a mission-aware coverage score in [0,1] from Stage-A summary lines.

    Preferred path: mission-aware coverage using `MISSION_CHECKS_COVERAGE` via schema helpers.
    Fallback: canonical slot coverage + checklist hint coverage.
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    checklist: List[str] = sample.get("checklist_lines", []) or []
    mission: Optional[str] = sample.get("mission")  # type: ignore
    if not lines:
        return 0.0

    mscore = _mission_specific_coverage(mission, lines)
    if isinstance(mscore, float):
        return max(0.0, min(1.0, mscore))

    text = "\n".join(str(x) for x in lines if isinstance(x, str)).strip()
    if not text:
        return 0.0

    hits = 0
    denom = 0

    for _, keywords in _SLOTS.items():
        denom += 1
        if any(kw in text for kw in keywords):
            hits += 1

    if checklist:
        covered = 0
        total = 0
        for hint in [str(h) for h in checklist if isinstance(h, str) and h.strip()]:
            hint_keywords = _extract_keywords([hint])
            if not hint_keywords:
                continue
            total += 1
            if any(kw in text for kw in hint_keywords):
                covered += 1
        if total > 0:
            denom += total
            hits += covered

    return float(hits / max(1, denom))
