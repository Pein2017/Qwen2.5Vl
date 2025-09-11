#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List


def _simple_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # character-level overlap ratio clipped to [0,1]
    aset = set(a)
    bset = set(b)
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    union = len(aset | bset)
    if union == 0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def mission_consistency_reward(sample: Dict[str, Any]) -> float:
    """Reward consistency between Stage‑B rationale/checklist and Stage‑A summaries.

    Expects in sample: summary_lines (List[str]), stage_b_reason (str|None), checklist_lines (List[str]|None).
    Returns [0,1].
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    reason: str = sample.get("stage_b_reason") or ""
    checklist: List[str] = sample.get("checklist_lines", []) or []
    if not lines:
        return 0.0
    score = 0.0
    # Overlap with reason
    if reason:
        score += max(_simple_overlap(reason, ln) for ln in lines)
    # Overlap with checklist items: average best-overlap per item
    if checklist:
        per_item = []
        for item in checklist:
            per_item.append(max(_simple_overlap(item, ln) for ln in lines) if lines else 0.0)
        score += sum(per_item) / max(1, len(per_item))
    # Normalize: reason (0..1) + checklist avg (0..1) => [0,1] after /2
    return float(max(0.0, min(1.0, score / 2.0)))
