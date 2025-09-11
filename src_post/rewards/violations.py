#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Set


# Minimal violation lexicon; extend as needed
VIOLATION_LEXICON: Set[str] = set([
    "未拧紧",
    "松动",
    "露铜",
    "复接",
    "生锈",
    "遮挡",
    "无遮挡被破坏",  # in case of negations you can expand logic later
    "弯曲半径违规",
    "弯曲半径不合理",
    "不合规",
])


def _count_violations(lines: List[str]) -> int:
    if not lines:
        return 0
    text = "\n".join([str(x) for x in lines if isinstance(x, str)]).strip()
    if not text:
        return 0
    return sum(1 for w in VIOLATION_LEXICON if w in text)


def violation_alignment_reward(sample: Dict[str, Any]) -> float:
    """Align violation mentions with GT label.

    - If GT=fail: reward increases when violations are mentioned in summaries or reason.
    - If GT=pass: penalize if violations are mentioned.

    Returns value in [-1, 1], caller can use a positive weight (for fail) or
    include this as a separate term with appropriate sign in config.

    Expected keys:
      - gt_label: "pass"|"fail"
      - summary_lines: List[str]
      - stage_b_reason: Optional[str]
    """
    try:
        gt = str(sample.get("gt_label", "")).strip().lower()
        lines: List[str] = sample.get("summary_lines", []) or []
        reason: str = sample.get("stage_b_reason") or ""
        num_in_summ = _count_violations(lines)
        num_in_reason = _count_violations([reason] if reason else [])
        total = int(num_in_summ + num_in_reason)
        if gt == "fail":
            # Map counts to [0,1]; cap at 3
            return min(1.0, float(total) / 3.0)
        if gt == "pass":
            # Negative alignment: any violation mention is bad
            return - min(1.0, float(total) / 2.0)
        return 0.0
    except Exception:
        return 0.0
