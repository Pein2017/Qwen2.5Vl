#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src_post.prompting.schema import get_mission_token_sets_by_outcome


def _count_hits(text: str, tokens: Set[str]) -> int:
    if not text or not tokens:
        return 0
    return sum(1 for t in tokens if t in text)


def negative_alignment_reward(sample: Dict[str, Any]) -> float:
    """Mission-aware negative alignment in [-1, 1].

    - If gt=fail: reward increases when any mission-specific FAIL tokens appear in Stage-A summaries
    - If gt=pass: penalize when any mission-specific FAIL tokens appear (should be avoided)
    业务约束：‘遮挡’与‘显示部分/显示完整’不参与判定（视为可见性描述，需过滤）。
    """
    gt = str(sample.get("gt_label", "")).strip().lower()
    mission: Optional[str] = sample.get("mission")  # type: ignore
    lines: List[str] = sample.get("summary_lines", []) or []
    text = "\n".join([str(x) for x in lines if isinstance(x, str)])
    _pass_set, fail_set = get_mission_token_sets_by_outcome(mission)
    # 过滤 dummy 可见性词：遮挡/只显示部分/显示完整
    dummy = {"遮挡", "只显示部分", "显示完整"}
    fail_set = {t for t in fail_set if t not in dummy}
    hits = _count_hits(text, fail_set)
    if gt == "fail":
        # encourage at least one negative; cap to 1.0 quickly
        return min(1.0, float(hits) / 1.0)
    if gt == "pass":
        # any negative mention is bad when pass
        return -min(1.0, float(hits) / 1.0)
    return 0.0


def positive_alignment_reward(sample: Dict[str, Any]) -> float:
    """Mission-aware positive alignment in [-1, 1].

    - If gt=pass: reward increases when PASS tokens appear in summaries
    - If gt=fail: penalize if PASS tokens appear (they can mask true failures)
    """
    gt = str(sample.get("gt_label", "")).strip().lower()
    mission: Optional[str] = sample.get("mission")  # type: ignore
    lines: List[str] = sample.get("summary_lines", []) or []
    text = "\n".join([str(x) for x in lines if isinstance(x, str)])
    pass_set, _fail_set = get_mission_token_sets_by_outcome(mission)
    # 过滤 dummy 可见性词
    dummy = {"遮挡", "只显示部分", "显示完整"}
    pass_set = {t for t in pass_set if t not in dummy}
    hits = _count_hits(text, pass_set)
    if gt == "pass":
        # reward modestly; cap at 1.0 slowly to not overpower negatives
        return min(1.0, float(hits) / 3.0)
    if gt == "fail":
        # discourage mentioning pass tokens in fail groups
        return -min(1.0, float(hits) / 2.0)
    return 0.0
