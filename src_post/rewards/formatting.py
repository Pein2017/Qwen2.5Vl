#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, List

from src_post.utils.text import compute_formatting_score


def formatting_reward(sample: Dict[str, Any]) -> float:
    """Heuristic reward in [0,1] encouraging on-domain vocabulary and clean format.

    Expects sample to contain keys: summary_lines (List[str]).
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    checklist: List[str] = sample.get("checklist_lines", []) or []
    return float(compute_formatting_score(lines, checklist))
