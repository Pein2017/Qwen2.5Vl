#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Set

from src_post.utils.text import _extract_keywords

# Heuristic slot keywords derived from data_details taxonomy and existing Chinese hints
_SLOTS: Dict[str, Set[str]] = {
    # Brand / device
    "brand": {"华为", "中兴", "爱立信", "BBU"},
    # Shield requirement & presence/mention
    "shield_requirement": {"需要挡风板", "无需挡风板"},
    "shield": {"挡风板"},
    # Shield attributes
    "shield_direction": {"安装方向正确", "安装方向错误"},
    "shield_obstruction": {"无遮挡", "有遮挡"},
    # Connection points (screws/connectors)
    "connect_compliance": {"合规", "不合规"},
    "connect_issues": {"未拧紧", "露铜", "复接", "生锈"},
    # Fibers
    "fiber_protection": {"有保护", "无保护", "套管", "蛇形管", "铠装"},
    "fiber_bend": {"弯曲半径合理", "弯曲半径违规", "弯曲半径不合理", "弯曲违规"},
    # Wires
    "wire_org": {"整齐", "杂乱"},
    # Labels
    "label_readable": {"标签", "清晰", "可读", "不可读"},
}


def coverage_reward(sample: Dict[str, Any]) -> float:
    """Compute a slot-coverage score in [0,1] from Stage-A summary lines.

    - Counts how many domain-relevant slots are mentioned in the summaries.
    - Optionally treats mission checklist hints as an extra coverage slot.

    Expected keys in sample:
      - summary_lines: List[str]
      - checklist_lines: Optional[List[str]]
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    checklist: List[str] = sample.get("checklist_lines", []) or []
    if not lines:
        return 0.0

    text = "\n".join(str(x) for x in lines if isinstance(x, str)).strip()
    if not text:
        return 0.0

    hits = 0
    denom = 0

    # Slot coverage
    for _, keywords in _SLOTS.items():
        denom += 1
        if any(kw in text for kw in keywords):
            hits += 1

    # Checklist coverage as an extra slot (robust to mission-specific phrasing)
    if checklist:
        ck = _extract_keywords([str(h) for h in checklist if isinstance(h, str)])
        if ck:
            denom += 1
            if any(kw in text for kw in ck):
                hits += 1

    return float(hits / max(1, denom))
