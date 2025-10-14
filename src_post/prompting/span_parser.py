#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, Optional


# Match variations like "总评: 通过" / "总评：通过" with flexible whitespace
_DECISION_RE = re.compile(r"总评\s*[:：]\s*(通过|不通过|合格|否)", re.IGNORECASE)
_REASON_RE = re.compile(r"原因\s*[:：]\s*(.+)")


def _normalize_label(label_text: str) -> str:
    text = label_text.strip().lower()
    if text in {"通过", "合格", "pass", "passed"}:
        return "pass"
    if text in {"不通过", "否", "fail", "failed"}:
        return "fail"
    return text


def parse_stage_b_output(text: str) -> Dict[str, Optional[str]]:
    """Parse Stage-B textual decision into structured fields.

    Returns dict with keys: `label` ("pass"|"fail" or None) and `reason` (str or None).
    """
    label: Optional[str] = None
    reason: Optional[str] = None

    m = _DECISION_RE.search(text)
    if m:
        label = _normalize_label(m.group(1))

    m2 = _REASON_RE.search(text)
    if m2:
        # Extract first line to avoid capturing trailing paragraphs
        reason_full = m2.group(1).strip()
        reason = reason_full.splitlines()[0].strip()

    return {"label": label, "reason": reason}
