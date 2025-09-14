#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from src_post.rewards.lexicon import FORBIDDEN_DECISION_WORDS, NEGATIVE_TOKENS

# Reuse Stage-B parser patterns
_DECISION_RE = re.compile(r"总评\s*[:：]\s*(通过|不通过)")
_REASON_RE = re.compile(r"原因\s*[:：]\s*(.+)")


def strict_decision_format(sample: Dict[str, Any]) -> float:
    """Return 1.0 only if Stage‑B output strictly matches required format.

    Required:
    - First decision line must contain exactly "总评: 通过" or "总评: 不通过"
    - If decision is 不通过, a reason line must be present with prefix "原因: "
    - Penalize ambiguous variants (forbidden list).
    """
    text = str(sample.get("stage_b_text", "") or "").strip()
    if not text:
        return 0.0
    # Must contain a decision token
    m = _DECISION_RE.search(text)
    if not m:
        return 0.0
    label = m.group(1)
    # If fail, require reason line
    if label == "不通过":
        if not _REASON_RE.search(text):
            return 0.0
    # Disallow ambiguous phrases
    if any(w in text for w in FORBIDDEN_DECISION_WORDS):
        return 0.0
    return 1.0


def pass_prior(sample: Dict[str, Any]) -> float:
    """Soft prior: if summaries contain no mission‑specific negatives, favor PASS.

    Returns in [0,1]. When no negatives are present, returns 1.0 if gt_label=pass,
    and 0.0 if gt_label=fail (so the weight should be small to not overpower).
    """
    gt = str(sample.get("gt_label", "")).strip().lower()
    lines: List[str] = sample.get("summary_lines", []) or []
    text = "\n".join(str(x) for x in lines if isinstance(x, str))
    has_neg = any(w in text for w in NEGATIVE_TOKENS)
    if has_neg:
        return 0.0
    if gt == "pass":
        return 1.0
    if gt == "fail":
        return 0.0
    return 0.0
