#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List


_SEP_RE = re.compile(r"[，,；;、\s]+")
_SPECIAL_RE = re.compile(r"<\|[^|>]+?\|>")


def _repetition_rate(text: str) -> float:
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    tokens = [t.strip() for t in _SEP_RE.split(text) if t.strip()]
    if len(tokens) <= 1:
        return 0.0
    # Count duplicate tokens beyond the first occurrence
    counts: Dict[str, int] = {}
    dup = 0
    for t in tokens:
        c = counts.get(t, 0) + 1
        counts[t] = c
        if c > 1:
            dup += 1
    rate_tokens = dup / max(1, len(tokens))
    # Character-level run penalty (e.g., 哈哈哈哈 / 螺丝螺丝)
    run_penalty = 0.0
    run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1] and text[i].strip():
            run += 1
            if run >= 4:
                run_penalty = 1.0
                break
        else:
            run = 1
    # Combine and clamp to [0,1]
    penalty = min(1.0, rate_tokens + 0.5 * run_penalty)
    return float(penalty)


def repetition_penalty(sample: Dict[str, Any]) -> float:
    """Return repetition penalty in [0,1]; higher when summaries repeat tokens excessively.

    Expects: sample["summary_lines"] as List[str].
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    if not lines:
        return 0.0
    total = 0.0
    for line in lines:
        total += _repetition_rate(line)
    return float(total / max(1, len(lines)))


def quote_penalty(sample: Dict[str, Any]) -> float:
    """Return 1.0 if any line contains quotes; else 0.0.
    Penalizes usage of quotes around the whole summary.
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    if not lines:
        return 0.0
    QUOTES = set(["'", '"', "“", "”", "‘", "’"])
    any_quote = any(any(ch in QUOTES for ch in str(line)) for line in lines)
    return 1.0 if any_quote else 0.0


def special_token_penalty(sample: Dict[str, Any]) -> float:
    """Penalty in [0,1] if special tokens like <|...|> leak into summaries.
    Returns 1.0 if any line matches, else 0.0.
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    if not lines:
        return 0.0
    leak = any(_SPECIAL_RE.search(str(line)) is not None for line in lines)
    return 1.0 if leak else 0.0
