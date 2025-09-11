#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, Any, List


_ILLEGAL_PATTERNS = [
    re.compile(r"<\|[^|>]+?\|>"),  # special tokens like <|...|>
    re.compile(r"\[\s*(?:-?\d+\s*(?:,\s*)?)+\s*\]?"),  # coordinate-like lists
]

_ILLEGAL_CHARS = set(["<", ">", "[", "]"])


def _line_cleanliness_score(text: str) -> float:
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    violations = 0
    # Regex-based patterns
    for pat in _ILLEGAL_PATTERNS:
        if pat.search(text):
            violations += 1
    # Direct illegal characters
    if any(ch in _ILLEGAL_CHARS for ch in text):
        violations += 1
    # Excessive latin letters/digits
    latin = sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9"))
    if latin > 4:
        violations += 1
    # Map to [0,1]: each violation subtracts 0.3, minimum 0
    score = max(0.0, 1.0 - 0.3 * violations)
    return float(score)


def cleanliness_reward(sample: Dict[str, Any]) -> float:
    """Return a positive cleanliness score in [0,1] for Stage-A summaries.

    High when no invalid tokens/characters are present; low when patterns leak.
    Expects: sample["summary_lines"] as List[str].
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    if not lines:
        return 0.0
    total = 0.0
    for line in lines:
        total += _line_cleanliness_score(line)
    return float(total / max(1, len(lines)))
