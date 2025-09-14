#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import List, Optional, Set


_CJK_TOKEN_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def _extract_keywords(hints: List[str]) -> Set[str]:
    """Extract simple keywords from hint sentences.

    - Split by non-alnum/CJK
    - Keep tokens with length >= 2
    - Drop purely numeric tokens
    """
    keywords: Set[str] = set()
    for hint in hints:
        if not isinstance(hint, str):
            continue
        for tok in _CJK_TOKEN_RE.split(hint):
            t = tok.strip()
            if not t:
                continue
            # skip short tokens and pure numbers
            if len(t) < 2:
                continue
            if all(ch.isdigit() for ch in t):
                continue
            keywords.add(t)
    return keywords


def _line_simplicity_score(line: str) -> float:
    """Heuristic simplicity/conciseness score in [0,1].

    Favors short, clean single-line phrases.
    """
    if not isinstance(line, str):
        return 0.0
    text = line.strip()
    if not text:
        return 0.0
    L = len(text)
    # Length-based tiers
    if L <= 36:
        base = 1.0
    elif L <= 48:
        base = 0.9
    elif L <= 64:
        base = 0.8
    elif L <= 80:
        base = 0.6
    else:
        base = 0.4
    # Light penalty for excessive punctuation/commas-like separators
    seps = len([ch for ch in text if ch in {',', '，', '；', ';', '、'}])
    sep_penalty = min(0.3, 0.05 * max(0, seps - 2))
    return max(0.0, min(1.0, base - sep_penalty))


def compute_formatting_score(lines: List[str], checklist: Optional[List[str]] = None) -> float:
    """Compute a formatting score in [0,1] for Stage-A summaries.

    Core idea: encourage coverage of mission-relevant hints (checklist), while
    lightly rewarding concise, clean single-line phrasing. This intentionally
    focuses positive signal on "what matters" to reduce hallucination pressure.

    Args:
        lines: Per-image one-line summaries.
        checklist: Optional list of mission hint sentences used as soft targets.

    Returns:
        A float in [0,1].
    """
    lines = [str(ln) for ln in (lines or []) if isinstance(ln, str)]
    if not lines:
        return 0.0

    # Hint coverage component
    cov = 0.0
    cov_denom = 0.0
    if checklist:
        keywords = _extract_keywords([str(h) for h in checklist if isinstance(h, str)])
        # If extraction fails or empty, fall back to simplicity only
        if keywords:
            # Treat each hint sentence as a unit: covered if any of its keywords appear in any line
            for hint in checklist:
                if not isinstance(hint, str) or not hint.strip():
                    continue
                hint_keywords = _extract_keywords([hint])
                if not hint_keywords:
                    continue
                is_covered = False
                for ln in lines:
                    if any(kw in ln for kw in hint_keywords):
                        is_covered = True
                        break
                cov += 1.0 if is_covered else 0.0
                cov_denom += 1.0
    cov_ratio = (cov / cov_denom) if cov_denom > 0 else 0.0

    # Simplicity component (averaged across lines)
    simp = 0.0
    for ln in lines:
        simp += _line_simplicity_score(ln)
    simp_avg = simp / max(1, len(lines))

    # Combine: prioritize hint coverage, retain some formatting pressure
    score = (0.75 * cov_ratio) + (0.25 * simp_avg)
    return float(max(0.0, min(1.0, score)))


# ---- Shared token parsing helpers ----

def split_candidates(s: str) -> List[str]:
    parts = re.split(r"[|/、，,；;]\s*", s or "")
    return [p.strip() for p in parts if p and p.strip()]


def extract_parenthesized_tokens(text: str) -> Set[str]:
    """Extract tokens inside Chinese/ASCII parentheses, split by candidate separators.

    Example: "by: (未拧紧/露铜)" -> {"未拧紧","露铜"}
    """
    tokens: Set[str] = set()
    for m in re.findall(r"[（(]([^（）()]+)[)）]", str(text) or ""):
        for t in split_candidates(m):
            tokens.add(t)
    return tokens
