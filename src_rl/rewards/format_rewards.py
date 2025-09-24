#!/usr/bin/env python3
"""
Format and domain rewards for dense-caption GRPO (pure text).

All functions are side-effect free and deterministic.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

# Canonical wrappers (literal tokens)
OBJ_S = "<|object_ref_start|>"
OBJ_E = "<|object_ref_end|>"
BOX_S = "<|box_start|>"
BOX_E = "<|box_end|>"
QUAD_S = "<|quad_start|>"
QUAD_E = "<|quad_end|>"
LINE_S = "<|line_start|>"
LINE_E = "<|line_end|>"

# Precompiled patterns
_NUM = re.compile(r"-?\d+")
_CHINESE_PUNCT = re.compile(r"[，、；：]")
_COORD_LIST = re.compile(r"\[(.*?)\]")


def _extract_coord_values(section: str) -> List[int]:
    vals = []
    for m in _NUM.findall(section):
        try:
            vals.append(int(m))
        except Exception:
            pass
    return vals


def check_wrappers(text: str) -> float:
    has_obj = OBJ_S in text and OBJ_E in text
    has_box = BOX_S in text and BOX_E in text
    has_quad = QUAD_S in text and QUAD_E in text
    has_line = LINE_S in text and LINE_E in text
    return 1.0 if has_obj and (has_box or has_quad or has_line) else 0.0


def check_ascii_separators(text: str) -> float:
    if _CHINESE_PUNCT.search(text):
        return 0.0
    # Encourage comma+space inside coord lists
    bad = 0
    lists = _COORD_LIST.findall(text)
    for sec in lists:
        if "," in sec and ", " not in sec:
            bad += 1
    if bad > 0:
        return 0.0
    return 1.0


def check_coords_counts(text: str) -> float:
    score = 1.0
    # box: 4 ints
    for m in re.finditer(re.escape(BOX_S) + r"\s*\[(.*?)\]\s*" + re.escape(BOX_E), text, re.DOTALL):
        if len(_extract_coord_values(m.group(1))) != 4:
            score *= 0.0
    # quad: 8 ints
    for m in re.finditer(re.escape(QUAD_S) + r"\s*\[(.*?)\]\s*" + re.escape(QUAD_E), text, re.DOTALL):
        if len(_extract_coord_values(m.group(1))) != 8:
            score *= 0.0
    # line: even >= 4
    for m in re.finditer(re.escape(LINE_S) + r"\s*\[(.*?)\]\s*" + re.escape(LINE_E), text, re.DOTALL):
        vals = _extract_coord_values(m.group(1))
        if len(vals) < 4 or len(vals) % 2 != 0:
            score *= 0.0
    return float(score)


_BANNED_TERMS = ["PPDU", "DCDU", "CPRI", "ODF", "光分路器"]


def check_banned_vocab(text: str) -> float:
    for term in _BANNED_TERMS:
        if term in text:
            return 0.0
    return 1.0


def parse_reward(text: str) -> float:
    # Simple success if at least one geometry block is present
    for pat in (BOX_S, QUAD_S, LINE_S):
        if pat in text:
            return 1.0
    return 0.0


def length_score(text: str, min_tokens: int = 16, max_tokens: int = 320) -> float:
    # Cheap proxy: count numbers + wrapper tokens occurrences
    tokens_like = len(_NUM.findall(text)) + text.count(OBJ_S)
    if tokens_like < min_tokens:
        return max(0.0, tokens_like / float(min_tokens))
    if tokens_like > max_tokens:
        return max(0.0, 1.0 - (tokens_like - max_tokens) / float(max_tokens))
    return 1.0


def compute_reward(text: str, weights: Dict[str, float]) -> float:
    comps = {
        "parse": parse_reward(text),
        "wrappers": check_wrappers(text),
        "coords": check_coords_counts(text),
        "separators": check_ascii_separators(text),
        "vocab": check_banned_vocab(text),
    }
    total_w = 0.0
    score = 0.0
    for k, v in weights.items():
        w = float(v)
        total_w += w
        score += w * float(comps.get(k, 0.0))
    if total_w <= 0:
        return 0.0
    return float(score / total_w)


__all__ = [
    "check_wrappers",
    "check_ascii_separators",
    "check_coords_counts",
    "check_banned_vocab",
    "parse_reward",
    "length_score",
    "compute_reward",
]
