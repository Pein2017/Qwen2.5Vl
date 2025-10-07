#!/usr/bin/env python3
"""
Format and domain rewards for dense-caption GRPO (pure text).

All functions are side-effect free and deterministic.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

# Canonical wrappers sourced from the single authority
from src_new.processing.special_tokens import GEOMETRY_TOKENS


OBJ_S, OBJ_E, BOX_S, BOX_E = (
    GEOMETRY_TOKENS["bbox_2d"][0],
    GEOMETRY_TOKENS["bbox_2d"][1],
    GEOMETRY_TOKENS["bbox_2d"][2],
    GEOMETRY_TOKENS["bbox_2d"][3],
)
QUAD_S, QUAD_E = GEOMETRY_TOKENS["quad"][2], GEOMETRY_TOKENS["quad"][3]
LINE_S, LINE_E = GEOMETRY_TOKENS["line"][2], GEOMETRY_TOKENS["line"][3]

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


def _has_any_geometry(text: str) -> bool:
    return (
        BOX_S in text
        and BOX_E in text
        or QUAD_S in text
        and QUAD_E in text
        or LINE_S in text
        and LINE_E in text
    )


def check_ascii_separators(text: str) -> float:
    # Do not reward separator formatting when there is no geometry at all
    if not _has_any_geometry(text):
        return 0.0
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


def _iter_coord_sections_within_wrappers(text: str) -> List[str]:
    """Return only the coordinate list contents inside geometry wrappers.

    This is stricter than global bracket search and avoids counting JSON-like
    fragments outside geometry tokens.
    """
    sections: List[str] = []
    # box
    for m in re.finditer(
        re.escape(BOX_S) + r"\s*\[(.*?)\]\s*" + re.escape(BOX_E), text, re.DOTALL
    ):
        sections.append(m.group(1))
    # quad
    for m in re.finditer(
        re.escape(QUAD_S) + r"\s*\[(.*?)\]\s*" + re.escape(QUAD_E), text, re.DOTALL
    ):
        sections.append(m.group(1))
    # line
    for m in re.finditer(
        re.escape(LINE_S) + r"\s*\[(.*?)\]\s*" + re.escape(LINE_E), text, re.DOTALL
    ):
        sections.append(m.group(1))
    return sections


def separators_score(
    text: str,
    *,
    chinese_penalty_alpha: float = 0.15,
    w_space: float = 0.6,
    w_completeness: float = 0.4,
) -> float:
    """Continuous separators reward in [0,1] with soft penalties.

    - Requires geometry wrappers present; otherwise returns 0.0
    - Scores only coordinate lists inside wrappers
    - Per-list score combines:
        * space-after-comma ratio: count(', ')/count(',')
        * comma completeness: observed_commas / expected_commas, expected=max(nums-1, 0)
    - Applies soft global penalty for Chinese punctuation occurrences
    """

    if not _has_any_geometry(text):
        return 0.0

    # Global soft penalty for Chinese punctuation occurrences (not hard zero)
    cn = len(_CHINESE_PUNCT.findall(text))
    global_penalty = math.exp(-float(chinese_penalty_alpha) * float(cn))

    lists = _iter_coord_sections_within_wrappers(text)
    if not lists:
        return 0.0

    per_list_scores: List[float] = []
    for sec in lists:
        # numbers and commas in the section
        nums = _NUM.findall(sec)
        num_numbers = len(nums)
        comma_count = sec.count(",")
        comma_space_count = sec.count(", ")

        # Space-after-comma ratio
        space_ratio = (
            1.0
            if comma_count == 0
            else float(comma_space_count) / float(max(comma_count, 1))
        )

        # Expected comma completeness ≈ (numbers - 1)
        expected_commas = max(num_numbers - 1, 0)
        completeness = (
            1.0
            if expected_commas == 0
            else min(1.0, float(comma_count) / float(expected_commas))
        )

        list_score = float(w_space) * space_ratio + float(w_completeness) * completeness
        per_list_scores.append(list_score)

    base = sum(per_list_scores) / float(len(per_list_scores))
    score = max(0.0, min(1.0, base * global_penalty))
    return float(score)


def check_coords_counts(text: str) -> float:
    """Continuous coordinate count correctness reward in [0,1].

    Returns the fraction of coordinate lists with correct counts:
    - bbox: exactly 4 ints
    - quad: exactly 8 ints
    - line: even count >= 4

    Provides smoother RL signal than binary all-or-nothing.
    """
    checks: List[bool] = []

    # box: 4 ints
    for m in re.finditer(
        re.escape(BOX_S) + r"\s*\[(.*?)\]\s*" + re.escape(BOX_E), text, re.DOTALL
    ):
        checks.append(len(_extract_coord_values(m.group(1))) == 4)

    # quad: 8 ints
    for m in re.finditer(
        re.escape(QUAD_S) + r"\s*\[(.*?)\]\s*" + re.escape(QUAD_E), text, re.DOTALL
    ):
        checks.append(len(_extract_coord_values(m.group(1))) == 8)

    # line: even >= 4
    for m in re.finditer(
        re.escape(LINE_S) + r"\s*\[(.*?)\]\s*" + re.escape(LINE_E), text, re.DOTALL
    ):
        vals = _extract_coord_values(m.group(1))
        checks.append(len(vals) >= 4 and len(vals) % 2 == 0)

    if not checks:
        return 0.0

    return float(sum(1.0 for c in checks if c) / len(checks))


_BANNED_TERMS = ["PPDU", "DCDU", "CPRI", "ODF", "光分路器"]


def check_banned_vocab(text: str) -> float:
    # Only evaluate when geometry is present; otherwise return neutral 0.0 to avoid rewarding empties
    if not _has_any_geometry(text):
        return 0.0
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


## Removed legacy proxy length rewards (length_score, length_window)


def length_vs_gt(
    text: str,
    *,
    meta: Dict | None = None,
    gen_len: int | None = None,
    gt_len: int | None = None,
    estimator: str = "tokenizer",
    lower: float = 0.7,
    upper: float = 1.2,
    gamma: float = 3.0,
    tail_numeric_weight: float = 0.4,
    alpha: float = 1.1,
) -> float:
    """Length-to-GT reward in [0,1] with strong overflow penalty.

    Always prefers tokenizer-aligned lengths injected via meta keys
    `gen_len_tokenizer` and `gt_len_tokenizer`. Falls back to simple
    numeric proxy only if unavailable.
    """
    # Prefer tokenizer-based lengths if meta provides them
    if gen_len is None and isinstance(meta, dict):
        try:
            val = meta.get("gen_len_tokenizer")
            if isinstance(val, int):
                gen_len = int(val)
        except Exception:
            pass
    if gt_len is None and isinstance(meta, dict):
        try:
            val = meta.get("gt_len_tokenizer")
            if isinstance(val, int):
                gt_len = int(val)
        except Exception:
            pass

    # Fallback estimators when explicit lengths not provided
    def _proxy_len(txt: str) -> int:
        return int(len(_NUM.findall(txt)) + txt.count(OBJ_S))

    if gen_len is None:
        gen_len = _proxy_len(text)
    if gt_len is None:
        gt_len = _proxy_len(text)

    gt_len = int(max(1, int(gt_len)))
    gen_len = int(max(0, int(gen_len)))

    r = float(gen_len) / float(gt_len)
    if r < float(lower):
        base = max(0.0, r / float(max(lower, 1e-6)))
    elif r <= float(upper):
        base = 1.0
    else:
        base = math.exp(-float(gamma) * (r - float(upper)))

    # Tail numeric penalty beyond alpha*gt_len
    tail_pen = 0.0
    try:
        cutoff = int(math.ceil(float(alpha) * float(gt_len)))
        if gen_len > cutoff and float(tail_numeric_weight) > 0.0:
            nums = len(_NUM.findall(text))
            digits = sum(ch.isdigit() for ch in text)
            total = max(1, len(text))
            frac_num = max(
                float(nums) / float(max(1, gen_len)), float(digits) / float(total)
            )
            tail_pen = float(tail_numeric_weight) * max(0.0, min(1.0, frac_num))
    except Exception:
        tail_pen = 0.0

    return max(0.0, min(1.0, float(base) - float(tail_pen)))


def compute_reward(text: str, weights: Dict[str, float]) -> float:
    comps = {
        "parse": parse_reward(text),
        "wrappers": check_wrappers(text),
        "coords": check_coords_counts(text),
        "separators": check_ascii_separators(text),
        "vocab": check_banned_vocab(text),
        # legacy length proxies removed
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
    "separators_score",
    "check_coords_counts",
    "check_banned_vocab",
    "parse_reward",
    "length_vs_gt",
    "compute_reward",
]
