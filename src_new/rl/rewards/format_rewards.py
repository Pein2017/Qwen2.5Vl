#!/usr/bin/env python3
"""
Format and domain rewards for dense-caption GRPO (pure text).

All functions are side-effect free and deterministic.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

from src_new.processing.parse_generated import parse_geometry_response

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


_BANNED_TERMS = [
    "PPDU",
    "DCDU",
    "CPRI",
    "ODF",
    "光分路器",
    # Common typos/variants observed in SFT generations
    "BBB",
    "BBB设备",
    "BBB端",
    "BBB端光纤插头",
    "BBB安装螺丝",
    "BB设备",
    "BB设备端",
    "BB设备电源线",
    "BB设备传输尾纤",
    "BB设备传输光纤",
    "BBUu",
    "BBUu设备",
]


def check_banned_vocab(text: str) -> float:
    # Only evaluate when geometry is present; otherwise return neutral 0.0 to avoid rewarding empties
    if not _has_any_geometry(text):
        return 0.0
    for term in _BANNED_TERMS:
        if term in text:
            return 0.0
    return 1.0


def parse_reward(text: str) -> float:
    """Binary parse success when any geometry wrapper is present."""
    for pat in (BOX_S, QUAD_S, LINE_S):
        if pat in text:
            return 1.0
    return 0.0


def pairing_ratio(text: str, *, meta: Dict | None = None) -> float:
    """Ratio of parsed objects to declared object refs (0..1).

    Encourages each object-ref block to correspond to exactly one geometry block
    that can be parsed by the tolerant parser.
    """
    try:
        obj_refs = int(text.count(OBJ_S))
    except Exception:
        obj_refs = 0
    try:
        parsed = parse_geometry_response(text, tolerant=True) or []
        parsed_count = int(len(parsed))
    except Exception:
        parsed_count = 0
    denom = max(obj_refs, 1)
    val = float(parsed_count) / float(denom)
    if val < 0.0:
        val = 0.0
    if val > 1.0:
        val = 1.0
    return float(val)


def duplicate_penalty(text: str) -> float:
    """Within-line duplicate penalty specialized for 'line' geometry (0..1).

    - Extract only line coordinate lists
    - Penalize zero-length segments and near-duplicate adjacent vertices
    - Returns 1.0 (no penalty) when no line geometry present
    Optional params are injected by runner via rewards_config.duplicate_penalty.line
    """
    # Extract line sections only
    sections: List[str] = []
    for m in re.finditer(
        re.escape(LINE_S) + r"\s*\[(.*?)\]\s*" + re.escape(LINE_E), text, re.DOTALL
    ):
        sections.append(m.group(1))

    if not sections:
        return 1.0

    # Defaults; runner may inject overrides by signature matching
    min_vertex_separation = 2
    zero_length_seg_penalty = 0.02
    per_duplicate_vertex = 0.01
    max_penalty = 0.20

    def _score_one(sec: str) -> float:
        vals = _extract_coord_values(sec)
        if len(vals) < 4 or len(vals) % 2 != 0:
            return 0.0
        pts: List[tuple[int, int]] = []
        it = iter(vals)
        for x in it:
            try:
                y = next(it)
            except StopIteration:
                break
            pts.append((int(x), int(y)))
        if len(pts) < 2:
            return 0.0
        zero_len = 0
        dup_adj = 0
        for i in range(1, len(pts)):
            dx = abs(pts[i][0] - pts[i - 1][0])
            dy = abs(pts[i][1] - pts[i - 1][1])
            if dx == 0 and dy == 0:
                zero_len += 1
            elif (dx + dy) < int(min_vertex_separation):
                dup_adj += 1
        penalty = zero_len * float(zero_length_seg_penalty) + dup_adj * float(
            per_duplicate_vertex
        )
        penalty = min(float(max_penalty), float(penalty))
        return float(max(0.0, 1.0 - penalty))

    scores = [_score_one(sec) for sec in sections]
    return float(sum(scores) / float(len(scores)))


def pattern_penalty(
    text: str,
    *,
    axis_run_max_ratio: float = 0.40,
    step_repeat_max_ratio: float = 0.40,
    per_overshoot: float = 0.05,
    max_penalty: float = 0.20,
) -> float:
    """Penalty in [0,1] for low-diversity line patterns (axis runs, repeated steps).

    Returns 1.0 when no line geometry is present. Runner can inject thresholds via
    rewards_config.pattern_penalty.line.*
    """
    # Extract line sections only
    sections: List[str] = []
    for m in re.finditer(
        re.escape(LINE_S) + r"\s*\[(.*?)\]\s*" + re.escape(LINE_E), text, re.DOTALL
    ):
        sections.append(m.group(1))
    if not sections:
        return 1.0

    def _ratios(sec: str) -> tuple[float, float]:
        vals = _extract_coord_values(sec)
        if len(vals) < 4 or len(vals) % 2 != 0:
            return 1.0, 1.0
        pts: List[tuple[int, int]] = []
        it = iter(vals)
        for x in it:
            try:
                y = next(it)
            except StopIteration:
                break
            pts.append((int(x), int(y)))
        if len(pts) < 2:
            return 1.0, 1.0
        segs = []
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            segs.append((dx, dy))
        if not segs:
            return 1.0, 1.0
        # Axis-run ratio
        axis = sum(1 for (dx, dy) in segs if dx == 0 or dy == 0) / float(len(segs))

        # Step-repeat ratio (most frequent delta)
        # Quantize tiny deltas to reduce sensitivity to small noise
        def _q(v: int) -> int:
            return int(v)

        hist: Dict[tuple[int, int], int] = {}
        for dx, dy in segs:
            key = (_q(dx), _q(dy))
            hist[key] = hist.get(key, 0) + 1
        repeat = max(hist.values()) / float(len(segs)) if hist else 1.0
        return float(axis), float(repeat)

    axes: List[float] = []
    reps: List[float] = []
    for sec in sections:
        a, r = _ratios(sec)
        axes.append(a)
        reps.append(r)
    # Average across lines in completion
    axis_ratio = sum(axes) / float(len(axes)) if axes else 1.0
    step_ratio = sum(reps) / float(len(reps)) if reps else 1.0
    overshoot = 0.0
    if axis_ratio > float(axis_run_max_ratio):
        overshoot += axis_ratio - float(axis_run_max_ratio)
    if step_ratio > float(step_repeat_max_ratio):
        overshoot += step_ratio - float(step_repeat_max_ratio)
    penalty = min(float(max_penalty), float(per_overshoot) * float(overshoot))
    return float(max(0.0, 1.0 - penalty))


## Removed legacy proxy length rewards (length_score, length_window)


def length_vs_gt(
    text: str,
    *,
    meta: Dict | None = None,
    # Minimal Gaussian params (wired from rewards_config.length_vs_gt)
    use_ratio: bool = True,
    sigma_ratio: float = 0.20,
    sigma_tokens: int = 128,
    min_reward: float = 0.0,
) -> float:
    """Gaussian length-vs-GT reward in [min_reward, 1.0].

    r = exp(-((L_pred - L_gt) / sigma)^2) with sigma derived from GT length
    when use_ratio=True, otherwise fixed token sigma.

    Prefers tokenizer-aligned lengths in meta: 'gt_len_tokenizer' and optionally
    'gen_len_tokenizer'. Falls back to a simple numeric proxy for gen_len only.
    """
    # Prefer tokenizer-based lengths if meta provides them
    gen_len = None
    gt_len = None
    if isinstance(meta, dict):
        try:
            val = meta.get("gen_len_tokenizer")
            if isinstance(val, int):
                gen_len = int(val)
        except Exception:
            pass
        try:
            val = meta.get("gt_len_tokenizer")
            if isinstance(val, int):
                gt_len = int(val)
        except Exception:
            pass

    # Fallback: estimate lengths when explicit not provided
    def _proxy_len(txt: str) -> int:
        return int(len(_NUM.findall(txt)) + txt.count(OBJ_S))

    if gen_len is None:
        gen_len = _proxy_len(text)

    # If GT still unavailable, try last-resort from meta.objects
    if gt_len is None:
        if isinstance(meta, dict) and meta.get("objects"):
            try:
                from src_new.processing.coordinate_converter import (
                    CoordinateTokenConverter,
                )

                conv = CoordinateTokenConverter()
                objs = meta.get("objects") or []
                gt_text = conv.convert_objects_to_tokens(objs)
                # Tokenizer not available here; use proxy on GT text
                gt_len = _proxy_len(gt_text)
            except Exception:
                pass
        if gt_len is None:
            # Neutral when true GT length unknown
            return max(0.5, float(min_reward))

    gt_len = int(max(1, int(gt_len)))
    gen_len = int(max(0, int(gen_len)))

    # Sigma selection
    if bool(use_ratio):
        sigma = max(1.0, float(sigma_ratio) * float(gt_len))
    else:
        sigma = max(1.0, float(sigma_tokens))

    delta = float(gen_len - gt_len)
    val = math.exp(-((delta / sigma) ** 2))
    if val < float(min_reward):
        val = float(min_reward)
    if val > 1.0:
        val = 1.0
    return float(val)


def compute_reward(text: str, weights: Dict[str, float]) -> float:
    comps = {
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
    "pairing_ratio",
    "duplicate_penalty",
    "pattern_penalty",
    "length_vs_gt",
    "compute_reward",
]
