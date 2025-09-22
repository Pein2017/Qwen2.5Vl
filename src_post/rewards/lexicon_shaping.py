#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from src_post.rewards.lexicon import CANONICAL_SLOTS
from src_post.prompting.schema import get_mission_token_sets_by_outcome, get_mission_checks
from src_post.utils.text import _extract_keywords, split_candidates
from difflib import SequenceMatcher


# A small curated set of Traditional-only characters observed in this domain.
# This is not exhaustive, but targets common leakages to encourage simplified forms.
_TRAD_CHARS: Set[str] = set(list(
    "處絲機櫃顯纜纖設備標籤無於與門電風擋盤臺體對點碼號規則關係採復試驗輸顯現務聯絡號碼資料檢視檢測螺絲於顯示"
))


def _allowed_token_set(mission: str | None) -> Set[str]:
    pass_tokens, fail_tokens = get_mission_token_sets_by_outcome(mission)
    allowed: Set[str] = set(pass_tokens) | set(fail_tokens)
    # Extend with canonical slot tokens
    for _, variants in CANONICAL_SLOTS.items():
        allowed |= set(variants)
    return {t for t in allowed if isinstance(t, str) and t.strip()}


def _extract_line_tokens(lines: List[str]) -> Tuple[Set[str], int]:
    tokens: Set[str] = set()
    total = 0
    for ln in lines:
        kws = _extract_keywords([ln])
        total += len(kws)
        tokens |= kws
    return tokens, total


def in_lexicon_reward(sample: Dict[str, Any]) -> float:
    """Reward ratio of Stage-A tokens that belong to mission/canonical lexicon.

    Returns value in [0,1]. When no tokens are extracted, returns 0.0.
    """
    lines: List[str] = [str(x) for x in (sample.get("summary_lines", []) or []) if isinstance(x, (str,))]
    mission = sample.get("mission")
    if not lines:
        return 0.0
    allowed = _allowed_token_set(mission)
    toks, total = _extract_line_tokens(lines)
    if total <= 0:
        return 0.0
    hits = sum(1 for t in toks if t in allowed)
    return float(max(0.0, min(1.0, hits / float(total))))


def oov_penalty(sample: Dict[str, Any]) -> float:
    """Penalty ratio of Stage-A tokens that are outside mission/canonical lexicon.

    Returns value in [0,1]. When no tokens are extracted, returns 0.0.
    """
    lines: List[str] = [str(x) for x in (sample.get("summary_lines", []) or []) if isinstance(x, (str,))]
    mission = sample.get("mission")
    if not lines:
        return 0.0
    allowed = _allowed_token_set(mission)
    toks, total = _extract_line_tokens(lines)
    if total <= 0:
        return 0.0
    oov = sum(1 for t in toks if t not in allowed)
    return float(max(0.0, min(1.0, oov / float(total))))


def traditional_char_penalty(sample: Dict[str, Any]) -> float:
    """Penalty in [0,1] for presence of Traditional-only characters in Stage-A lines.

    Heuristic: compute the fraction of CJK characters that are in a curated
    Traditional-only set; cap to 1.0. If no CJK characters present, returns 0.0.
    """
    lines: List[str] = [str(x) for x in (sample.get("summary_lines", []) or []) if isinstance(x, (str,))]
    if not lines:
        return 0.0
    text = "\n".join(lines)
    import re as _re
    cjk_chars = _re.findall(r"[\u4e00-\u9fff]", text)
    if not cjk_chars:
        return 0.0
    trad_count = sum(1 for ch in cjk_chars if ch in _TRAD_CHARS)
    frac = trad_count / float(len(cjk_chars))
    return float(max(0.0, min(1.0, frac)))


def table_lexicon_conformity(sample: Dict[str, Any]) -> float:
    """Unified reward in [0,1] to encourage table.json lexicon and resist unrelated tokens.

    Rules:
      - Build allowed lexicon from mission (pass+fail items) and canonical slots.
      - Consider tokens BEFORE any "备注:" or "备注：" substring as "strict" segment.
      - Positive: higher ratio of allowed tokens in the strict segment.
      - Negative: OOV tokens ratio (strict segment) and Traditional-only character fraction.
      - Tokens AFTER the first "备注:" are not penalized for OOV (treated as freeform remarks).

    Returns:
      A score in [0,1], higher is better (more conformant to the spec lexicon).
    """
    lines: List[str] = [str(x) for x in (sample.get("summary_lines", []) or []) if isinstance(x, (str,))]
    mission = sample.get("mission")
    if not lines:
        return 0.0

    # Split at first occurrence of 备注: / 备注： per line and keep pre-remarks part for strict evaluation
    strict_parts: List[str] = []
    for ln in lines:
        idx = -1
        for mark in ["备注:", "备注："]:
            j = ln.find(mark)
            if j != -1:
                idx = j if idx == -1 else min(idx, j)
        strict_parts.append(ln if idx == -1 else ln[:idx])

    allowed = _allowed_token_set(mission)

    # Token extraction on strict parts
    strict_tokens, strict_total = _extract_line_tokens(strict_parts)
    if strict_total <= 0:
        # If nothing to evaluate strictly, fall back to a minimal Traditional penalty across all lines
        tp = traditional_char_penalty({"summary_lines": lines})
        return float(max(0.0, min(1.0, 1.0 - tp)))

    allowed_hits = sum(1 for t in strict_tokens if t in allowed)
    oov_hits = sum(1 for t in strict_tokens if t not in allowed)
    allowed_ratio = allowed_hits / float(strict_total)
    oov_ratio = oov_hits / float(strict_total)

    # Traditional char penalty on strict segment only
    import re as _re
    strict_text = "\n".join(strict_parts)
    cjk_chars = _re.findall(r"[\u4e00-\u9fff]", strict_text)
    if cjk_chars:
        trad_frac = sum(1 for ch in cjk_chars if ch in _TRAD_CHARS) / float(len(cjk_chars))
    else:
        trad_frac = 0.0

    # Generic misuse penalty: in BBU missions, penalize using generic '设备' without any 'BBU'/'BBU设备' token
    def _generic_misuse_penalty(mission_name: Any, tokens: Set[str]) -> float:
        try:
            m = (str(mission_name) or "").lower()
        except Exception:
            m = ""
        if "bbu" in m:
            has_generic = ("设备" in tokens)
            has_bbu_specific = any((("bbu" in t.lower()) or ("BBU设备" in t)) for t in tokens)
            if has_generic and not has_bbu_specific:
                return 1.0
        return 0.0

    misuse = _generic_misuse_penalty(mission, strict_tokens)

    # Combine: prioritize allowed tokens, reduce oov and Traditional usage
    score = (0.7 * allowed_ratio) + (0.2 * (1.0 - oov_ratio)) + (0.1 * (1.0 - trad_frac))
    # Apply misuse penalty (subtract a fixed margin)
    if misuse > 0.0:
        score = score - 0.3 * misuse
    return float(max(0.0, min(1.0, score)))


def _allowed_phrases(mission: str | None) -> Set[str]:
    """Collect full phrases from table.json for fuzzy alignment."""
    phrases: Set[str] = set()
    try:
        for _, _toks, check_name in get_mission_checks(mission):
            if isinstance(check_name, str) and check_name.strip():
                phrases.add(check_name.strip())
    except Exception:
        pass
    # Also include canonical slot variants as phrases
    for _, variants in CANONICAL_SLOTS.items():
        for v in variants:
            if isinstance(v, str) and v.strip():
                phrases.add(v.strip())
    return phrases


def _best_fuzzy_ratio(text: str, candidates: Set[str]) -> float:
    if not isinstance(text, str) or not text.strip() or not candidates:
        return 0.0
    # Light normalization: collapse spaces/commas-like
    s = "".join(ch for ch in text if ch.strip())
    best = 0.0
    for c in candidates:
        try:
            t = "".join(ch for ch in c if ch.strip())
            r = SequenceMatcher(None, s, t).ratio()
            if r > best:
                best = r
        except Exception:
            continue
    return float(max(0.0, min(1.0, best)))


def soft_lexicon_alignment(sample: Dict[str, Any]) -> float:
    """Soft, noise-tolerant alignment reward in [0,1] for Stage-A summaries.

    Design goals:
      - Encourage segments that softly match table.json phrases/canonical slots.
      - Tolerate minor typos/noise; avoid hard rejection of near-miss segments.
      - Limit constraints to pre-remarks content (before the first '备注:'/ '备注：').

    Method:
      - Split each pre-remarks line into segments by separators (逗号/斜杠/顿号等)。
      - For each segment, compute best fuzzy similarity to allowed phrases/tokens.
      - Accept matches with ratio >= THR (default 0.60). Compute coverage and avg similarity.
      - Penalize noise by character fraction of low-similarity segments.
      - Small bonus for presence of counting pattern '×N'.
    """
    lines: List[str] = [str(x) for x in (sample.get("summary_lines", []) or []) if isinstance(x, (str,))]
    mission = sample.get("mission")
    if not lines:
        return 0.0

    # Pre-remarks strict parts
    strict_parts: List[str] = []
    for ln in lines:
        cut = len(ln)
        for mark in ["备注:", "备注："]:
            j = ln.find(mark)
            if j != -1:
                cut = min(cut, j)
        strict_parts.append(ln[:cut])

    # Build candidate lexicon
    phrases = _allowed_phrases(mission)
    allowed_tokens = _allowed_token_set(mission)
    # Use phrases ∪ tokens for matching
    cand: Set[str] = set(phrases) | set(allowed_tokens)
    if not cand:
        return 0.0

    # Split into segments
    segments: List[str] = []
    for part in strict_parts:
        segments.extend(split_candidates(part))
    segments = [seg for seg in segments if isinstance(seg, str) and seg.strip()]
    if not segments:
        return 0.0

    total_chars = sum(len(seg) for seg in segments)
    if total_chars <= 0:
        return 0.0

    # Compute per-segment best fuzzy ratio
    THR = 0.60
    matched_count = 0
    matched_sim_sum = 0.0
    noise_chars = 0
    has_count = 0
    import re as _re
    count_re = _re.compile(r"×\s*\d+")
    for seg in segments:
        r = _best_fuzzy_ratio(seg, cand)
        if r >= THR:
            matched_count += 1
            matched_sim_sum += r
        else:
            noise_chars += len(seg)
        if count_re.search(seg) is not None:
            has_count += 1

    coverage = matched_count / float(len(segments))
    sim_avg = (matched_sim_sum / float(max(1, matched_count))) if matched_count > 0 else 0.0
    noise_ratio = noise_chars / float(total_chars)
    count_ratio = has_count / float(len(segments))

    score = (0.50 * coverage) + (0.35 * sim_avg) + (0.10 * count_ratio) + (0.05 * (1.0 - noise_ratio))
    return float(max(0.0, min(1.0, score)))


