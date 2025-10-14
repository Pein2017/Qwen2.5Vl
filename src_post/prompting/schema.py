#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from src_post.utils.text import _extract_keywords, extract_parenthesized_tokens


def _normalize_mission(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(str(name).strip().lower().split())


def _resolve_table_path() -> str:
    root_guess = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = [
        os.path.join(root_guess, "group_annotation", "table.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "group_annotation", "table.json"),
        os.path.join(os.getcwd(), "group_annotation", "table.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("group_annotation/table.json not found in expected locations")


@lru_cache(maxsize=1)
def _build_parsed_checks() -> Dict[str, List[Tuple[str, Set[str], str]]]:
    """Build mission checks directly from group_annotation/table.json.

    Returns mission→list of (status, tokens, check_name)
    - status: "covered" (single status used)
    - tokens: a non-empty set of strings to match in text
    - check_name: the original item string (for logging/diagnostics)
    """
    path = _resolve_table_path()
    data = json.loads(open(path, "r", encoding="utf-8").read())
    mapping: Dict[str, List[Tuple[str, Set[str], str]]] = {}
    for row in data:
        mission_raw = row.get("mission")
        mission = _normalize_mission(mission_raw)
        items = row.get("desc_summary") or []
        if not isinstance(items, list):
            continue
        cur = mapping.setdefault(mission, [])
        for it in items:
            if not isinstance(it, str):
                continue
            s = it.strip()
            if not s:
                continue
            # Tokens: extract from parentheses if present; always include the full phrase
            toks = set(extract_parenthesized_tokens(s))
            toks.add(s)
            cur.append(("covered", toks, s))
    return mapping


def get_mission_checks(mission: Optional[str]) -> List[Tuple[str, Set[str], str]]:
    parsed = _build_parsed_checks()
    key = _normalize_mission(mission)
    if not key or key not in parsed:
        return []
    return list(parsed[key])


def get_mission_token_set(mission: Optional[str]) -> Set[str]:
    toks: Set[str] = set()
    for status, ts, _ in get_mission_checks(mission):
        toks |= set(ts)
    return toks


@lru_cache(maxsize=64)
def get_mission_token_sets_by_outcome(mission: Optional[str]) -> Tuple[Set[str], Set[str]]:
    """Return two token sets for the mission: (pass_tokens, fail_tokens).

    Tokens are collected from group_annotation/table.json by mission and
    split by the row's global_pass flag. We include:
      - the full phrase
      - tokens extracted from parentheses
      - simple keywords extracted from the phrase (CJK/alnum splits)
    """
    if not mission:
        return set(), set()
    path = _resolve_table_path()
    try:
        data = json.loads(open(path, "r", encoding="utf-8").read())
    except Exception as e:
        raise FileNotFoundError(f"Failed reading group_annotation table: {path} ({e})")
    key = _normalize_mission(mission)
    pass_tokens: Set[str] = set()
    fail_tokens: Set[str] = set()
    for row in data:
        m_raw = row.get("mission")
        m = _normalize_mission(m_raw)
        if m != key:
            continue
        gp = str(row.get("global_pass", "")).strip()
        items = row.get("desc_summary") or []
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, str):
                continue
            s = it.strip()
            if not s:
                continue
            toks: Set[str] = set()
            toks |= set(extract_parenthesized_tokens(s))
            toks |= set(_extract_keywords([s]))
            toks.add(s)
            if gp == "通过":
                pass_tokens |= toks
            elif gp == "不通过":
                fail_tokens |= toks
    return pass_tokens, fail_tokens
