#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Set, Tuple, Optional

from src_post.prompting.conversation import MISSION_CHECKS_COVERAGE
from src_post.utils.text import extract_parenthesized_tokens


@lru_cache(maxsize=1)
def _build_parsed_checks() -> Dict[str, List[Tuple[str, Set[str], str]]]:
    """Parse `MISSION_CHECKS_COVERAGE` into mission→list of (status, tokens, check_name).

    status: "covered"|"partial"
    tokens: canonical tokens that satisfy the check
    check_name: human-readable check description
    """
    mapping: Dict[str, List[Tuple[str, Set[str], str]]] = {}
    for mission, items in MISSION_CHECKS_COVERAGE.items():
        cur: List[Tuple[str, Set[str], str]] = []
        for it in items:
            status = str(it.get("status", "covered")).strip()
            if status not in {"covered", "partial"}:
                continue
            check = str(it.get("check", ""))
            by = str(it.get("by", ""))
            toks = set()
            toks |= extract_parenthesized_tokens(by)
            toks |= extract_parenthesized_tokens(check)
            if "标签/无法识别" in by or "标签/无法识别" in check:
                toks.add("标签/无法识别")
            if toks:
                cur.append((status, toks, check))
        mapping[mission] = cur
    return mapping


def get_mission_checks(mission: Optional[str]) -> List[Tuple[str, Set[str], str]]:
    parsed = _build_parsed_checks()
    if not mission or mission not in parsed:
        return []
    return list(parsed[mission])


def get_mission_token_set(mission: Optional[str]) -> Set[str]:
    toks: Set[str] = set()
    for status, ts, _ in get_mission_checks(mission):
        toks |= set(ts)
    return toks
