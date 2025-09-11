#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set


@lru_cache(maxsize=1)
def _load_taxonomy() -> Dict[str, Set[str]]:
    base = Path.cwd()
    candidates = [
        base / "hierarchical_attribute_mapping.json",
        base / "attribute_taxonomy.json",
        Path("hierarchical_attribute_mapping.json"),
        Path("attribute_taxonomy.json"),
    ]
    vocab: Set[str] = set()
    for p in candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                # Collect leaf strings from nested dict/list structures
                def collect(obj: Any) -> None:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(k, str):
                                vocab.add(k.strip())
                            collect(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            collect(it)
                    elif isinstance(obj, str):
                        vocab.add(obj.strip())
                collect(data)
        except Exception:
            continue
    # Pre-filter very short tokens
    vocab = {t for t in vocab if t and len(t) >= 2}
    return {"vocab": vocab}


def taxonomy_reward(sample: Dict[str, Any]) -> float:
    """Reward mission-relevant vocabulary hits in Stage‑A summaries.

    Expects sample to contain: summary_lines (List[str]). Returns [0,1].
    """
    lines: List[str] = sample.get("summary_lines", []) or []
    if not lines:
        return 0.0
    tax = _load_taxonomy()
    vocab = tax.get("vocab", set())
    if not vocab:
        return 0.0
    total = 0.0
    for ln in lines:
        if not isinstance(ln, str):
            continue
        text = ln.strip()
        if not text:
            continue
        hits = sum(1 for w in vocab if w in text)
        off = 0
        # Off-domain heuristic: latin blobs or obvious noise reduce score
        off += sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z")) > 6
        off += 1 if any(sym in text for sym in ["<", ">", "[", "]"]) else 0
        # Normalize: favor a few relevant hits; clip penalties
        score = max(0.0, min(1.0, (hits * 0.25) - (0.2 * off)))
        total += score
    return float(total / max(1, len(lines)))
