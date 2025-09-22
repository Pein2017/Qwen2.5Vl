#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any


def soft_overlong_penalty(inputs: Dict[str, Any]) -> float:
    """Penalty when Stage-B reply hits max length without EOS.

    Expected inputs keys (subset used):
      - stage_b_text: Optional[str]
      - meta: Optional[dict] may include {"stage_b_hit_max": bool, "soft_overlong_penalty_weight": float}

    Returns:
      negative penalty <= 0.0 when hit_max is True, else 0.0
    """
    meta = inputs.get("meta") if isinstance(inputs.get("meta"), dict) else {}
    hit_max = bool(meta.get("stage_b_hit_max", False))
    if not hit_max:
        return 0.0
    weight = meta.get("soft_overlong_penalty_weight", 0.0)
    try:
        w = float(weight)
    except Exception:
        w = 0.0
    if w < 0.0:
        w = 0.0
    # return negative penalty when hit max
    return -w
