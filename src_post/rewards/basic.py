#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict


def exact_label_match_reward(sample: Dict[str, Any]) -> float:
    """Binary reward: 1.0 if pred_label == gt_label else 0.0.

    This is reply-aware in Stage-B (pred_label parsed from reply), and becomes
    a no-op in Stage-A where pred_label is None.
    """
    try:
        gt = str(sample.get("gt_label", "")).strip().lower()
        pred = str(sample.get("pred_label", "") or "").strip().lower()
        if not gt or not pred:
            return 0.0
        return 1.0 if gt == pred else 0.0
    except Exception:
        return 0.0
