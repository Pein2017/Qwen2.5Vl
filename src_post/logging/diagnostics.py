#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional

from src_post.rewards import REGISTRY


def _safe_clip_01(x: float) -> float:
    try:
        if x != x or x == float("inf") or x == float("-inf"):
            return 0.0
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def compute_phase_a_diagnostics(summary_lines: List[str], checklist_lines: List[str], mission: Optional[str] = None) -> Dict[str, float]:
    """Compute Stage‑A diagnostics in [0,1] for formatting/coverage/cleanliness/taxonomy/consistency.

    This function wraps the corresponding reward functions to produce a diagnostic snapshot.
    It is device‑agnostic and safe to call on CPU.
    """
    inputs = {
        "summary_lines": list(summary_lines or []),
        "checklist_lines": list(checklist_lines or []),
        "stage_b_reason": "",
        # Other fields are unused by these reward functions
        "gt_label": "",
        "pred_label": None,
        "tf_p_pass": 0.0,
        "tf_p_fail": 0.0,
        # Mission for mission-aware coverage
        "mission": (str(mission) if mission is not None else None),
    }
    out: Dict[str, float] = {}
    for name in ("formatting", "coverage", "cleanliness", "taxonomy", "consistency"):
        fn = REGISTRY.get(name)
        if fn is None:
            out[name] = 0.0
            continue
        try:
            out[name] = _safe_clip_01(float(fn(inputs)))
        except Exception:
            out[name] = 0.0
    return out
