#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src_post.rewards import REGISTRY


def compose_reward(
    gt_label: str,
    pred_label: Optional[str],
    summary_lines: List[str],
    reason: Optional[str],
    checklist_lines: List[str],
    reward_names: List[str],
    reward_weights: List[float],
    tf_p_pass: float,
    tf_p_fail: float,
    stage_b_text: Optional[str] = None,
    mission: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> float:
    if len(reward_names) != len(reward_weights):
        raise ValueError(
            f"reward_weights length ({len(reward_weights)}) must match reward_fns length ({len(reward_names)})"
        )
    unknown = [n for n in reward_names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown reward names: {unknown}. Known: {sorted(list(REGISTRY.keys()))}.")
    inputs: Dict[str, Any] = {
        "gt_label": str(gt_label).strip().lower(),
        "pred_label": str(pred_label) if pred_label is not None else None,
        "summary_lines": list(summary_lines),
        "stage_b_reason": reason,
        "checklist_lines": list(checklist_lines),
        "tf_p_pass": float(tf_p_pass),
        "tf_p_fail": float(tf_p_fail),
        # Optional for strict_decision_format
        "stage_b_text": (str(stage_b_text) if stage_b_text is not None else None),
        # Mission for mission-aware rewards (coverage/taxonomy etc.)
        "mission": (str(mission) if mission is not None else None),
        # Meta for auxiliary rewards (e.g., overlong penalty)
        "meta": meta if isinstance(meta, dict) else {},
    }
    composed = 0.0
    denom = 0.0
    for w, fn_name in zip(reward_weights, reward_names):
        fn = REGISTRY[fn_name]
        try:
            r = float(fn(inputs))
        except KeyError as e:
            raise KeyError(f"Reward '{fn_name}' missing input key: {e}. Provided keys: {list(inputs.keys())}.")
        except Exception as e:
            raise RuntimeError(f"Reward '{fn_name}' failed with error: {e}.")
        if not math.isfinite(r):
            raise ValueError(f"Reward '{fn_name}' returned non-finite value: {r}.")
        composed += float(w) * r
        denom += abs(float(w))
    if denom <= 0:
        raise ValueError("Sum of absolute reward weights is zero; please set non-zero weights.")
    return composed / denom
