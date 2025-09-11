#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict


def decision_prob_reward(sample: Dict[str, Any]) -> float:
    """Return a dense, label-aligned score from teacher-forcing decision probabilities.

    Expected keys in sample (best-effort, returns 0.0 if missing):
      - gt_label: "pass"|"fail"
      - tf_p_pass: Optional[float]  # P("总评: 通过") from TF over Stage-B prompt
      - tf_p_fail: Optional[float]  # P("总评: 不通过") from TF over Stage-B prompt
      - tf_logp_pass/tf_logp_fail (optional): if provided, will be preferred to compute probs via softmax
    Returns a float in [0,1].
    """
    try:
        gt = str(sample.get("gt_label", "")).strip().lower()
        # Prefer direct probabilities when available
        p_pass = sample.get("tf_p_pass", None)
        p_fail = sample.get("tf_p_fail", None)
        # Fallback: derive from log-probs if present
        if (p_pass is None or p_fail is None):
            lp_pass = sample.get("tf_logp_pass", None)
            lp_fail = sample.get("tf_logp_fail", None)
            if isinstance(lp_pass, (float, int)) and isinstance(lp_fail, (float, int)):
                import math
                # Softmax over two items (stable)
                m = max(float(lp_pass), float(lp_fail))
                e1 = math.exp(float(lp_pass) - m)
                e2 = math.exp(float(lp_fail) - m)
                s = e1 + e2 if (e1 + e2) > 0 else 1.0
                p_pass = e1 / s
                p_fail = e2 / s
        if not isinstance(p_pass, (float, int)) or not isinstance(p_fail, (float, int)):
            return 0.0
        p_pass = float(p_pass)
        p_fail = float(p_fail)
        if gt == "pass":
            return max(0.0, min(1.0, p_pass))
        if gt == "fail":
            return max(0.0, min(1.0, p_fail))
        return 0.0
    except Exception:
        return 0.0
