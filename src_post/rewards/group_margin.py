#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any, Dict


def group_margin(sample: Dict[str, Any]) -> float:
    """Return TF-based decision margin: log(p_pass+eps) - log(p_fail+eps).

    Requires inputs to contain:
      - tf_p_pass: float in (0,1)
      - tf_p_fail: float in (0,1)

    Raises:
      ValueError with actionable guidance when inputs are missing/invalid.
    """
    eps = 1e-9
    try:
        p_pass = sample.get("tf_p_pass", None)
        p_fail = sample.get("tf_p_fail", None)
        if not isinstance(p_pass, (float, int)) or not isinstance(p_fail, (float, int)):
            raise ValueError(
                "group_margin: missing/invalid TF probabilities; ensure tf_decision_probs is called before reward composition"
            )
        p_pass_f = float(p_pass)
        p_fail_f = float(p_fail)
        if not (0.0 <= p_pass_f <= 1.0 and 0.0 <= p_fail_f <= 1.0):
            raise ValueError(
                "group_margin: TF probabilities must be in [0,1]; got tf_p_pass={} tf_p_fail={}".format(
                    p_pass_f, p_fail_f
                )
            )
        # Stable margin in log-space; eps guards log(0)
        return float(math.log(p_pass_f + eps) - math.log(p_fail_f + eps))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            "group_margin: missing/invalid TF probabilities; ensure tf_decision_probs is called before reward composition"
        ) from e
