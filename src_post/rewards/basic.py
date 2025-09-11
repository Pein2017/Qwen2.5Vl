#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any


def exact_label_match_reward(sample: Dict[str, Any]) -> float:
    """Binary reward: 1.0 if pred_label == gt_label else 0.0.

    Expects sample to contain keys: gt_label (str), pred_label (str).
    """
    gt = str(sample.get("gt_label", "")).strip().lower()
    pred = str(sample.get("pred_label", "")).strip().lower()
    return 1.0 if gt and pred and gt == pred else 0.0
