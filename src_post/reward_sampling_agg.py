#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Any, Dict


@dataclass
class SummaryCandidate:
    text: str
    token_ids: List[int]
    step_logprobs: List[float]


def compute_reward(gt_label: str, pred_label: Optional[str]) -> int:
    return int(isinstance(pred_label, str) and gt_label.strip().lower() == pred_label.strip().lower())


# The following helpers are thin utilities intended to be used by GRPO runner.
# Actual generation/encoding occurs in the runner to keep device/processor context.

def pack_stage_b_reward_inputs(
    gt_label: str,
    pred_label: Optional[str],
    summary_lines: List[str],
    reason: Optional[str],
    checklist_lines: List[str],
) -> Dict[str, Any]:
    return {
        "gt_label": str(gt_label).strip().lower(),
        "pred_label": str(pred_label) if pred_label is not None else None,
        "summary_lines": list(summary_lines),
        "stage_b_reason": reason,
        "checklist_lines": list(checklist_lines),
    }
