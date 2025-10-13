#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics adapter to project structured diagnostics into flat logging keys.

- Accepts diagnostics with 'group_losses' structured as:
  diagnostics['group_losses'] = {
    'teacher': {'caption': T, 'grounding': T, 'formatting': T},
    'student': {'caption': S, 'grounding': S, 'formatting': S},
  }
- Emits flat keys like:
  'eval/teacher_caption_loss', 'eval/student_grounding_loss', etc.
- Also emits combined totals: 'eval/group_caption_loss_total', etc.
"""
from typing import Dict

import torch

from .metrics_registry import ROLE_NAMES, compute_group_totals_from_flat


def adapt_group_losses(prefix: str, diagnostics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not diagnostics or "group_losses" not in diagnostics:
        return out
    gl = diagnostics.get("group_losses")
    if not isinstance(gl, dict):
        return out
    # Per-role flat
    for role in ROLE_NAMES:
        role_dict = gl.get(role)
        if isinstance(role_dict, dict):
            for name, tensor_val in role_dict.items():
                key = f"{prefix}/{role}_{name}_loss"
                try:
                    out[key] = float(tensor_val.detach().float().item())
                except Exception:
                    pass
    # Combined totals via shared helper
    # Build a transient flat map to reuse the central utility
    flat: Dict[str, float] = {}
    for role in ROLE_NAMES:
        role_dict = gl.get(role)
        if isinstance(role_dict, dict):
            for name, tensor_val in role_dict.items():
                try:
                    flat[f"{role}_{name}_loss"] = float(tensor_val.detach().float().item())
                except Exception:
                    pass
    out.update(compute_group_totals_from_flat(flat, prefix=prefix))
    return out
