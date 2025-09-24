#!/usr/bin/env python3
"""
Parity check utility to compare RL vs SFT pre-generation inputs.

- Verifies that text tensors have equal shapes
- Checks that input_ids are identical (full equality) and reports first mismatch
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _tensor_info(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}" if torch.is_tensor(t) else str(type(t))


def parity_check(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Tuple[bool, str]:
    """
    Compare two pre-generation input dicts for parity.

    Returns (ok, message). On failure, message explains the first mismatch.
    Required keys: input_ids, attention_mask
    """
    required = ("input_ids", "attention_mask")
    for key in required:
        if key not in a or key not in b:
            return False, f"missing required key '{key}' in inputs"
        ta, tb = a[key], b[key]
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            return False, f"{key} must be tensors: a={_tensor_info(ta)}, b={_tensor_info(tb)}"
        if ta.shape != tb.shape:
            return False, f"{key} shape mismatch: a={_tensor_info(ta)}, b={_tensor_info(tb)}"

    # Normalize to 2D [B, S]
    a_ids = a["input_ids"].unsqueeze(0) if a["input_ids"].dim() == 1 else a["input_ids"]
    b_ids = b["input_ids"].unsqueeze(0) if b["input_ids"].dim() == 1 else b["input_ids"]

    # Full equality check on input_ids
    diff = (a_ids != b_ids)
    if diff.any():
        # Find first mismatch index for diagnostics
        mismatch = torch.nonzero(diff, as_tuple=False)[0]
        b_idx, s_idx = int(mismatch[0].item()), int(mismatch[1].item()) if mismatch.numel() >= 2 else (0, 0)
        return False, f"input_ids differ at batch={b_idx}, pos={s_idx}: a={int(a_ids[b_idx, s_idx].item())}, b={int(b_ids[b_idx, s_idx].item())}"

    return True, "ok"


__all__ = ["parity_check"]
