#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Set

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


def compute_phase_a_diagnostics(
    summary_lines: List[str], checklist_lines: List[str], mission: Optional[str] = None
) -> Dict[str, float]:
    """Compute Stage‑A diagnostics in [0,1].

    With the slimmed reward set, we report coverage only to reflect mission token hits.
    """
    inputs = {
        "summary_lines": list(summary_lines or []),
        "checklist_lines": list(checklist_lines or []),
        "stage_b_reason": "",
        # Other fields are unused by coverage
        "gt_label": "",
        "pred_label": None,
        "tf_p_pass": 0.0,
        "tf_p_fail": 0.0,
        "mission": (str(mission) if mission is not None else None),
    }
    out: Dict[str, float] = {}
    fn = REGISTRY.get("coverage")
    if fn is None:
        out["coverage"] = 0.0
        return out
    try:
        out["coverage"] = _safe_clip_01(float(fn(inputs)))
    except Exception:
        out["coverage"] = 0.0
    return out


# ---- New: item-level diagnostics (per-image and group-level) ----
from src_post.prompting.schema import get_mission_token_sets_by_outcome
from src_post.rewards.lexicon import NEGATIVE_TOKENS


def build_item_level_diagnostics(
    summary_lines: List[str], mission: Optional[str]
) -> Dict[str, any]:
    """Analyze Stage‑A summaries at item/token level.

    Returns a dict with:
      - per_image: list of {present_pass_tokens, present_fail_tokens}
      - group_missing_pass_tokens: sorted list of pass tokens not covered by any image
      - group_fail_tokens: sorted list of fail/negative tokens covered by any image

    Notes:
      - pass/fail token sets are assembled from mission table (pass_tokens, fail_tokens) and
        merged with a canonical negative lexicon for fail side.
      - This function is stateless and safe to call in logging or lightweight diagnostics.
    """
    lines = [str(x or "") for x in (summary_lines or [])]
    pass_tokens: Set[str]
    fail_tokens_raw: Set[str]
    pass_tokens, fail_tokens_raw = get_mission_token_sets_by_outcome(mission)
    fail_tokens = set(fail_tokens_raw) | set(NEGATIVE_TOKENS)

    per_image: List[Dict[str, List[str]]] = []
    group_pass_hits: Set[str] = set()
    group_fail_hits: Set[str] = set()

    for text in lines:
        hits_p = sorted({tok for tok in pass_tokens if tok and tok in text})
        hits_f = sorted({tok for tok in fail_tokens if tok and tok in text})
        group_pass_hits |= set(hits_p)
        group_fail_hits |= set(hits_f)
        per_image.append(
            {
                "present_pass_tokens": hits_p,
                "present_fail_tokens": hits_f,
            }
        )

    missing_pass = sorted([t for t in pass_tokens if t not in group_pass_hits])
    group_fail = sorted(list(group_fail_hits))

    return {
        "per_image": per_image,
        "group_missing_pass_tokens": missing_pass,
        "group_fail_tokens": group_fail,
    }


def classify_mismatch(
    gt_label: str, pred_label: Optional[str], diag: Dict[str, any]
) -> str:
    """Classify mismatch root-cause using item-level signals.

    Returns one of: "一致", "少识别了", "多识别了", "识别错误了".
    """
    g = str(gt_label or "").strip().lower()
    p = (str(pred_label or "").strip().lower()) if pred_label is not None else None
    if not p or g == p:
        return "一致"
    miss = set(diag.get("group_missing_pass_tokens", []) or [])
    negs = set(diag.get("group_fail_tokens", []) or [])
    if g == "pass" and p == "fail":
        if len(negs) > 0:
            return "多识别了"
        return "少识别了"
    if g == "fail" and p == "pass":
        if len(negs) == 0 and len(miss) > 0:
            return "少识别了"
        return "识别错误了"
    return "识别错误了"
