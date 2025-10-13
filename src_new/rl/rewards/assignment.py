#!/usr/bin/env python3
"""Assignment-based reward for dense captioning via Hungarian matching.

Combines geometry and caption similarity into a single per-completion reward:

  cost = (1 - geometry_score) + lambda_caption * (1 - caption_score)

Reward aggregates matched quality and penalizes FP/FN:

  reward = clamp01(alpha * mean_matched_score - beta_fp * fp_rate - beta_fn * fn_rate)

Geometry scoring mirrors existing rewards:
  - bbox/quad: mapped GIoU in [0,1]
  - line: buffered line GIoU in [0,1] when Shapely available; fallback to endpoint L1 → similarity

Caption scoring uses token-level F1 on the object "desc".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from data_conversion.coordinate_manager import CoordinateManager
from src_new.rl.rewards import detection_rewards as dr


def _mapped(val: float) -> float:
    """Clamp to [0,1]."""
    v = float(val)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _geom_score_bbox(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    # mapped GIoU ∈ [0,1]
    g = dr.giou_bbox(tuple(a), tuple(b))
    return _mapped(0.5 * (g + 1.0))


def _geom_score_quad(a_flat: List[int], b_flat: List[int]) -> float:
    # Canonicalize order, then polygon GIoU mapped to [0,1]
    a_pts = [(int(a_flat[i]), int(a_flat[i + 1])) for i in range(0, 8, 2)]
    b_pts = [(int(b_flat[i]), int(b_flat[i + 1])) for i in range(0, 8, 2)]
    a_ord = CoordinateManager._canonical_quad_ordering(a_pts)
    b_ord = CoordinateManager._canonical_quad_ordering(b_pts)
    g = dr._giou_polygons_from_points(a_ord, b_ord)
    return _mapped(0.5 * (g + 1.0))


def _geom_score_line(
    a_flat: List[int], b_flat: List[int], *, buffer_width: float, scale: float
) -> float:
    # Buffered line GIoU is required; raise if invalid
    g = dr._giou_buffered_lines(a_flat, b_flat, float(buffer_width))
    return _mapped(0.5 * (g + 1.0))


def _build_objects(text: str, meta: Optional[Dict[str, object]]):
    pred = dr.parse_dense_caption(text)
    gt = meta.get("objects", []) if isinstance(meta, dict) else []
    if not isinstance(gt, list):
        gt = []
    return pred, gt


def _geometry_of(obj: Dict[str, object]) -> str:
    if "bbox_2d" in obj:
        return "bbox_2d"
    if "quad" in obj:
        return "quad"
    if "line" in obj:
        return "line"
    return "none"


def _build_cost_matrix(
    pred: List[Dict[str, object]],
    gt: List[Dict[str, object]],
    *,
    lambda_caption: float,
    line_buffer_frac: float,
    meta: Optional[Dict[str, object]],
) -> List[List[float]]:
    # Precompute normalization/line buffer width
    scale = dr._normalization_scale(meta)
    buffer_width = max(1.0, float(line_buffer_frac) * float(scale))

    costs: List[List[float]] = []
    for p in pred:
        prow: List[float] = []
        p_geom = _geometry_of(p)
        for g in gt:
            g_geom = _geometry_of(g)
            if p_geom != g_geom:
                # High cost for mismatched geometry types
                prow.append(1.0 + float(lambda_caption))
                continue
            # Geometry score in [0,1]
            if p_geom == "bbox_2d":
                pa = tuple(int(v) for v in p["bbox_2d"])  # type: ignore[index]
                ga = tuple(int(v) for v in g["bbox_2d"])  # type: ignore[index]
                geom_s = _geom_score_bbox(pa, ga)
            elif p_geom == "quad":
                pa = [int(v) for v in p["quad"]][:8]  # type: ignore[index]
                ga = [int(v) for v in g["quad"]][:8]  # type: ignore[index]
                geom_s = _geom_score_quad(pa, ga)
            elif p_geom == "line":
                pa = [int(v) for v in p["line"]]  # type: ignore[index]
                ga = [int(v) for v in g["line"]]  # type: ignore[index]
                geom_s = _geom_score_line(
                    pa, ga, buffer_width=buffer_width, scale=scale
                )
            else:
                geom_s = 0.0

            # Caption score
            pred_desc = p.get("desc", "")
            gt_desc = g.get("desc", "")
            if not isinstance(pred_desc, str):
                pred_desc = ""
            if not isinstance(gt_desc, str):
                gt_desc = ""
            cap_s = dr._desc_f1(pred_desc, gt_desc)

            # Combine into cost
            cost = (1.0 - float(geom_s)) + float(lambda_caption) * (1.0 - float(cap_s))
            # Clamp to non-negative
            if cost < 0.0:
                cost = 0.0
            prow.append(float(cost))
        costs.append(prow)
    return costs


def _hungarian(costs: List[List[float]]) -> List[Tuple[int, int]]:
    # Try SciPy if available; otherwise greedy fallback
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import linear_sum_assignment as _lsa  # type: ignore

        cm = np.asarray(costs, dtype=float)
        if cm.size == 0:
            return []
        rows, cols = _lsa(cm)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        # Greedy fallback: O(n^2)
        pairs: List[Tuple[int, int]] = []
        if not costs:
            return pairs
        used_cols: set[int] = set()
        for r, row in enumerate(costs):
            best_c = None
            best_v = float("inf")
            for c, v in enumerate(row):
                if c in used_cols:
                    continue
                if v < best_v:
                    best_v = float(v)
                    best_c = c
            if best_c is not None:
                used_cols.add(best_c)
                pairs.append((r, best_c))
        return pairs


def assignment_f1(
    text: str,
    *,
    meta: Optional[Dict[str, object]] = None,
    lambda_caption: float = 0.3,
    alpha: float = 1.0,
    beta_fp: float = 0.5,
    beta_fn: float = 0.5,
    line_buffer_frac: float = 0.01,
    **_: object,
) -> float:
    """Compute assignment-based reward combining geometry and caption.

    Returns a scalar in [0,1].
    """
    pred, gt = _build_objects(text, meta)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        # All FP or all FN → penalize via rates below; zero quality
        mean_quality = 0.0
        n_pred = len(pred)
        n_gt = len(gt)
        fp_rate = 1.0 if n_pred > 0 else 0.0
        fn_rate = 1.0 if n_gt > 0 else 0.0
        reward = (
            float(alpha) * float(mean_quality)
            - float(beta_fp) * fp_rate
            - float(beta_fn) * fn_rate
        )
        return _mapped(reward)

    costs = _build_cost_matrix(
        pred,
        gt,
        lambda_caption=lambda_caption,
        line_buffer_frac=line_buffer_frac,
        meta=meta,
    )
    pairs = _hungarian(costs)

    matched_scores: List[float] = []
    used_pred = set(i for i, _ in pairs)
    used_gt = set(j for _, j in pairs)
    # Recompute combined score for matched pairs as (1 - normalized cost)
    for i, j in pairs:
        # Normalize: cost = a + b*lambda_caption, max when geom=0, cap=0
        denom = 1.0 + float(lambda_caption)
        raw_cost = float(costs[i][j])
        s = 1.0 - max(0.0, min(1.0, raw_cost / denom))
        matched_scores.append(_mapped(s))

    mean_quality = (
        float(sum(matched_scores) / len(matched_scores)) if matched_scores else 0.0
    )

    # FP/FN rates
    n_pred = len(pred)
    n_gt = len(gt)
    fp = max(0, n_pred - len(used_pred))
    fn = max(0, n_gt - len(used_gt))
    fp_rate = float(fp) / float(max(n_pred, 1))
    fn_rate = float(fn) / float(max(n_gt, 1))

    reward = (
        float(alpha) * float(mean_quality)
        - float(beta_fp) * fp_rate
        - float(beta_fn) * fn_rate
    )
    return _mapped(reward)


__all__ = ["assignment_f1"]
