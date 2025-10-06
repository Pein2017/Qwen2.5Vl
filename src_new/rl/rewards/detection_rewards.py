#!/usr/bin/env python3
"""Detection-aware reward components for dense caption RL."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from data_conversion.coordinate_manager import CoordinateManager
from src_new.augmentation.utils import bbox_from_quad_flat
from src_new.processing.parse_generated import parse_geometry_response


try:  # Optional dependency for optimal assignment
    from scipy.optimize import linear_sum_assignment as _lsa  # type: ignore
except Exception:  # pragma: no cover - SciPy optional
    _lsa = None  # type: ignore


def parse_dense_caption(text: str) -> List[Dict[str, object]]:
    """Parse dense-caption output into structured objects."""
    return [dict(obj) for obj in parse_geometry_response(text, tolerant=True)]


def _iter_gt_boxes(
    objects: Iterable[Dict[str, object]],
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for obj in objects:
        if (
            "bbox_2d" in obj
            and isinstance(obj["bbox_2d"], list)
            and len(obj["bbox_2d"]) >= 4
        ):
            x1, y1, x2, y2 = map(int, obj["bbox_2d"][:4])
            boxes.append((x1, y1, x2, y2))
        elif "quad" in obj and isinstance(obj["quad"], list) and len(obj["quad"]) >= 8:
            boxes.append(bbox_from_quad_flat([int(v) for v in obj["quad"][:8]]))
    return boxes


# ------------------- Geometry helpers -------------------


def _bbox_area(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return float(w * h)


def _intersection(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return _bbox_area((ix1, iy1, ix2, iy2))


def _enclosing(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    cx1 = min(ax1, bx1)
    cy1 = min(ay1, by1)
    cx2 = max(ax2, bx2)
    cy2 = max(ay2, by2)
    return (cx1, cy1, cx2, cy2)


def giou_bbox(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> float:
    """Compute Generalized IoU between two axis-aligned boxes.

    Returns a value in [-1, 1].
    """
    area_a = _bbox_area(box_a)
    area_b = _bbox_area(box_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter = _intersection(box_a, box_b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    iou = inter / union
    c_box = _enclosing(box_a, box_b)
    c_area = _bbox_area(c_box)
    if c_area <= 0.0:
        return iou
    giou = iou - (c_area - union) / c_area
    # Keep within valid numeric range
    if giou < -1.0:
        giou = -1.0
    if giou > 1.0:
        giou = 1.0
    return float(giou)


def _normalization_scale(meta: Optional[Dict[str, object]]) -> float:
    """Choose a reasonable normalization scale for L1-like geometry errors."""
    candidates: List[int] = []
    if isinstance(meta, dict):
        w = meta.get("width")
        h = meta.get("height")
        if isinstance(w, (int, float)):
            candidates.append(int(w))
        if isinstance(h, (int, float)):
            candidates.append(int(h))
    scale = max(candidates) if candidates else 1
    return float(max(1, scale))


def _l1_distance(a: List[int], b: List[int]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(sum(abs(int(a[i]) - int(b[i])) for i in range(n))) / float(n)


def _greedy_match(
    preds: List[List[int]], gts: List[List[int]], dist_fn
) -> List[Tuple[int, int]]:
    """Greedy 1-1 matching of predicted and ground-truth lists using a distance."""
    if not preds or not gts:
        return []
    remaining_gts = set(range(len(gts)))
    matches: List[Tuple[int, int]] = []
    for p_idx, p in enumerate(preds):
        # pick nearest available gt
        best = None
        best_d = float("inf")
        for g_idx in list(remaining_gts):
            d = dist_fn(p, gts[g_idx])
            if d < best_d:
                best_d = d
                best = g_idx
        if best is not None:
            remaining_gts.remove(best)
            matches.append((p_idx, best))
        if not remaining_gts:
            break
    return matches


def _hungarian_assign(costs: List[List[float]]) -> Optional[List[Tuple[int, int]]]:
    """Run Hungarian assignment if SciPy is available; otherwise return None."""
    if _lsa is None:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        return None
    cm = np.asarray(costs, dtype=float)
    if cm.size == 0:
        return []
    row_ind, col_ind = _lsa(cm)
    rows = row_ind.tolist()
    cols = col_ind.tolist()
    return list(zip(rows, cols))


def _assign_pairs(
    preds: List[List[int]] | List[Tuple[int, int, int, int]],
    gts: List[List[int]] | List[Tuple[int, int, int, int]],
    cost_fn,
) -> List[Tuple[int, int]]:
    """Assign predictions to ground-truth using Hungarian if possible, else greedy.

    cost_fn takes (pred, gt) and returns a non-negative scalar cost (lower is better).
    """
    if not preds or not gts:
        return []
    # Build cost matrix
    costs: List[List[float]] = []
    for p in preds:
        row: List[float] = []
        for g in gts:
            row.append(float(cost_fn(p, g)))
        costs.append(row)
    # Try Hungarian
    pairs = _hungarian_assign(costs)
    if pairs is not None:
        return pairs

    # Fallback to greedy on precomputed matrix
    def _dist_from_matrix(pi: int, gi: int) -> float:
        return float(costs[pi][gi])

    return _greedy_match(
        list(range(len(preds))),
        list(range(len(gts))),
        lambda pi, gi: _dist_from_matrix(pi, gi),
    )


# ------------------- Rewards -------------------


def reward_coverage(
    text: str, *, meta: Optional[Dict[str, object]] = None, **_: object
) -> float:
    pred = parse_dense_caption(text)
    gt_objects = meta.get("objects", []) if isinstance(meta, dict) else []
    if not isinstance(gt_objects, list):
        gt_objects = []
    target = max(len(gt_objects), 1)
    delta = abs(len(pred) - len(gt_objects))
    return max(0.0, 1.0 - (delta / target))


def reward_geometry_sanity(
    text: str,
    *,
    meta: Optional[Dict[str, object]] = None,
    **_: object,
) -> float:
    pred = parse_dense_caption(text)
    if not pred:
        return 0.0

    limit = None
    if isinstance(meta, dict):
        width = meta.get("width")
        height = meta.get("height")
        numeric = [int(v) for v in (width, height) if isinstance(v, (int, float))]
        if numeric:
            limit = max(numeric)

    def _check(values: List[int]) -> bool:
        if limit is not None:
            for v in values:
                if v < 0 or v > limit:
                    return False
        if len(values) == 4:  # bbox
            x1, y1, x2, y2 = values
            return x1 < x2 and y1 < y2
        return True

    for obj in pred:
        coords = []
        if "bbox_2d" in obj:
            coords = [int(x) for x in obj["bbox_2d"]]
        elif "quad" in obj:
            coords = [int(x) for x in obj["quad"]]
        elif "line" in obj:
            coords = [int(x) for x in obj["line"]]
        if not _check(coords):
            return 0.0
    return 1.0


def reward_bbox_iou(
    text: str,
    *,
    meta: Optional[Dict[str, object]] = None,
    **_: object,
) -> float:
    """Greedy/Hungarian one-to-one matching reward using Generalized IoU for bboxes.

    For quads, compare GIoU on their axis-aligned bounding boxes. GIoU in [-1,1]
    is mapped to [0,1] for stability, and the reward is the mean mapped value over
    matched pairs.
    """
    pred = parse_dense_caption(text)
    pred_boxes = _iter_gt_boxes(pred)
    if not pred_boxes:
        return 0.0

    gt_objects = meta.get("objects", []) if isinstance(meta, dict) else []
    if not isinstance(gt_objects, list) or not gt_objects:
        return 0.0
    gt_boxes = _iter_gt_boxes(gt_objects)
    if not gt_boxes:
        return 0.0

    def _mapped_giou(
        a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
    ) -> float:
        g = giou_bbox(tuple(a), tuple(b))
        m = 0.5 * (g + 1.0)
        if m < 0.0:
            m = 0.0
        if m > 1.0:
            m = 1.0
        return float(m)

    pairs = _assign_pairs(pred_boxes, gt_boxes, lambda a, b: 1.0 - _mapped_giou(a, b))
    if not pairs:
        return 0.0

    scores = [_mapped_giou(pred_boxes[i], gt_boxes[j]) for (i, j) in pairs]
    return float(sum(scores) / len(scores))


# Backward compatible alias and explicit GIoU name
reward_bbox_giou = reward_bbox_iou


def reward_quad_l1(
    text: str,
    *,
    meta: Optional[Dict[str, object]] = None,
    **_: object,
) -> float:
    """L1-based proximity for quad coordinates (8-dim) with 1-1 matching.

    Returns a value in (0, 1], higher is better. If no quads or GT quads, returns 0.0.
    """
    pred_objs = parse_dense_caption(text)
    pred_quads: List[List[int]] = []
    for o in pred_objs:
        if "quad" in o and isinstance(o["quad"], list) and len(o["quad"]) >= 8:
            pred_quads.append([int(v) for v in o["quad"][:8]])
    if not pred_quads:
        return 0.0

    gt_quads: List[List[int]] = []
    if isinstance(meta, dict):
        for o in meta.get("objects", []) or []:
            if (
                isinstance(o, dict)
                and "quad" in o
                and isinstance(o["quad"], list)
                and len(o["quad"]) >= 8
            ):
                gt_quads.append([int(v) for v in o["quad"][:8]])
    if not gt_quads:
        return 0.0

    # Canonicalize order for both predicted and GT
    def _canon(points_flat: List[int]) -> List[int]:
        pts = [(int(points_flat[i]), int(points_flat[i + 1])) for i in range(0, 8, 2)]
        ordered = CoordinateManager._canonical_quad_ordering(pts)
        out: List[int] = []
        for x, y in ordered:
            out.extend([int(x), int(y)])
        return out

    pred_canon = [_canon(q) for q in pred_quads]
    gt_canon = [_canon(q) for q in gt_quads]

    scale = _normalization_scale(meta)

    def _dist(a: List[int], b: List[int]) -> float:
        return _l1_distance(a, b) / scale

    pairs = _assign_pairs(pred_canon, gt_canon, _dist)
    if not pairs:
        return 0.0

    errs = [_dist(pred_canon[i], gt_canon[j]) for (i, j) in pairs]
    mean_err = float(sum(errs) / len(errs))
    return float(1.0 / (1.0 + mean_err))


def reward_line_l1(
    text: str,
    *,
    meta: Optional[Dict[str, object]] = None,
    **_: object,
) -> float:
    """L1-based proximity for line endpoints (4-dim) with 1-1 matching.

    Uses canonical line direction. If no lines or GT lines, returns 0.0.
    """
    pred_objs = parse_dense_caption(text)
    pred_lines: List[List[int]] = []
    for o in pred_objs:
        if (
            "line" in o
            and isinstance(o["line"], list)
            and len(o["line"]) >= 4
            and len(o["line"]) % 2 == 0
        ):
            pred_lines.append([int(v) for v in o["line"]])
    if not pred_lines:
        return 0.0

    gt_lines: List[List[int]] = []
    if isinstance(meta, dict):
        for o in meta.get("objects", []) or []:
            if (
                isinstance(o, dict)
                and "line" in o
                and isinstance(o["line"], list)
                and len(o["line"]) >= 4
                and len(o["line"]) % 2 == 0
            ):
                gt_lines.append([int(v) for v in o["line"]])
    if not gt_lines:
        return 0.0

    # Reduce to endpoints with canonical direction
    def _endpoints(points_flat: List[int]) -> List[int]:
        pts = [
            (int(points_flat[i]), int(points_flat[i + 1]))
            for i in range(0, len(points_flat), 2)
        ]
        ordered = CoordinateManager._canonical_line_ordering(pts)
        a = ordered[0]
        b = ordered[-1]
        return [int(a[0]), int(a[1]), int(b[0]), int(b[1])]

    pred_end = [_endpoints(p) for p in pred_lines]
    gt_end = [_endpoints(g) for g in gt_lines]

    scale = _normalization_scale(meta)

    def _dist(a: List[int], b: List[int]) -> float:
        return _l1_distance(a, b) / scale

    pairs = _assign_pairs(pred_end, gt_end, _dist)
    if not pairs:
        return 0.0

    errs = [_dist(pred_end[i], gt_end[j]) for (i, j) in pairs]
    mean_err = float(sum(errs) / len(errs))
    return float(1.0 / (1.0 + mean_err))


def reward_ordering(
    text: str,
    *,
    meta: Optional[Dict[str, object]] = None,
    **_: object,
) -> float:
    """Ordering reward: quads clockwise from top-left; lines start from leftmost endpoint.

    Returns the mean correctness ratio over all predicted quads and lines. If none exist, returns 1.0.
    """
    objs = parse_dense_caption(text)

    def _quad_ok(q: List[int]) -> bool:
        if len(q) < 8:
            return True
        pts = [(int(q[i]), int(q[i + 1])) for i in range(0, 8, 2)]
        ordered = CoordinateManager._canonical_quad_ordering(pts)
        flat: List[int] = []
        for x, y in ordered:
            flat.extend([int(x), int(y)])
        return flat == [int(v) for v in q[:8]]

    def _line_ok(line: List[int]) -> bool:
        if len(line) < 4 or len(line) % 2 != 0:
            return True
        pts = [(int(line[i]), int(line[i + 1])) for i in range(0, len(line), 2)]
        ordered = CoordinateManager._canonical_line_ordering(pts)
        flat: List[int] = []
        for x, y in ordered:
            flat.extend([int(x), int(y)])
        # Either identical or exact reverse indicates wrong direction; we need identical
        return flat == [int(v) for v in line]

    checks: List[bool] = []
    for o in objs:
        if "quad" in o and isinstance(o["quad"], list) and len(o["quad"]) >= 8:
            checks.append(_quad_ok([int(v) for v in o["quad"]]))
        if (
            "line" in o
            and isinstance(o["line"], list)
            and len(o["line"]) >= 4
            and len(o["line"]) % 2 == 0
        ):
            checks.append(_line_ok([int(v) for v in o["line"]]))

    if not checks:
        return 1.0
    return float(sum(1.0 for c in checks if c) / float(len(checks)))


__all__ = [
    "parse_dense_caption",
    "reward_coverage",
    "reward_geometry_sanity",
    "reward_bbox_iou",
    "reward_bbox_giou",
    "reward_quad_l1",
    "reward_line_l1",
    "reward_ordering",
]
