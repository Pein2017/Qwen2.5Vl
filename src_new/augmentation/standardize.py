from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


GeometryObject = Dict[str, Any]


def bbox_to_quad(bbox: Sequence[int]) -> List[int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox_to_quad: expected 4 ints, got {bbox}")
    x1, y1, x2, y2 = bbox
    if not (x1 < x2 and y1 < y2):
        # Enforce bbox canonical min/max quickly
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    # Clockwise from top-left
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _top_left_index(points: List[Tuple[int, int]]) -> int:
    # pick smallest y, then smallest x among ties
    best_idx = 0
    best = points[0]
    for i, p in enumerate(points):
        if (p[1] < best[1]) or (p[1] == best[1] and p[0] < best[0]):
            best_idx = i
            best = p
    return best_idx


def canonicalize_quad(points_xy: List[Tuple[int, int]]) -> List[int]:
    """Return quad as [x1,y1, x2,y2, x3,y3, x4,y4] in clockwise order from top-left.

    Robust to arbitrary input vertex ordering; uses centroid-based corner classification
    with a row-wise fallback to avoid self-crossing or rounding-induced duplicates.
    """
    if len(points_xy) != 4:
        raise ValueError(f"canonicalize_quad: expected 4 points, got {len(points_xy)}")

    pts = [(int(x), int(y)) for (x, y) in points_xy]
    # Centroid
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    def classify_corner(p: Tuple[int, int]) -> Tuple[int, float]:
        x, y = float(p[0]), float(p[1])
        # Corner id and tie-break score (mirrors vis_generation style)
        if x <= cx and y <= cy:  # top-left
            return (0, -(x + y))  # minimize x+y
        if x >= cx and y <= cy:  # top-right
            return (1, x - y)  # maximize x-y
        if x >= cx and y >= cy:  # bottom-right
            return (2, x + y)  # maximize x+y
        # bottom-left
        return (3, -x + y)  # maximize -x+y

    sorted_points = sorted(pts, key=classify_corner)

    # Ensure distinct-ish corners; if not, fallback to row-wise ordering
    corner_ids = [classify_corner(p)[0] for p in sorted_points]
    if len(set(corner_ids)) != 4:
        by_y = sorted(pts, key=lambda p: p[1])
        top = sorted(by_y[:2], key=lambda p: p[0])
        bottom = sorted(by_y[2:], key=lambda p: p[0])
        ordered = [top[0], top[1], bottom[1], bottom[0]]  # TL, TR, BR, BL
    else:
        # Map to strict TL, TR, BR, BL order by corner id
        buckets: Dict[int, Tuple[int, int]] = {}
        for p in sorted_points:
            cid, _ = classify_corner(p)
            if cid not in buckets:
                buckets[cid] = p
        ordered = [buckets[0], buckets[1], buckets[2], buckets[3]]

    flat: List[int] = []
    for x, y in ordered:
        flat.extend([int(x), int(y)])
    return flat


def sort_objects_top_bottom_left_right(
    objects: List[GeometryObject],
) -> List[GeometryObject]:
    def first_point(obj: GeometryObject) -> Tuple[int, int]:
        if "quad" in obj:
            q = obj["quad"]
            return (q[0], q[1])
        if "bbox_2d" in obj:
            b = obj["bbox_2d"]
            return (b[0], b[1])
        if "line" in obj and len(obj["line"]) >= 2:
            l = obj["line"]
            return (l[0], l[1])
        return (0, 0)

    return sorted(objects, key=lambda o: (first_point(o)[1], first_point(o)[0]))


def validate_sample_after_transform(
    sample: Dict[str, Any], width: int, height: int
) -> None:
    if "objects" not in sample or not sample["objects"]:
        raise ValueError("Augmentation produced empty objects list")
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid canvas size after rotation: width={width}, height={height}"
        )
    for obj in sample["objects"]:
        if "desc" not in obj:
            raise ValueError("Object missing 'desc' after augmentation")
        # Geometry keys
        keys = [k for k in ("bbox_2d", "quad", "line") if k in obj]
        if len(keys) != 1:
            raise ValueError(f"Object must have exactly one geometry key, got {keys}")
        g = keys[0]
        coords = obj[g]
        if g == "bbox_2d":
            if len(coords) != 4:
                raise ValueError(f"bbox_2d must have 4 ints, got {coords}")
        elif g == "quad":
            if len(coords) != 8:
                raise ValueError(f"quad must have 8 ints, got {coords}")
        elif g == "line":
            if len(coords) < 4 or (len(coords) % 2) != 0:
                raise ValueError(f"line must have even length >= 4, got {len(coords)}")
        # Bounds check
        for i in range(0, len(coords), 2):
            x = coords[i]
            y = coords[i + 1]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"Coordinate out of bounds after rotation: (x={x}, y={y}) not in [0..{width - 1}]x[0..{height - 1}]"
                )
