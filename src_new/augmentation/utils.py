from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from PIL import Image


@dataclass(frozen=True)
class CanvasTransform:
    """Describes a rotation about the original image center and an optional translation.

    angle_rad: rotation angle in radians
    translate_xy: translation to apply after rotation (e.g., when expanding canvas)
    out_size_wh: (width, height) of the output canvas
    """

    angle_rad: float
    translate_xy: Tuple[float, float]
    out_size_wh: Tuple[int, int]


def deg2rad(angle_deg: float) -> float:
    return angle_deg * math.pi / 180.0


def rotate_point_about_center(
    x: float, y: float, cx: float, cy: float, angle_rad: float
) -> Tuple[float, float]:
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)
    dx = x - cx
    dy = y - cy
    rx = cos_t * dx - sin_t * dy
    ry = sin_t * dx + cos_t * dy
    return rx + cx, ry + cy


def compute_expanded_canvas_and_offset(
    width: int, height: int, angle_rad: float
) -> Tuple[int, int, float, float]:
    """Compute expanded canvas size and translation offset after rotation.

    Uses floor/ceil bounds to better align with PIL.rotate(expand=True) behavior.
    Returns (out_w, out_h, offset_x, offset_y) where offset should be added post rotation.
    """
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    corners = (
        (0.0, 0.0),
        (width - 1.0, 0.0),
        (width - 1.0, height - 1.0),
        (0.0, height - 1.0),
    )
    rotated = [rotate_point_about_center(x, y, cx, cy, angle_rad) for (x, y) in corners]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    min_x = math.floor(min(xs))
    min_y = math.floor(min(ys))
    max_x = math.ceil(max(xs))
    max_y = math.ceil(max(ys))
    out_w = int(max_x - min_x + 1)
    out_h = int(max_y - min_y + 1)
    offset_x = -float(min_x)
    offset_y = -float(min_y)
    return out_w, out_h, offset_x, offset_y


def pil_interpolation(mode: str) -> int:
    mode_lower = mode.lower()
    if mode_lower == "nearest":
        return Image.NEAREST
    if mode_lower == "bilinear":
        return Image.BILINEAR
    if mode_lower == "bicubic":
        return Image.BICUBIC
    raise ValueError(f"Unsupported interpolation mode: {mode}")


def round_and_clamp_points(
    points_xy: Sequence[Tuple[float, float]], out_w: int, out_h: int
) -> List[Tuple[int, int]]:
    rounded: List[Tuple[int, int]] = []
    for x, y in points_xy:
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(out_w - 1, xi))
        yi = max(0, min(out_h - 1, yi))
        rounded.append((xi, yi))
    return rounded
