from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from PIL import Image

from src_new_json.augmentation.standardize import (
    canonicalize_quad,
    validate_sample_after_transform,
)
from src_new_json.augmentation.utils import (
    CanvasTransform,
    compute_expanded_canvas_and_offset,
    deg2rad,
    rotate_point_about_center,
    round_and_clamp_points,
)
from src_new_json.config.augmentation_config import ImageGeomConfig


def _build_canvas_transform(
    width: int, height: int, angle_deg: float
) -> CanvasTransform:
    angle_rad = deg2rad(angle_deg)
    out_w, out_h, off_x, off_y = compute_expanded_canvas_and_offset(
        width, height, angle_rad
    )
    return CanvasTransform(
        angle_rad=angle_rad, translate_xy=(off_x, off_y), out_size_wh=(out_w, out_h)
    )


def _rotate_image(img: Image.Image, angle_deg: float) -> Image.Image:
    return img.rotate(angle=-angle_deg, resample=Image.BILINEAR, expand=True)


def _rotate_point_list(
    xy: List[Tuple[int, int]], width: int, height: int, t: CanvasTransform
) -> List[Tuple[int, int]]:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    rotated: List[Tuple[float, float]] = []
    for x, y in xy:
        rx, ry = rotate_point_about_center(float(x), float(y), cx, cy, t.angle_rad)
        rx += t.translate_xy[0]
        ry += t.translate_xy[1]
        rotated.append((rx, ry))
    return round_and_clamp_points(rotated, t.out_size_wh[0], t.out_size_wh[1])


def apply_image_geom(
    images: List[Image.Image],
    sample: Dict[str, Any],
    cfg: ImageGeomConfig,
    rng: random.Random,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    width = int(sample["width"])  # fail-fast
    height = int(sample["height"])  # fail-fast
    # Rotation only v1; TODO: translate_pct, scale_range, perspective_pct, crop_pct, multiscale
    angle_deg = rng.uniform(
        float(cfg.rotate_deg_range[0]), float(cfg.rotate_deg_range[1])
    )

    transform = _build_canvas_transform(width, height, angle_deg)
    out_w, out_h = transform.out_size_wh
    rotated_images = [_rotate_image(img, angle_deg) for img in images]

    new_objects: List[Dict[str, Any]] = []
    for obj in sample["objects"]:
        if "line" in obj:
            l = obj["line"]
            pts = [(int(l[i]), int(l[i + 1])) for i in range(0, len(l), 2)]
            rot = _rotate_point_list(pts, width, height, transform)
            flat: List[int] = []
            for x, y in rot:
                flat.extend([x, y])
            new_objects.append({"line": flat, "desc": obj["desc"]})
            continue
        if "bbox_2d" in obj:
            b = obj["bbox_2d"]
            x1, y1, x2, y2 = b
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            rot = _rotate_point_list(corners, width, height, transform)
            canon_quad = canonicalize_quad(rot)
            new_objects.append({"quad": canon_quad, "desc": obj["desc"]})
            continue
        if "quad" in obj:
            q = obj["quad"]
            pts = [
                (int(q[0]), int(q[1])),
                (int(q[2]), int(q[3])),
                (int(q[4]), int(q[5])),
                (int(q[6]), int(q[7])),
            ]
            rot = _rotate_point_list(pts, width, height, transform)
            canon_quad = canonicalize_quad(rot)
            new_objects.append({"quad": canon_quad, "desc": obj["desc"]})
            continue
        raise ValueError(f"Unsupported geometry in object: keys={list(obj.keys())}")

    new_sample = dict(sample)
    new_sample["objects"] = new_objects
    new_sample["width"] = out_w
    new_sample["height"] = out_h

    validate_sample_after_transform(new_sample, out_w, out_h)
    return rotated_images, new_sample
