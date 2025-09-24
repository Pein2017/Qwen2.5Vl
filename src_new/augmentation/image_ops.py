from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from PIL import Image

from src_new.augmentation.standardize import (
    canonicalize_quad,
    validate_sample_after_transform,
)
from src_new.augmentation.utils import (
    CanvasTransform,
    compute_expanded_canvas_and_offset,
    deg2rad,
    rotate_point_about_center,
    round_and_clamp_points,
    smart_resize_dimensions,
)
from src_new.config.augmentation_config import ImageGeomConfig, SmartResizeConfig


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


def _scale_bbox(
    bbox: List[int], scale_x: float, scale_y: float, out_w: int, out_h: int
) -> List[int]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got {bbox}")
    max_x = max(out_w - 1, 0)
    max_y = max(out_h - 1, 0)
    x1 = int(round(bbox[0] * scale_x))
    y1 = int(round(bbox[1] * scale_y))
    x2 = int(round(bbox[2] * scale_x))
    y2 = int(round(bbox[3] * scale_y))
    x1 = max(0, min(max_x, x1))
    y1 = max(0, min(max_y, y1))
    x2 = max(0, min(max_x, x2))
    y2 = max(0, min(max_y, y2))
    if x1 >= x2:
        if x1 >= max_x:
            x1 = max(0, max_x - 1)
            x2 = max_x
        else:
            x2 = min(max_x, x1 + 1)
    if y1 >= y2:
        if y1 >= max_y:
            y1 = max(0, max_y - 1)
            y2 = max_y
        else:
            y2 = min(max_y, y1 + 1)
    return [x1, y1, x2, y2]


def _scale_points(
    points: List[Tuple[int, int]], scale_x: float, scale_y: float, out_w: int, out_h: int
) -> List[Tuple[int, int]]:
    scaled = [(p[0] * scale_x, p[1] * scale_y) for p in points]
    return round_and_clamp_points(scaled, out_w, out_h)


def apply_smart_resize(
    images: List[Image.Image],
    sample: Dict[str, Any],
    cfg: SmartResizeConfig,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    if not images or not cfg.enabled:
        return images, sample

    if "width" not in sample or "height" not in sample:
        raise ValueError("Sample missing required 'width'/'height' keys for smart resize")

    width = int(sample["width"])
    height = int(sample["height"])
    target_w, target_h = smart_resize_dimensions(
        width=width,
        height=height,
        factor=int(cfg.factor),
        min_pixels=int(cfg.min_pixels),
        max_pixels=int(cfg.max_pixels),
        max_ratio=float(cfg.max_ratio),
    )

    if target_w == width and target_h == height:
        return images, sample

    scale_x = float(target_w) / float(width)
    scale_y = float(target_h) / float(height)

    resized_images = [
        img.resize((target_w, target_h), resample=Image.BICUBIC)
        for img in images
    ]

    objects = sample.get("objects", [])
    new_objects: List[Dict[str, Any]] = []
    for obj in objects:
        new_obj = dict(obj)
        if "bbox_2d" in obj:
            bbox = [int(v) for v in obj["bbox_2d"]]
            new_obj["bbox_2d"] = _scale_bbox(bbox, scale_x, scale_y, target_w, target_h)
        elif "quad" in obj:
            q = obj["quad"]
            pts = [
                (int(q[0]), int(q[1])),
                (int(q[2]), int(q[3])),
                (int(q[4]), int(q[5])),
                (int(q[6]), int(q[7])),
            ]
            scaled_pts = _scale_points(pts, scale_x, scale_y, target_w, target_h)
            flat: List[int] = []
            for x, y in scaled_pts:
                flat.extend([x, y])
            new_obj["quad"] = flat
        elif "line" in obj:
            line = obj["line"]
            pts = [(int(line[i]), int(line[i + 1])) for i in range(0, len(line), 2)]
            scaled_pts = _scale_points(pts, scale_x, scale_y, target_w, target_h)
            flat: List[int] = []
            for x, y in scaled_pts:
                flat.extend([x, y])
            new_obj["line"] = flat
        else:
            raise ValueError(f"Unsupported geometry in object: keys={list(obj.keys())}")
        new_objects.append(new_obj)

    new_sample = dict(sample)
    new_sample["objects"] = new_objects
    new_sample["width"] = target_w
    new_sample["height"] = target_h

    if new_objects:
        validate_sample_after_transform(new_sample, target_w, target_h)

    return resized_images, new_sample


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
