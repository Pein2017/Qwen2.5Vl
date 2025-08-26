from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from src_new.augmentation.standardize import (
    canonicalize_quad,
    validate_sample_after_transform,
)
from src_new.augmentation.utils import (
    CanvasTransform,
    compute_expanded_canvas_and_offset,
    deg2rad,
    pil_interpolation,
    rotate_point_about_center,
    round_and_clamp_points,
)
from src_new.config.augmentation_config import (
    AlbumentationsRandAugConfig,
    AngleRotateConfig,
    AugmentationConfig,
    ObjectLocalAffineConfig,
    validate_angle_rotate_config,
    validate_augmentation_config,
)


try:
    import albumentations as A
except Exception:  # pragma: no cover
    A = None


@dataclass(frozen=True)
class AngleRotateOp:
    config: AngleRotateConfig

    def _sample_angle_deg(self, rng: random.Random) -> float:
        mode = self.config.sample_mode
        if mode == "fixed":
            assert self.config.fixed_angle_deg is not None
            return float(self.config.fixed_angle_deg)
        if mode == "uniform_range":
            assert (
                self.config.angle_min_deg is not None
                and self.config.angle_max_deg is not None
            )
            return rng.uniform(
                float(self.config.angle_min_deg), float(self.config.angle_max_deg)
            )
        if mode == "set":
            assert (
                self.config.angles_set_deg is not None
                and len(self.config.angles_set_deg) > 0
            )
            return float(rng.choice(self.config.angles_set_deg))
        raise ValueError(f"Unsupported sample_mode: {mode}")

    def _build_canvas_transform(
        self, width: int, height: int, angle_deg: float
    ) -> CanvasTransform:
        angle_rad = deg2rad(angle_deg)
        if self.config.expand:
            out_w, out_h, off_x, off_y = compute_expanded_canvas_and_offset(
                width, height, angle_rad
            )
            return CanvasTransform(
                angle_rad=angle_rad,
                translate_xy=(off_x, off_y),
                out_size_wh=(out_w, out_h),
            )
        return CanvasTransform(
            angle_rad=angle_rad, translate_xy=(0.0, 0.0), out_size_wh=(width, height)
        )

    def _rotate_image(self, img: Image.Image, angle_deg: float) -> Image.Image:
        interp = pil_interpolation(self.config.interpolation)
        if self.config.expand:
            return img.rotate(
                angle=-angle_deg,
                resample=interp,
                expand=True,
                fillcolor=self.config.fill_color,
            )
        return img.rotate(
            angle=-angle_deg,
            resample=interp,
            expand=False,
            fillcolor=self.config.fill_color,
        )

    def _rotate_point_list(
        self, xy: List[Tuple[int, int]], width: int, height: int, t: CanvasTransform
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

    def apply(
        self, sample: Dict[str, Any], images: List[Image.Image], rng: random.Random
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        width = int(sample["width"])  # fail-fast on missing key
        height = int(sample["height"])  # fail-fast on missing key
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid sample width/height: {width}x{height}")

        angle_deg = self._sample_angle_deg(rng)
        is_identity = abs(angle_deg) < 1e-6

        transform = self._build_canvas_transform(width, height, angle_deg)
        out_w, out_h = transform.out_size_wh
        rotated_images = [self._rotate_image(img, angle_deg) for img in images]

        new_objects: List[Dict[str, Any]] = []
        if is_identity:
            # Preserve types; canonicalize and clamp quads to satisfy validation
            for obj in sample["objects"]:
                if "bbox_2d" in obj:
                    new_objects.append(obj)
                elif "quad" in obj:
                    q = obj["quad"]
                    pts = [
                        (int(q[0]), int(q[1])),
                        (int(q[2]), int(q[3])),
                        (int(q[4]), int(q[5])),
                        (int(q[6]), int(q[7])),
                    ]
                    clamped = round_and_clamp_points(pts, width, height)
                    canon_quad = canonicalize_quad(clamped)
                    new_objects.append({"quad": canon_quad, "desc": obj["desc"]})
                elif "line" in obj:
                    new_objects.append(obj)
                else:
                    raise ValueError(
                        f"Unsupported geometry in object: keys={list(obj.keys())}"
                    )
        else:
            for obj in sample["objects"]:
                if "line" in obj:
                    # Rotate each vertex of the line and keep as line
                    l = obj["line"]
                    pts = [(int(l[i]), int(l[i + 1])) for i in range(0, len(l), 2)]
                    rotated_pts = self._rotate_point_list(pts, width, height, transform)
                    flat: List[int] = []
                    for x, y in rotated_pts:
                        flat.extend([x, y])
                    new_objects.append({"line": flat, "desc": obj["desc"]})
                    continue
                if "bbox_2d" in obj:
                    b = obj["bbox_2d"]
                    x1, y1, x2, y2 = b
                    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    rotated_pts = self._rotate_point_list(
                        corners, width, height, transform
                    )
                    canon_quad = canonicalize_quad(rotated_pts)
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
                    rotated_pts = self._rotate_point_list(pts, width, height, transform)
                    canon_quad = canonicalize_quad(rotated_pts)
                    new_objects.append({"quad": canon_quad, "desc": obj["desc"]})
                    continue
                raise ValueError(
                    f"Unsupported geometry in object: keys={list(obj.keys())}"
                )

        new_sample = dict(sample)
        new_sample["objects"] = new_objects
        new_sample["width"] = out_w
        new_sample["height"] = out_h

        validate_sample_after_transform(new_sample, out_w, out_h)
        return rotated_images, new_sample


def _build_albumentations_policy(cfg: AlbumentationsRandAugConfig, rng: random.Random):
    if not cfg.enabled:
        return None
    if A is None:
        raise ImportError(
            "Albumentations is not installed but albumentations_rand.enabled=True was provided in config"
        )
    m = float(cfg.magnitude)

    # Construct candidates (no geometry). Respect safe_ops_only flag.
    safe_ops = [
        A.RandomBrightnessContrast(
            brightness_limit=0.3 * m, contrast_limit=0.3 * m, p=1.0
        ),
        A.RandomGamma(gamma_limit=(int(70 + 30 * (1 - m)), int(130 + 70 * m)), p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=int(8 * m),
            sat_shift_limit=int(20 * m),
            val_shift_limit=int(10 * m),
            p=1.0,
        ),
        A.RGBShift(
            r_shift_limit=int(10 * m),
            g_shift_limit=int(10 * m),
            b_shift_limit=int(10 * m),
            p=1.0,
        ),
        A.Sharpen(alpha=(0.05, 0.2 * m + 0.05), lightness=(0.9, 1.1), p=1.0),
    ]
    extended_ops = safe_ops + [
        A.ImageCompression(
            quality_lower=max(35, int(85 - 50 * m)), quality_upper=95, p=1.0
        ),
        A.GaussNoise(var_limit=(5.0, 50.0 * m), p=1.0),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=1.0,
        ),
        A.CoarseDropout(max_holes=int(5 + 10 * m), max_height=2, max_width=2, p=1.0),
    ]

    ops = safe_ops if cfg.safe_ops_only else extended_ops

    def make_compose():
        return A.Compose(
            [
                A.SomeOf(ops, n=cfg.num_ops, replace=False, p=cfg.apply_prob),
            ]
        )

    return make_compose


def _apply_albumentations(
    images: List[Image.Image], maker, rng: random.Random
) -> List[Image.Image]:
    assert maker is not None, "albumentations policy maker is None"
    aug = maker()
    out: List[Image.Image] = []
    for img in images:
        img_np = np.array(img)
        res = aug(image=img_np)
        out_img = Image.fromarray(res["image"])
        out.append(out_img)
    return out


@dataclass(frozen=True)
class AugmentationPipeline:
    config: AugmentationConfig
    op: AngleRotateOp

    @staticmethod
    def from_config(config: AugmentationConfig) -> "AugmentationPipeline":
        validate_augmentation_config(config)
        validate_angle_rotate_config(config.op)
        op = AngleRotateOp(config=config.op)
        return AugmentationPipeline(config=config, op=op)

    def _should_skip_due_to_lines(self, sample: Dict[str, Any]) -> bool:
        has_line = any(
            ("line" in o) for o in sample["objects"]
        )  # fail-fast on missing key
        if not has_line:
            return False
        policy = self.config.lines_policy
        if policy == "identity":
            return True
        if policy == "drop_objects":
            objs = [o for o in sample["objects"] if "line" not in o]
            sample["objects"] = objs
            return False
        if policy == "transform":
            # allow transform at image level (rotation) for line: no skip
            return False
        if policy == "error":
            raise ValueError(
                "lines_policy=error and line objects present; cannot rotate lines in v1"
            )
        raise ValueError(f"Unknown lines_policy: {policy}")

    def apply(
        self, sample: Dict[str, Any], images: List[Image.Image], sample_index: int
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        base_seed = int(self.config.rng_seed)
        rng = random.Random(base_seed ^ (sample_index & 0x7FFFFFFF))

        # Build albumentations policy maker lazily per call for determinism
        alb_maker = None
        if self.config.albumentations_rand is not None:
            alb_maker = _build_albumentations_policy(
                self.config.albumentations_rand, rng
            )

        if self._should_skip_due_to_lines(sample):
            if alb_maker is not None:
                images = _apply_albumentations(images, alb_maker, rng)
            return images, sample

        rotated_images, new_sample = self.op.apply(
            sample=sample, images=images, rng=rng
        )

        if alb_maker is not None:
            rotated_images = _apply_albumentations(rotated_images, alb_maker, rng)

        # Optional per-object local affine
        ola = self.config.object_local_affine
        if ola is not None and ola.enabled and ola.apply_pixels:
            before = new_sample
            new_sample = _apply_per_object_local_affine(new_sample, ola, rng)
            rotated_images = apply_per_object_pixels(
                rotated_images, before, new_sample, ola
            )

        # Optional object copy-paste (weapon)
        ocp = getattr(self.config, "object_copy_paste", None)
        if ocp is not None and ocp.enabled:
            rotated_images, new_sample = _apply_object_copy_paste(
                rotated_images, new_sample, ocp, rng
            )

        # Optional object blur (weapon)
        obl = getattr(self.config, "object_blur", None)
        if obl is not None and obl.enabled:
            rotated_images, new_sample = _apply_object_blur(
                rotated_images, new_sample, obl, rng
            )

        # Optional RandAug pool: sample N ops among included weapons
        rp = getattr(self.config, "rand_pool", None)
        if rp is not None and rp.enabled and rng.random() < rp.apply_prob:
            # Build callable ops referencing current config knobs
            ops = []
            if (
                rp.include_object_affine
                and ola is not None
                and ola.enabled
                and ola.apply_pixels
            ):
                ops.append(
                    lambda imgs, s: (apply_per_object_pixels(imgs, s, s, ola), s)[1]
                )  # placeholder
            if rp.include_object_copy_paste and ocp is not None and ocp.enabled:
                ops.append(lambda imgs, s: _apply_object_copy_paste(imgs, s, ocp, rng))
            if rp.include_object_blur and obl is not None and obl.enabled:
                ops.append(lambda imgs, s: _apply_object_blur(imgs, s, obl, rng))
            # Sample distinct ops
            rng.shuffle(ops)
            for op_fn in ops[: rp.num_ops]:
                res = op_fn(rotated_images, new_sample)
                # op may return Tuple[List[Image], sample] or (images,sample) shaped
                if isinstance(res, tuple) and len(res) == 2:
                    rotated_images, new_sample = res

        # Apply criteria/guards (e.g., occlusion annotations)
        new_sample = _apply_occlusion_criterion(
            new_sample, getattr(self.config, "criteria", None)
        )

        return rotated_images, new_sample


def _polygon_area(pts: List[Tuple[int, int]]) -> float:
    area2 = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def _point_in_quad(px: float, py: float, quad: List[Tuple[int, int]]) -> bool:
    # Using barycentric via split into two triangles
    def tri_area(a, b, c):
        return abs(
            (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
        )

    A = quad
    whole = _polygon_area(A)
    a1 = tri_area((px, py), A[0], A[1])
    a2 = tri_area((px, py), A[1], A[2])
    a3 = tri_area((px, py), A[2], A[3])
    a4 = tri_area((px, py), A[3], A[0])
    return abs((a1 + a2 + a3 + a4) - whole) < 1e-3


def _quad_contains_quad(
    inner: List[Tuple[int, int]], outer: List[Tuple[int, int]]
) -> bool:
    for x, y in inner:
        if not _point_in_quad(x, y, outer):
            return False
    return True


def _iou_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _quad_bbox(quad: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_from_quad_flat(q_flat: List[int]) -> Tuple[int, int, int, int]:
    xs = q_flat[0::2]
    ys = q_flat[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def _rasterize_quads_to_mask(
    quads: List[List[Tuple[int, int]]], size: Tuple[int, int]
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for q in quads:
        draw.polygon(q, outline=255, fill=255)
    return mask


def _build_occupancy_mask(
    sample: Dict[str, Any], width: int, height: int, down: int, margin: int
) -> Image.Image:
    quads: List[List[Tuple[int, int]]] = []
    for o in sample.get("objects", []):
        if "quad" in o:
            q = o["quad"]
            quads.append([(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])])
        elif "bbox_2d" in o:
            b = o["bbox_2d"]
            x1, y1, x2, y2 = b
            quads.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        elif "line" in o:
            # approximate with small-width polygon around line; skip for occupancy simplicity
            continue
    if not quads:
        return Image.new("L", (max(1, width // down), max(1, height // down)), 0)

    base = _rasterize_quads_to_mask(quads, (width, height))
    if margin > 0:
        base = base.filter(ImageFilter.MaxFilter(size=margin | 1))
    low = base.resize((max(1, width // down), max(1, height // down)), Image.NEAREST)
    return low


def _apply_object_copy_paste(
    images: List[Image.Image],
    sample: Dict[str, Any],
    cfg,
    rng: random.Random,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    if not images:
        return images, sample
    if not cfg.enabled:
        return images, sample

    img_rgba = images[0].convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))

    width = int(sample["width"])
    height = int(sample["height"])

    # Build groups: for any object fully contained by another, attach as child to the container
    objects = sample.get("objects", [])
    idx_to_quad: List[List[Tuple[int, int]]] = []
    for o in objects:
        if "quad" in o:
            q = o["quad"]
            idx_to_quad.append([(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])])
        elif "bbox_2d" in o:
            b = o["bbox_2d"]
            x1, y1, x2, y2 = b
            idx_to_quad.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        else:
            idx_to_quad.append([])

    groups: List[List[int]] = []
    assigned = set()
    for i, qi in enumerate(idx_to_quad):
        if not qi or i in assigned:
            continue
        group = [i]
        for j, qj in enumerate(idx_to_quad):
            if i == j or not qj:
                continue
            if _quad_contains_quad(qj, qi):
                group.append(j)
        for g in group:
            assigned.add(g)
        groups.append(sorted(group))

    # Occupancy mask
    occ_mask = _build_occupancy_mask(
        sample, width, height, cfg.occ_grid_downscale, cfg.occ_margin_px
    )

    def occ_fraction_for_quad(cand: List[Tuple[int, int]]) -> float:
        hi = Image.new("L", (width, height), 0)
        ImageDraw.Draw(hi).polygon(cand, outline=255, fill=255)
        low = hi.resize(occ_mask.size, Image.NEAREST)
        # compute mean over candidate region
        cand_sum = 0
        cand_cnt = 0
        pix_occ = occ_mask.load()
        pix_c = low.load()
        for y in range(low.size[1]):
            for x in range(low.size[0]):
                if pix_c[x, y] > 0:
                    cand_cnt += 1
                    cand_sum += 1 if pix_occ[x, y] > 0 else 0
        return (cand_sum / cand_cnt) if cand_cnt > 0 else 0.0

    def transform_quad(
        quad: List[Tuple[int, int]], angle_deg: float, scale: float, dx: int, dy: int
    ) -> List[Tuple[int, int]]:
        from math import cos, radians, sin

        cx = sum(p[0] for p in quad) / 4.0
        cy = sum(p[1] for p in quad) / 4.0
        ang = radians(angle_deg)
        cs, sn = cos(ang), sin(ang)
        out = []
        for x, y in quad:
            # scale around center
            sx = cx + (x - cx) * scale
            sy = cy + (y - cy) * scale
            # rotate around center
            rx = cs * (sx - cx) - sn * (sy - cy) + cx
            ry = sn * (sx - cx) + cs * (sy - cy) + cy
            out.append((int(round(rx + dx)), int(round(ry + dy))))
        return out

    new_objects = list(objects)

    def extract_patch(quad: List[Tuple[int, int]]) -> Image.Image:
        # local crop to bbox, masked by quad
        x1, y1, x2, y2 = _quad_bbox(quad)
        x2i, y2i = x2 + 1, y2 + 1
        local_src = img_rgba.crop((x1, y1, x2i, y2i))
        mask = Image.new("L", (x2i - x1, y2i - y1), 0)
        pts_local = [(px - x1, py - y1) for (px, py) in quad]
        ImageDraw.Draw(mask).polygon(pts_local, outline=255, fill=255)
        return Image.composite(
            local_src, Image.new("RGBA", local_src.size, (0, 0, 0, 0)), mask
        )

    # For each group (container + any contained), attempt copies
    for group in groups:
        root_idx = group[0]
        root_quad = idx_to_quad[root_idx]
        if not root_quad:
            continue
        # Optional per-object filtering by type
        if cfg.allowed_types is not None:
            desc = objects[root_idx].get("desc", "")
            if not any(t in desc for t in cfg.allowed_types):
                continue
        if rng.random() > cfg.per_object_prob:
            continue

        # Build group relative offsets
        group_quads = [idx_to_quad[i] for i in group]
        group_offsets = []
        for q in group_quads:
            group_offsets.append(
                [
                    (q[k][0] - root_quad[k][0], q[k][1] - root_quad[k][1])
                    for k in range(4)
                ]
            )

        # Prepare root patch
        root_patch = extract_patch(root_quad)

        for _ in range(cfg.num_copies_per_object):
            placed = False
            for _try in range(cfg.attempts):
                dx = rng.randint(-cfg.translate_px, cfg.translate_px)
                dy = rng.randint(-cfg.translate_px, cfg.translate_px)
                ang = rng.uniform(-cfg.rotation_jitter_deg, cfg.rotation_jitter_deg)
                sc = rng.uniform(cfg.scale_jitter_min, cfg.scale_jitter_max)

                cand_root = transform_quad(root_quad, ang, sc, dx, dy)
                # bounds check
                if any(not (0 <= x < width and 0 <= y < height) for x, y in cand_root):
                    continue
                # occupancy check
                if occ_fraction_for_quad(cand_root) > cfg.max_occ_fraction:
                    continue
                # IoU check vs existing
                bb_cand = _quad_bbox(cand_root)
                too_close = False
                for o in new_objects:
                    if "quad" in o:
                        bb = _quad_bbox(
                            [(o["quad"][i], o["quad"][i + 1]) for i in range(0, 8, 2)]
                        )
                    elif "bbox_2d" in o:
                        b = o["bbox_2d"]
                        bb = (
                            min(b[0], b[2]),
                            min(b[1], b[3]),
                            max(b[0], b[2]),
                            max(b[1], b[3]),
                        )
                    else:
                        continue
                    if _iou_bbox(bb_cand, bb) > cfg.max_iou_with_existing:
                        too_close = True
                        break
                if too_close:
                    continue

                # Paste root patch rotated+scaled around its center
                pr = root_patch.rotate(angle=-ang, resample=Image.BILINEAR, expand=True)
                cx = sum(p[0] for p in root_quad) / 4.0
                cy = sum(p[1] for p in root_quad) / 4.0
                paste_x = int(round(cx + dx - pr.width / 2))
                paste_y = int(round(cy + dy - pr.height / 2))
                overlay.paste(pr, (paste_x, paste_y), pr)

                # Add root object clone
                rq = [coord for xy in cand_root for coord in xy]
                new_objects.append({"quad": rq, "desc": objects[root_idx]["desc"]})

                # Paste and add each contained member, preserving relative offsets
                for idx, offsets in zip(group[1:], group_offsets[1:]):
                    src_q = idx_to_quad[idx]
                    # apply same transform to src_q center-wise
                    # derive by applying offsets to transformed root quad corners
                    member_q = [
                        (
                            cand_root[k][0] + offsets[k][0],
                            cand_root[k][1] + offsets[k][1],
                        )
                        for k in range(4)
                    ]
                    # bounds check
                    if any(
                        not (0 <= x < width and 0 <= y < height) for x, y in member_q
                    ):
                        continue
                    # extract member patch and paste
                    mem_patch = extract_patch(src_q).rotate(
                        angle=-ang, resample=Image.BILINEAR, expand=True
                    )
                    mcx = sum(p[0] for p in src_q) / 4.0
                    mcy = sum(p[1] for p in src_q) / 4.0
                    mpx = int(round(mcx + dx - mem_patch.width / 2))
                    mpy = int(round(mcy + dy - mem_patch.height / 2))
                    overlay.paste(mem_patch, (mpx, mpy), mem_patch)
                    new_objects.append(
                        {
                            "quad": [c for xy in member_q for c in xy],
                            "desc": objects[idx]["desc"],
                        }
                    )

                placed = True
                break
            # if not placed, skip silently

    composed = Image.alpha_composite(img_rgba, overlay).convert("RGB")
    return [composed], {**sample, "objects": new_objects}


def _apply_object_blur(
    images: List[Image.Image], sample: Dict[str, Any], cfg, rng: random.Random
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    if not images or not cfg.enabled:
        return images, sample
    img = images[0].convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out = img.copy()

    def blur_region(
        pil_img: Image.Image, mask: Image.Image, radius: float
    ) -> Image.Image:
        if cfg.blur_type == "gaussian":
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius))
        else:
            # approximate box blur via multiple BoxBlur calls
            blurred = pil_img.filter(ImageFilter.BoxBlur(radius))
        return Image.composite(blurred, pil_img, mask)

    w, h = img.size
    for obj in sample.get("objects", []):
        if rng.random() > cfg.per_object_prob:
            continue
        # Skip OCR-sensitive targets
        try:
            if isinstance(obj.get("desc"), str) and ("标签" in obj["desc"]):
                continue
        except Exception:
            pass
        if "quad" in obj:
            q = obj["quad"]
            pts = [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
        elif "bbox_2d" in obj:
            b = obj["bbox_2d"]
            pts = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
        else:
            continue
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).polygon(pts, outline=255, fill=255)
        radius = rng.uniform(cfg.radius_min, cfg.radius_max)
        out = blur_region(out, mask, radius)

    return [out.convert("RGB")], sample


def _rotate_point(
    px: float, py: float, cx: float, cy: float, angle_rad: float
) -> Tuple[int, int]:
    from math import cos, sin

    dx = px - cx
    dy = py - cy
    rx = cos(angle_rad) * dx - sin(angle_rad) * dy + cx
    ry = sin(angle_rad) * dx + cos(angle_rad) * dy + cy
    return int(round(rx)), int(round(ry))


def _apply_per_object_local_affine(
    sample: Dict[str, Any], cfg: ObjectLocalAffineConfig, rng: random.Random
) -> Dict[str, Any]:
    # If not applying pixels, skip to avoid coordinates-only noise
    if not cfg.apply_pixels:
        return sample
    width, height = int(sample["width"]), int(sample["height"])
    objects = sample.get("objects", [])

    # Precompute current bboxes for overlap checking
    def obj_bbox(o: Dict[str, Any]) -> Tuple[int, int, int, int]:
        if "quad" in o:
            qf = o["quad"]
            return _bbox_from_quad_flat(qf)
        if "bbox_2d" in o:
            b = o["bbox_2d"]
            return (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
        if "line" in o:
            l = o["line"]
            xs = l[0::2]
            ys = l[1::2]
            return (min(xs), min(ys), max(xs), max(ys))
        return (0, 0, 0, 0)

    def quad_inside_bounds(q: List[Tuple[int, int]]) -> bool:
        for x, y in q:
            if not (0 <= x < width and 0 <= y < height):
                return False
        return True

    def quad_area(q: List[Tuple[int, int]]) -> float:
        # Signed area /2
        area2 = 0.0
        for i in range(4):
            x1, y1 = q[i]
            x2, y2 = q[(i + 1) % 4]
            area2 += x1 * y2 - x2 * y1
        return abs(area2) * 0.5

    def object_quad(o: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        if "quad" in o:
            q = o["quad"]
            return [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
        if "bbox_2d" in o:
            b = o["bbox_2d"]
            return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
        return None

    def line_points(o: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        if "line" in o:
            l = o["line"]
            return [(l[i], l[i + 1]) for i in range(0, len(l), 2)]
        return None

    current_bboxes = [obj_bbox(o) for o in objects]
    updated_objects: List[Dict[str, Any]] = [{} for _ in range(len(objects))]

    from math import radians

    visited: set[int] = set()

    for idx, o in enumerate(objects):
        if idx in visited:
            continue
        # Only consider roots with polygon geometry
        root_poly = object_quad(o)
        if root_poly is None:
            updated_objects[idx] = o
            visited.add(idx)
            continue
        if rng.random() > cfg.per_object_prob:
            updated_objects[idx] = o
            visited.add(idx)
            continue

        # Root center
        cx = sum(x for x, _y in root_poly) / 4.0
        cy = sum(y for _x, y in root_poly) / 4.0

        # Build children set: objects fully inside root polygon
        children: List[int] = []
        for j, oj in enumerate(objects):
            if j == idx or j in visited:
                continue
            qp = object_quad(oj)
            if qp is not None:
                if _quad_contains_quad(qp, root_poly):
                    children.append(j)
                continue
            lp = line_points(oj)
            if lp is not None:
                if all(_point_in_quad(px, py, root_poly) for (px, py) in lp):
                    children.append(j)

        success = False
        attempt = 0
        new_root_quad: List[int] = [c for xy in root_poly for c in xy]
        new_children_geoms: Dict[int, Dict[str, Any]] = {}
        while attempt <= cfg.max_resample and not success:
            attempt += 1
            angle_deg = rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)
            tx = rng.randint(-cfg.translate_px, cfg.translate_px)
            ty = rng.randint(-cfg.translate_px, cfg.translate_px)
            ang = radians(angle_deg)

            # Transform root
            rpts: List[Tuple[int, int]] = []
            for x, y in root_poly:
                rx, ry = _rotate_point(x, y, cx, cy, ang)
                rpts.append((rx + tx, ry + ty))

            if not quad_inside_bounds(rpts):
                continue
            canon_root = canonicalize_quad(rpts)
            nb = _bbox_from_quad_flat(canon_root)
            if (
                quad_area(
                    [
                        (canon_root[0], canon_root[1]),
                        (canon_root[2], canon_root[3]),
                        (canon_root[4], canon_root[5]),
                        (canon_root[6], canon_root[7]),
                    ]
                )
                < 1.0
            ):
                continue
            if cfg.avoid_overlap:
                overlap_ok = True
                for j, bb in enumerate(current_bboxes):
                    if j == idx:
                        continue
                    if _iou_bbox(nb, bb) > cfg.iou_thresh:
                        overlap_ok = False
                        break
                if not overlap_ok:
                    continue

            # Transform children with same (ang, tx, ty) around root center
            valid_children = True
            new_children_geoms.clear()
            for j in children:
                oj = objects[j]
                qp = object_quad(oj)
                if qp is not None:
                    pts = []
                    for x, y in qp:
                        rx, ry = _rotate_point(x, y, cx, cy, ang)
                        pts.append((rx + tx, ry + ty))
                    if not quad_inside_bounds(pts):
                        valid_children = False
                        break
                    canon = canonicalize_quad(pts)
                    new_children_geoms[j] = {"quad": canon, "desc": oj["desc"]}
                    continue
                lp = line_points(oj)
                if lp is not None:
                    pts = []
                    for x, y in lp:
                        rx, ry = _rotate_point(x, y, cx, cy, ang)
                        pts.append((rx + tx, ry + ty))
                    # clamp
                    pts = round_and_clamp_points(pts, width, height)
                    flat: List[int] = []
                    for x, y in pts:
                        flat.extend([x, y])
                    new_children_geoms[j] = {"line": flat, "desc": oj["desc"]}
                    continue

            if not valid_children:
                continue

            new_root_quad = canon_root
            success = True

        # Commit updates for root and children
        if success:
            updated_objects[idx] = {"quad": new_root_quad, "desc": o["desc"]}
            current_bboxes[idx] = _bbox_from_quad_flat(new_root_quad)
            visited.add(idx)
            for j in children:
                val = new_children_geoms[j] if j in new_children_geoms else objects[j]
                updated_objects[j] = val
                visited.add(j)
        else:
            # No change; keep originals
            updated_objects[idx] = o
            visited.add(idx)
            for j in children:
                if j not in visited:
                    updated_objects[j] = objects[j]
                    visited.add(j)

    # Fill untouched objects
    for i in range(len(objects)):
        if not updated_objects[i]:
            updated_objects[i] = objects[i]

    sample_out = dict(sample)
    sample_out["objects"] = updated_objects
    validate_sample_after_transform(sample_out, width, height)
    return sample_out


def apply_per_object_pixels(
    images: List[Image.Image],
    sample_in: Dict[str, Any],
    sample_out: Dict[str, Any],
    cfg: ObjectLocalAffineConfig,
) -> List[Image.Image]:
    """When per-object local affine produced new quads, optionally rotate+paste pixels for the affected objects.

    For v1, we only support single-image samples with a simple rotate+translate of the object patch.
    Implementation: infer angle from TL->TR vector change and translate by center delta. Overlaps allowed.
    """
    if not cfg.apply_pixels:
        return images
    if not images:
        return images
    from PIL import ImageDraw

    src_rgba = images[0].convert("RGBA")  # immutable sampling source
    base_rgba = images[0].convert("RGBA")  # will erase old regions progressively

    overlay = Image.new("RGBA", src_rgba.size, (0, 0, 0, 0))

    def to_pts(o: Dict[str, Any]) -> List[Tuple[int, int]]:
        if "quad" in o:
            q = o["quad"]
            return [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
        if "bbox_2d" in o:
            b = o["bbox_2d"]
            x1, y1, x2, y2 = b
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return []

    def center(pts: List[Tuple[int, int]]) -> Tuple[float, float]:
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        return cx, cy

    def bbox(pts: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    transparent = Image.new("RGBA", src_rgba.size, (0, 0, 0, 0))

    for o_prev, o_new in zip(
        sample_in.get("objects", []), sample_out.get("objects", [])
    ):
        # Only process geometry-bearing objects
        if "quad" not in o_new and "bbox_2d" not in o_new:
            continue

        prev_pts = to_pts(o_prev)
        new_pts = to_pts(o_new)
        if len(prev_pts) != 4 or len(new_pts) != 4:
            continue

        # Build global mask for erasing old region
        mask_prev_global = Image.new("L", src_rgba.size, 0)
        ImageDraw.Draw(mask_prev_global).polygon(prev_pts, outline=255, fill=255)

        # Create local patch around previous bbox
        x1, y1, x2, y2 = bbox(prev_pts)
        # Ensure inclusive bounds
        x2_i, y2_i = x2 + 1, y2 + 1
        local_src = src_rgba.crop((x1, y1, x2_i, y2_i))
        local_mask = Image.new("L", (x2_i - x1, y2_i - y1), 0)
        prev_pts_local = [(px - x1, py - y1) for (px, py) in prev_pts]
        ImageDraw.Draw(local_mask).polygon(prev_pts_local, outline=255, fill=255)
        patch = Image.composite(
            local_src, Image.new("RGBA", local_src.size, (0, 0, 0, 0)), local_mask
        )

        # Infer rotation between TL->TR edges
        from math import atan2, degrees

        def edge_angle(pts: List[Tuple[int, int]]) -> float:
            tl, tr = pts[0], pts[1]
            return atan2(tr[1] - tl[1], tr[0] - tl[0])

        ang_prev = edge_angle(prev_pts)
        ang_new = edge_angle(new_pts)
        ang_deg = degrees(ang_new - ang_prev)

        # Rotate the patch around its own center
        patch_rot = patch.rotate(angle=-ang_deg, resample=Image.BILINEAR, expand=True)

        # Erase original region from base
        base_rgba = Image.composite(transparent, base_rgba, mask_prev_global)

        # Translate by center delta
        cx_prev, cy_prev = center(prev_pts)
        cx_new, cy_new = center(new_pts)
        dx = int(round(cx_new - cx_prev))
        dy = int(round(cy_new - cy_prev))

        # Paste rotated+shifted patch onto overlay (overlaps allowed)
        paste_x = int(round(cx_prev + dx - patch_rot.width / 2))
        paste_y = int(round(cy_prev + dy - patch_rot.height / 2))
        overlay.paste(patch_rot, (paste_x, paste_y), patch_rot)

    composed = Image.alpha_composite(base_rgba, overlay).convert("RGB")
    return [composed]


def _apply_occlusion_criterion(sample: Dict[str, Any], cfg) -> Dict[str, Any]:
    if cfg is None or cfg.occlusion is None or not cfg.occlusion.enabled:
        return sample
    occ = cfg.occlusion
    width = int(sample["width"])
    height = int(sample["height"])

    # Prepare masks for quads/bboxes and for lines
    down = max(1, occ.mask_downscale)
    w_low = max(1, width // down)
    h_low = max(1, height // down)

    # Build per-object masks at low resolution
    masks: List[Tuple[str, Image.Image]] = []  # (type, mask)
    for o in sample.get("objects", []):
        m = Image.new("L", (w_low, h_low), 0)
        draw = ImageDraw.Draw(m)
        if "quad" in o:
            q = o["quad"]
            pts = [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
            hi = Image.new("L", (width, height), 0)
            ImageDraw.Draw(hi).polygon(pts, outline=255, fill=255)
            m = hi.resize((w_low, h_low), Image.NEAREST)
            masks.append(("area", m))
        elif "bbox_2d" in o:
            b = o["bbox_2d"]
            hi = Image.new("L", (width, height), 0)
            ImageDraw.Draw(hi).rectangle(
                [b[0], b[1], b[2], b[3]], outline=255, fill=255
            )
            m = hi.resize((w_low, h_low), Image.NEAREST)
            masks.append(("area", m))
        elif "line" in o:
            l = o["line"]
            pts = [(l[i], l[i + 1]) for i in range(0, len(l), 2)]
            hi = Image.new("L", (width, height), 0)
            ImageDraw.Draw(hi).line(pts, fill=255, width=max(1, occ.line_width_px))
            m = hi.resize((w_low, h_low), Image.NEAREST)
            masks.append(("line", m))
        else:
            masks.append(("other", m))

    # Compute occlusion by pairwise intersections
    objs = sample.get("objects", [])
    updated = []
    for i, o in enumerate(objs):
        typ_i, mask_i = masks[i]
        if typ_i not in ("area", "line"):
            updated.append(o)
            continue
        # union of all other masks
        union = Image.new("L", mask_i.size, 0)
        for j, (typ_j, mask_j) in enumerate(masks):
            if j == i:
                continue
            union = ImageChops.lighter(union, mask_j)
        # overlap fraction
        inter = ImageChops.multiply(mask_i, union)
        inter_sum = sum(inter.getdata())
        area_sum = sum(mask_i.getdata())
        frac = (inter_sum / area_sum) if area_sum > 0 else 0.0
        thresh = (
            occ.min_overlap_fraction_line
            if typ_i == "line"
            else occ.min_overlap_fraction_bbox
        )
        if frac >= thresh:
            # Ensure '有遮挡' present in desc
            desc = o.get("desc", "")
            if isinstance(desc, str) and ("有遮挡" not in desc):
                if "无遮挡" in desc:
                    desc = desc.replace("无遮挡", "有遮挡")
                else:
                    desc = (desc + ",有遮挡") if desc else "有遮挡"
            new_o = dict(o)
            new_o["desc"] = desc
            updated.append(new_o)
        else:
            updated.append(o)

    out = dict(sample)
    out["objects"] = updated
    return out
