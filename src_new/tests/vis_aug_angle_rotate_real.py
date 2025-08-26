from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from src_new.augmentation.base import AugmentationPipeline
from src_new.augmentation.viz_debug import overlay_quads
from src_new.config.augmentation_config import (
    AlbumentationsRandAugConfig,
    AngleRotateConfig,
    AugmentationConfig,
    ColorJitterConfig,
    ObjectBlurConfig,
    ObjectCopyPasteConfig,
    ObjectLocalAffineConfig,
    RandAugPoolConfig,
)
from src_new.utils.path_manager import create_path_manager


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_image(img_rel_path: str, data_root: str) -> Image.Image:
    pm = create_path_manager(data_root)
    p = pm.resolve_path(img_rel_path)
    return Image.open(str(p)).convert("RGB")


def build_pipeline(
    angle_cfg: dict,
    lines_policy: str = "identity",
    cj_cfg: Dict[str, Any] | None = None,
    alb_cfg: Dict[str, Any] | None = None,
    ola_cfg: Dict[str, Any] | None = None,
    ocp_cfg: Dict[str, Any] | None = None,
    ob_cfg: Dict[str, Any] | None = None,
    rp_cfg: Dict[str, Any] | None = None,
) -> AugmentationPipeline:
    color_jitter = None
    if cj_cfg is not None:

        def _to_tuple(name: str):
            v = cj_cfg.get(name)
            return tuple(v) if isinstance(v, list) else v

        color_jitter = ColorJitterConfig(
            enabled=bool(cj_cfg.get("enabled", False)),
            apply_prob=float(cj_cfg.get("apply_prob", 1.0)),
            brightness=_to_tuple("brightness"),
            contrast=_to_tuple("contrast"),
            saturation=_to_tuple("saturation"),
            sharpness=_to_tuple("sharpness"),
            order=cj_cfg.get("order"),
        )

    alb_rand = None
    if alb_cfg is not None:
        alb_rand = AlbumentationsRandAugConfig(
            enabled=bool(alb_cfg.get("enabled", False)),
            apply_prob=float(alb_cfg.get("apply_prob", 1.0)),
            num_ops=int(alb_cfg.get("num_ops", 2)),
            magnitude=float(alb_cfg.get("magnitude", 0.5)),
            safe_ops_only=bool(alb_cfg.get("safe_ops_only", True)),
        )

    ola = None
    if ola_cfg is not None:
        ola = ObjectLocalAffineConfig(
            enabled=bool(ola_cfg.get("enabled", False)),
            per_object_prob=float(ola_cfg.get("per_object_prob", 0.5)),
            max_rotation_deg=float(ola_cfg.get("max_rotation_deg", 8.0)),
            translate_px=int(ola_cfg.get("translate_px", 6)),
            avoid_overlap=bool(ola_cfg.get("avoid_overlap", True)),
            iou_thresh=float(ola_cfg.get("iou_thresh", 0.05)),
            max_resample=int(ola_cfg.get("max_resample", 10)),
            apply_pixels=bool(ola_cfg.get("apply_pixels", False)),
        )

    ocp = None
    if ocp_cfg is not None:
        ocp = ObjectCopyPasteConfig(
            enabled=bool(ocp_cfg.get("enabled", False)),
            per_object_prob=float(ocp_cfg.get("per_object_prob", 0.5)),
            num_copies_per_object=int(ocp_cfg.get("num_copies_per_object", 1)),
            translate_px=int(ocp_cfg.get("translate_px", 24)),
            rotation_jitter_deg=float(ocp_cfg.get("rotation_jitter_deg", 10.0)),
            scale_jitter_min=float(ocp_cfg.get("scale_jitter_min", 0.95)),
            scale_jitter_max=float(ocp_cfg.get("scale_jitter_max", 1.05)),
            occ_grid_downscale=int(ocp_cfg.get("occ_grid_downscale", 8)),
            occ_margin_px=int(ocp_cfg.get("occ_margin_px", 8)),
            max_occ_fraction=float(ocp_cfg.get("max_occ_fraction", 0.1)),
            max_iou_with_existing=float(ocp_cfg.get("max_iou_with_existing", 0.2)),
            attempts=int(ocp_cfg.get("attempts", 20)),
            allowed_types=ocp_cfg.get("allowed_types"),
        )

    ob = None
    if ob_cfg is not None:
        ob = ObjectBlurConfig(
            enabled=bool(ob_cfg.get("enabled", False)),
            per_object_prob=float(ob_cfg.get("per_object_prob", 0.5)),
            blur_type=str(ob_cfg.get("blur_type", "gaussian")),
            radius_min=float(ob_cfg.get("radius_min", 2.0)),
            radius_max=float(ob_cfg.get("radius_max", 6.0)),
        )

    rp = None
    if rp_cfg is not None:
        rp = RandAugPoolConfig(
            enabled=bool(rp_cfg.get("enabled", False)),
            apply_prob=float(rp_cfg.get("apply_prob", 1.0)),
            num_ops=int(rp_cfg.get("num_ops", 2)),
            include_object_affine=bool(rp_cfg.get("include_object_affine", False)),
            include_object_copy_paste=bool(
                rp_cfg.get("include_object_copy_paste", False)
            ),
            include_object_blur=bool(rp_cfg.get("include_object_blur", False)),
        )

    aug_cfg = AugmentationConfig(
        enabled=True,
        rng_seed=12345,
        apply_to_teachers=False,
        lines_policy=lines_policy,
        debug_visualization=False,
        debug_output_dir=None,
        op=AngleRotateConfig(
            sample_mode=angle_cfg["mode"],
            fixed_angle_deg=angle_cfg.get("fixed"),
            angle_min_deg=angle_cfg.get("min"),
            angle_max_deg=angle_cfg.get("max"),
            angles_set_deg=angle_cfg.get("set"),
            expand=angle_cfg.get("expand", True),
            interpolation=angle_cfg.get("interp", "bilinear"),
            fill_color=tuple(angle_cfg.get("fill", (0, 0, 0))),
        ),
        color_jitter=color_jitter,
        albumentations_rand=alb_rand,
        object_local_affine=ola,
        object_copy_paste=ocp,
        object_blur=ob,
        rand_pool=rp,
    )
    return AugmentationPipeline.from_config(aug_cfg)


def visualize_sample(
    sample: Dict[str, Any], data_root: str, out_dir: str, idx: int
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    img_path = sample["images"][0]
    base_img = load_image(img_path, data_root)

    base_vis = overlay_quads(base_img, sample["objects"])

    variants = [
        {
            "name": "fixed_+60_expand",
            "cfg": {"mode": "fixed", "fixed": 60.0, "expand": True},
            "cj": None,
            "alb": None,
        },
        {
            "name": "fixed_-60_expand",
            "cfg": {"mode": "fixed", "fixed": -60.0, "expand": True},
            "cj": None,
            "alb": None,
        },
        {
            "name": "range_(-75,75)_expand",
            "cfg": {"mode": "uniform_range", "min": -75.0, "max": 75.0, "expand": True},
            "cj": None,
            "alb": None,
        },
        {
            "name": "alb_rand_strong",
            "cfg": {"mode": "fixed", "fixed": 0.0, "expand": False},
            "cj": None,
            "alb": {
                "enabled": True,
                "apply_prob": 1.0,
                "num_ops": 3,
                "magnitude": 0.8,
                "safe_ops_only": True,
            },
            "ola": None,
        },
        {
            "name": "object_affine_v1",
            "cfg": {"mode": "fixed", "fixed": 0.0, "expand": False},
            "cj": None,
            "alb": None,
            "ola": {
                "enabled": True,
                "per_object_prob": 1.0,
                "max_rotation_deg": 25.0,
                "translate_px": 24,
                "avoid_overlap": False,
                "iou_thresh": 0.05,
                "max_resample": 5,
                "apply_pixels": True,
            },
        },
        {
            "name": "object_copy_paste_v1",
            "cfg": {"mode": "fixed", "fixed": 0.0, "expand": False},
            "cj": None,
            "alb": None,
            "ola": None,
            "ocp": {
                "enabled": True,
                "per_object_prob": 0.8,
                "num_copies_per_object": 1,
                "translate_px": 64,
                "rotation_jitter_deg": 10.0,
                "scale_jitter_min": 0.95,
                "scale_jitter_max": 1.05,
                "occ_grid_downscale": 8,
                "occ_margin_px": 8,
                "max_occ_fraction": 0.1,
                "max_iou_with_existing": 0.2,
                "attempts": 30,
            },
        },
        {
            "name": "object_blur_v1",
            "cfg": {"mode": "fixed", "fixed": 0.0, "expand": False},
            "ob": {
                "enabled": True,
                "per_object_prob": 0.9,
                "blur_type": "gaussian",
                "radius_min": 0.5,
                "radius_max": 1.2,
            },
        },
        {
            "name": "rand_pool_mixed",
            "cfg": {"mode": "fixed", "fixed": 0.0, "expand": False},
            "ola": {
                "enabled": True,
                "per_object_prob": 1.0,
                "max_rotation_deg": 15.0,
                "translate_px": 16,
                "avoid_overlap": False,
                "iou_thresh": 0.2,
                "max_resample": 5,
                "apply_pixels": True,
            },
            "ocp": {
                "enabled": True,
                "per_object_prob": 0.6,
                "num_copies_per_object": 1,
                "translate_px": 48,
                "rotation_jitter_deg": 8.0,
                "scale_jitter_min": 0.95,
                "scale_jitter_max": 1.05,
                "occ_grid_downscale": 8,
                "occ_margin_px": 6,
                "max_occ_fraction": 0.15,
                "max_iou_with_existing": 0.25,
                "attempts": 20,
            },
            "ob": {
                "enabled": True,
                "per_object_prob": 0.5,
                "blur_type": "gaussian",
                "radius_min": 0.5,
                "radius_max": 1.2,
            },
            "rp": {
                "enabled": True,
                "apply_prob": 1.0,
                "num_ops": 2,
                "include_object_affine": True,
                "include_object_copy_paste": True,
                "include_object_blur": True,
            },
        },
    ]

    grids: List[tuple[str, Image.Image]] = [("baseline", base_vis)]

    for variant in variants:
        pipeline = build_pipeline(
            variant["cfg"],
            lines_policy="transform",
            cj_cfg=variant.get("cj"),
            alb_cfg=variant.get("alb"),
            ola_cfg=variant.get("ola"),
            ocp_cfg=variant.get("ocp"),
            ob_cfg=variant.get("ob"),
            rp_cfg=variant.get("rp"),
        )
        aug_img_list, aug_sample = pipeline.apply(
            sample=sample.copy(),
            images=[base_img],
            sample_index=random.randint(0, 1_000_000),
        )
        vis = overlay_quads(aug_img_list[0], aug_sample["objects"])
        grids.append((variant["name"], vis))

    # Use a uniform cell size based on the largest variant to avoid overlap
    max_w = max(im.size[0] for _, im in grids)
    max_h = max(im.size[1] for _, im in grids)

    cols = 2
    rows = (len(grids) + cols - 1) // cols

    pad = 24
    label_h = 28

    out_w = cols * max_w + (cols + 1) * pad
    out_h = rows * (max_h + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (out_w, out_h), color=(30, 30, 30))

    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(canvas)
    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (name, im) in enumerate(grids):
        r = i // cols
        c = i % cols
        cell_x = pad + c * (max_w + pad)
        cell_y = pad + r * (max_h + label_h + pad)

        # Title bar
        draw.rectangle(
            [cell_x, cell_y, cell_x + max_w, cell_y + label_h - 2], fill=(50, 50, 50)
        )
        draw.text((cell_x + 6, cell_y + 6), name, fill=(220, 220, 220), font=font)

        # Center the image within the cell
        off_x = (max_w - im.size[0]) // 2
        off_y = (max_h - im.size[1]) // 2
        canvas.paste(im, (cell_x + off_x, cell_y + label_h + off_y))

    out_path = Path(out_dir) / f"aug_vis_{idx}.jpg"
    canvas.save(out_path)
    print(f"Saved visualization: {out_path}")


def main():
    data_root = "data/ds_v2_full"
    jsonl_path = str(Path(data_root) / "all_samples.jsonl")

    if not Path(jsonl_path).exists():
        raise FileNotFoundError(
            f"Missing {jsonl_path}. Please ensure data_root structure matches DataResolver expectations."
        )

    samples = read_jsonl(jsonl_path)

    rng = random.Random(123)
    chosen_indices = rng.sample(range(len(samples)), k=min(10, len(samples)))

    out_dir = "outputs/aug_vis"
    for i, idx in enumerate(chosen_indices):
        s = samples[idx]
        if "images" not in s or not s["images"]:
            continue
        visualize_sample(s, data_root=data_root, out_dir=out_dir, idx=i)


if __name__ == "__main__":
    main()
