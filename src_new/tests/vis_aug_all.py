#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src_new.augmentation import ObjectAwareAugmentationPipeline
from src_new.augmentation.standardize import canonicalize_quad
from src_new.augmentation.viz_debug import overlay_quads
from src_new.config.augmentation_config import (
    AugmentationConfig,
    CriteriaConfig,
    ImageGeomConfig,
    LineAugConfig,
    OcclusionCriterionConfig,
    PhotometricConfig,
)
from src_new.utils.path_manager import create_path_manager


OUT_DIR = Path("/data3/Qwen2.5-VL-main/src_new/tests/vis")
DATA_ROOT = Path("/data3/Qwen2.5-VL-main/data/ds_v2_full")
SAMPLES_JSONL = DATA_ROOT / "all_samples.jsonl"


def _load_samples(num: int = 3) -> List[Tuple[Dict[str, Any], List[Image.Image]]]:
    import json

    assert SAMPLES_JSONL.exists(), f"Missing {SAMPLES_JSONL}"
    results: List[Tuple[Dict[str, Any], List[Image.Image]]] = []
    with open(SAMPLES_JSONL, "r", encoding="utf-8") as f:
        pm = create_path_manager(str(DATA_ROOT))
        for line in f:
            if len(results) >= num:
                break
            line = line.strip()
            if not line:
                continue
            samp = json.loads(line)
            images: List[Image.Image] = []
            for rel in samp.get("images", [])[:1]:
                p = pm.resolve_path(rel)
                images.append(Image.open(p).convert("RGB"))
            if images:
                results.append((samp, images))
    if not results:
        raise RuntimeError("No samples found in all_samples.jsonl")
    return results


def _save_grid(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / f"{name}.jpeg",
        format="jpeg",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _canon_quads(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for o in objects:
        if "quad" in o:
            q = o["quad"]
            pts = [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
            cq = canonicalize_quad(pts)
            out.append({"quad": cq, "desc": o["desc"]})
        else:
            out.append(o)
    return out


def make_cfg(stage: str) -> AugmentationConfig:
    crit = CriteriaConfig(
        occlusion=OcclusionCriterionConfig(
            enabled=False,
            min_overlap_fraction_bbox=0.2,
            min_overlap_fraction_line=0.2,
            mask_downscale=8,
            line_width_px=2,
        )
    )
    if stage == "image_geom":
        return AugmentationConfig(
            enabled=True,
            rng_seed=123,
            apply_to_teachers=False,
            lines_policy="transform",
            debug_visualization=False,
            debug_output_dir=None,
            image_geom=ImageGeomConfig(
                rotate_deg_range=(-15.0, 15.0),
                translate_pct=0.0,
                scale_range=(1.0, 1.0),
                perspective_pct=0.0,
                crop_pct=0.0,
                multiscale_short_edges=None,
            ),
            photometric=None,
            lines=None,
            type_policies=None,
            ocr=None,
            criteria=crit,
        )
    if stage == "photometric":
        return AugmentationConfig(
            enabled=True,
            rng_seed=123,
            apply_to_teachers=False,
            lines_policy="transform",
            debug_visualization=False,
            debug_output_dir=None,
            image_geom=None,
            photometric=PhotometricConfig(
                enabled=True,
                apply_prob=1.0,
                num_ops=2,
                magnitude=0.6,
                ocr_safe_pool=True,
            ),
            lines=None,
            type_policies=None,
            ocr=None,
            criteria=crit,
        )
    if stage == "lines":
        return AugmentationConfig(
            enabled=True,
            rng_seed=123,
            apply_to_teachers=False,
            lines_policy="transform",
            debug_visualization=False,
            debug_output_dir=None,
            image_geom=None,
            photometric=None,
            lines=LineAugConfig(
                enabled=True,
                jitter_px_minmax=(2, 4),
                resample_points=32,
                min_length_px=10,
            ),
            type_policies=None,
            ocr=None,
            criteria=crit,
        )
    if stage == "criteria":
        return AugmentationConfig(
            enabled=True,
            rng_seed=123,
            apply_to_teachers=False,
            lines_policy="transform",
            debug_visualization=False,
            debug_output_dir=None,
            image_geom=None,
            photometric=None,
            lines=None,
            type_policies=None,
            ocr=None,
            criteria=CriteriaConfig(
                occlusion=OcclusionCriterionConfig(
                    enabled=True,
                    min_overlap_fraction_bbox=0.2,
                    min_overlap_fraction_line=0.2,
                    mask_downscale=4,
                    line_width_px=2,
                )
            ),
        )
    # baseline (identity)
    return AugmentationConfig(
        enabled=True,
        rng_seed=123,
        apply_to_teachers=False,
        lines_policy="transform",
        debug_visualization=False,
        debug_output_dir=None,
        image_geom=None,
        photometric=None,
        lines=None,
        type_policies=None,
        ocr=None,
        criteria=crit,
    )


def _overlay_np(img: Image.Image, objs: List[Dict[str, Any]]) -> np.ndarray:
    objs_c = _canon_quads(objs)
    over = overlay_quads(img, objs_c, color=(0, 255, 0))
    return np.array(over)


def _overlay_changed_np(
    img: Image.Image,
    before_objs: List[Dict[str, Any]],
    after_objs: List[Dict[str, Any]],
) -> np.ndarray:
    # Draw unchanged in green; changed (desc contains '有遮挡' but not in before) in red
    base = img.copy().convert("RGB")
    draw = ImageDraw.Draw(base)

    def _quad_pts(q):
        return [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7]), (q[0], q[1])]

    def _draw_obj(o, color):
        if "quad" in o:
            q = o["quad"]
            pts = _quad_pts(
                canonicalize_quad(
                    [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
                )
            )
            draw.line(pts, fill=color, width=2)
        elif "bbox_2d" in o:
            b = o["bbox_2d"]
            draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
        elif "line" in o:
            l = o["line"]
            pts = [(l[i], l[i + 1]) for i in range(0, len(l), 2)]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)

    # Build quick lookup by (geometry, first coord) to approximate matching
    def _key(o: Dict[str, Any]) -> Tuple[str, int, int]:
        if "quad" in o:
            q = o["quad"]
            return ("quad", q[0], q[1])
        if "bbox_2d" in o:
            b = o["bbox_2d"]
            return ("bbox_2d", b[0], b[1])
        if "line" in o:
            l = o["line"]
            return ("line", l[0], l[1])
        return ("other", 0, 0)

    before_keys = {_key(o) for o in before_objs}

    for o in after_objs:
        k = _key(o)
        is_changed = ("有遮挡" in str(o.get("desc", ""))) and (k not in before_keys)
        _draw_obj(o, (255, 0, 0) if is_changed else (0, 255, 0))

    return np.array(base)


def plot_stage_grid(
    stage: str, samples_and_images: List[Tuple[Dict[str, Any], List[Image.Image]]]
) -> None:
    import matplotlib.pyplot as plt

    cfg = make_cfg(stage)
    pipe = ObjectAwareAugmentationPipeline.from_config(cfg)

    n = len(samples_and_images)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = np.array([[axes[0], axes[1]]])  # normalize shape

    for row, (sample, images) in enumerate(samples_and_images):
        base_img = images[0]
        # original overlay
        orig_np = _overlay_np(base_img, sample["objects"])  # type: ignore[index]
        # augmented
        imgs_aug, samp_aug = pipe.apply(
            sample=sample.copy(), images=[base_img], sample_index=row
        )
        if stage == "criteria":
            aug_np = _overlay_changed_np(
                imgs_aug[0], sample["objects"], samp_aug["objects"]
            )  # type: ignore[index]
        else:
            aug_np = _overlay_np(imgs_aug[0], samp_aug["objects"])  # type: ignore[index]

        ax_l = axes[row, 0]
        ax_r = axes[row, 1]
        ax_l.imshow(orig_np)
        ax_l.set_title(f"{stage} — Original", fontsize=12)
        ax_l.axis("off")
        ax_r.imshow(aug_np)
        # Include a simple id tag if available
        sid = str(sample.get("images", [""])[0])
        ax_r.set_title(f"{stage} — Augmented ({Path(sid).name})", fontsize=12)
        ax_r.axis("off")

    fig.suptitle(f"Augmentation: {stage}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_grid(fig, f"aug_{stage}_grid")


def main() -> None:
    random.seed(0)
    samples_and_images = _load_samples(3)

    for stage in ("baseline", "image_geom", "photometric", "lines", "criteria"):
        plot_stage_grid(stage, samples_and_images)


if __name__ == "__main__":
    main()
