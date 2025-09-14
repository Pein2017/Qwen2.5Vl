from __future__ import annotations

import random
from typing import List

import albumentations as A
import numpy as np
from PIL import Image

from src_new_json.config.augmentation_config import PhotometricConfig


def _build_compose(cfg: PhotometricConfig, ocr_guard: bool):
    if not cfg.enabled:
        return None
    m = float(cfg.magnitude)

    # Safe ops pool
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
    # Extended ops include JPEG/noise/blur/dropout but reduced if ocr_guard
    extended_ops = safe_ops + [
        A.ImageCompression(
            quality_lower=max(60, int(92 - 35 * m)), quality_upper=96, p=1.0
        ),
        A.GaussNoise(var_limit=(5.0, max(10.0, 40.0 * m)), p=1.0),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=1.0,
        ),
        # Narrow dropout when not OCR-guarded. If OCR-guarded, we will not include dropout at all.
        A.CoarseDropout(max_holes=int(2 + 4 * m), max_height=2, max_width=2, p=1.0),
    ]
    ops = safe_ops if (ocr_guard and cfg.ocr_safe_pool) else extended_ops
    return A.Compose(
        [A.SomeOf(ops, n=max(0, int(cfg.num_ops)), replace=False, p=cfg.apply_prob)],
        p=1.0,
    )


def apply_photometric(
    images: List[Image.Image],
    cfg: PhotometricConfig,
    rng: random.Random,
    ocr_guard: bool,
) -> List[Image.Image]:
    compose = _build_compose(cfg, ocr_guard)
    if compose is None:
        return images
    out: List[Image.Image] = []

    # Derive a per-sample seed from rng; apply to numpy + random for determinism
    np_seed = rng.randrange(0, 2**31)

    # Save states to avoid leaking to caller
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(np_seed)
        np.random.seed(np_seed)
        for img in images:
            img_np = np.array(img)
            res = compose(image=img_np)
            out.append(Image.fromarray(res["image"]))
    finally:
        # Restore caller RNG states
        random.setstate(py_state)
        np.random.set_state(np_state)

    return out
