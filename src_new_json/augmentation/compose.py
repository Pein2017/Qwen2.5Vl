from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src_new_json.config.augmentation_config import (
    AugmentationConfig,
    validate_augmentation_config,
)

from . import image_ops, line_ops
from . import photometric as photometric_ops


@dataclass(frozen=True)
class ObjectAwareAugmentationPipeline:
    config: AugmentationConfig

    @staticmethod
    def from_config(config: AugmentationConfig) -> "ObjectAwareAugmentationPipeline":
        validate_augmentation_config(config)
        return ObjectAwareAugmentationPipeline(config=config)

    def apply(
        self,
        sample: Dict[str, Any],
        images: List[Image.Image],
        sample_index: int,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        if not images:
            return images, sample
        base_seed = int(self.config.rng_seed)
        if rng is None:
            rng = random.Random(base_seed ^ (sample_index & 0x7FFFFFFF))

        out_images = images
        out_sample = dict(sample)

        # 0) smart resize
        if (
            self.config.smart_resize is not None
            and self.config.smart_resize.enabled
        ):
            out_images, out_sample = image_ops.apply_smart_resize(
                images=out_images,
                sample=out_sample,
                cfg=self.config.smart_resize,
            )

        # 1) image_geom
        if self.config.image_geom is not None:
            out_images, out_sample = image_ops.apply_image_geom(
                images=out_images,
                sample=out_sample,
                cfg=self.config.image_geom,
                rng=rng,
            )

        # 2) photometric (OCR/brand/line-safe when applicable)
        ocr_guard = False
        if self.config.ocr is not None and self.config.ocr.label_protect:
            objs = out_sample["objects"]
            has_label = any((("desc" in o) and isinstance(o["desc"], str) and ("标签" in o["desc"])) for o in objs)
            # Brand text often appears on BBU/shield; guard when vendor tokens or '品牌' are mentioned
            descs = [o["desc"] for o in objs if (isinstance(o, dict) and ("desc" in o) and isinstance(o["desc"], str))]
            brand_tokens = ("华为", "中兴", "爱立信", "品牌")
            brand_mention = any(any(bt in d for bt in brand_tokens) for d in descs)
            # Thin-line geometry (fiber/wire) is sensitive to blur/dropout
            has_thin_lines = any(("line" in o) for o in objs)
            ocr_guard = has_label or brand_mention or has_thin_lines
        if self.config.photometric is not None and self.config.photometric.enabled:
            out_images = photometric_ops.apply_photometric(
                images=out_images,
                cfg=self.config.photometric,
                rng=rng,
                ocr_guard=ocr_guard,
            )

        # 3) line ops (fiber/wire)
        if self.config.lines is not None and self.config.lines.enabled:
            out_sample = line_ops.apply_line_ops(
                sample=out_sample, cfg=self.config.lines, rng=rng
            )

        # Skipping object-level move/copy/blur in this initial commit.
        # TODO: implement object_ops.move/copy_paste/blur with per-type policies.

        # 4) occlusion criteria removed (no-op)
        return out_images, out_sample
