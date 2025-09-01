from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src_new.config.augmentation_config import (
    AugmentationConfig,
    validate_augmentation_config,
)

from . import image_ops, line_ops
from . import photometric as photometric_ops
from .validators import apply_occlusion_criterion


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

        # 1) image_geom
        if self.config.image_geom is not None:
            out_images, out_sample = image_ops.apply_image_geom(
                images=out_images,
                sample=out_sample,
                cfg=self.config.image_geom,
                rng=rng,
            )

        # 2) photometric (OCR-safe when label present and enabled)
        ocr_guard = False
        if self.config.ocr is not None and self.config.ocr.label_protect:
            ocr_guard = any(("标签" in o["desc"]) for o in out_sample["objects"])
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

        # 4) occlusion criteria (monitoring only; no desc mutation)
        if (
            self.config.criteria is not None
            and self.config.criteria.occlusion is not None
        ):
            out_sample = apply_occlusion_criterion(out_sample, self.config.criteria)

        return out_images, out_sample
