from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from src_new.config.augmentation_config import (
    AugmentationConfig,
    CriteriaConfig,
    ImageGeomConfig,
    LineAugConfig,
    OcclusionCriterionConfig,
    OCRPolicyConfig,
    PhotometricConfig,
    SmartResizeConfig,
    TypePolicyConfig,
)


PresetName = Literal["off", "conservative", "moderate", "aggressive"]


@dataclass(frozen=True)
class PresetOptions:
    preset: PresetName
    rng_seed: int = 12345
    apply_to_teachers: bool = False
    lines_policy: Literal["identity", "drop_objects", "error", "transform"] = (
        "transform"
    )
    debug_visualization: bool = False
    debug_output_dir: Optional[str] = None
    smart_resize_enabled: Optional[bool] = None
    smart_resize_factor: Optional[int] = None
    smart_resize_min_pixels: Optional[int] = None
    smart_resize_max_pixels: Optional[int] = None
    smart_resize_max_ratio: Optional[float] = None


_DEF_TYPE_POLICIES: Dict[str, TypePolicyConfig] = {
    "label": TypePolicyConfig(
        allow_move=True,
        allow_copy_paste=True,
        allow_blur=False,
        occluder_prob=0.2,
        inpaint_source=True,
        max_iou_with_existing=0.25,
        max_occ_fraction=0.15,
        occ_grid_downscale=8,
        occ_margin_px=6,
        same_plane_constraint=True,
        copy_paste_attempts=20,
        alpha_feather_px=1.5,
        allowed_copy_types=["label"],
    ),
    "connect_point": TypePolicyConfig(
        allow_move=True,
        allow_copy_paste=True,
        allow_blur=True,
        occluder_prob=0.2,
        inpaint_source=True,
        max_iou_with_existing=0.25,
        max_occ_fraction=0.15,
        occ_grid_downscale=8,
        occ_margin_px=6,
        same_plane_constraint=True,
        copy_paste_attempts=20,
        alpha_feather_px=1.5,
        allowed_copy_types=["connect_point"],
    ),
    "bbu": TypePolicyConfig(
        allow_move=False,
        allow_copy_paste=False,
        allow_blur=False,
        occluder_prob=0.1,
        inpaint_source=False,
        same_plane_constraint=True,
    ),
    "bbu_shield": TypePolicyConfig(
        allow_move=True,
        allow_copy_paste=False,
        allow_blur=False,
        occluder_prob=0.2,
        inpaint_source=True,
        max_iou_with_existing=0.25,
        max_occ_fraction=0.15,
        occ_grid_downscale=8,
        occ_margin_px=6,
        same_plane_constraint=True,
    ),
    "fiber": TypePolicyConfig(
        allow_move=False,
        allow_copy_paste=False,
        allow_blur=False,
        occluder_prob=0.15,
        inpaint_source=False,
        same_plane_constraint=True,
    ),
    "wire": TypePolicyConfig(
        allow_move=False,
        allow_copy_paste=False,
        allow_blur=False,
        occluder_prob=0.15,
        inpaint_source=False,
        same_plane_constraint=True,
    ),
}


def _build_smart_resize_from_options(opts: PresetOptions) -> SmartResizeConfig:
    if (
        opts.smart_resize_enabled is None
        or opts.smart_resize_factor is None
        or opts.smart_resize_min_pixels is None
        or opts.smart_resize_max_pixels is None
        or opts.smart_resize_max_ratio is None
    ):
        raise ValueError(
            "Preset '{}' requires smart_resize configuration (enabled, factor, min_pixels, max_pixels, "
            "max_ratio). Provide these under augmentation.smart_resize in YAML.".format(opts.preset)
        )
    return SmartResizeConfig(
        enabled=bool(opts.smart_resize_enabled),
        factor=int(opts.smart_resize_factor),
        min_pixels=int(opts.smart_resize_min_pixels),
        max_pixels=int(opts.smart_resize_max_pixels),
        max_ratio=float(opts.smart_resize_max_ratio),
    )


def build_augmentation_config_from_preset(opts: PresetOptions) -> AugmentationConfig:
    preset = opts.preset
    if preset == "off":
        return AugmentationConfig(
            enabled=False,
            rng_seed=opts.rng_seed,
            apply_to_teachers=opts.apply_to_teachers,
            lines_policy=opts.lines_policy,
            debug_visualization=opts.debug_visualization,
            debug_output_dir=opts.debug_output_dir,
            smart_resize=None,
            image_geom=None,
            photometric=None,
            lines=None,
            type_policies=_DEF_TYPE_POLICIES,
            ocr=OCRPolicyConfig(
                label_protect=True, force_unreadable_on_strong_distortion=True
            ),
            criteria=CriteriaConfig(
                occlusion=OcclusionCriterionConfig(
                    enabled=True,
                    min_overlap_fraction_bbox=0.2,
                    min_overlap_fraction_line=0.2,
                    mask_downscale=8,
                    line_width_px=2,
                )
            ),
        )

    # Shared OCR + criteria for all non-off presets
    ocr = OCRPolicyConfig(
        label_protect=True, force_unreadable_on_strong_distortion=True
    )
    crit = CriteriaConfig(
        occlusion=OcclusionCriterionConfig(
            enabled=True,
            min_overlap_fraction_bbox=0.2,
            min_overlap_fraction_line=0.2,
            mask_downscale=8,
            line_width_px=2,
        )
    )

    if preset == "conservative":
        return AugmentationConfig(
            enabled=True,
            rng_seed=opts.rng_seed,
            apply_to_teachers=opts.apply_to_teachers,
            lines_policy=opts.lines_policy,
            debug_visualization=opts.debug_visualization,
            debug_output_dir=opts.debug_output_dir,
            smart_resize=_build_smart_resize_from_options(opts),
            image_geom=ImageGeomConfig(
                rotate_deg_range=(-8.0, 8.0),
                translate_pct=0.10,
                scale_range=(0.95, 1.05),
                perspective_pct=0.02,
                crop_pct=0.03,
                multiscale_short_edges=[896, 1024, 1280],
            ),
            photometric=PhotometricConfig(
                enabled=True,
                apply_prob=0.8,
                num_ops=1,
                magnitude=0.4,
                ocr_safe_pool=True,
            ),
            lines=LineAugConfig(
                enabled=True,
                jitter_px_minmax=(2, 4),
                resample_points=32,
                min_length_px=24,
            ),
            type_policies=_DEF_TYPE_POLICIES,
            ocr=ocr,
            criteria=crit,
        )

    if preset == "moderate":
        return AugmentationConfig(
            enabled=True,
            rng_seed=opts.rng_seed,
            apply_to_teachers=opts.apply_to_teachers,
            lines_policy=opts.lines_policy,
            debug_visualization=opts.debug_visualization,
            debug_output_dir=opts.debug_output_dir,
            smart_resize=_build_smart_resize_from_options(opts),
            image_geom=ImageGeomConfig(
                rotate_deg_range=(-10.0, 10.0),
                translate_pct=0.10,
                scale_range=(0.9, 1.1),
                perspective_pct=0.03,
                crop_pct=0.05,
                multiscale_short_edges=[896, 1024, 1280],
            ),
            photometric=PhotometricConfig(
                enabled=True,
                apply_prob=0.8,
                num_ops=2,
                magnitude=0.5,
                ocr_safe_pool=True,
            ),
            lines=LineAugConfig(
                enabled=True,
                jitter_px_minmax=(2, 6),
                resample_points=32,
                min_length_px=24,
            ),
            type_policies=_DEF_TYPE_POLICIES,
            ocr=ocr,
            criteria=crit,
        )

    # aggressive
    return AugmentationConfig(
        enabled=True,
        rng_seed=opts.rng_seed,
        apply_to_teachers=opts.apply_to_teachers,
        lines_policy=opts.lines_policy,
        debug_visualization=opts.debug_visualization,
        debug_output_dir=opts.debug_output_dir,
        smart_resize=_build_smart_resize_from_options(opts),
        image_geom=ImageGeomConfig(
            rotate_deg_range=(-15.0, 15.0),
            translate_pct=0.12,
            scale_range=(0.85, 1.15),
            perspective_pct=0.05,
            crop_pct=0.05,
            multiscale_short_edges=[896, 1024, 1280],
        ),
        photometric=PhotometricConfig(
            enabled=True, apply_prob=0.8, num_ops=3, magnitude=0.7, ocr_safe_pool=True
        ),
        lines=LineAugConfig(
            enabled=True, jitter_px_minmax=(3, 7), resample_points=32, min_length_px=24
        ),
        type_policies=_DEF_TYPE_POLICIES,
        ocr=ocr,
        criteria=crit,
    )
