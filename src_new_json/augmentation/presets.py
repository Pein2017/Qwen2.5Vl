from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from src_new_json.config.augmentation_config import (
    AugmentationConfig,
    CriteriaConfig,
    ImageGeomConfig,
    LineAugConfig,
    OcclusionCriterionConfig,
    OCRPolicyConfig,
    PhotometricConfig,
    TypePolicyConfig,
)


PresetName = Literal["off", "conservative", "moderate", "aggressive"]


@dataclass(frozen=True)
class PresetOptions:
    preset: PresetName
    rng_seed: int = 12345
    apply_to_teachers: bool = False
    lines_policy: Literal["identity", "drop_objects", "error", "transform"] = (
        "identity"
    )
    debug_visualization: bool = False
    debug_output_dir: Optional[str] = None


_DEF_TYPE_POLICIES: Dict[str, TypePolicyConfig] = {
    "label": TypePolicyConfig(
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
        copy_paste_attempts=20,
        alpha_feather_px=1.5,
        allowed_copy_types=["label"],
    ),
    "connect_point": TypePolicyConfig(
        allow_move=True,
        allow_copy_paste=False,
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


def _disable_moves(tp: Dict[str, TypePolicyConfig]) -> Dict[str, TypePolicyConfig]:
    """Return a copy of type policies with allow_move set to False for all types."""
    out: Dict[str, TypePolicyConfig] = {}
    for k, p in tp.items():
        out[k] = TypePolicyConfig(
            allow_move=False,
            allow_copy_paste=p.allow_copy_paste,
            allow_blur=p.allow_blur,
            occluder_prob=p.occluder_prob,
            inpaint_source=p.inpaint_source,
            max_iou_with_existing=p.max_iou_with_existing,
            max_occ_fraction=p.max_occ_fraction,
            occ_grid_downscale=p.occ_grid_downscale,
            occ_margin_px=p.occ_margin_px,
            same_plane_constraint=p.same_plane_constraint,
            copy_paste_attempts=p.copy_paste_attempts,
            alpha_feather_px=p.alpha_feather_px,
            allowed_copy_types=p.allowed_copy_types,
        )
    return out


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
            image_geom=None,
            photometric=None,
            lines=None,  # keep line objects unchanged
            type_policies=_disable_moves(_DEF_TYPE_POLICIES),
            ocr=OCRPolicyConfig(
                label_protect=True, force_unreadable_on_strong_distortion=True
            ),
            criteria=None,  # occlusion removed
        )

    # Shared OCR for all non-off presets (criteria removed)
    ocr = OCRPolicyConfig(
        label_protect=True, force_unreadable_on_strong_distortion=True
    )

    if preset == "conservative":
        return AugmentationConfig(
            enabled=True,
            rng_seed=opts.rng_seed,
            apply_to_teachers=opts.apply_to_teachers,
            lines_policy=opts.lines_policy,
            debug_visualization=opts.debug_visualization,
            debug_output_dir=opts.debug_output_dir,
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
            lines=None,  # keep line objects unchanged
            type_policies=_disable_moves(_DEF_TYPE_POLICIES),
            ocr=ocr,
            criteria=None,
        )

    if preset == "moderate":
        return AugmentationConfig(
            enabled=True,
            rng_seed=opts.rng_seed,
            apply_to_teachers=opts.apply_to_teachers,
            lines_policy=opts.lines_policy,
            debug_visualization=opts.debug_visualization,
            debug_output_dir=opts.debug_output_dir,
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
            lines=None,  # keep line objects unchanged
            type_policies=_disable_moves(_DEF_TYPE_POLICIES),
            ocr=ocr,
            criteria=None,
        )

    # aggressive
    return AugmentationConfig(
        enabled=True,
        rng_seed=opts.rng_seed,
        apply_to_teachers=opts.apply_to_teachers,
        lines_policy=opts.lines_policy,
        debug_visualization=opts.debug_visualization,
        debug_output_dir=opts.debug_output_dir,
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
        lines=None,  # keep line objects unchanged
        type_policies=_disable_moves(_DEF_TYPE_POLICIES),
        ocr=ocr,
        criteria=None,
    )
