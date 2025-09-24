from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


# ---------------- Legacy / common blocks ----------------


@dataclass(frozen=True)
class AngleRotateConfig:
    """Configuration for arbitrary-angle rotation augmentation.

    All fields are explicit; callers must provide the required companions for the selected sample_mode.
    """

    # Sampling mode for the rotation angle
    sample_mode: Literal["fixed", "uniform_range", "set"]

    # Used when sample_mode == "fixed"
    fixed_angle_deg: Optional[float]

    # Used when sample_mode == "uniform_range"
    angle_min_deg: Optional[float]
    angle_max_deg: Optional[float]

    # Used when sample_mode == "set"
    angles_set_deg: Optional[List[float]]

    # Canvas behavior and image resampling
    expand: bool
    interpolation: Literal["nearest", "bilinear", "bicubic"]
    fill_color: Optional[Tuple[int, int, int]]  # RGB


@dataclass(frozen=True)
class AlbumentationsRandAugConfig:
    """Albumentations-based RandomAug photometric policy (image-only)."""

    enabled: bool
    apply_prob: float
    num_ops: int  # e.g., 2 or 3
    magnitude: float  # 0..1 for scaling ranges (coarse control)
    safe_ops_only: bool  # restrict to safe photometric transforms (no heavy artifacts)


@dataclass(frozen=True)
class ObjectLocalAffineConfig:
    """Per-object small rotation + translation in image coordinates (coords-only).

    For v1 we do not change pixels; we constrain to avoid overlaps via IoU threshold.
    Applies to bbox_2d and quad; lines are skipped.
    """

    enabled: bool
    per_object_prob: float  # probability to attempt transform per object
    max_rotation_deg: float  # maximum absolute rotation (degrees)
    translate_px: int  # max absolute translation (pixels) along x/y
    avoid_overlap: bool  # if True, resample to avoid IoU>threshold
    iou_thresh: float  # IoU threshold to avoid when avoid_overlap=True
    max_resample: int  # max resampling attempts per object
    apply_pixels: bool = (
        False  # if True, also rotate/paste pixels (masked) in addition to coords
    )


@dataclass(frozen=True)
class ObjectCopyPasteConfig:
    """Duplicate selected objects and paste into relatively empty regions.

    Does not erase the source object; creates new objects and updates pixels.
    Works for bbox_2d and quad; lines are not copied.
    """

    enabled: bool
    per_object_prob: float
    num_copies_per_object: int
    translate_px: int
    rotation_jitter_deg: float
    scale_jitter_min: float
    scale_jitter_max: float
    occ_grid_downscale: int
    occ_margin_px: int
    max_occ_fraction: float
    max_iou_with_existing: float
    attempts: int
    allowed_types: Optional[List[str]]
    # Soft blending radius (in pixels) applied to the alpha mask of pasted patches
    # to reduce hard seams/halo artifacts. Set 0.0 to disable.
    alpha_feather_px: float = 0.0


@dataclass(frozen=True)
class ObjectBlurConfig:
    """Object-level blur (虚化) effect using a mask over the object geometry.

    Applies blur only within the object region; supports gaussian/box blur.
    """

    enabled: bool
    per_object_prob: float
    blur_type: Literal["gaussian", "box"]
    radius_min: float
    radius_max: float


@dataclass(frozen=True)
class RandAugPoolConfig:
    """Randomly apply a subset of object-level augmentations from a pool."""

    enabled: bool
    apply_prob: float
    num_ops: int
    include_object_affine: bool
    include_object_copy_paste: bool
    include_object_blur: bool


@dataclass(frozen=True)
class OcclusionCriterionConfig:
    """Annotate occlusions when occluders overlap targets."""

    enabled: bool
    min_overlap_fraction_bbox: float  # fraction of polygon area overlapped
    min_overlap_fraction_line: float  # fraction of line pixels overlapped
    mask_downscale: int  # performance knob for mask-based intersection
    line_width_px: int  # stroke width when rasterizing lines for overlap


@dataclass(frozen=True)
class CriteriaConfig:
    occlusion: Optional[OcclusionCriterionConfig] = None


# ---------------- Object-aware new blocks ----------------


@dataclass(frozen=True)
class ImageGeomConfig:
    rotate_deg_range: Tuple[float, float]
    translate_pct: float
    scale_range: Tuple[float, float]
    perspective_pct: float
    crop_pct: float
    multiscale_short_edges: Optional[List[int]]


@dataclass(frozen=True)
class SmartResizeConfig:
    enabled: bool
    factor: int
    min_pixels: int
    max_pixels: int
    max_ratio: float


@dataclass(frozen=True)
class PhotometricConfig:
    enabled: bool
    apply_prob: float
    num_ops: int
    magnitude: float
    ocr_safe_pool: bool


@dataclass(frozen=True)
class LineAugConfig:
    enabled: bool
    jitter_px_minmax: Tuple[int, int]
    resample_points: int
    min_length_px: int


@dataclass(frozen=True)
class TypePolicyConfig:
    allow_move: bool
    allow_copy_paste: bool
    allow_blur: bool
    occluder_prob: float
    inpaint_source: bool
    max_iou_with_existing: Optional[float] = None
    max_occ_fraction: Optional[float] = None
    occ_grid_downscale: Optional[int] = None
    occ_margin_px: Optional[int] = None
    same_plane_constraint: bool = True
    copy_paste_attempts: Optional[int] = None
    alpha_feather_px: Optional[float] = None
    allowed_copy_types: Optional[List[str]] = None


@dataclass(frozen=True)
class OCRPolicyConfig:
    label_protect: bool
    force_unreadable_on_strong_distortion: bool


@dataclass(frozen=True)
class AugmentationConfig:
    """Top-level augmentation configuration.

    This object is attached to the main training Config as `augmentation` when enabled.
    """

    enabled: bool
    rng_seed: int
    apply_to_teachers: bool
    lines_policy: Literal["identity", "drop_objects", "error", "transform"]

    debug_output_dir: Optional[str]

    # Criteria/guards
    criteria: Optional[CriteriaConfig] = None
    debug_visualization: bool = False

    # New object-aware blocks (optional)
    smart_resize: Optional[SmartResizeConfig] = None
    image_geom: Optional[ImageGeomConfig] = None
    photometric: Optional[PhotometricConfig] = None
    lines: Optional[LineAugConfig] = None
    type_policies: Optional[Dict[str, TypePolicyConfig]] = None
    ocr: Optional[OCRPolicyConfig] = None


# ---------------- Validators ----------------


def validate_image_geom_config(cfg: ImageGeomConfig) -> None:
    if cfg.rotate_deg_range[0] > cfg.rotate_deg_range[1]:
        raise ValueError("ImageGeomConfig.rotate_deg_range min must be <= max")
    if cfg.translate_pct < 0:
        raise ValueError("ImageGeomConfig.translate_pct must be >=0")
    if cfg.scale_range[0] <= 0 or cfg.scale_range[0] > cfg.scale_range[1]:
        raise ValueError("ImageGeomConfig.scale_range must satisfy 0<min<=max")
    if cfg.perspective_pct < 0 or cfg.crop_pct < 0:
        raise ValueError("ImageGeomConfig perspective/crop must be >=0")


def validate_smart_resize_config(cfg: SmartResizeConfig) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError("SmartResizeConfig.enabled must be bool")
    if cfg.factor <= 0:
        raise ValueError("SmartResizeConfig.factor must be positive")
    if cfg.min_pixels <= 0 or cfg.max_pixels <= 0:
        raise ValueError("SmartResizeConfig pixel thresholds must be positive")
    if cfg.min_pixels > cfg.max_pixels:
        raise ValueError("SmartResizeConfig.min_pixels must be <= max_pixels")
    if cfg.max_ratio <= 0:
        raise ValueError("SmartResizeConfig.max_ratio must be positive")


def validate_photometric2_config(cfg: PhotometricConfig) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError("PhotometricConfig.enabled must be bool")
    if not (0.0 <= cfg.apply_prob <= 1.0):
        raise ValueError("PhotometricConfig.apply_prob must be in [0,1]")
    if cfg.num_ops < 0:
        raise ValueError("PhotometricConfig.num_ops must be >=0")
    if not (0.0 <= cfg.magnitude <= 1.0):
        raise ValueError("PhotometricConfig.magnitude must be in [0,1]")


def validate_line_aug_config(cfg: LineAugConfig) -> None:
    mn, mx = cfg.jitter_px_minmax
    if mn < 0 or mx < 0 or mn > mx:
        raise ValueError(
            "LineAugConfig.jitter_px_minmax must be non-negative and min<=max"
        )
    if cfg.resample_points < 2:
        raise ValueError("LineAugConfig.resample_points must be >=2")
    if cfg.min_length_px < 0:
        raise ValueError("LineAugConfig.min_length_px must be >=0")


def validate_type_policies(tp: Dict[str, TypePolicyConfig]) -> None:
    for t, p in tp.items():
        if p.occluder_prob < 0 or p.occluder_prob > 1:
            raise ValueError(f"TypePolicyConfig.occluder_prob for {t} must be in [0,1]")
        if p.max_iou_with_existing is not None and not (
            0.0 <= p.max_iou_with_existing <= 1.0
        ):
            raise ValueError(
                f"TypePolicyConfig.max_iou_with_existing for {t} must be in [0,1]"
            )
        if p.max_occ_fraction is not None and not (0.0 <= p.max_occ_fraction <= 1.0):
            raise ValueError(
                f"TypePolicyConfig.max_occ_fraction for {t} must be in [0,1]"
            )
        if p.occ_grid_downscale is not None and p.occ_grid_downscale < 1:
            raise ValueError(f"TypePolicyConfig.occ_grid_downscale for {t} must be >=1")
        if p.occ_margin_px is not None and p.occ_margin_px < 0:
            raise ValueError(f"TypePolicyConfig.occ_margin_px for {t} must be >=0")
        if p.copy_paste_attempts is not None and p.copy_paste_attempts < 1:
            raise ValueError(
                f"TypePolicyConfig.copy_paste_attempts for {t} must be >=1"
            )
        if p.alpha_feather_px is not None and p.alpha_feather_px < 0:
            raise ValueError(f"TypePolicyConfig.alpha_feather_px for {t} must be >=0")


def validate_ocr_policy(cfg: OCRPolicyConfig) -> None:
    if not isinstance(cfg.label_protect, bool):
        raise ValueError("OCRPolicyConfig.label_protect must be bool")
    if not isinstance(cfg.force_unreadable_on_strong_distortion, bool):
        raise ValueError(
            "OCRPolicyConfig.force_unreadable_on_strong_distortion must be bool"
        )


def validate_criteria_config(cfg: CriteriaConfig) -> None:
    if cfg.occlusion is not None:
        occ = cfg.occlusion
        if not isinstance(occ.enabled, bool):
            raise ValueError("OcclusionCriterionConfig.enabled must be bool")
        if not (0.0 <= occ.min_overlap_fraction_bbox <= 1.0):
            raise ValueError("min_overlap_fraction_bbox must be in [0,1]")
        if not (0.0 <= occ.min_overlap_fraction_line <= 1.0):
            raise ValueError("min_overlap_fraction_line must be in [0,1]")
        if occ.mask_downscale < 1:
            raise ValueError("mask_downscale must be >=1")
        if occ.line_width_px < 1:
            raise ValueError("line_width_px must be >=1")


def validate_augmentation_config(cfg: AugmentationConfig) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError(
            f"AugmentationConfig.enabled must be bool, got {type(cfg.enabled)}: {cfg.enabled!r}"
        )
    if not isinstance(cfg.rng_seed, int):
        raise ValueError(
            f"AugmentationConfig.rng_seed must be int, got {type(cfg.rng_seed)}: {cfg.rng_seed!r}"
        )
    if not isinstance(cfg.apply_to_teachers, bool):
        raise ValueError(
            f"AugmentationConfig.apply_to_teachers must be bool, got {type(cfg.apply_to_teachers)}: {cfg.apply_to_teachers!r}"
        )
    if cfg.lines_policy not in ("identity", "drop_objects", "error", "transform"):
        raise ValueError(
            f"AugmentationConfig.lines_policy must be one of identity|drop_objects|error|transform, got {cfg.lines_policy!r}"
        )

    if not isinstance(cfg.debug_visualization, bool):
        raise ValueError(
            f"AugmentationConfig.debug_visualization must be bool, got {type(cfg.debug_visualization)}: {cfg.debug_visualization!r}"
        )

    if cfg.criteria is not None:
        validate_criteria_config(cfg.criteria)
    if cfg.smart_resize is not None:
        validate_smart_resize_config(cfg.smart_resize)
    if cfg.image_geom is not None:
        validate_image_geom_config(cfg.image_geom)
    if cfg.photometric is not None:
        validate_photometric2_config(cfg.photometric)
    if cfg.lines is not None:
        validate_line_aug_config(cfg.lines)
    if cfg.type_policies is not None:
        validate_type_policies(cfg.type_policies)
    if cfg.ocr is not None:
        validate_ocr_policy(cfg.ocr)
