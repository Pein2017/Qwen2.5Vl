from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple


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
class ColorJitterConfig:
    """Image-only color jitter configuration (does not change coordinates)."""

    enabled: bool
    apply_prob: float
    brightness: Optional[Tuple[float, float]]
    contrast: Optional[Tuple[float, float]]
    saturation: Optional[Tuple[float, float]]
    sharpness: Optional[Tuple[float, float]]
    order: Optional[List[str]]


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


@dataclass(frozen=True)
class AugmentationConfig:
    """Top-level augmentation configuration.

    This object is attached to the main training Config as `augmentation` when enabled.
    """

    enabled: bool
    rng_seed: int
    apply_to_teachers: bool
    lines_policy: Literal["identity", "drop_objects", "error", "transform"]

    debug_visualization: bool
    debug_output_dir: Optional[str]

    # Single op plan: arbitrary-angle rotation (required)
    op: AngleRotateConfig

    # Optional image-only photometric augmentations
    color_jitter: Optional[ColorJitterConfig] = (
        None  # legacy manual jitter (deprecated in favor of Albumentations)
    )
    albumentations_rand: Optional[AlbumentationsRandAugConfig] = None

    # Optional per-object local affine (coords-only)
    object_local_affine: Optional[ObjectLocalAffineConfig] = None

    # Optional object copy-paste
    object_copy_paste: Optional[ObjectCopyPasteConfig] = None

    # Optional object blur
    object_blur: Optional[ObjectBlurConfig] = None

    # Optional randomized pool
    rand_pool: Optional[RandAugPoolConfig] = None

    # Criteria/guards
    criteria: Optional[CriteriaConfig] = None


def validate_angle_rotate_config(cfg: AngleRotateConfig) -> None:
    mode = cfg.sample_mode

    # Helper for near-zero test
    def is_near_zero(x: float) -> bool:
        return abs(float(x)) < 1e-6

    if mode == "fixed":
        if cfg.fixed_angle_deg is None:
            raise ValueError(
                "AngleRotateConfig: fixed_angle_deg must be set when sample_mode='fixed'"
            )
    elif mode == "uniform_range":
        if cfg.angle_min_deg is None or cfg.angle_max_deg is None:
            raise ValueError(
                "AngleRotateConfig: angle_min_deg and angle_max_deg must be set when sample_mode='uniform_range'"
            )
        if cfg.angle_min_deg > cfg.angle_max_deg:
            raise ValueError(
                f"AngleRotateConfig: angle_min_deg ({cfg.angle_min_deg}) must be <= angle_max_deg ({cfg.angle_max_deg})"
            )
    elif mode == "set":
        if not cfg.angles_set_deg or len(cfg.angles_set_deg) == 0:
            raise ValueError(
                "AngleRotateConfig: angles_set_deg must be a non-empty list when sample_mode='set'"
            )
    else:
        raise ValueError(f"AngleRotateConfig: unsupported sample_mode: {mode}")

    if cfg.interpolation not in ("nearest", "bilinear", "bicubic"):
        raise ValueError(
            f"AngleRotateConfig: unsupported interpolation '{cfg.interpolation}'. Choose from nearest|bilinear|bicubic"
        )

    # Fail-fast: forbid keep-size for any non-zero rotation (prevents boundary clipping)
    if cfg.expand is False:
        if (
            mode == "fixed"
            and cfg.fixed_angle_deg is not None
            and not is_near_zero(cfg.fixed_angle_deg)
        ):
            raise ValueError(
                "AngleRotateConfig: expand=False with non-zero fixed_angle is not allowed. "
                "Set expand=True or use fixed_angle_deg=0.0."
            )
        if mode == "uniform_range" and (
            (cfg.angle_min_deg is not None and not is_near_zero(cfg.angle_min_deg))
            or (cfg.angle_max_deg is not None and not is_near_zero(cfg.angle_max_deg))
        ):
            raise ValueError(
                "AngleRotateConfig: expand=False with a non-zero angle range is not allowed. "
                "Set expand=True or restrict the range to 0.0."
            )
        if mode == "set" and any(
            not is_near_zero(a) for a in (cfg.angles_set_deg or [])
        ):
            raise ValueError(
                "AngleRotateConfig: expand=False with a set containing non-zero angles is not allowed. "
                "Set expand=True or include only 0.0."
            )


def validate_color_jitter_config(cfg: ColorJitterConfig) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError(
            f"ColorJitterConfig.enabled must be bool, got {type(cfg.enabled)}: {cfg.enabled!r}"
        )
    if not (0.0 <= cfg.apply_prob <= 1.0):
        raise ValueError(
            f"ColorJitterConfig.apply_prob must be in [0,1], got {cfg.apply_prob}"
        )

    def _validate_range(name: str, rng: Optional[Tuple[float, float]]):
        if rng is None:
            return
        if not isinstance(rng, tuple) or len(rng) != 2:
            raise ValueError(
                f"ColorJitterConfig.{name} must be a tuple(min,max), got {rng}"
            )
        lo, hi = rng
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError(f"ColorJitterConfig.{name} invalid range: {rng}")

    _validate_range("brightness", cfg.brightness)
    _validate_range("contrast", cfg.contrast)
    _validate_range("saturation", cfg.saturation)
    _validate_range("sharpness", cfg.sharpness)

    if cfg.order is not None:
        valid = {"brightness", "contrast", "saturation", "sharpness"}
        if any(x not in valid for x in cfg.order):
            raise ValueError(
                f"ColorJitterConfig.order contains invalid transform; valid: {sorted(valid)}"
            )


def validate_albumentations_rand_config(cfg: AlbumentationsRandAugConfig) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError(
            f"AlbumentationsRandAugConfig.enabled must be bool, got {type(cfg.enabled)}: {cfg.enabled!r}"
        )
    if not (0.0 <= cfg.apply_prob <= 1.0):
        raise ValueError(
            f"AlbumentationsRandAugConfig.apply_prob must be in [0,1], got {cfg.apply_prob}"
        )
    if not isinstance(cfg.num_ops, int) or cfg.num_ops < 1:
        raise ValueError(
            f"AlbumentationsRandAugConfig.num_ops must be >=1, got {cfg.num_ops}"
        )
    if not (0.0 <= cfg.magnitude <= 1.0):
        raise ValueError(
            f"AlbumentationsRandAugConfig.magnitude must be in [0,1], got {cfg.magnitude}"
        )


def validate_object_copy_paste_config(cfg: ObjectCopyPasteConfig) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError("ObjectCopyPasteConfig.enabled must be bool")
    if not (0.0 <= cfg.per_object_prob <= 1.0):
        raise ValueError("ObjectCopyPasteConfig.per_object_prob must be in [0,1]")
    if cfg.num_copies_per_object < 1:
        raise ValueError("ObjectCopyPasteConfig.num_copies_per_object must be >=1")
    if cfg.translate_px < 0:
        raise ValueError("ObjectCopyPasteConfig.translate_px must be >=0")
    if cfg.rotation_jitter_deg < 0:
        raise ValueError("ObjectCopyPasteConfig.rotation_jitter_deg must be >=0")
    if not (0.0 < cfg.scale_jitter_min <= cfg.scale_jitter_max):
        raise ValueError("scale_jitter_min/max must satisfy 0 < min <= max")
    if cfg.occ_grid_downscale < 1:
        raise ValueError("occ_grid_downscale must be >=1")
    if cfg.occ_margin_px < 0:
        raise ValueError("occ_margin_px must be >=0")
    if not (0.0 <= cfg.max_occ_fraction <= 1.0):
        raise ValueError("max_occ_fraction must be in [0,1]")
    if not (0.0 <= cfg.max_iou_with_existing <= 1.0):
        raise ValueError("max_iou_with_existing must be in [0,1]")
    if cfg.attempts < 1:
        raise ValueError("attempts must be >=1")


def validate_object_blur_config(cfg: ObjectBlurConfig) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError("ObjectBlurConfig.enabled must be bool")
    if not (0.0 <= cfg.per_object_prob <= 1.0):
        raise ValueError("ObjectBlurConfig.per_object_prob must be in [0,1]")
    if cfg.blur_type not in ("gaussian", "box"):
        raise ValueError("ObjectBlurConfig.blur_type must be 'gaussian' or 'box'")
    if cfg.radius_min <= 0 or cfg.radius_max <= 0 or cfg.radius_min > cfg.radius_max:
        raise ValueError("ObjectBlurConfig.radius_min/max must be >0 and min<=max")


def validate_rand_pool_config(
    cfg: RandAugPoolConfig, aug_cfg: "AugmentationConfig"
) -> None:
    if not isinstance(cfg.enabled, bool):
        raise ValueError("RandAugPoolConfig.enabled must be bool")
    if not (0.0 <= cfg.apply_prob <= 1.0):
        raise ValueError("RandAugPoolConfig.apply_prob must be in [0,1]")
    includes = [
        cfg.include_object_affine,
        cfg.include_object_copy_paste,
        cfg.include_object_blur,
    ]
    if cfg.num_ops < 1:
        raise ValueError("RandAugPoolConfig.num_ops must be >=1")
    if not any(includes):
        raise ValueError("RandAugPoolConfig: at least one include_* must be True")
    if cfg.num_ops > sum(1 for x in includes if x):
        raise ValueError(
            "RandAugPoolConfig.num_ops cannot exceed number of included ops"
        )
    # Ensure required sub-configs exist when included
    if cfg.include_object_affine and aug_cfg.object_local_affine is None:
        raise ValueError(
            "RandAugPoolConfig requires object_local_affine when include_object_affine=True"
        )
    if cfg.include_object_copy_paste and aug_cfg.object_copy_paste is None:
        raise ValueError(
            "RandAugPoolConfig requires object_copy_paste when include_object_copy_paste=True"
        )
    if cfg.include_object_blur and aug_cfg.object_blur is None:
        raise ValueError(
            "RandAugPoolConfig requires object_blur when include_object_blur=True"
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

    validate_angle_rotate_config(cfg.op)
    if cfg.color_jitter is not None:
        validate_color_jitter_config(cfg.color_jitter)
    if cfg.albumentations_rand is not None:
        validate_albumentations_rand_config(cfg.albumentations_rand)
    if cfg.object_local_affine is not None:
        ola = cfg.object_local_affine
        if not (0.0 <= ola.per_object_prob <= 1.0):
            raise ValueError(
                f"ObjectLocalAffine.per_object_prob must be in [0,1], got {ola.per_object_prob}"
            )
        if ola.max_rotation_deg < 0:
            raise ValueError("ObjectLocalAffine.max_rotation_deg must be >= 0")
        if ola.translate_px < 0:
            raise ValueError("ObjectLocalAffine.translate_px must be >= 0")
        if ola.avoid_overlap and not (0.0 <= ola.iou_thresh <= 1.0):
            raise ValueError("ObjectLocalAffine.iou_thresh must be in [0,1]")
        if ola.max_resample < 0:
            raise ValueError("ObjectLocalAffine.max_resample must be >= 0")
    if cfg.object_copy_paste is not None:
        validate_object_copy_paste_config(cfg.object_copy_paste)
    if cfg.object_blur is not None:
        validate_object_blur_config(cfg.object_blur)
    if cfg.rand_pool is not None and cfg.rand_pool.enabled:
        validate_rand_pool_config(cfg.rand_pool, cfg)
    if cfg.criteria is not None:
        validate_criteria_config(cfg.criteria)
