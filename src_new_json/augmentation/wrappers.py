from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from src_new_json.config.augmentation_config import AugmentationConfig

from .presets import PresetOptions, build_augmentation_config_from_preset


PresetName = Literal["off", "conservative", "moderate", "aggressive"]


def _parse_smart_resize_dict(
    preset: PresetName, smart_resize: Optional[Dict[str, object]]
) -> Tuple[
    Optional[bool],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[float],
]:
    if smart_resize is None:
        return None, None, None, None, None
    if not isinstance(smart_resize, dict):
        raise ValueError(
            f"smart_resize for preset '{preset}' must be a mapping, got {type(smart_resize)}"
        )
    required = {"enabled", "factor", "min_pixels", "max_pixels", "max_ratio"}
    missing = required.difference(smart_resize.keys())
    if missing:
        raise ValueError(
            f"smart_resize for preset '{preset}' missing required keys: {sorted(missing)}"
        )
    return (
        bool(smart_resize["enabled"]),
        int(smart_resize["factor"]),
        int(smart_resize["min_pixels"]),
        int(smart_resize["max_pixels"]),
        float(smart_resize["max_ratio"]),
    )


def get_preset_config(
    preset: PresetName,
    rng_seed: int = 12345,
    apply_to_teachers: bool = False,
    smart_resize: Optional[Dict[str, object]] = None,
) -> AugmentationConfig:
    """Return an AugmentationConfig built from a named preset.

    Example:
        cfg = get_preset_config("moderate", rng_seed=17)
    """
    sr_enabled, sr_factor, sr_min, sr_max, sr_ratio = _parse_smart_resize_dict(
        preset, smart_resize
    )
    opts = PresetOptions(
        preset=preset,
        rng_seed=int(rng_seed),
        apply_to_teachers=bool(apply_to_teachers),
        smart_resize_enabled=sr_enabled,
        smart_resize_factor=sr_factor,
        smart_resize_min_pixels=sr_min,
        smart_resize_max_pixels=sr_max,
        smart_resize_max_ratio=sr_ratio,
    )
    return build_augmentation_config_from_preset(opts)


@dataclass(frozen=True)
class CurriculumPhase:
    start_epoch: int
    preset: PresetName


def make_curriculum(phases: List[Tuple[int, PresetName]]) -> List[dict]:
    """Build a curriculum list suitable for YAML config (augmentation_schedule).

    Args:
        phases: List of (start_epoch, preset) pairs, e.g.,
            [(0, "off"), (1, "conservative"), (3, "moderate"), (6, "off")]

    Returns:
        List of dicts: [{start_epoch: int, preset: str}, ...]
    """
    out: List[dict] = []
    for start, name in sorted(phases, key=lambda p: p[0]):
        out.append({"start_epoch": int(start), "preset": name})
    return out
