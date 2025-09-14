from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from src_new_json.config.augmentation_config import AugmentationConfig

from .presets import PresetOptions, build_augmentation_config_from_preset


PresetName = Literal["off", "conservative", "moderate", "aggressive"]


def get_preset_config(
    preset: PresetName, rng_seed: int = 12345, apply_to_teachers: bool = False
) -> AugmentationConfig:
    """Return an AugmentationConfig built from a named preset.

    Example:
        cfg = get_preset_config("moderate", rng_seed=17)
    """
    opts = PresetOptions(
        preset=preset,
        rng_seed=int(rng_seed),
        apply_to_teachers=bool(apply_to_teachers),
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
