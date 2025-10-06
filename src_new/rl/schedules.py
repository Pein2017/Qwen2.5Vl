"""Schedules for sampling temperature, KL beta, and curriculum stages."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence


ScheduleType = Literal["constant", "linear", "cosine"]


def temperature_at(
    step: int,
    schedule: ScheduleType,
    base: float,
    *,
    total_steps: int | None = None,
    min_temperature: float = 1e-4,
) -> float:
    """Compute the sampling temperature for the given ``step``."""

    if step < 0:
        step = 0
    if base <= 0:
        return float(min_temperature)
    if schedule == "constant" or total_steps is None or total_steps <= 0:
        return float(max(base, min_temperature))

    progress = min(max(step / float(total_steps), 0.0), 1.0)
    if schedule == "linear":
        value = base * (1.0 - progress)
    elif schedule == "cosine":
        import math

        value = base * (0.5 + 0.5 * math.cos(math.pi * progress))
    else:
        value = base
    return float(max(value, min_temperature))


def beta_at(
    step: int,
    beta_start: float,
    *,
    anneal: dict | None = None,
) -> float:
    """Return the KL coefficient for PPO/GRPO style regularization."""

    if beta_start <= 0:
        return 0.0
    if not anneal:
        return float(beta_start)

    anneal_type = anneal.get("type", "constant")
    steps = int(anneal.get("steps", 0) or 0)
    if steps <= 0:
        return float(beta_start)

    progress = min(max(step / float(steps), 0.0), 1.0)
    if anneal_type == "linear":
        return float(beta_start * (1.0 - progress))
    if anneal_type == "cosine":
        import math

        return float(beta_start * (0.5 + 0.5 * math.cos(math.pi * progress)))
    return float(beta_start)


def curriculum_stage_at(step: int, switch_steps: Sequence[int] | None) -> int:
    """Determine the curriculum stage index based on ``switch_steps``."""

    if not switch_steps:
        return 0

    for idx, boundary in enumerate(switch_steps, start=1):
        if step < boundary:
            return idx - 1
    return len(switch_steps)


__all__ = ["temperature_at", "beta_at", "curriculum_stage_at"]
