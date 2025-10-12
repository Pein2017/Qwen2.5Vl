"""Diagnostics and debug settings (env-driven, read once).

This module centralizes optional knobs for logging and memory cleanup
without changing core algorithms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DebugSettings:
    debug_timing: bool
    generation_warn_threshold: float
    clear_cache_policy: str  # "always" | "boundary" | "never"


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "y")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if isinstance(val, str) and len(val) > 0 else default


SETTINGS = DebugSettings(
    debug_timing=_env_bool("DEBUG_TIMING", False),
    generation_warn_threshold=_env_float("GENERATION_WARN_THRESHOLD", 60.0),
    clear_cache_policy=_env_str("RL_CLEAR_CACHE", "always"),
)


__all__ = ["DebugSettings", "SETTINGS"]


