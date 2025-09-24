"""Compatibility layer exposing the layered configuration system."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .loader import load_layered_config
from .schema import TrainingConfig


# Backward-compatible globals consumed by logger_factory fallbacks
_CONFIGURED_LOGGERS: set[str] = set()
_GLOBAL_LOG_LEVEL: int = logging.INFO


def _normalize_log_level(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    try:
        return mapping[level.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported log level: {level}") from exc


def set_global_log_level(level: Union[str, int]) -> None:
    """Set global log level for all config-managed loggers."""
    global _GLOBAL_LOG_LEVEL
    numeric_level = _normalize_log_level(level)

    try:
        from ..utils.rank_aware_logging import set_global_log_level as _set_rank_level

        _set_rank_level(numeric_level)
    except Exception:
        logging.getLogger("config").warning(
            "Rank-aware logging unavailable; falling back to standard log level propagation"
        )

    for logger_name in _CONFIGURED_LOGGERS:
        logging.getLogger(logger_name).setLevel(numeric_level)

    _GLOBAL_LOG_LEVEL = numeric_level


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def save_config(config: TrainingConfig, output_path: Union[str, Path]) -> None:
    """Persist a configuration to YAML for inspection/debugging."""
    payload = _dataclass_to_dict(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_config(config_name: str) -> TrainingConfig:
    """Load a layered configuration given a CLI-style identifier."""
    return load_layered_config(config_name)


Config = TrainingConfig


__all__ = [
    "Config",
    "load_config",
    "save_config",
    "set_global_log_level",
    "_CONFIGURED_LOGGERS",
    "_GLOBAL_LOG_LEVEL",
]
