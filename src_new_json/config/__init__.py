"""
Configuration management for Qwen2.5-VL training.

This module provides a unified configuration system with direct YAML mapping
and comprehensive validation.
"""

from typing import TYPE_CHECKING

# Re-export core components
from src_new_json.config.config import Config, load_config, save_config


if TYPE_CHECKING:
    # Type-only imports
    pass

__all__ = [
    "Config",
    "load_config",
    "save_config",
]
