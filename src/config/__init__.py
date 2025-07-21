"""
Simplified Configuration Module for Qwen2.5-VL Training

This module provides direct access to configuration values using the flat DirectConfig approach.

Usage:
    # Initialize config once at startup
    from src.config import init_config
    init_config("configs/base_flat_det.yaml")

    # Access anywhere in the codebase
    from src.config import config
    learning_rate = config.learning_rate
    model_path = config.model_path
    batch_size = config.per_device_train_batch_size
"""

# Single configuration system - flat DirectConfig
from .global_config import (
    DirectConfig,
    config,
    get_config,
    init_config,
    reset_config,
)


__all__ = [
    "DirectConfig",
    "config",
    "get_config",
    "init_config",
    "reset_config",
]
