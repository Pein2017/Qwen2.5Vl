"""
Direct Configuration Module for Qwen2.5-VL Training

This module provides direct, flat access to configuration values without any
parameter passing or nested structures. All config values are defined once
in YAML and accessed directly.

Usage:
    # Initialize once at application startup
    from src.config import init_config
    init_config("configs/base_flat.yaml")

    # Access anywhere in the codebase - direct and simple
    from src.config import config

    learning_rate = config.learning_rate
    model_path = config.model_path
    batch_size = config.per_device_train_batch_size
    data_root = config.data_root
"""

# Direct configuration system
from .global_config import (
    # Configuration class (for type hints)
    DirectConfig,
    get_config,
    # Core functions
    init_config,
    reset_config,
)


class ConfigAccessor:
    """Module-level accessor for the global config singleton."""

    def __getattr__(self, name):
        from .global_config import config as _config

        if _config is None:
            raise RuntimeError("Config not initialized. Call init_config() first.")
        return getattr(_config, name)

    def __bool__(self):
        from .global_config import config as _config

        return _config is not None


# Create the module-level config accessor
config = ConfigAccessor()

__all__ = [
    # Core functions
    "init_config",
    "reset_config",
    "get_config",
    # Direct config access
    "config",
    # Configuration class
    "DirectConfig",
]
