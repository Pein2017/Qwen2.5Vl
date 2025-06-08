"""
Configuration module for BBU training.
Provides both legacy and new configuration systems during transition.
"""

# Legacy system (for backward compatibility)
from .base import Config
from .config_manager import ConfigManager
from .manager import ConfigurationManager, list_available_profiles, load_config

# New system (recommended)
from .schema import TrainingConfiguration

__all__ = [
    # Legacy system
    "Config",
    "ConfigManager",
    # New system
    "TrainingConfiguration",
    "ConfigurationManager",
    "load_config",
    "list_available_profiles",
]
