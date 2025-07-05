"""
Configuration Module for Qwen2.5-VL Training

This module provides both the new domain-specific configuration system and
backward compatibility with the legacy DirectConfig approach.

New Domain-Specific Usage:
    # Initialize domain-specific configs
    from src.config import init_config_manager
    manager = init_config_manager("configs/base_flat.yaml")
    
    # Access domain-specific configs
    model_settings = manager.model
    training_params = manager.training
    data_config = manager.data

Legacy Direct Access (Backward Compatible):
    # Initialize legacy config
    from src.config import init_config  
    init_config("configs/base_flat.yaml")
    
    # Access anywhere in the codebase
    from src.config import config
    learning_rate = config.learning_rate
"""

# New domain-specific configuration system
from .config_manager import (
    ConfigManager,
    init_config_manager,
    get_config_manager,
    reset_config_manager,
)

from .domain_configs import (
    ModelConfig,
    TrainingConfig, 
    DataConfig,
    DetectionConfig,
    InfrastructureConfig,
)

# Legacy configuration system (backward compatibility)
from .global_config import (
    DirectConfig,
    get_config,
    init_config,
    reset_config,
)


class ConfigAccessor:
    """
    Module-level accessor that supports both new and legacy config systems.
    
    Provides seamless backward compatibility while enabling migration to
    the new domain-specific configuration approach.
    """
    
    def __getattr__(self, name):
        # Try new config manager first
        try:
            manager = get_config_manager()
            return getattr(manager, name)
        except RuntimeError:
            pass
        
        # Fall back to legacy config system
        from .global_config import config as _config
        
        if _config is None:
            raise RuntimeError(
                "Config not initialized. Call either init_config_manager() for new system "
                "or init_config() for legacy system."
            )
        return getattr(_config, name)
    
    def __bool__(self):
        # Check if either config system is initialized
        try:
            manager = get_config_manager()
            return manager.is_initialized()
        except RuntimeError:
            pass
        
        from .global_config import config as _config
        return _config is not None
    
    @property
    def manager(self):
        """Access to the new domain-specific config manager."""
        return get_config_manager()


# Create the module-level config accessor
config = ConfigAccessor()

__all__ = [
    # New domain-specific system
    "ConfigManager",
    "init_config_manager", 
    "get_config_manager",
    "reset_config_manager",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig", 
    "DetectionConfig",
    "InfrastructureConfig",
    # Legacy system (backward compatibility)
    "init_config",
    "reset_config", 
    "get_config",
    "DirectConfig",
    # Unified access
    "config",
]
