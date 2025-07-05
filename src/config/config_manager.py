"""
Configuration Manager for Qwen2.5-VL Training

This module provides centralized configuration management, validation, and coordination
between different domain-specific configuration classes. It replaces the monolithic
DirectConfig approach with a more modular and maintainable system.

Key Features:
- Domain-specific config validation
- Cross-config dependency checking
- Automatic learning rate scaling
- Clean configuration interfaces
- Comprehensive error handling
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import fields, asdict

from .domain_configs import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    DetectionConfig,
    InfrastructureConfig
)


class ConfigManager:
    """
    Centralized configuration manager that coordinates between domain-specific configs.
    
    Provides clean interfaces for:
    - Loading and validating configurations
    - Cross-config dependency checking
    - Automatic parameter scaling and derivation
    - Unified configuration access
    """
    
    def __init__(self):
        self.model: Optional[ModelConfig] = None
        self.training: Optional[TrainingConfig] = None
        self.data: Optional[DataConfig] = None
        self.detection: Optional[DetectionConfig] = None
        self.infrastructure: Optional[InfrastructureConfig] = None
        self._initialized = False
    
    def load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from YAML file and populate domain configs.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        self._populate_configs(config_dict)
        self._validate_all_configs()
        self._apply_cross_config_logic()
        self._initialized = True
    
    def _populate_configs(self, config_dict: Dict[str, Any]) -> None:
        """Populate domain-specific configs from flat YAML dictionary."""
        
        # Helper to extract and convert fields for a specific config class
        def extract_config_fields(config_class, prefix: str = "") -> Dict[str, Any]:
            config_fields = {}
            field_names = {f.name for f in fields(config_class)}
            
            for key, value in config_dict.items():
                # Remove prefix if present (e.g., "detection_" -> "")
                field_name = key
                if prefix and key.startswith(prefix):
                    field_name = key[len(prefix):]
                
                if field_name in field_names:
                    config_fields[field_name] = self._convert_type(
                        value, config_class, field_name
                    )
            
            return config_fields
        
        # Populate each domain config
        try:
            self.model = ModelConfig(**extract_config_fields(ModelConfig))
            self.training = TrainingConfig(**extract_config_fields(TrainingConfig))
            self.data = DataConfig(**extract_config_fields(DataConfig))
            self.detection = DetectionConfig(**extract_config_fields(DetectionConfig, "detection_"))
            self.infrastructure = InfrastructureConfig(**extract_config_fields(InfrastructureConfig))
            
        except TypeError as e:
            raise ValueError(f"Configuration error: Missing or invalid fields. {e}")
    
    def _convert_type(self, value: Any, config_class: type, field_name: str) -> Any:
        """Convert YAML value to appropriate type for config field."""
        if value is None:
            return None
        
        # Get target type from dataclass field
        field_map = {f.name: f.type for f in fields(config_class)}
        target_type = field_map.get(field_name)
        
        if target_type is None:
            return value
        
        # Handle Optional types
        from typing import get_origin, get_args, Union
        if get_origin(target_type) is Union:
            # Assumes Optional[T] is Union[T, NoneType]
            actual_type = next(
                (t for t in get_args(target_type) if t is not type(None)), None
            )
            if actual_type:
                target_type = actual_type
        
        # Perform type conversion
        try:
            if target_type is bool and isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            else:
                return target_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Config error: Could not convert '{field_name}' with value '{value}' "
                f"to type {target_type.__name__}"
            ) from e
    
    def _validate_all_configs(self) -> None:
        """Validate all domain-specific configurations."""
        configs_to_validate = [
            (self.model, "ModelConfig"),
            (self.training, "TrainingConfig"), 
            (self.data, "DataConfig"),
            (self.detection, "DetectionConfig"),
            (self.infrastructure, "InfrastructureConfig")
        ]
        
        for config, name in configs_to_validate:
            if config is None:
                raise ValueError(f"{name} not properly initialized")
            
            # Call validate method if it exists
            if hasattr(config, 'validate'):
                try:
                    config.validate()
                except Exception as e:
                    raise ValueError(f"{name} validation failed: {e}")
    
    def _apply_cross_config_logic(self) -> None:
        """Apply logic that depends on multiple configs."""
        self._apply_learning_rate_scaling()
        self._validate_cross_config_dependencies()
    
    def _apply_learning_rate_scaling(self) -> None:
        """Apply automatic learning rate scaling based on batch size and collator type."""
        if not self.training.auto_scale_lr:
            return
        
        # Apply batch size scaling
        self._apply_batch_size_lr_scaling()
        
        # Apply collator type scaling
        self._apply_collator_lr_scaling()
    
    def _apply_batch_size_lr_scaling(self) -> None:
        """Scale learning rates based on effective batch size."""
        reference_bs = self.training.lr_reference_batch_size
        if reference_bs <= 0:
            return
        
        effective_bs = self.training.effective_batch_size
        scale = effective_bs / reference_bs
        
        if abs(scale - 1.0) < 1e-6:
            return
        
        # Scale all learning rate fields
        lr_fields = [
            'learning_rate', 'vision_lr', 'merger_lr', 
            'llm_lr', 'detection_lr', 'adapter_lr'
        ]
        
        for field_name in lr_fields:
            if not hasattr(self.training, field_name):
                raise AttributeError(f"Training config missing required field: {field_name}")
            current_lr = getattr(self.training, field_name)
            if current_lr is not None and current_lr > 0:
                setattr(self.training, field_name, current_lr * scale)
        
        print(f"🔄 Auto LR scaling: effective_bs={effective_bs}, reference_bs={reference_bs}, "
              f"scale={scale:.2f}. Learning rates updated accordingly.")
    
    def _apply_collator_lr_scaling(self) -> None:
        """Apply token-length-aware learning rate scaling for different collator types."""
        disable_scaling = os.getenv("DISABLE_COLLATOR_LR_SCALING", "false").lower() == "true"
        if disable_scaling:
            return
        
        try:
            from src.lr_scaling import create_token_length_scaler
            
            scaler = create_token_length_scaler(
                auto_scale_lr=self.training.auto_scale_lr,
                base_collator_type="standard"
            )
            
            # Prepare config dict for scaling
            config_dict = {
                "collator_type": self.data.collator_type,
                "learning_rate": self.training.learning_rate,
                "llm_lr": self.training.llm_lr,
                "adapter_lr": self.training.adapter_lr,
                "vision_lr": self.training.vision_lr,
                "merger_lr": self.training.merger_lr,
                "detection_lr": self.training.detection_lr,
            }
            
            # Apply scaling
            scaled_config = scaler.scale_learning_rates(
                config=config_dict,
                collator_type=self.data.collator_type
            )
            
            # Update training config with scaled values
            for key, value in scaled_config.items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
                    
        except ImportError as e:
            raise RuntimeError(f"lr_scaling module required for auto_scale_lr=True but not available: {e}")
    
    def _validate_cross_config_dependencies(self) -> None:
        """Validate dependencies between different config domains."""
        
        # Validate detection dependencies
        if self.detection.detection_enabled and not self.training.tune_detection:
            if self.training.detection_lr <= 0:
                raise ValueError(
                    "Detection is enabled but detection_lr is 0. "
                    "Set detection_lr > 0 to train detection head."
                )
        
        # Validate data dependencies
        if not Path(self.data.train_data_path).exists():
            raise ValueError(f"Training data file not found: {self.data.train_data_path}")
        
        if not Path(self.data.val_data_path).exists():
            raise ValueError(f"Validation data file not found: {self.data.val_data_path}")
        
        # Validate model dependencies
        if not Path(self.model.model_path).exists():
            raise ValueError(f"Model path not found: {self.model.model_path}")
        
        # Validate infrastructure dependencies
        if self.infrastructure.eval_steps > 0 and self.infrastructure.eval_strategy == "no":
            raise ValueError("eval_steps > 0 but eval_strategy is 'no'")
    
    def get_legacy_config_dict(self) -> Dict[str, Any]:
        """
        Generate a flat dictionary compatible with the old DirectConfig system.
        This enables backward compatibility during the transition period.
        """
        if not self._initialized:
            raise RuntimeError("ConfigManager not initialized. Call load_from_yaml() first.")
        
        # Combine all config dictionaries
        legacy_dict = {}
        
        # Add model config fields
        legacy_dict.update(asdict(self.model))
        
        # Add training config fields  
        legacy_dict.update(asdict(self.training))
        
        # Add data config fields
        legacy_dict.update(asdict(self.data))
        
        # Add detection config fields with "detection_" prefix
        detection_dict = asdict(self.detection)
        for key, value in detection_dict.items():
            if not key.startswith("detection_"):
                legacy_dict[f"detection_{key}"] = value
            else:
                legacy_dict[key] = value
        
        # Add infrastructure config fields
        legacy_dict.update(asdict(self.infrastructure))
        
        return legacy_dict
    
    def is_initialized(self) -> bool:
        """Check if configuration manager is properly initialized."""
        return self._initialized
    
    def __getattr__(self, name: str) -> Any:
        """
        Provide backward compatibility by allowing direct access to config fields.
        This enables a smooth transition from the old DirectConfig system.
        """
        if not self._initialized:
            raise RuntimeError("ConfigManager not initialized. Call load_from_yaml() first.")
        
        # Search for the attribute in all domain configs
        configs = [self.model, self.training, self.data, self.detection, self.infrastructure]
        
        for config in configs:
            if hasattr(config, name):
                return getattr(config, name)
        
        # Check for detection fields with prefix
        if name.startswith("detection_"):
            field_name = name[len("detection_"):]
            if hasattr(self.detection, field_name):
                return getattr(self.detection, field_name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Global configuration manager instance
config_manager: Optional[ConfigManager] = None


def init_config_manager(config_path: str) -> ConfigManager:
    """
    Initialize global configuration manager from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ConfigManager: Initialized configuration manager
        
    Raises:
        RuntimeError: If config manager is already initialized
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    global config_manager
    
    if config_manager is not None:
        raise RuntimeError(
            "Config manager already initialized. Call reset_config_manager() first if needed."
        )
    
    config_manager = ConfigManager()
    config_manager.load_from_yaml(config_path)
    
    return config_manager


def get_config_manager() -> ConfigManager:
    """
    Get global configuration manager instance.
    
    Returns:
        ConfigManager: Global configuration manager
        
    Raises:
        RuntimeError: If config manager not initialized
    """
    if config_manager is None:
        raise RuntimeError("Config manager not initialized. Call init_config_manager() first.")
    
    return config_manager


def reset_config_manager():
    """Reset global configuration manager (useful for testing)."""
    global config_manager
    config_manager = None