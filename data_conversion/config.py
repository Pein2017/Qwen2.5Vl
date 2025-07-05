#!/usr/bin/env python3
"""
Configuration Management for Data Conversion Pipeline

Provides structured, type-safe configuration with validation.
Replaces environment variables and complex command-line arguments.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


@dataclass
class DataConversionConfig:
    """Configuration for the data conversion pipeline."""
    
    # Core paths
    input_dir: str = "ds"
    output_dir: str = "data"
    output_image_dir: str = "ds_output"
    
    # Language and processing options
    language: str = "chinese"  # "chinese" or "english"
    response_types: List[str] = field(default_factory=lambda: ["object_type", "property"])
    
    # Image processing
    resize_enabled: bool = True
    
    # Data splitting
    val_ratio: float = 0.1
    max_teachers: int = 10
    seed: int = 42
    
    # Token mapping
    token_map_path: Optional[str] = None
    hierarchy_path: Optional[str] = "data_conversion/label_hierarchy.json"
    
    # Processing parameters
    log_level: str = "INFO"
    fail_fast: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_paths()
        self._validate_parameters()
        self._setup_token_map()
    
    def _validate_paths(self) -> None:
        """Validate and create necessary directories."""
        input_path = Path(self.input_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        # Create output directories
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.resize_enabled:
            output_image_path = Path(self.output_image_dir)
            output_image_path.mkdir(parents=True, exist_ok=True)
    
    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        if self.language not in ["chinese", "english"]:
            raise ValueError(f"Unsupported language: {self.language}")
        
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")
        
        if self.max_teachers < 0:
            raise ValueError(f"max_teachers must be non-negative, got {self.max_teachers}")
        
        if not self.response_types:
            raise ValueError("response_types cannot be empty")
        
        valid_response_types = {"object_type", "property", "extra_info"}
        for resp_type in self.response_types:
            if resp_type not in valid_response_types:
                raise ValueError(f"Invalid response type: {resp_type}")
    
    def _setup_token_map(self) -> None:
        """Setup token map path based on language if not explicitly provided."""
        if self.token_map_path is None:
            if self.language == "chinese":
                self.token_map_path = "data_conversion/token_map_zh.json"
            elif self.language == "english":
                self.token_map_path = "data_conversion/token_map.json"
        
        # Validate token map file exists if provided
        if self.token_map_path:
            token_map_path = Path(self.token_map_path)
            if not token_map_path.exists():
                if self.language == "english":
                    raise FileNotFoundError(f"Token map file required for English: {token_map_path}")
                else:
                    logger.warning(f"Token map file not found: {token_map_path}")
                    self.token_map_path = None
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for compatibility."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "output_image_dir": self.output_image_dir,
            "language": self.language,
            "response_types": self.response_types,
            "resize": self.resize_enabled,
            "val_ratio": self.val_ratio,
            "max_teachers": self.max_teachers,
            "seed": self.seed,
            "token_map_path": self.token_map_path,
            "hierarchy_path": self.hierarchy_path,
            "log_level": self.log_level
        }
    
    @classmethod
    def from_args(cls, args) -> "DataConversionConfig":
        """Create config from command line arguments."""
        config_dict = {}
        
        # Map argument names to config fields
        arg_mapping = {
            "input_dir": "input_dir",
            "output_dir": "output_dir",
            "output_image_dir": "output_image_dir",
            "language": "language",
            "response_types": "response_types",
            "resize": "resize_enabled",
            "val_ratio": "val_ratio",
            "max_teachers": "max_teachers",
            "seed": "seed",
            "token_map_path": "token_map_path",
            "hierarchy_path": "hierarchy_path",
            "log_level": "log_level"
        }
        
        for arg_name, config_field in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    config_dict[config_field] = value
        
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "DataConversionConfig":
        """Create config from environment variables (for backward compatibility)."""
        import os
        
        config_dict = {}
        
        # Map environment variables to config fields
        env_mapping = {
            "INPUT_DIR": "input_dir",
            "OUTPUT_DIR": "output_dir",
            "OUTPUT_IMAGE_DIR": "output_image_dir",
            "LANGUAGE": "language",
            "RESIZE": "resize_enabled",
            "VAL_RATIO": "val_ratio",
            "MAX_TEACHERS": "max_teachers",
            "SEED": "seed",
            "TOKEN_MAP_EN": "token_map_path",
            "TOKEN_MAP_ZH": "token_map_path",
            "LOG_LEVEL": "log_level"
        }
        
        for env_var, config_field in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_field == "resize_enabled":
                    config_dict[config_field] = value.lower() in ("true", "1", "yes")
                elif config_field == "val_ratio":
                    config_dict[config_field] = float(value)
                elif config_field in ("max_teachers", "seed"):
                    config_dict[config_field] = int(value)
                elif config_field == "response_types":
                    config_dict[config_field] = value.split()
                else:
                    config_dict[config_field] = value
        
        # Handle language-specific token map selection
        language = config_dict.get("language", "chinese")
        if "token_map_path" not in config_dict:
            if language == "chinese" and "TOKEN_MAP_ZH" in os.environ:
                config_dict["token_map_path"] = os.environ["TOKEN_MAP_ZH"]
            elif language == "english" and "TOKEN_MAP_EN" in os.environ:
                config_dict["token_map_path"] = os.environ["TOKEN_MAP_EN"]
        
        return cls(**config_dict)


def setup_logging(config: DataConversionConfig) -> None:
    """Setup logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        encoding="utf-8"
    )
    
    logger.info(f"Logging configured at {config.log_level} level")


def validate_config(config: DataConversionConfig) -> None:
    """Additional validation for configuration consistency."""
    # Check if hierarchy file exists
    if config.hierarchy_path:
        hierarchy_path = Path(config.hierarchy_path)
        if not hierarchy_path.exists():
            logger.warning(f"Label hierarchy file not found: {hierarchy_path}")
    
    # Log configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Input: {config.input_dir} → Output: {config.output_dir}")
    logger.info(f"  Language: {config.language}")
    logger.info(f"  Response Types: {config.response_types}")
    logger.info(f"  Image Resize: {'Enabled' if config.resize_enabled else 'Disabled'}")
    logger.info(f"  Teachers: {config.max_teachers}, Val Ratio: {config.val_ratio}")
    if config.token_map_path:
        logger.info(f"  Token Map: {config.token_map_path}")