"""
Direct Global Configuration System for Qwen2.5-VL Training

This module provides direct access to configuration values without any parameter passing
or nested structures. All config values are defined once in YAML and accessed directly.

Usage:
    # Initialize once at application startup
    init_config("configs/base.yaml")

    # Access anywhere in the codebase - direct and flat
    from src.config import config

    learning_rate = config.learning_rate
    model_path = config.model_path
    batch_size = config.batch_size
    data_root = config.data_root
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class DirectConfig:
    """
    Direct configuration access - all values are flat and accessible directly.
    No nested structures, no parameter passing, no conversions.
    """

    def __init__(self):
        # Model settings
        self.model_path: str = ""
        self.model_size: str = ""
        self.model_max_length: int = 0
        self.attn_implementation: str = ""
        self.torch_dtype: str = ""
        self.use_cache: bool = False

        # Training settings
        self.num_train_epochs: int = 0
        self.per_device_train_batch_size: int = 0
        self.per_device_eval_batch_size: int = 0
        self.gradient_accumulation_steps: int = 0
        self.learning_rate: float = 0.0
        self.vision_lr: float = 0.0
        self.mlp_lr: float = 0.0
        self.llm_lr: float = 0.0
        self.warmup_ratio: float = 0.0
        self.weight_decay: float = 0.0
        self.max_grad_norm: float = 0.0
        self.lr_scheduler_type: str = ""
        self.gradient_checkpointing: bool = False
        self.bf16: bool = False
        self.fp16: bool = False

        # Data settings
        self.train_data_path: str = ""
        self.val_data_path: str = ""
        self.data_root: str = ""
        self.max_total_length: int = 0
        self.use_candidates: bool = False
        self.candidates_file: str = ""
        self.collator_type: str = ""
        self.multi_round: bool = False
        self.max_examples: int = 0

        # Evaluation settings
        self.eval_strategy: str = ""
        self.eval_steps: int = 0
        self.save_strategy: str = ""
        self.save_steps: int = 0
        self.save_total_limit: int = 0

        # Logging settings
        self.logging_steps: int = 0
        self.logging_dir: Optional[str] = None
        self.log_level: str = ""
        self.report_to: str = ""
        self.verbose: bool = False
        self.disable_tqdm: bool = False

        # Monitoring settings
        self.enable_monitoring: bool = False
        self.monitor_log_dir: str = ""
        self.save_predictions: bool = False
        self.save_token_analysis: bool = False
        self.save_raw_text: bool = False

        # Detection loss configuration (Open Vocabulary Dense Captioning)
        self.detection_num_queries: int = 50  # Number of object queries
        self.detection_max_caption_length: int = 32  # Maximum caption length in tokens

        # Loss component weights
        self.detection_bbox_weight: float = 5.0  # L1 + GIoU loss weight
        self.detection_giou_weight: float = 2.0  # GIoU loss weight
        self.detection_objectness_weight: float = 1.0  # Object presence loss weight
        self.detection_caption_weight: float = 2.0  # Caption generation loss weight

        self.detection_enabled: bool = True  # Enable detection training

        # Detection learning rate (NEW)
        self.detection_lr: float = 0.0  # Learning rate for detection head parameters

        # Model architecture (match official Qwen2.5-VL 3B)
        self.model_hidden_size: int = 3584  # 3B model hidden size
        self.model_num_layers: int = 28  # Number of transformer layers
        self.model_num_attention_heads: int = 28  # Number of attention heads
        self.model_vocab_size: int = 152064  # Vocabulary size

        # Training configuration
        self.use_flash_attention: bool = True  # Use Flash Attention 2
        self.mixed_precision: str = "bf16"  # Use bfloat16 for training

        # Performance settings
        self.dataloader_num_workers: int = 0
        self.pin_memory: bool = False
        self.prefetch_factor: int = 0
        self.batching_strategy: str = ""
        self.remove_unused_columns: bool = False

        # Output settings
        self.output_base_dir: str = ""
        self.run_name: Optional[str] = None
        self.tb_dir: str = ""

        # Stability settings
        self.max_consecutive_nan: int = 0
        self.max_consecutive_zero: int = 0
        self.max_nan_ratio: float = 0.0
        self.nan_monitoring_window: int = 0
        self.allow_occasional_nan: bool = False
        self.nan_recovery_enabled: bool = False
        self.learning_rate_reduction_factor: float = 0.0
        self.gradient_clip_reduction_factor: float = 0.0

        # Debug settings
        self.test_samples: int = 0
        self.test_forward_pass: bool = False

    @property
    def tune_vision(self) -> bool:
        """Auto-determine if vision encoder should be trained based on learning rate."""
        return self.vision_lr > 0

    @property
    def tune_mlp(self) -> bool:
        """Auto-determine if MLP connector should be trained based on learning rate."""
        return self.mlp_lr > 0

    @property
    def tune_llm(self) -> bool:
        """Auto-determine if LLM should be trained based on learning rate."""
        return self.llm_lr > 0

    @property
    def tune_detection(self) -> bool:
        """Auto-determine if detection head should be trained based on learning rate."""
        return self.detection_lr > 0

    @property
    def use_differential_lr(self) -> bool:
        """Auto-determine if differential learning rates should be used."""
        lrs = [self.vision_lr, self.mlp_lr, self.llm_lr, self.detection_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1

    @property
    def output_dir(self) -> str:
        """Get full output directory path."""
        if self.run_name:
            return f"{self.output_base_dir}/{self.run_name}"
        return self.output_base_dir


# Global singleton instance
config: Optional[DirectConfig] = None


def init_config(
    config_path: str, overrides: Optional[Dict[str, Any]] = None
) -> DirectConfig:
    """
    Initialize global configuration from flat YAML file.

    Args:
        config_path: Path to YAML configuration file
        overrides: Optional dictionary of override values

    Returns:
        DirectConfig: Initialized configuration

    Raises:
        RuntimeError: If config is already initialized
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    global config
    if config is not None:
        raise RuntimeError(
            "Config already initialized. Call reset_config() first if needed."
        )

    # Load YAML configuration
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    # Apply overrides if provided
    if overrides:
        config_dict.update(overrides)

    # Create and populate config
    config = DirectConfig()

    # Directly set all attributes from flat YAML with type conversion
    # Skip computed properties that can't be set directly
    computed_properties = {"output_dir"}

    for key, value in config_dict.items():
        if key in computed_properties:
            print(
                f"Info: Skipping computed property '{key}' - value derived automatically"
            )
            continue

        if hasattr(config, key):
            # Get the expected type from the default value
            default_value = getattr(config, key)
            expected_type = type(default_value)

            # Convert value to expected type if needed
            if expected_type is not type(None) and value is not None:
                try:
                    if expected_type is bool:
                        # Handle boolean conversion properly
                        converted_value = (
                            bool(value)
                            if not isinstance(value, str)
                            else value.lower() in ("true", "1", "yes", "on")
                        )
                    else:
                        converted_value = expected_type(value)
                    setattr(config, key, converted_value)
                except (ValueError, TypeError) as e:
                    print(
                        f"Warning: Could not convert '{key}' value '{value}' to {expected_type.__name__}: {e}"
                    )
                    setattr(config, key, value)  # Use original value as fallback
            else:
                setattr(config, key, value)
        else:
            print(f"Warning: Unknown config key '{key}' ignored")

    # Validate configuration
    _validate_config(config)

    return config


def reset_config():
    """Reset global configuration (useful for testing)."""
    global config
    config = None


def get_config() -> DirectConfig:
    """
    Get global configuration instance.

    Returns:
        DirectConfig: Global configuration

    Raises:
        RuntimeError: If config not initialized
    """
    if config is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return config


def _validate_config(cfg: DirectConfig):
    """Validate configuration for common issues."""

    # Check that at least one learning rate is non-zero
    if cfg.vision_lr == 0 and cfg.mlp_lr == 0 and cfg.llm_lr == 0:
        raise ValueError(
            "At least one learning rate (vision_lr, mlp_lr, llm_lr) must be > 0"
        )

    # Check use_cache and gradient_checkpointing compatibility
    if cfg.use_cache and cfg.gradient_checkpointing:
        raise ValueError(
            "use_cache=True is incompatible with gradient_checkpointing=True. "
            "Set either use_cache=False or gradient_checkpointing=False."
        )

    # Check paths exist
    if not Path(cfg.model_path).exists():
        raise ValueError(f"Model path does not exist: {cfg.model_path}")

    data_root = Path(cfg.data_root)
    train_path = data_root / cfg.train_data_path
    val_path = data_root / cfg.val_data_path

    if not train_path.exists():
        raise ValueError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise ValueError(f"Validation data not found: {val_path}")
