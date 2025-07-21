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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class DirectConfig:
    """
    Direct configuration access - all values are flat and accessible directly.
    No nested structures, no parameter passing, no conversions.

    NOTE: All fields without defaults must come first to avoid dataclass errors.
    """

    # === ALL REQUIRED FIELDS (no defaults) ===

    # Model settings
    model_path: str
    model_size: str
    model_max_length: int
    attn_implementation: str
    torch_dtype: str
    use_cache: bool  # For training (typically False to save memory)
    use_cache_inference: bool  # For inference/post-training
    model_hidden_size: int
    model_num_layers: int
    model_num_attention_heads: int
    model_vocab_size: int

    # Training settings
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    vision_lr: float
    merger_lr: float
    llm_lr: float
    coordinate_lr: float  # Learning rate for coordinate token components
    adapter_lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    use_flash_attention: bool
    mixed_precision: str

    # Data settings
    train_data_path: str
    val_data_path: str
    data_root: str
    max_total_length: int
    teacher_pool_file: str
    num_teacher_samples: int
    collator_type: str
    teacher_ratio: float  # Ratio of samples that use teachers during training
    max_examples: int
    language: str

    # Evaluation settings
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int

    # Logging settings - simplified with rank-aware logging
    logging_steps: int
    logging_dir: str
    log_level: str
    report_to: str
    disable_tqdm: bool
    verbose: bool

    # Coordinate token configuration (replaces legacy detection)
    coordinate_tokens_enabled: bool
    coordinate_config_enable_coordinate_tokens: bool
    coordinate_config_max_coord_value: int
    coordinate_config_coord_token_init_std: float
    coordinate_config_coordinate_loss_weight: float
    coordinate_config_regular_loss_weight: float
    coordinate_config_soft_expectation_temperature: float
    coordinate_config_focal_loss_alpha: float
    coordinate_config_focal_loss_gamma: float
    coordinate_config_use_official_box_tokens: bool
    chat_processor_enable_coordinate_tokens: bool
    chat_processor_max_coord_value: int
    chat_processor_use_official_box_tokens: bool

    # Essential settings
    remove_unused_columns: bool

    # DataLoader performance settings
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int

    # Output settings
    output_dir: str
    run_name: str
    tb_dir: str


    # Teacher-Student Loss Weights
    teacher_loss_weight: float
    student_loss_weight: float


    # Vision processing parameters
    patch_size: int
    merge_size: int
    temporal_patch_size: int

    # === OPTIONAL FIELDS (with defaults) ===

    # Runtime properties (added dynamically during initialization)
    run_output_dir: str = ""
    tensorboard_dir: str = ""
    log_file_dir: str = ""

    @property
    def tune_vision(self) -> bool:
        """Auto-determine if vision encoder should be trained based on learning rate."""
        return self.vision_lr > 0

    @property
    def tune_merger(self) -> bool:
        """Auto-determine if merger should be trained based on learning rate."""
        return self.merger_lr > 0

    @property
    def tune_llm(self) -> bool:
        """Auto-determine if LLM should be trained based on learning rate."""
        return self.llm_lr > 0

    @property
    def tune_coordinate_tokens(self) -> bool:
        """Auto-determine if coordinate tokens should be trained based on learning rate."""
        return self.coordinate_lr > 0

    @property
    def use_differential_lr(self) -> bool:
        """Auto-determine if differential learning rates should be used."""
        lrs = [self.vision_lr, self.merger_lr, self.llm_lr, self.coordinate_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1


# Global singleton instance
config: Optional[DirectConfig] = None


def _flatten_nested_config(
    config_dict: dict, parent_key: str = "", sep: str = "_"
) -> dict:
    """
    Flatten nested configuration dictionary.

    Args:
        config_dict: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator character

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in config_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_nested_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def init_config(config_path: str) -> DirectConfig:
    """
    Initialize global configuration from flat YAML file.

    Args:
        config_path: Path to YAML configuration file

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
        raw_config_dict = yaml.safe_load(f)

    # Flatten nested configurations
    config_dict = _flatten_nested_config(raw_config_dict)

    # Manually convert types before dataclass instantiation
    from dataclasses import fields
    from typing import get_args, get_origin

    field_map = {f.name: f.type for f in fields(DirectConfig)}
    converted_dict = {}

    for key, value in config_dict.items():
        if key not in field_map:
            continue  # Let the dataclass handle extra keys

        target_type = field_map[key]
        origin_type = get_origin(target_type)

        # Handle Optional[T]
        if origin_type is Union:
            # Assumes Optional[T] is Union[T, NoneType]
            actual_type = next(
                (t for t in get_args(target_type) if t is not type(None)), None
            )
            if actual_type:
                target_type = actual_type
            else:
                converted_dict[key] = None
                continue

        if value is None:
            converted_dict[key] = None
            continue

        # Perform type conversion
        try:
            if target_type is bool and isinstance(value, str):
                converted_dict[key] = value.lower() in ("true", "1", "yes")
            else:
                converted_dict[key] = target_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Config error: Could not convert '{key}' with value '{value}' to type {target_type.__name__}"
            ) from e

    # Create and populate config using dictionary unpacking
    try:
        new_config = DirectConfig(**converted_dict)
    except TypeError as e:
        raise ValueError(f"Configuration error: Missing or extra keys in YAML. {e}")

    # --- Automatically derive and set paths ---
    if not new_config.run_name:
        raise ValueError("`run_name` must be defined in the configuration.")

    # 1. Main output directory for the run
    new_config.run_output_dir = str(Path(new_config.output_dir) / new_config.run_name)

    # 2. TensorBoard directory
    new_config.tensorboard_dir = str(Path(new_config.tb_dir) / new_config.run_name)

    # 3. Log file directory
    new_config.log_file_dir = str(Path(new_config.run_output_dir) / "logs")

    # Create directories
    Path(new_config.run_output_dir).mkdir(parents=True, exist_ok=True)

    # Validate the final configuration
    _validate_config(new_config)

    # Set global config and return
    config = new_config
    return config


def _validate_config(config: DirectConfig) -> None:
    """Validate configuration values."""
    if config.per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")

    if config.coordinate_lr < 0:
        raise ValueError("coordinate_lr must be non-negative")

    # Coordinate token validation
    if config.coordinate_tokens_enabled:
        if not hasattr(config, 'coordinate_config_max_coord_value'):
            raise ValueError("coordinate_config_max_coord_value required when coordinate tokens enabled")
        if config.coordinate_config_max_coord_value <= 0:
            raise ValueError("coordinate_config_max_coord_value must be positive")
        if config.coordinate_config_coordinate_loss_weight < 0:
            raise ValueError("coordinate_config_coordinate_loss_weight must be non-negative")
        if config.coordinate_config_regular_loss_weight < 0:
            raise ValueError("coordinate_config_regular_loss_weight must be non-negative")
        if config.coordinate_config_soft_expectation_temperature <= 0:
            raise ValueError("coordinate_config_soft_expectation_temperature must be positive")
        if not (0.0 <= config.coordinate_config_focal_loss_alpha <= 1.0):
            raise ValueError("coordinate_config_focal_loss_alpha must be between 0 and 1")
        if config.coordinate_config_focal_loss_gamma < 0:
            raise ValueError("coordinate_config_focal_loss_gamma must be non-negative")

    if config.coordinate_config_max_coord_value <= 0:
        raise ValueError("coordinate_config_max_coord_value must be positive")


def reset_config() -> None:
    """Reset global configuration (for testing purposes)."""
    global config
    config = None


def get_config() -> DirectConfig:
    """Get the global configuration instance."""
    if config is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return config
