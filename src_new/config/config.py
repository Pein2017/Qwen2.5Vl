"""
Unified configuration management for Qwen2.5-VL training.

This module provides a single configuration class that directly maps to the
existing bbu_v2.yaml file without requiring any modifications to the YAML.

Key Features:
- Direct field mapping from YAML to dataclass fields
- Comprehensive type annotations and validation
- Fail-fast error handling with detailed error messages
- Strict validation requiring all fields to be explicitly defined
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Global logging configuration (compatibility with legacy callers)
# These values mirror the state managed by src_new.utils.rank_aware_logging.
_GLOBAL_LOG_LEVEL = logging.INFO
_CONFIGURED_LOGGERS = set()


def set_global_log_level(level: str) -> None:
    """Set global log level for all loggers in the system.

    This delegates to the centralized rank-aware logging utilities to ensure
    consistent behavior across ranks and modules. The local variables remain
    only for backward compatibility with legacy callers.
    """
    global _GLOBAL_LOG_LEVEL

    # Normalize to string for the rank-aware API, which accepts str or int
    normalized_level = level
    try:
        # Defer to rank-aware system for propagation across all loggers
        from ..utils.rank_aware_logging import set_global_log_level as _set_global

        _set_global(normalized_level)
    except Exception:
        # Fallback: update root logger if rank-aware utilities are unavailable
        if isinstance(level, str):
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            resolved = level_map.get(level.upper(), logging.INFO)
        else:
            resolved = int(level)
        logging.getLogger().setLevel(resolved)
        for logger_name in _CONFIGURED_LOGGERS:
            logging.getLogger(logger_name).setLevel(resolved)
    finally:
        # Keep local mirror updated for compatibility (e.g., script fallback)
        if isinstance(level, str):
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            _GLOBAL_LOG_LEVEL = level_map.get(level.upper(), logging.INFO)
        else:
            _GLOBAL_LOG_LEVEL = int(level)


def get_config_logger() -> logging.Logger:
    """Get rank-aware logger for config module."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("config")
    except ImportError:
        # Fallback to original implementation
        logger = logging.getLogger("config")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(_GLOBAL_LOG_LEVEL)
            _CONFIGURED_LOGGERS.add("config")
        return logger


logger = get_config_logger()


def _convert_scientific_notation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert scientific notation strings to floats in configuration data.

    Args:
        data: Raw configuration dictionary from YAML

    Returns:
        Configuration dictionary with converted numeric values
    """
    converted_data = {}

    for key, value in data.items():
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                # Try to convert any numeric string to float
                float_value = float(value)
                converted_data[key] = float_value
                logger.debug(f"Converted {key}: {value} (str) -> {float_value} (float)")
            except ValueError:
                # If conversion fails, keep as string
                converted_data[key] = value
        else:
            # Keep non-string values as-is
            converted_data[key] = value

    return converted_data


@dataclass
class Config:
    """
    Unified configuration class with direct mapping to bbu_v2.yaml.

    All field names and types match exactly with the existing YAML structure
    to ensure zero-modification compatibility. Every parameter is explicitly
    defined with proper type hints and validation.

    Note: All required fields (no defaults) must come first in dataclass definition.
    """

    # === REQUIRED FIELDS (no defaults) ===
    # Model settings
    model_path: str
    model_size: str
    model_max_length: int
    attn_implementation: str
    torch_dtype: str

    # Training settings
    num_train_epochs: int
    per_device_train_batch_size: int
    learning_rate: float
    vision_lr: float
    merger_lr: float
    llm_lr: float

    # Data settings
    train_data_path: str
    val_data_path: str
    data_root: str
    teacher_pool_file: str

    # Output settings
    output_dir: str
    run_name: str
    max_coord_value: int
    model_hidden_size: int

    # === OPTIONAL FIELDS (with defaults) ===
    # Model settings with defaults
    use_cache: bool = False
    use_cache_inference: bool = True
    model_num_layers: int = 36
    model_num_attention_heads: int = 16
    model_vocab_size: int = 151665

    # Training settings with defaults
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    adapter_lr: float = 0.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0001
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    use_flash_attention: bool = True
    mixed_precision: str = "bf16"

    # Data settings with defaults
    max_total_length: int = 12000
    num_teacher_samples: int = 1
    collator_type: str = "packed"
    teacher_ratio: float = 0.5
    language: str = "chinese"

    # Dataset size limiting (optional, primarily for debugging/testing)
    # Set to None or 0 to use full dataset, or specify a positive integer to limit samples
    max_dataset_size: Optional[int] = None

    # Evaluation settings with defaults
    eval_strategy: str = "steps"
    eval_steps: int = 20
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    save_on_each_node: bool = False  # EFFICIENCY: Only rank 0 saves checkpoints

    # Best checkpoint tracking settings with defaults
    load_best_model_at_end: bool = True  # Enable automatic best checkpoint saving
    metric_for_best_model: str = "eval_loss"  # Track evaluation loss for best model
    greater_is_better: bool = False  # Lower eval_loss is better

    # Unified checkpoint management settings
    best_checkpoint_metric: str = "eval_loss"  # Metric to track for best checkpoints
    best_checkpoint_greater_is_better: bool = (
        False  # Whether higher metric values are better
    )

    # Logging settings with defaults
    logging_steps: int = 10
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    disable_tqdm: bool = True
    verbose: bool = False

    # Coordinate token configuration with defaults
    coordinate_tokens_enabled: bool = True

    coordinate_loss_weight: float = 0.05
    regular_loss_weight: float = 1.0
    new_geometry_tokens: Optional[List[str]] = None

    # Essential settings with defaults
    remove_unused_columns: bool = False

    # Dataloader performance settings with defaults
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Output settings with defaults
    tb_dir: str = "tb"

    # Teacher-student loss weights with defaults
    teacher_loss_weight: float = 0.3
    student_loss_weight: float = 1.0

    # Vision processing parameters with defaults
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2
    max_pixels: int = (
        401408  # 512 * 28 * 28 - controls Qwen2VL image processor pixel limit
    )

    # Training control flags with defaults
    training_prompt_style: bool = True
    use_consistent_prompts: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Performs fail-fast validation to catch configuration errors early
        with detailed error messages to help debugging.

        Raises:
            ValueError: If any configuration values are invalid
            FileNotFoundError: If required paths don't exist
        """
        self._validate_model_settings()
        self._validate_training_settings()
        self._validate_data_settings()
        self._validate_coordinate_settings()

        logger.info("✅ Configuration validation passed")

    def _validate_model_settings(self) -> None:
        """Validate model-related settings."""
        if not self.model_path:
            raise ValueError("model_path cannot be empty")

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        if self.model_max_length <= 0:
            raise ValueError(
                f"model_max_length must be positive, got {self.model_max_length}"
            )

        if self.attn_implementation not in ["flash_attention_2", "eager", "sdpa"]:
            raise ValueError(f"Invalid attn_implementation: {self.attn_implementation}")

        if self.torch_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(f"Invalid torch_dtype: {self.torch_dtype}")

        if self.max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {self.max_pixels}")

    def _validate_training_settings(self) -> None:
        """Validate training-related settings."""
        if self.num_train_epochs <= 0:
            raise ValueError(
                f"num_train_epochs must be positive, got {self.num_train_epochs}"
            )

        if self.per_device_train_batch_size <= 0:
            raise ValueError(
                f"per_device_train_batch_size must be positive, got {self.per_device_train_batch_size}"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.vision_lr < 0:
            raise ValueError(f"vision_lr cannot be negative, got {self.vision_lr}")

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError(
                f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio}"
            )

    def _validate_data_settings(self) -> None:
        """Validate data-related settings."""
        if not self.train_data_path:
            raise ValueError("train_data_path cannot be empty")

        if not self.val_data_path:
            raise ValueError("val_data_path cannot be empty")

        if not self.teacher_pool_file:
            raise ValueError("teacher_pool_file cannot be empty")

        if self.teacher_ratio < 0 or self.teacher_ratio > 1:
            raise ValueError(
                f"teacher_ratio must be between 0 and 1, got {self.teacher_ratio}"
            )

        if self.collator_type not in ["packed", "standard"]:
            raise ValueError(f"Invalid collator_type: {self.collator_type}")

        if self.language not in ["chinese", "english"]:
            raise ValueError(f"Unsupported language: {self.language}")

    def _validate_coordinate_settings(self) -> None:
        """Validate coordinate token settings."""
        if self.max_coord_value <= 0:
            raise ValueError(
                f"max_coord_value must be positive, got {self.max_coord_value}"
            )

        if self.coordinate_loss_weight < 0:
            raise ValueError(
                f"coordinate_loss_weight cannot be negative, got {self.coordinate_loss_weight}"
            )

        if self.regular_loss_weight <= 0:
            raise ValueError(
                f"regular_loss_weight must be positive, got {self.regular_loss_weight}"
            )

        # Initialize new_geometry_tokens if not provided
        if self.coordinate_tokens_enabled and self.new_geometry_tokens is None:
            # Only add line tokens - quad tokens already exist in Qwen2.5-VL
            self.new_geometry_tokens = [
                "<|line_start|>",
                "<|line_end|>",
            ]


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file with comprehensive validation.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML file is malformed
        ValueError: If configuration contains invalid values
        TypeError: If configuration values have wrong types
    """
    # FAIL-FAST: Validate config path
    if not config_path:
        raise ValueError("config_path cannot be empty")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not config_file.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")

    # Load YAML with error handling
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read config file {config_path}: {e}")

    # FAIL-FAST: Validate YAML loaded correctly
    if data is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(data)}")

    # Convert scientific notation strings to floats
    data = _convert_scientific_notation(data)

    # === Unified dataset path defaults ===
    # If only data_root is provided, auto-derive standard file paths inside it
    # Expected structure under data_root:
    #   - images/  (image files referenced in JSONL as ./images/xxx.jpeg)
    #   - train.jsonl
    #   - val.jsonl
    #   - teacher_pool.jsonl
    try:
        data_root_value = data.get("data_root")
        if data_root_value:
            # Normalize to Path for safe joining but keep string form in final dict
            data_root_path = Path(data_root_value)

            # Derive when missing
            if not data.get("train_data_path"):
                data["train_data_path"] = str(data_root_path / "train.jsonl")
            if not data.get("val_data_path"):
                data["val_data_path"] = str(data_root_path / "val.jsonl")
            if not data.get("teacher_pool_file"):
                # Use .jsonl per project convention
                data["teacher_pool_file"] = str(data_root_path / "teacher_pool.jsonl")
    except Exception:
        # Do not block config loading if derivation fails; validation will catch later
        pass

    # Create config with comprehensive error handling
    try:
        config = Config(**data)
    except TypeError as e:
        raise TypeError(f"Configuration contains invalid field types: {e}")
    except Exception as e:
        raise ValueError(f"Failed to create configuration from {config_path}: {e}")

    logger.info(f"✅ Configuration loaded successfully from {config_path}")
    logger.info(f"📋 Model: {config.model_size} at {config.model_path}")
    logger.info(f"📋 Data: train={config.train_data_path}, val={config.val_data_path}")
    logger.info(
        f"📋 Coordinate tokens: {'enabled' if config.coordinate_tokens_enabled else 'disabled'}"
    )
    logger.info(f"📋 Teacher ratio: {config.teacher_ratio}")

    return config


def save_config(config: "Config", output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        output_path: Path where to save the YAML file

    Raises:
        ValueError: If config is None or output_path is empty
        OSError: If unable to write to the output path
    """
    if config is None:
        raise ValueError("config cannot be None")

    if not output_path:
        raise ValueError("output_path cannot be empty")

    # Convert dataclass to dictionary
    config_dict = {
        field_name: getattr(config, field_name)
        for field_name in config.__dataclass_fields__
    }

    # Write to YAML file
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=True)

    except Exception as e:
        raise OSError(f"Failed to save config to {output_path}: {e}")

    logger.info(f"✅ Configuration saved to {output_path}")
