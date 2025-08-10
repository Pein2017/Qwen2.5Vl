"""
Modern Configuration System for BBU Training

This system eliminates redundancy by using Pydantic's automatic YAML parsing.
Add/remove fields by simply editing YAML - no Python code changes needed!

Key Benefits:
- Single source of truth (YAML only)
- Automatic type validation and conversion
- No more double-definition maintenance
- Built-in documentation
- Extensible with custom validators
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, computed_field, field_validator


class BBUConfig(BaseModel):
    """
    BBU Configuration - automatically parsed from YAML

    To add a new parameter:
    1. Add it to your YAML file
    2. That's it! It's automatically available

    For better IDE support, optionally add type annotation here,
    but it's not required for basic usage.
    """

    model_config = {
        "extra": "forbid",  # Prevent typos in YAML
        "validate_assignment": True,  # Validate field updates
        "str_strip_whitespace": True,  # Auto-strip strings
    }

    # === MODEL SETTINGS ===
    model_path: str = Field(description="Path to the model directory")
    model_size: Literal["3B", "7B"] = Field(description="Model size variant")
    model_max_length: int = Field(gt=0, description="Maximum sequence length")
    attn_implementation: Literal["flash_attention_2", "eager", "sdpa"] = Field(
        description="Attention implementation"
    )
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        description="PyTorch data type"
    )
    use_cache: bool = Field(description="Enable KV cache during training")
    use_cache_inference: bool = Field(description="Enable KV cache during inference")
    model_hidden_size: int = Field(gt=0, description="Model hidden dimension")
    model_num_layers: int = Field(gt=0, description="Number of transformer layers")
    model_num_attention_heads: int = Field(
        gt=0, description="Number of attention heads"
    )
    model_vocab_size: int = Field(gt=0, description="Vocabulary size")

    # === TRAINING SETTINGS ===
    num_train_epochs: int = Field(gt=0, description="Number of training epochs")
    per_device_train_batch_size: int = Field(
        gt=0, description="Training batch size per device"
    )
    per_device_eval_batch_size: int = Field(
        gt=0, description="Evaluation batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        gt=0, description="Gradient accumulation steps"
    )
    learning_rate: float = Field(gt=0, description="Main learning rate")
    vision_lr: float = Field(ge=0, description="Vision encoder learning rate")
    merger_lr: float = Field(ge=0, description="Vision-language merger learning rate")
    llm_lr: float = Field(ge=0, description="LLM learning rate")
    adapter_lr: float = Field(ge=0, description="Adapter learning rate")
    warmup_ratio: float = Field(ge=0, le=1, description="Learning rate warmup ratio")
    weight_decay: float = Field(ge=0, description="Weight decay coefficient")
    max_grad_norm: float = Field(gt=0, description="Maximum gradient norm for clipping")
    lr_scheduler_type: str = Field(description="Learning rate scheduler type")
    gradient_checkpointing: bool = Field(description="Enable gradient checkpointing")
    bf16: bool = Field(description="Use bfloat16 precision")
    fp16: bool = Field(description="Use float16 precision")
    use_flash_attention: bool = Field(description="Use Flash Attention optimization")
    mixed_precision: Literal["bf16", "fp16", "no"] = Field(
        description="Mixed precision training mode"
    )

    # === DATA SETTINGS ===
    train_data_path: str = Field(description="Path to training data")
    val_data_path: str = Field(description="Path to validation data")
    data_root: str = Field(description="Root directory for data files")
    max_total_length: int = Field(gt=0, description="Maximum total sequence length")
    teacher_pool_file: str = Field(description="Path to teacher pool file")
    num_teacher_samples: int = Field(gt=0, description="Number of teacher samples")
    collator_type: Literal["standard", "packed"] = Field(
        description="Data collator type"
    )
    teacher_ratio: float = Field(
        ge=0, le=1, description="Ratio of teacher vs student samples"
    )
    language: Literal["chinese", "english"] = Field(description="Primary language")
    max_dataset_size: Optional[int] = Field(
        default=None,
        description="Maximum dataset size for debugging (optional, -1 = use all)",
    )

    @field_validator("max_dataset_size")
    @classmethod
    def validate_max_dataset_size(cls, v):
        """Validate max_dataset_size: None, -1, or positive integer."""
        if v is not None and v != -1 and v <= 0:
            raise ValueError(
                "max_dataset_size must be None, -1 (use all), or a positive integer"
            )
        return v

    # === COORDINATE TOKEN CONFIGURATION ===
    coordinate_tokens_enabled: bool = Field(
        description="Enable coordinate token system"
    )
    max_coord_value: int = Field(gt=0, description="Maximum coordinate value")
    coordinate_loss_weight: float = Field(
        ge=0, description="Weight for coordinate loss"
    )
    regular_loss_weight: float = Field(ge=0, description="Weight for regular LLM loss")

    # === VISION PROCESSING ===
    patch_size: int = Field(gt=0, description="Vision patch size")
    merge_size: int = Field(gt=0, description="Vision merge size")
    temporal_patch_size: int = Field(gt=0, description="Temporal patch size")

    # === TRAINING CONTROL ===
    training_prompt_style: bool = Field(description="Use training prompt style")
    use_consistent_prompts: bool = Field(description="Use consistent prompting")

    # === PERFORMANCE SETTINGS ===
    dataloader_num_workers: int = Field(
        ge=0, description="Number of dataloader workers"
    )
    pin_memory: bool = Field(description="Pin memory in dataloader")
    prefetch_factor: int = Field(gt=0, description="Dataloader prefetch factor")
    remove_unused_columns: bool = Field(
        description="Remove unused columns from dataset"
    )

    # === OUTPUT SETTINGS ===
    output_dir: str = Field(description="Base output directory")
    run_name: str = Field(description="Name of this training run")
    tb_dir: str = Field(description="TensorBoard log directory")

    # === LOSS WEIGHTS ===
    teacher_loss_weight: float = Field(ge=0, description="Weight for teacher loss")
    student_loss_weight: float = Field(ge=0, description="Weight for student loss")

    # === EVALUATION SETTINGS ===
    eval_strategy: Literal["steps", "epoch", "no"] = Field(
        description="Evaluation strategy"
    )
    eval_steps: int = Field(gt=0, description="Steps between evaluations")
    save_strategy: Literal["steps", "epoch", "no"] = Field(
        description="Model saving strategy"
    )
    save_steps: int = Field(gt=0, description="Steps between model saves")
    save_total_limit: int = Field(gt=0, description="Maximum number of saved models")

    # === LOGGING SETTINGS ===
    logging_steps: int = Field(gt=0, description="Steps between log outputs")
    logging_dir: str = Field(description="Directory for log files")
    report_to: Literal["tensorboard", "wandb", "none"] = Field(
        description="Experiment tracking system"
    )
    disable_tqdm: bool = Field(description="Disable tqdm progress bars")
    verbose: bool = Field(description="Enable verbose logging")

    # === COMPUTED PROPERTIES (No YAML needed) ===
    @computed_field
    @property
    def run_output_dir(self) -> str:
        """Computed output directory path."""
        return str(Path(self.output_dir) / self.run_name)

    @computed_field
    @property
    def tensorboard_dir(self) -> str:
        """Computed TensorBoard directory path."""
        return str(Path(self.tb_dir) / self.run_name)

    @computed_field
    @property
    def log_file_dir(self) -> str:
        """Computed log file directory path."""
        return str(Path(self.run_output_dir) / "logs")

    @computed_field
    @property
    def use_differential_lr(self) -> bool:
        """Auto-determine if differential learning rates should be used."""
        lrs = [self.vision_lr, self.merger_lr, self.llm_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1

    # === VALIDATORS ===
    @field_validator("coordinate_loss_weight")
    @classmethod
    def validate_coordinate_loss(cls, v: float, info) -> float:
        """Validate coordinate loss weight when coordinate tokens are enabled."""
        if info.data.get("coordinate_tokens_enabled") and v <= 0:
            raise ValueError(
                "coordinate_loss_weight must be > 0 when coordinate tokens enabled"
            )
        return v

    @field_validator("vision_lr", "merger_lr", "llm_lr")
    @classmethod
    def validate_component_lrs(cls, v: float, info) -> float:
        """Validate that component learning rates are reasonable."""
        main_lr = info.data.get("learning_rate", 1.0)
        if v > main_lr * 10:  # Allow some flexibility
            raise ValueError(
                f"Component LR {v} seems too high compared to main LR {main_lr}"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to create directories."""
        Path(self.run_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_file_dir).mkdir(parents=True, exist_ok=True)


def load_config(yaml_path: str) -> BBUConfig:
    """
    Load configuration from YAML with automatic validation.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        BBUConfig instance with all parameters validated

    Raises:
        ValidationError: If any field is invalid or missing
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    # FAIL-FAST: Validate yaml_path
    if not yaml_path:
        raise ValueError("yaml_path cannot be empty")

    yaml_file = Path(yaml_path)

    # FAIL-FAST: Validate file exists and is a file
    if not yaml_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    if not yaml_file.is_file():
        raise ValueError(f"Configuration path is not a file: {yaml_path}")

    # FAIL-FAST: Load and parse YAML with explicit error handling
    try:
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read configuration file: {e}")

    # FAIL-FAST: Validate yaml_data is a dictionary
    if not isinstance(yaml_data, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(yaml_data)}")

    # FAIL-FAST: Create BBUConfig with explicit error handling
    try:
        config = BBUConfig(**yaml_data)
    except Exception as e:
        raise ValueError(f"Failed to create configuration: {e}")

    return config


# === DOMAIN CONFIG EXTRACTORS (Clean, validated, non-redundant!) ===
@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration extractor with validation."""

    learning_rate: float
    batch_size: int
    epochs: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    vision_lr: float
    merger_lr: float
    llm_lr: float
    adapter_lr: float
    use_flash_attention: bool
    mixed_precision: str
    teacher_loss_weight: float
    student_loss_weight: float

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "TrainingConfig":
        """Create training config from main config with validation."""
        return cls(
            learning_rate=config.learning_rate,
            batch_size=config.per_device_train_batch_size,
            epochs=config.num_train_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            lr_scheduler_type=config.lr_scheduler_type,
            gradient_checkpointing=config.gradient_checkpointing,
            bf16=config.bf16,
            fp16=config.fp16,
            vision_lr=config.vision_lr,
            merger_lr=config.merger_lr,
            llm_lr=config.llm_lr,
            adapter_lr=config.adapter_lr,
            use_flash_attention=config.use_flash_attention,
            mixed_precision=config.mixed_precision,
            teacher_loss_weight=config.teacher_loss_weight,
            student_loss_weight=config.student_loss_weight,
        )

    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs: {self.epochs}")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"Invalid warmup_ratio: {self.warmup_ratio}")


@dataclass(frozen=True)
class CoordinateConfig:
    """Coordinate configuration extractor with validation."""

    coordinate_tokens_enabled: bool
    max_coord_value: int
    coordinate_loss_weight: float
    regular_loss_weight: float

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "CoordinateConfig":
        """Create coordinate config from main config with validation."""
        return cls(
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
            max_coord_value=config.max_coord_value,
            coordinate_loss_weight=config.coordinate_loss_weight,
            regular_loss_weight=config.regular_loss_weight,
        )

    def __post_init__(self):
        """Validate coordinate configuration."""
        if self.coordinate_tokens_enabled:
            if self.max_coord_value <= 0:
                raise ValueError(f"Invalid max_coord_value: {self.max_coord_value}")
            if self.coordinate_loss_weight < 0:
                raise ValueError(
                    f"Invalid coordinate_loss_weight: {self.coordinate_loss_weight}"
                )
            if self.regular_loss_weight < 0:
                raise ValueError(
                    f"Invalid regular_loss_weight: {self.regular_loss_weight}"
                )


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration extractor with validation."""

    model_path: str
    model_size: str
    model_max_length: int
    attn_implementation: str
    torch_dtype: str
    use_cache: bool
    use_cache_inference: bool
    model_hidden_size: int
    model_num_layers: int
    model_num_attention_heads: int
    model_vocab_size: int

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "ModelConfig":
        """Create model config from main config with validation."""
        return cls(
            model_path=config.model_path,
            model_size=config.model_size,
            model_max_length=config.model_max_length,
            attn_implementation=config.attn_implementation,
            torch_dtype=config.torch_dtype,
            use_cache=config.use_cache,
            use_cache_inference=config.use_cache_inference,
            model_hidden_size=config.model_hidden_size,
            model_num_layers=config.model_num_layers,
            model_num_attention_heads=config.model_num_attention_heads,
            model_vocab_size=config.model_vocab_size,
        )

    def __post_init__(self):
        """Validate model configuration."""
        if not self.model_path:
            raise ValueError("model_path cannot be empty")
        if self.model_max_length <= 0:
            raise ValueError(f"Invalid model_max_length: {self.model_max_length}")
        if self.model_hidden_size <= 0:
            raise ValueError(f"Invalid model_hidden_size: {self.model_hidden_size}")
        if self.model_num_layers <= 0:
            raise ValueError(f"Invalid model_num_layers: {self.model_num_layers}")
        if self.model_num_attention_heads <= 0:
            raise ValueError(
                f"Invalid model_num_attention_heads: {self.model_num_attention_heads}"
            )


@dataclass(frozen=True)
class DataConfig:
    """Data configuration extractor with validation."""

    train_data_path: str
    val_data_path: str
    data_root: str
    max_total_length: int
    teacher_pool_file: str
    num_teacher_samples: int
    collator_type: str
    teacher_ratio: float
    max_examples: int
    language: str
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int
    remove_unused_columns: bool
    max_dataset_size: Optional[int]

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "DataConfig":
        """Create data config from main config with validation."""
        return cls(
            train_data_path=config.train_data_path,
            val_data_path=config.val_data_path,
            data_root=config.data_root,
            max_total_length=config.max_total_length,
            teacher_pool_file=config.teacher_pool_file,
            num_teacher_samples=config.num_teacher_samples,
            collator_type=config.collator_type,
            teacher_ratio=config.teacher_ratio,
            max_examples=config.max_examples,
            language=config.language,
            dataloader_num_workers=config.dataloader_num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            remove_unused_columns=config.remove_unused_columns,
            max_dataset_size=config.max_dataset_size,
        )

    def __post_init__(self):
        """Validate data configuration."""
        if not self.train_data_path:
            raise ValueError("train_data_path cannot be empty")
        if not self.val_data_path:
            raise ValueError("val_data_path cannot be empty")
        if not self.data_root:
            raise ValueError("data_root cannot be empty")
        if self.max_total_length <= 0:
            raise ValueError(f"Invalid max_total_length: {self.max_total_length}")
        if self.num_teacher_samples <= 0:
            raise ValueError(f"Invalid num_teacher_samples: {self.num_teacher_samples}")
        if not 0 <= self.teacher_ratio <= 1:
            raise ValueError(f"Invalid teacher_ratio: {self.teacher_ratio}")
        if self.dataloader_num_workers < 0:
            raise ValueError(
                f"Invalid dataloader_num_workers: {self.dataloader_num_workers}"
            )


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration extractor with validation."""

    logging_steps: int
    logging_dir: str
    report_to: str
    disable_tqdm: bool
    verbose: bool
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "LoggingConfig":
        """Create logging config from main config with validation."""
        return cls(
            logging_steps=config.logging_steps,
            logging_dir=config.logging_dir,
            report_to=config.report_to,
            disable_tqdm=config.disable_tqdm,
            verbose=config.verbose,
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
        )

    def __post_init__(self):
        """Validate logging configuration."""
        if self.logging_steps <= 0:
            raise ValueError(f"Invalid logging_steps: {self.logging_steps}")
        if not self.logging_dir:
            raise ValueError("logging_dir cannot be empty")
        valid_eval_strategies = ["steps", "epoch", "no"]
        if self.eval_strategy not in valid_eval_strategies:
            raise ValueError(
                f"Invalid eval_strategy: {self.eval_strategy}. Must be one of {valid_eval_strategies}"
            )


@dataclass(frozen=True)
class VisionConfig:
    """Vision configuration extractor with validation."""

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    training_prompt_style: bool
    use_consistent_prompts: bool

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "VisionConfig":
        """Create vision config from main config with validation."""
        return cls(
            patch_size=config.patch_size,
            merge_size=config.merge_size,
            temporal_patch_size=config.temporal_patch_size,
            training_prompt_style=config.training_prompt_style,
            use_consistent_prompts=config.use_consistent_prompts,
        )

    def __post_init__(self):
        """Validate vision configuration."""
        if self.patch_size <= 0:
            raise ValueError(f"Invalid patch_size: {self.patch_size}")
        if self.merge_size <= 0:
            raise ValueError(f"Invalid merge_size: {self.merge_size}")
        if self.temporal_patch_size <= 0:
            raise ValueError(f"Invalid temporal_patch_size: {self.temporal_patch_size}")


import threading


# Global config for modules that need it
_global_config: BBUConfig | None = None
_config_lock = threading.RLock()  # Reentrant lock for thread safety


def get_config() -> BBUConfig:
    """
    Get the global configuration instance (thread-safe).

    Returns:
        BBUConfig instance

    Raises:
        RuntimeError: If config has not been initialized
    """
    with _config_lock:
        if _global_config is None:
            raise RuntimeError("Config not initialized. Call init_config() first.")
        return _global_config


def init_config(yaml_path: str) -> BBUConfig:
    """
    Initialize the global configuration from a YAML file (thread-safe).

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        BBUConfig instance

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the YAML file is invalid
        RuntimeError: If config is already initialized (call reset_config() first)
    """
    global _global_config

    with _config_lock:
        # FAIL-FAST: Check if already initialized
        if _global_config is not None:
            raise RuntimeError(
                "Config already initialized. Call reset_config() first if you need to reinitialize."
            )

        # FAIL-FAST: Validate yaml_path
        if not yaml_path:
            raise ValueError("yaml_path cannot be empty")

        # FAIL-FAST: Validate file exists
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        if not yaml_file.is_file():
            raise ValueError(f"Configuration path is not a file: {yaml_path}")

        _global_config = load_config(yaml_path)
        return _global_config


def reset_config() -> None:
    """
    Reset the global configuration (thread-safe).

    This is primarily for testing purposes.
    """
    global _global_config
    with _config_lock:
        _global_config = None


def is_config_initialized() -> bool:
    """
    Check if the global configuration is initialized (thread-safe).

    Returns:
        True if config is initialized, False otherwise
    """
    with _config_lock:
        return _global_config is not None


# === DOMAIN CONFIG PROPERTY GROUPS (Added after class definition) ===


def _add_domain_properties():
    """Add domain config property methods to BBUConfig class."""

    def training_config(self) -> "TrainingConfig":
        """Access training-specific configuration as a typed object."""
        return TrainingConfig.from_bbu_config(self)

    def coordinate_config(self) -> "CoordinateConfig":
        """Access coordinate token configuration as a typed object."""
        return CoordinateConfig.from_bbu_config(self)

    def model_config(self) -> "ModelConfig":
        """Access model configuration as a typed object."""
        return ModelConfig.from_bbu_config(self)

    def data_config(self) -> "DataConfig":
        """Access data configuration as a typed object."""
        return DataConfig.from_bbu_config(self)

    def logging_config(self) -> "LoggingConfig":
        """Access logging configuration as a typed object."""
        return LoggingConfig.from_bbu_config(self)

    def vision_config(self) -> "VisionConfig":
        """Access vision configuration as a typed object."""
        return VisionConfig.from_bbu_config(self)

    # Add the property methods to BBUConfig class
    BBUConfig.training_config = property(training_config)
    BBUConfig.coordinate_config = property(coordinate_config)
    BBUConfig.model_config = property(model_config)
    BBUConfig.data_config = property(data_config)
    BBUConfig.logging_config = property(logging_config)
    BBUConfig.vision_config = property(vision_config)


# Call the function to add the properties
_add_domain_properties()
