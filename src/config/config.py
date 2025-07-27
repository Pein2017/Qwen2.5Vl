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

from pathlib import Path
from typing import Any, Literal

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
    coordinate_lr: float = Field(ge=0, description="Coordinate token learning rate")
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
    max_examples: int = Field(gt=0, description="Maximum number of examples")
    language: Literal["chinese", "english"] = Field(description="Primary language")

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
    detection_freeze_epochs: int = Field(
        ge=0, description="Epochs to freeze detection training"
    )

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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        description="Logging level"
    )
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
        lrs = [self.vision_lr, self.merger_lr, self.llm_lr, self.coordinate_lr]
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


# === DOMAIN CONFIG EXTRACTORS (No more redundant classes!) ===
class TrainingConfig:
    """Training configuration extractor - no redundant field definitions!"""

    def __init__(self, config: BBUConfig):
        self.learning_rate = config.learning_rate
        self.batch_size = config.per_device_train_batch_size
        self.epochs = config.num_train_epochs
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.warmup_ratio = config.warmup_ratio
        self.weight_decay = config.weight_decay
        self.max_grad_norm = config.max_grad_norm
        self.lr_scheduler_type = config.lr_scheduler_type
        self.gradient_checkpointing = config.gradient_checkpointing
        self.bf16 = config.bf16
        self.fp16 = config.fp16
        self.vision_lr = config.vision_lr
        self.merger_lr = config.merger_lr
        self.llm_lr = config.llm_lr
        self.coordinate_lr = config.coordinate_lr
        self.adapter_lr = config.adapter_lr
        self.detection_freeze_epochs = config.detection_freeze_epochs
        self.use_flash_attention = config.use_flash_attention
        self.mixed_precision = config.mixed_precision
        self.teacher_loss_weight = config.teacher_loss_weight
        self.student_loss_weight = config.student_loss_weight

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "TrainingConfig":
        """Create training config from main config."""
        return cls(config)


class CoordinateConfig:
    """Coordinate configuration extractor - no redundant field definitions!"""

    def __init__(self, config: BBUConfig):
        self.coordinate_tokens_enabled = config.coordinate_tokens_enabled
        self.max_coord_value = config.max_coord_value
        self.coordinate_loss_weight = config.coordinate_loss_weight
        self.regular_loss_weight = config.regular_loss_weight

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "CoordinateConfig":
        """Create coordinate config from main config."""
        return cls(config)


class ModelConfig:
    """Model configuration extractor - no redundant field definitions!"""

    def __init__(self, config: BBUConfig):
        self.model_path = config.model_path
        self.model_size = config.model_size
        self.model_max_length = config.model_max_length
        self.attn_implementation = config.attn_implementation
        self.torch_dtype = config.torch_dtype
        self.use_cache = config.use_cache
        self.use_cache_inference = config.use_cache_inference
        self.model_hidden_size = config.model_hidden_size
        self.model_num_layers = config.model_num_layers
        self.model_num_attention_heads = config.model_num_attention_heads
        self.model_vocab_size = config.model_vocab_size

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "ModelConfig":
        """Create model config from main config."""
        return cls(config)


class DataConfig:
    """Data configuration extractor - no redundant field definitions!"""

    def __init__(self, config: BBUConfig):
        self.train_data_path = config.train_data_path
        self.val_data_path = config.val_data_path
        self.data_root = config.data_root
        self.max_total_length = config.max_total_length
        self.teacher_pool_file = config.teacher_pool_file
        self.num_teacher_samples = config.num_teacher_samples
        self.collator_type = config.collator_type
        self.teacher_ratio = config.teacher_ratio
        self.max_examples = config.max_examples
        self.language = config.language
        self.dataloader_num_workers = config.dataloader_num_workers
        self.pin_memory = config.pin_memory
        self.prefetch_factor = config.prefetch_factor
        self.remove_unused_columns = config.remove_unused_columns

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "DataConfig":
        """Create data config from main config."""
        return cls(config)


class LoggingConfig:
    """Logging configuration extractor - no redundant field definitions!"""

    def __init__(self, config: BBUConfig):
        self.logging_steps = config.logging_steps
        self.logging_dir = config.logging_dir
        self.log_level = config.log_level
        self.report_to = config.report_to
        self.disable_tqdm = config.disable_tqdm
        self.verbose = config.verbose
        self.eval_strategy = config.eval_strategy
        self.eval_steps = config.eval_steps
        self.save_strategy = config.save_strategy
        self.save_steps = config.save_steps
        self.save_total_limit = config.save_total_limit

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "LoggingConfig":
        """Create logging config from main config."""
        return cls(config)


class VisionConfig:
    """Vision configuration extractor - no redundant field definitions!"""

    def __init__(self, config: BBUConfig):
        self.patch_size = config.patch_size
        self.merge_size = config.merge_size
        self.temporal_patch_size = config.temporal_patch_size
        self.training_prompt_style = config.training_prompt_style
        self.use_consistent_prompts = config.use_consistent_prompts

    @classmethod
    def from_bbu_config(cls, config: BBUConfig) -> "VisionConfig":
        """Create vision config from main config."""
        return cls(config)


# Global config for modules that need it
_global_config: BBUConfig | None = None


def get_config() -> BBUConfig:
    """
    Get the global configuration instance.

    Returns:
        BBUConfig instance

    Raises:
        RuntimeError: If config has not been initialized
    """
    if _global_config is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return _global_config


def init_config(yaml_path: str) -> BBUConfig:
    """
    Initialize the global configuration from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        BBUConfig instance

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the YAML file is invalid
    """
    # FAIL-FAST: Validate yaml_path
    if not yaml_path:
        raise ValueError("yaml_path cannot be empty")

    # FAIL-FAST: Validate file exists
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    if not yaml_file.is_file():
        raise ValueError(f"Configuration path is not a file: {yaml_path}")

    global _global_config
    _global_config = load_config(yaml_path)
    return _global_config
