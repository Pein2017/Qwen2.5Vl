"""
Explicit Configuration System for Qwen2.5-VL Training

This module provides a strict configuration system that eliminates all implicit defaults
and requires every parameter to be explicitly defined in YAML configuration files.

Key Features:
- No implicit defaults - all parameters must be in YAML
- Fail-fast validation with clear error messages
- Type safety with comprehensive validation
- Eliminates 200+ fallback patterns from codebase

Usage:
    config = load_explicit_config("configs/training.yaml")
    # All parameters guaranteed to be explicitly set
    learning_rate = config.learning_rate  # No fallbacks needed
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(kw_only=True)
class ExplicitDirectConfig:
    """
    Explicit configuration schema - NO DEFAULTS ALLOWED
    Every field must be provided in YAML configuration
    """

    # === CORE MODEL SETTINGS (REQUIRED) ===
    model_path: str
    model_size: str  # "3B" | "7B"
    model_max_length: int
    attn_implementation: str  # "flash_attention_2" | "eager" | "sdpa"
    torch_dtype: str  # "bfloat16" | "float16" | "float32"
    use_cache: bool
    use_cache_inference: bool
    model_hidden_size: int
    model_num_layers: int
    model_num_attention_heads: int
    model_vocab_size: int

    # === TRAINING SETTINGS (REQUIRED) ===
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    vision_lr: float
    merger_lr: float
    llm_lr: float
    coordinate_lr: float
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

    # === DATA SETTINGS (REQUIRED) ===
    train_data_path: str
    val_data_path: str
    data_root: str
    max_total_length: int
    teacher_pool_file: str
    num_teacher_samples: int
    collator_type: str  # "standard" | "packed"
    teacher_ratio: float
    max_examples: int
    language: str  # "chinese" | "english"

    # === COORDINATE TOKEN CONFIGURATION (SIMPLIFIED) ===
    coordinate_tokens_enabled: bool  # Enable coordinate token system
    max_coord_value: (
        int  # Maximum coordinate value (creates tokens [0, max_coord_value-1])
    )
    coordinate_loss_weight: float  # Weight for coordinate loss
    regular_loss_weight: float  # Weight for regular LLM loss

    # === TRAINING STABILITY SETTINGS (SIMPLIFIED) ===
    # max_grad_norm already defined above in training settings

    # === VISION PROCESSING (ALL REQUIRED) ===
    patch_size: int
    merge_size: int
    temporal_patch_size: int

    # === TRAINING CONTROL (ALL REQUIRED) ===
    training_prompt_style: bool
    use_consistent_prompts: bool
    detection_freeze_epochs: int

    # === PERFORMANCE SETTINGS (ALL REQUIRED) ===
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int
    remove_unused_columns: bool

    # === OUTPUT SETTINGS (ALL REQUIRED) ===
    output_dir: str
    run_name: str
    tb_dir: str

    # === LOSS WEIGHTS (ALL REQUIRED) ===
    teacher_loss_weight: float
    student_loss_weight: float

    # === EVALUATION SETTINGS (ALL REQUIRED) ===
    eval_strategy: str  # "steps" | "epoch" | "no"
    eval_steps: int
    save_strategy: str  # "steps" | "epoch" | "no"
    save_steps: int
    save_total_limit: int

    # === LOGGING SETTINGS (ALL REQUIRED) ===
    logging_steps: int
    logging_dir: str
    log_level: str  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    report_to: str  # "tensorboard" | "wandb" | "none"
    disable_tqdm: bool
    verbose: bool

    # === CHAT PROCESSOR SETTINGS ===
    # No longer needed - max_coord_value is used directly

    # === RUNTIME PROPERTIES (COMPUTED, NOT IN YAML) ===
    run_output_dir: str = field(init=False)
    tensorboard_dir: str = field(init=False)
    log_file_dir: str = field(init=False)

    @property
    def use_differential_lr(self) -> bool:
        """Auto-determine if differential learning rates should be used."""
        lrs = [self.vision_lr, self.merger_lr, self.llm_lr, self.coordinate_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1

    def __post_init__(self):
        """Compute runtime properties and validate configuration."""
        # Compute derived paths
        self.run_output_dir = str(Path(self.output_dir) / self.run_name)
        self.tensorboard_dir = str(Path(self.tb_dir) / self.run_name)
        self.log_file_dir = str(Path(self.run_output_dir) / "logs")

        # Create directories
        Path(self.run_output_dir).mkdir(parents=True, exist_ok=True)

        # Convert scientific notation strings to proper types
        self._convert_scientific_notation()

        # Validate configuration
        self._validate_all_parameters()

    def _convert_scientific_notation(self):
        """Convert string scientific notation to proper float types."""
        # Direct conversion without getattr/setattr - fast-fail mode
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)
        if isinstance(self.vision_lr, str):
            self.vision_lr = float(self.vision_lr)
        if isinstance(self.merger_lr, str):
            self.merger_lr = float(self.merger_lr)
        if isinstance(self.llm_lr, str):
            self.llm_lr = float(self.llm_lr)
        if isinstance(self.coordinate_lr, str):
            self.coordinate_lr = float(self.coordinate_lr)
        if isinstance(self.adapter_lr, str):
            self.adapter_lr = float(self.adapter_lr)
        if isinstance(self.warmup_ratio, str):
            self.warmup_ratio = float(self.warmup_ratio)
        if isinstance(self.weight_decay, str):
            self.weight_decay = float(self.weight_decay)
        if isinstance(self.max_grad_norm, str):
            self.max_grad_norm = float(self.max_grad_norm)
        if isinstance(self.teacher_ratio, str):
            self.teacher_ratio = float(self.teacher_ratio)
        if isinstance(self.coordinate_loss_weight, str):
            self.coordinate_loss_weight = float(self.coordinate_loss_weight)
        if isinstance(self.regular_loss_weight, str):
            self.regular_loss_weight = float(self.regular_loss_weight)
        if isinstance(self.teacher_loss_weight, str):
            self.teacher_loss_weight = float(self.teacher_loss_weight)
        if isinstance(self.student_loss_weight, str):
            self.student_loss_weight = float(self.student_loss_weight)

    def _validate_all_parameters(self):
        """Comprehensive validation with fail-fast behavior."""
        errors = []

        # Validate learning rates
        lr_fields = [
            ("learning_rate", self.learning_rate),
            ("vision_lr", self.vision_lr),
            ("merger_lr", self.merger_lr),
            ("llm_lr", self.llm_lr),
            ("coordinate_lr", self.coordinate_lr),
            ("adapter_lr", self.adapter_lr),
        ]

        for field_name, lr_value in lr_fields:
            if lr_value < 0:
                errors.append(
                    f"Learning rate '{field_name}' cannot be negative: {lr_value}"
                )
            elif lr_value > 1.0:
                errors.append(
                    f"Learning rate '{field_name}' seems too high: {lr_value}"
                )

        # Validate coordinate configuration
        if self.coordinate_tokens_enabled:
            if self.max_coord_value <= 0:
                errors.append(
                    "max_coord_value must be positive when coordinate tokens are enabled"
                )
            if self.coordinate_loss_weight < 0:
                errors.append("coordinate_loss_weight cannot be negative")
            if self.regular_loss_weight < 0:
                errors.append("regular_loss_weight cannot be negative")

        # Validate enum values
        valid_torch_dtypes = ["bfloat16", "float16", "float32"]
        if self.torch_dtype not in valid_torch_dtypes:
            errors.append(
                f"Invalid torch_dtype: {self.torch_dtype}. Must be one of {valid_torch_dtypes}"
            )

        valid_mixed_precision = ["bf16", "fp16", "no"]
        if self.mixed_precision not in valid_mixed_precision:
            errors.append(
                f"Invalid mixed_precision: {self.mixed_precision}. Must be one of {valid_mixed_precision}"
            )

        valid_collator_types = ["standard", "packed"]
        if self.collator_type not in valid_collator_types:
            errors.append(
                f"Invalid collator_type: {self.collator_type}. Must be one of {valid_collator_types}"
            )

        valid_languages = ["chinese", "english"]
        if self.language not in valid_languages:
            errors.append(
                f"Invalid language: {self.language}. Must be one of {valid_languages}"
            )

        # Fail fast if any errors found
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_msg)


def load_explicit_config(yaml_path: str) -> ExplicitDirectConfig:
    """
    Load configuration with strict validation - no defaults allowed.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        ExplicitDirectConfig instance with all parameters validated

    Raises:
        ValueError: If any required field is missing or invalid
        FileNotFoundError: If YAML file doesn't exist
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)

    try:
        # This will fail immediately if any required field is missing
        config = ExplicitDirectConfig(**yaml_data)
        # Set global config instance
        set_explicit_config(config)
        return config
    except TypeError as e:
        # Convert TypeError to more helpful message
        missing_fields = _extract_missing_fields(str(e))
        raise ValueError(
            f"Missing required configuration fields in {yaml_path}:\n"
            f"  {missing_fields}\n"
            f"All parameters must be explicitly defined. No defaults are provided."
        ) from e


def _extract_missing_fields(error_message: str) -> str:
    """Extract missing field names from TypeError message."""
    if "missing" in error_message and "required" in error_message:
        # Extract field names from error message
        # Example: "__init__() missing 1 required keyword-only argument: 'field_name'"
        import re

        match = re.search(r"argument(?:s)?: (.+)", error_message)
        if match:
            return match.group(1)
    return error_message


def validate_config_completeness(yaml_path: str) -> List[str]:
    """
    Validate that YAML file contains all required fields.

    Returns:
        List of missing field names (empty if all fields present)
    """
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Get all required fields from dataclass
    from dataclasses import fields

    required_fields = {f.name for f in fields(ExplicitDirectConfig) if f.init}

    # Remove computed fields
    computed_fields = {"run_output_dir", "tensorboard_dir", "log_file_dir"}
    required_fields -= computed_fields

    # Check which fields are missing
    provided_fields = set(yaml_data.keys())
    missing_fields = required_fields - provided_fields

    return sorted(missing_fields)


# Global configuration instance (will be set by load_explicit_config)
explicit_config: Optional[ExplicitDirectConfig] = None


def get_explicit_config() -> ExplicitDirectConfig:
    """Get the global explicit configuration instance."""
    if explicit_config is None:
        raise RuntimeError(
            "Configuration not initialized. Call load_explicit_config() first."
        )
    return explicit_config


def set_explicit_config(config: ExplicitDirectConfig) -> None:
    """Set the global explicit configuration instance."""
    global explicit_config
    explicit_config = config
