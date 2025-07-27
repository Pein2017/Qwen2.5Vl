"""
Domain-specific Configuration Classes for BBU Training

This module provides typed, validated configuration classes for different
domain areas, extracted from the main BBUConfig. Each domain config
includes validation and fail-fast error handling.

Usage:
    # Load main config
    bbu_config = load_config("configs/bbu_v2.yaml")

    # Extract domain configs with validation
    training_config = TrainingConfig.from_bbu_config(bbu_config)
    model_config = ModelConfig.from_bbu_config(bbu_config)
    data_config = DataConfig.from_bbu_config(bbu_config)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config.config import BBUConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Training-specific configuration with validation."""

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

    # Differential learning rates
    vision_lr: float
    merger_lr: float
    llm_lr: float
    coordinate_lr: float
    adapter_lr: float

    # Training control
    detection_freeze_epochs: int
    use_flash_attention: bool
    mixed_precision: str

    # Loss weights
    teacher_loss_weight: float
    student_loss_weight: float

    @classmethod
    def from_bbu_config(cls, config: "BBUConfig") -> "TrainingConfig":
        """Extract training config with validation."""
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
            coordinate_lr=config.coordinate_lr,
            adapter_lr=config.adapter_lr,
            detection_freeze_epochs=config.detection_freeze_epochs,
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
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"Invalid gradient_accumulation_steps: {self.gradient_accumulation_steps}"
            )
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"Invalid warmup_ratio: {self.warmup_ratio}")


@dataclass(frozen=True)
class ModelConfig:
    """Model-specific configuration with validation."""

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
    def from_bbu_config(cls, config: "BBUConfig") -> "ModelConfig":
        """Extract model config with validation."""
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
    """Data-specific configuration with validation."""

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

    # Performance settings
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int
    remove_unused_columns: bool

    @classmethod
    def from_bbu_config(cls, config: "BBUConfig") -> "DataConfig":
        """Extract data config with validation."""
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
class CoordinateConfig:
    """Coordinate token configuration with validation."""

    coordinate_tokens_enabled: bool
    max_coord_value: int
    coordinate_loss_weight: float
    regular_loss_weight: float

    @classmethod
    def from_bbu_config(cls, config: "BBUConfig") -> "CoordinateConfig":
        """Extract coordinate config with validation."""
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
class LoggingConfig:
    """Logging configuration with validation."""

    logging_steps: int
    logging_dir: str
    log_level: str
    report_to: str
    disable_tqdm: bool
    verbose: bool

    # Evaluation settings
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int

    @classmethod
    def from_bbu_config(cls, config: "BBUConfig") -> "LoggingConfig":
        """Extract logging config with validation."""
        return cls(
            logging_steps=config.logging_steps,
            logging_dir=config.logging_dir,
            log_level=config.log_level,
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
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}"
            )
        valid_eval_strategies = ["steps", "epoch", "no"]
        if self.eval_strategy not in valid_eval_strategies:
            raise ValueError(
                f"Invalid eval_strategy: {self.eval_strategy}. Must be one of {valid_eval_strategies}"
            )


@dataclass(frozen=True)
class VisionConfig:
    """Vision processing configuration with validation."""

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    training_prompt_style: bool
    use_consistent_prompts: bool

    @classmethod
    def from_bbu_config(cls, config: "BBUConfig") -> "VisionConfig":
        """Extract vision config with validation."""
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
