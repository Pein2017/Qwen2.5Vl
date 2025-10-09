"""Dataclass schema for the layered Qwen2.5-VL configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Sequence, Tuple

from .augmentation_config import (
    AugmentationConfig,
    PhotometricConfig,
    SmartResizeConfig,
)


class SchemaError(ValueError):
    """Raised when a schema section fails validation."""


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    attn_implementation: str
    torch_dtype: str
    use_cache: bool
    merge_size: int
    max_pixels: int

    def __post_init__(self) -> None:
        if not self.model_path:
            raise SchemaError("model.model_path cannot be empty")
        if not self.attn_implementation:
            raise SchemaError("model.attn_implementation cannot be empty")
        if not self.torch_dtype:
            raise SchemaError("model.torch_dtype cannot be empty")
        if self.merge_size <= 0:
            raise SchemaError("model.merge_size must be > 0")
        if self.max_pixels <= 0:
            raise SchemaError("model.max_pixels must be > 0")


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    teacher_pool_file: Optional[str] = None
    max_total_length: int = 0
    num_teacher_samples: int = 0
    max_dataset_size: int = -1

    def __post_init__(self) -> None:
        if not self.data_root:
            raise SchemaError("data.data_root cannot be empty")
        if self.max_total_length <= 0:
            raise SchemaError("data.max_total_length must be > 0")
        if self.num_teacher_samples < 0:
            raise SchemaError("data.num_teacher_samples must be >= 0")


@dataclass(frozen=True)
class TrainingConfigSection:
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    vision_lr: float
    merger_lr: float
    llm_lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    collator_type: str
    dataloader_num_workers: int
    dataloader_pin_memory: bool  # Fixed field name to match config
    prefetch_factor: int
    remove_unused_columns: bool
    max_steps: int = -1  # Add missing max_steps field
    conversation_variant_ratios: Optional[Dict[str, float]] = None
    lr_merger: Optional[float] = None
    lr_full_model: Optional[float] = None

    def __post_init__(self) -> None:
        required_positive = (
            ("num_train_epochs", self.num_train_epochs),
            ("per_device_train_batch_size", self.per_device_train_batch_size),
            ("per_device_eval_batch_size", self.per_device_eval_batch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
        )
        for key, value in required_positive:
            if value <= 0:
                raise SchemaError(f"training.{key} must be > 0")
        if self.learning_rate <= 0:
            raise SchemaError("training.learning_rate must be > 0")
        if self.max_grad_norm <= 0:
            raise SchemaError("training.max_grad_norm must be > 0")
        if self.weight_decay < 0:
            raise SchemaError("training.weight_decay must be >= 0")
        if self.prefetch_factor < 0:
            raise SchemaError("training.prefetch_factor must be >= 0")
        if self.dataloader_num_workers < 0:
            raise SchemaError("training.dataloader_num_workers must be >= 0")


@dataclass(frozen=True)
class GroupLossWeights:
    caption: float
    grounding: float
    formatting: float

    def __post_init__(self) -> None:
        if self.caption < 0 or self.grounding < 0 or self.formatting < 0:
            raise SchemaError("loss.grouped weights must be >= 0")


@dataclass(frozen=True)
class LossConfig:
    teacher_loss_weight: float
    student_loss_weight: float
    grouped: GroupLossWeights

    def __post_init__(self) -> None:
        if self.teacher_loss_weight < 0:
            raise SchemaError("loss.teacher_loss_weight must be >= 0")
        if self.student_loss_weight < 0:
            raise SchemaError("loss.student_loss_weight must be >= 0")
        if self.teacher_loss_weight + self.student_loss_weight <= 0:
            raise SchemaError(
                "loss.teacher_loss_weight + loss.student_loss_weight must be > 0"
            )


@dataclass(frozen=True)
class AugmentationScheduleItem:
    start_epoch: int
    preset: str
    smart_resize: Optional[SmartResizeConfig] = None

    def __post_init__(self) -> None:
        if self.start_epoch < 0:
            raise SchemaError("features.augmentation_schedule.start_epoch must be >= 0")
        if not self.preset:
            raise SchemaError("features.augmentation_schedule.preset cannot be empty")


@dataclass(frozen=True)
class AugmentationFeatures:
    enabled: bool
    config: Optional[AugmentationConfig]
    schedule: Tuple[AugmentationScheduleItem, ...]
    smart_resize_defaults: Optional[SmartResizeConfig]

    def __post_init__(self) -> None:
        if self.enabled and (self.config is None and not self.schedule):
            raise SchemaError(
                "features.augmentation requires config or schedule when enabled"
            )
        if not self.enabled and self.config is not None:
            raise SchemaError(
                "features.augmentation.config must be null when augmentation is disabled"
            )


@dataclass(frozen=True)
class TeacherAugmentationConfig:
    enabled: bool
    config: Optional[AugmentationConfig]

    def __post_init__(self) -> None:
        if self.enabled:
            if self.config is None:
                raise SchemaError(
                    "features.teacher_augmentation.config is required when enabled"
                )
            if not isinstance(self.config.photometric, PhotometricConfig):
                raise SchemaError(
                    "features.teacher_augmentation requires photometric block"
                )
        elif self.config is not None:
            raise SchemaError(
                "features.teacher_augmentation.config must be null when disabled"
            )


@dataclass(frozen=True)
class TeacherPairingConfig:
    enabled: bool
    teacher_ratio: Optional[float]
    dynamic_pairing_enabled: bool
    dynamic_pair_cross_bucket_explore_prob: Optional[float]

    def __post_init__(self) -> None:
        if self.enabled:
            if self.teacher_ratio is None:
                raise SchemaError(
                    "features.teacher_pairing.teacher_ratio is required when enabled"
                )
            if self.teacher_ratio < 0:
                raise SchemaError("features.teacher_pairing.teacher_ratio must be >= 0")
            if self.dynamic_pairing_enabled and (
                self.dynamic_pair_cross_bucket_explore_prob is None
            ):
                raise SchemaError(
                    "features.teacher_pairing.dynamic_pair_cross_bucket_explore_prob is required when dynamic pairing is enabled"
                )
        else:
            if self.teacher_ratio not in (None, 0.0):
                raise SchemaError(
                    "features.teacher_pairing.teacher_ratio must be null/0 when disabled"
                )


@dataclass(frozen=True)
class PhaseConfig:
    enabled: bool
    name: Optional[str]
    llm_top_k_block: Optional[int]
    vision_top_k_block: Optional[int]
    freeze_patch_embed: Optional[bool]

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not self.name:
            raise SchemaError("features.phase.name is required when enabled")
        normalized = self.name.lower()
        if normalized == "phase_2":
            if self.llm_top_k_block is None:
                raise SchemaError(
                    "features.phase.llm_top_k_block is required for phase_2"
                )
            if self.llm_top_k_block not in (-1,) and self.llm_top_k_block <= 0:
                raise SchemaError(
                    "features.phase.llm_top_k_block must be >0 or -1 when phase is phase_2"
                )
        if normalized == "phase_1" and self.llm_top_k_block not in (None, 0):
            raise SchemaError(
                "features.phase.llm_top_k_block must be 0 when phase is phase_1"
            )


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool
    save_steps: Optional[int]
    save_total_limit: Optional[int]
    metric_for_best_model: Optional[str]
    greater_is_better: Optional[bool]
    save_strategy: Optional[str]
    load_best_model_at_end: Optional[bool]
    min_interval_steps: Optional[int]
    interval_multiplier_of_eval_steps: Optional[int]

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        required = (
            ("save_steps", self.save_steps),
            ("save_total_limit", self.save_total_limit),
            ("metric_for_best_model", self.metric_for_best_model),
            ("greater_is_better", self.greater_is_better),
            ("save_strategy", self.save_strategy),
            ("load_best_model_at_end", self.load_best_model_at_end),
        )
        for key, value in required:
            if value is None:
                raise SchemaError(f"features.checkpoint.{key} is required when enabled")
        provided = sum(
            1
            for v in (self.min_interval_steps, self.interval_multiplier_of_eval_steps)
            if v is not None
        )
        if provided != 1:
            raise SchemaError(
                "features.checkpoint requires exactly one of min_interval_steps or interval_multiplier_of_eval_steps"
            )
        if self.min_interval_steps is not None and self.min_interval_steps < 0:
            raise SchemaError(
                "features.checkpoint.min_interval_steps must be >= 0 when provided"
            )
        if (
            self.interval_multiplier_of_eval_steps is not None
            and self.interval_multiplier_of_eval_steps <= 0
        ):
            raise SchemaError(
                "features.checkpoint.interval_multiplier_of_eval_steps must be > 0"
            )


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool
    logging_steps: Optional[int]
    report_to: Optional[str]
    disable_tqdm: Optional[bool]

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.logging_steps is None or self.logging_steps <= 0:
            raise SchemaError("features.logging.logging_steps must be > 0 when enabled")
        if not self.report_to:
            raise SchemaError("features.logging.report_to is required when enabled")
        if self.disable_tqdm is None:
            raise SchemaError("features.logging.disable_tqdm is required when enabled")


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool
    strategy: Optional[str]
    eval_steps: Optional[int]

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not self.strategy:
            raise SchemaError("features.evaluation.strategy is required when enabled")
        if self.eval_steps is None or self.eval_steps <= 0:
            raise SchemaError("features.evaluation.eval_steps must be > 0 when enabled")


@dataclass(frozen=True)
class FeaturesConfig:
    augmentation: AugmentationFeatures
    teacher_augmentation: TeacherAugmentationConfig
    teacher_pairing: TeacherPairingConfig
    phase: PhaseConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig


@dataclass(frozen=True)
class OutputConfig:
    run_name: str
    output_dir: str
    tb_dir: str

    def __post_init__(self) -> None:
        if not self.run_name:
            raise SchemaError("output.run_name cannot be empty")
        if not self.output_dir:
            raise SchemaError("output.output_dir cannot be empty")
        if not self.tb_dir:
            raise SchemaError("output.tb_dir cannot be empty")

    @property
    def run_output_dir(self) -> str:
        return str(Path(self.output_dir) / self.run_name)

    @property
    def tensorboard_dir(self) -> str:
        return str(Path(self.tb_dir) / self.run_name)

    @property
    def log_file_dir(self) -> str:
        return str(Path(self.run_output_dir) / "logs")


@dataclass(frozen=True)
class RuntimeConfig:
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.seed is not None and self.seed < 0:
            raise SchemaError("runtime.seed must be >= 0 when provided")


@dataclass(frozen=True)
class AdvancedConfig:
    span_include_im_end_in_labels: bool
    debug_alignment: bool
    augmentation_smart_resize_defaults: Optional[SmartResizeConfig]
    trainable_token_strings: Optional[Tuple[str, ...]] = None
    # Legacy loss weights (disabled in src_new)
    teacher_loss_weight: float = 0.5
    student_loss_weight: float = 1.0
    caption_loss_weight: float = 1.0
    grounding_loss_weight: float = 1.0
    formatting_loss_weight: float = 1.0  # Use new geometry tokens format


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfigSection
    loss: LossConfig
    features: FeaturesConfig
    output: OutputConfig
    runtime: RuntimeConfig
    advanced: AdvancedConfig

    _FLAT_ATTR_MAP: ClassVar[Dict[str, Sequence[str]]] = {
        "model_path": ("model", "model_path"),
        "attn_implementation": ("model", "attn_implementation"),
        "torch_dtype": ("model", "torch_dtype"),
        "use_cache": ("model", "use_cache"),
        "merge_size": ("model", "merge_size"),
        "max_pixels": ("model", "max_pixels"),
        "data_root": ("data", "data_root"),
        "train_data_path": ("data", "train_data_path"),
        "val_data_path": ("data", "val_data_path"),
        "teacher_pool_file": ("data", "teacher_pool_file"),
        "max_total_length": ("data", "max_total_length"),
        "num_teacher_samples": ("data", "num_teacher_samples"),
        "max_dataset_size": ("data", "max_dataset_size"),
        "num_train_epochs": ("training", "num_train_epochs"),
        "per_device_train_batch_size": ("training", "per_device_train_batch_size"),
        "per_device_eval_batch_size": ("training", "per_device_eval_batch_size"),
        "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
        "learning_rate": ("training", "learning_rate"),
        "vision_lr": ("training", "vision_lr"),
        "merger_lr": ("training", "merger_lr"),
        "llm_lr": ("training", "llm_lr"),
        "warmup_ratio": ("training", "warmup_ratio"),
        "weight_decay": ("training", "weight_decay"),
        "max_grad_norm": ("training", "max_grad_norm"),
        "lr_scheduler_type": ("training", "lr_scheduler_type"),
        "gradient_checkpointing": ("training", "gradient_checkpointing"),
        "bf16": ("training", "bf16"),
        "fp16": ("training", "fp16"),
        "collator_type": ("training", "collator_type"),
        "dataloader_num_workers": ("training", "dataloader_num_workers"),
        "dataloader_pin_memory": ("training", "dataloader_pin_memory"),
        "max_steps": ("training", "max_steps"),
        "prefetch_factor": ("training", "prefetch_factor"),
        "remove_unused_columns": ("training", "remove_unused_columns"),
        "conversation_variant_ratios": ("training", "conversation_variant_ratios"),
        "lr_merger": ("training", "lr_merger"),
        "lr_full_model": ("training", "lr_full_model"),
        "teacher_loss_weight": ("loss", "teacher_loss_weight"),
        "student_loss_weight": ("loss", "student_loss_weight"),
        "caption_loss_weight": ("loss", "grouped", "caption"),
        "grounding_loss_weight": ("loss", "grouped", "grounding"),
        "formatting_loss_weight": ("loss", "grouped", "formatting"),
        "output_dir": ("output", "output_dir"),
        "run_name": ("output", "run_name"),
        "tb_dir": ("output", "tb_dir"),
        "use_aug": ("features", "augmentation", "enabled"),
        "augmentation": ("features", "augmentation", "config"),
        "augmentation_schedule": ("features", "augmentation", "schedule"),
        "augmentation_smart_resize_defaults": (
            "advanced",
            "augmentation_smart_resize_defaults",
        ),
        "teacher_augmentation": (
            "features",
            "teacher_augmentation",
            "config",
        ),
        "teacher_ratio": ("features", "teacher_pairing", "teacher_ratio"),
        "dynamic_pairing_enabled": (
            "features",
            "teacher_pairing",
            "dynamic_pairing_enabled",
        ),
        "dynamic_pair_cross_bucket_explore_prob": (
            "features",
            "teacher_pairing",
            "dynamic_pair_cross_bucket_explore_prob",
        ),
        "phase_name": ("features", "phase", "name"),
        "llm_top_k_block": ("features", "phase", "llm_top_k_block"),
        "vision_top_k_block": ("features", "phase", "vision_top_k_block"),
        "freeze_patch_embed": ("features", "phase", "freeze_patch_embed"),
        "save_strategy": ("features", "checkpoint", "save_strategy"),
        "save_steps": ("features", "checkpoint", "save_steps"),
        "save_total_limit": ("features", "checkpoint", "save_total_limit"),
        "load_best_model_at_end": (
            "features",
            "checkpoint",
            "load_best_model_at_end",
        ),
        "metric_for_best_model": (
            "features",
            "checkpoint",
            "metric_for_best_model",
        ),
        "greater_is_better": (
            "features",
            "checkpoint",
            "greater_is_better",
        ),
        "eval_strategy": ("features", "evaluation", "strategy"),
        "eval_steps": ("features", "evaluation", "eval_steps"),
        "logging_steps": ("features", "logging", "logging_steps"),
        "report_to": ("features", "logging", "report_to"),
        "disable_tqdm": ("features", "logging", "disable_tqdm"),
        "seed": ("runtime", "seed"),
        "span_include_im_end_in_labels": (
            "advanced",
            "span_include_im_end_in_labels",
        ),
        "debug_alignment": ("advanced", "debug_alignment"),
        "trainable_token_strings": (
            "advanced",
            "trainable_token_strings",
        ),
        "teacher_loss_weight": (
            "advanced",
            "teacher_loss_weight",
        ),
        "student_loss_weight": (
            "advanced",
            "student_loss_weight",
        ),
        "caption_loss_weight": (
            "advanced",
            "caption_loss_weight",
        ),
        "grounding_loss_weight": (
            "advanced",
            "grounding_loss_weight",
        ),
        "formatting_loss_weight": (
            "advanced",
            "formatting_loss_weight",
        ),
    }

    def __getattr__(self, name: str) -> object:
        mapping = self._FLAT_ATTR_MAP.get(name)
        if mapping is None:
            raise AttributeError(name)
        value: object = self
        for attr in mapping:
            value = getattr(value, attr)
        return value

    def __post_init__(self) -> None:  # type: ignore[override]
        # Enforce bf16-only policy at structured config level
        dtype_norm = str(self.model.torch_dtype).lower()
        if dtype_norm not in {"bfloat16", "bf16"}:
            raise SchemaError("model.torch_dtype must be 'bfloat16'")
        if not bool(self.training.bf16):
            raise SchemaError("training.bf16 must be true")
        if bool(self.training.fp16):
            raise SchemaError("training.fp16 must be false (bf16-only policy)")

    @property
    def run_output_dir(self) -> str:
        return self.output.run_output_dir

    @property
    def tensorboard_dir(self) -> str:
        return self.output.tensorboard_dir

    @property
    def log_file_dir(self) -> str:
        return self.output.log_file_dir


__all__ = [
    "SchemaError",
    "ModelConfig",
    "DataConfig",
    "TrainingConfigSection",
    "GroupLossWeights",
    "LossConfig",
    "AugmentationScheduleItem",
    "AugmentationFeatures",
    "TeacherAugmentationConfig",
    "TeacherPairingConfig",
    "PhaseConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "EvaluationConfig",
    "FeaturesConfig",
    "OutputConfig",
    "RuntimeConfig",
    "AdvancedConfig",
    "TrainingConfig",
]
