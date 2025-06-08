"""
Unified Configuration Schema - Single Source of Truth
All parameters defined here with types, constraints, and documentation.
No defaults - all values must be explicitly provided.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Model configuration parameters."""

    model_path: str = Field(..., description="Path to the model checkpoint")
    model_size: str = Field(..., description="Model size identifier (e.g., '3B', '7B')")
    model_max_length: int = Field(..., gt=0, description="Maximum sequence length")
    attn_implementation: Literal["flash_attention_2", "eager"] = Field(
        ..., description="Attention implementation"
    )
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        ..., description="Model precision"
    )
    use_cache: bool = Field(
        ..., description="Enable KV cache (incompatible with gradient checkpointing)"
    )
    use_model_wrapper: bool = Field(
        ..., description="Use ModelWrapper (true) or direct loading (false)"
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")
        return v


class DataConfig(BaseModel):
    """Data configuration parameters."""

    train_data_path: str = Field(..., description="Path to training data file")
    val_data_path: str = Field(..., description="Path to validation data file")
    data_root: str = Field(..., description="Root directory for data files")
    collator_type: Literal["standard", "flatten"] = Field(
        ..., description="Data collator type"
    )
    max_total_length: int = Field(
        ..., gt=0, description="Maximum total sequence length"
    )

    # Preprocessing
    use_candidates: bool = Field(
        ..., description="Use candidate phrases for preprocessing"
    )
    candidates_file: str = Field(..., description="Path to candidate phrases file")
    multi_round: bool = Field(..., description="Enable multi-round conversation format")
    max_examples: int = Field(..., gt=0, description="Maximum examples per sample")

    @model_validator(mode="after")
    def validate_data_paths(self):
        data_root = Path(self.data_root)
        train_path = data_root / self.train_data_path
        val_path = data_root / self.val_data_path

        if not train_path.exists():
            raise ValueError(f"Training data not found: {train_path}")
        if not val_path.exists():
            raise ValueError(f"Validation data not found: {val_path}")

        return self


class LearningRateConfig(BaseModel):
    """Learning rate configuration parameters."""

    vision_lr: float = Field(
        ..., ge=0, description="Vision encoder learning rate (0 = freeze)"
    )
    mlp_lr: float = Field(
        ..., ge=0, description="MLP connector learning rate (0 = freeze)"
    )
    llm_lr: float = Field(..., ge=0, description="LLM learning rate (0 = freeze)")
    learning_rate: float = Field(..., gt=0, description="Fallback learning rate")

    @model_validator(mode="after")
    def validate_learning_rates(self):
        if self.vision_lr == 0 and self.mlp_lr == 0 and self.llm_lr == 0:
            raise ValueError(
                "At least one learning rate (vision_lr, mlp_lr, llm_lr) must be > 0"
            )

        return self

    @property
    def tune_vision(self) -> bool:
        return self.vision_lr > 0

    @property
    def tune_mlp(self) -> bool:
        return self.mlp_lr > 0

    @property
    def tune_llm(self) -> bool:
        return self.llm_lr > 0

    @property
    def use_differential_lr(self) -> bool:
        active_lrs = [lr for lr in [self.vision_lr, self.mlp_lr, self.llm_lr] if lr > 0]
        return len(set(active_lrs)) > 1


class TrainingConfig(BaseModel):
    """Training configuration parameters."""

    num_train_epochs: int = Field(..., gt=0, description="Number of training epochs")
    per_device_train_batch_size: int = Field(
        ..., gt=0, description="Training batch size per device"
    )
    per_device_eval_batch_size: int = Field(
        ..., gt=0, description="Evaluation batch size per device"
    )
    total_batch_size: int = Field(..., gt=0, description="Total effective batch size")
    gradient_accumulation_steps: int = Field(
        ..., gt=0, description="Gradient accumulation steps"
    )

    # Optimization
    warmup_ratio: float = Field(..., ge=0, le=1, description="Warmup ratio")
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = Field(
        ..., description="Learning rate scheduler"
    )
    max_grad_norm: float = Field(
        ..., gt=0, description="Maximum gradient norm for clipping"
    )
    weight_decay: float = Field(..., ge=0, description="Weight decay coefficient")
    gradient_checkpointing: bool = Field(
        ..., description="Enable gradient checkpointing for memory efficiency"
    )

    # Precision
    bf16: bool = Field(..., description="Enable bfloat16 training")
    fp16: bool = Field(..., description="Enable float16 training")

    @model_validator(mode="after")
    def validate_precision(self):
        if self.bf16 and self.fp16:
            raise ValueError("Cannot enable both bf16 and fp16 simultaneously")

        return self


class EvaluationConfig(BaseModel):
    """Evaluation and saving configuration parameters."""

    eval_strategy: Literal["no", "steps", "epoch"] = Field(
        ..., description="Evaluation strategy"
    )
    eval_steps: int = Field(..., gt=0, description="Evaluation frequency in steps")
    save_strategy: Literal["no", "steps", "epoch"] = Field(
        ..., description="Save strategy"
    )
    save_steps: int = Field(..., gt=0, description="Save frequency in steps")
    save_total_limit: int = Field(
        ..., gt=0, description="Maximum number of checkpoints to keep"
    )


class LoggingConfig(BaseModel):
    """Logging configuration parameters."""

    logging_steps: int = Field(..., gt=0, description="Logging frequency in steps")
    logging_dir: Optional[str] = Field(
        None, description="Logging directory (auto-generated if None)"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        ..., description="Logging level"
    )
    report_to: Literal["tensorboard", "wandb", "none"] = Field(
        ..., description="Experiment tracking platform"
    )
    verbose: bool = Field(..., description="Enable verbose logging")
    disable_tqdm: bool = Field(..., description="Disable tqdm progress bars")


class MonitoringConfig(BaseModel):
    """Training monitoring configuration parameters."""

    enable_monitoring: bool = Field(..., description="Enable training monitoring")
    monitor_log_dir: str = Field(..., description="Directory for monitoring logs")
    save_predictions: bool = Field(
        ..., description="Save model predictions during evaluation"
    )
    save_token_analysis: bool = Field(..., description="Save token-level analysis")
    save_raw_text: bool = Field(
        ..., description="Save raw text predictions and ground truth"
    )


class LossConfig(BaseModel):
    """Loss function configuration parameters."""

    loss_type: Literal["object_detection", "language_modeling"] = Field(
        ..., description="Loss function type"
    )
    lm_weight: float = Field(..., gt=0, description="Language modeling loss weight")
    bbox_weight: float = Field(..., ge=0, description="Bounding box loss weight")
    giou_weight: float = Field(..., ge=0, description="GIoU loss weight")
    class_weight: float = Field(..., ge=0, description="Classification loss weight")
    hungarian_matching: bool = Field(
        ..., description="Use Hungarian matching for object detection"
    )
    detection_mode: Literal["training", "inference"] = Field(
        ..., description="Detection mode"
    )
    inference_frequency: int = Field(
        ..., gt=0, description="Inference frequency during training"
    )
    max_generation_length: int = Field(
        ..., gt=0, description="Maximum generation length for inference"
    )
    use_semantic_similarity: bool = Field(
        ..., description="Use semantic similarity in loss computation"
    )


class PerformanceConfig(BaseModel):
    """Performance optimization configuration parameters."""

    dataloader_num_workers: int = Field(
        ..., ge=0, description="Number of dataloader workers"
    )
    pin_memory: bool = Field(..., description="Pin memory for faster GPU transfer")
    prefetch_factor: int = Field(
        ..., gt=0, description="Prefetch factor for dataloader"
    )
    batching_strategy: Literal["standard", "dynamic"] = Field(
        ..., description="Batching strategy"
    )
    remove_unused_columns: bool = Field(
        ..., description="Remove unused columns from dataset"
    )


class OutputConfig(BaseModel):
    """Output configuration parameters."""

    output_base_dir: str = Field(..., description="Base directory for outputs")
    run_name: Optional[str] = Field(
        None, description="Run name (auto-generated if None)"
    )
    tb_dir: str = Field(..., description="TensorBoard log directory")

    @property
    def output_dir(self) -> str:
        if self.run_name:
            return f"{self.output_base_dir}/{self.run_name}"
        return self.output_base_dir


class StabilityConfig(BaseModel):
    """Training stability and monitoring configuration parameters."""

    max_consecutive_nan: int = Field(
        ..., gt=0, description="Maximum consecutive NaN losses before stopping"
    )
    max_consecutive_zero: int = Field(
        ..., gt=0, description="Maximum consecutive zero losses before stopping"
    )
    max_nan_ratio: float = Field(
        ..., ge=0, le=1, description="Maximum ratio of NaN losses in monitoring window"
    )
    nan_monitoring_window: int = Field(
        ..., gt=0, description="Window size for NaN monitoring"
    )
    allow_occasional_nan: bool = Field(..., description="Allow occasional NaN losses")
    nan_recovery_enabled: bool = Field(
        ..., description="Enable automatic recovery from NaN losses"
    )
    learning_rate_reduction_factor: float = Field(
        ..., gt=0, lt=1, description="Learning rate reduction factor for recovery"
    )
    gradient_clip_reduction_factor: float = Field(
        ..., gt=0, lt=1, description="Gradient clipping reduction factor for recovery"
    )


class DebugConfig(BaseModel):
    """Debug mode configuration parameters."""

    test_samples: int = Field(..., gt=0, description="Number of samples for testing")
    test_forward_pass: bool = Field(
        ..., description="Test forward pass during initialization"
    )


class TrainingConfiguration(BaseModel):
    """
    Complete training configuration schema.
    Single source of truth for all parameters.
    """

    # Core configuration sections
    model: ModelConfig
    data: DataConfig
    learning_rates: LearningRateConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    loss: LossConfig
    performance: PerformanceConfig
    output: OutputConfig
    stability: StabilityConfig
    debug: DebugConfig

    @model_validator(mode="after")
    def validate_compatibility(self):
        """Validate parameter compatibility across sections."""
        # Check use_cache and gradient_checkpointing compatibility
        if self.model.use_cache and self.training.gradient_checkpointing:
            raise ValueError(
                "use_cache=True is incompatible with gradient_checkpointing=True. "
                "Set either use_cache=False or gradient_checkpointing=False."
            )

        # Check max_total_length consistency
        if self.data.max_total_length > self.model.model_max_length:
            raise ValueError(
                f"max_total_length ({self.data.max_total_length}) cannot exceed "
                f"model_max_length ({self.model.model_max_length})"
            )

        return self

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Prevent extra fields
        use_enum_values = True
