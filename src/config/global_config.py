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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class DirectConfig:
    """
    Direct configuration access - all values are flat and accessible directly.
    No nested structures, no parameter passing, no conversions.
    """

    # Model settings
    model_path: str
    model_size: str
    model_max_length: int
    attn_implementation: str
    torch_dtype: str
    use_cache: bool

    # Training settings
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    vision_lr: float
    merger_lr: float
    llm_lr: float
    detection_lr: float
    adapter_lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool

    # Data settings
    train_data_path: str
    val_data_path: str
    data_root: str
    max_total_length: int
    use_candidates: bool
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

    # Logging settings
    logging_steps: int
    logging_dir: str
    log_level: str
    report_to: str
    verbose: bool
    disable_tqdm: bool

    # Detection loss configuration (End-to-End Training)
    detection_enabled: bool
    detection_num_queries: int
    detection_max_caption_length: int

    # Detection Head Architecture
    detection_decoder_dim_feedforward_factor: float
    detection_decoder_num_layers: int
    detection_caption_decoder_dim_feedforward_factor: float
    detection_caption_decoder_num_layers: int
    detection_head_dropout: float

    # Adapter hyperparameters for detection head
    detection_adapter_bottleneck_ratio: int
    detection_adapter_num_layers: int

    # Loss component weights
    detection_bbox_weight: float
    detection_giou_weight: float
    detection_objectness_weight: float
    detection_caption_weight: float

    # Focal Loss specific weights
    detection_focal_loss_gamma: float
    detection_focal_loss_alpha: float

    # Model architecture (match official Qwen2.5-VL)
    model_hidden_size: int
    model_num_layers: int
    model_num_attention_heads: int
    model_vocab_size: int

    # Training configuration
    use_flash_attention: bool
    mixed_precision: str

    # Performance settings
    dataloader_num_workers: int
    pin_memory: bool
    prefetch_factor: int
    batching_strategy: str
    remove_unused_columns: bool

    # Output settings
    output_dir: str
    run_name: Optional[str]
    tb_dir: str

    # Stability settings
    max_consecutive_nan: int
    max_consecutive_zero: int
    max_nan_ratio: float
    nan_monitoring_window: int
    allow_occasional_nan: bool
    nan_recovery_enabled: bool
    learning_rate_reduction_factor: float
    gradient_clip_reduction_factor: float

    # Debug settings
    test_samples: int
    test_forward_pass: bool

    # Training schedule tweaks
    detection_freeze_epochs: int

    # Vision processing parameters (Qwen2.5-VL image processor)
    patch_size: int  # Spatial patch size of vision encoder (default: 14)
    merge_size: int  # Merge size from vision encoder to LLM encoder (default: 2)
    temporal_patch_size: int  # Temporal patch size of vision encoder (default: 2)

    # Teacher-Student Loss Weights
    teacher_loss_weight: float  # Weight for teacher loss in backpropagation
    student_loss_weight: float  # Weight for student loss in backpropagation

    # --- Fields with defaults (must be last in dataclass) ---
    candidates_file: Optional[str] = None
    lr_reference_batch_size: int = 0
    auto_scale_lr: bool = True
    run_output_dir: str = field(init=False)
    tensorboard_dir: str = field(init=False)
    log_file_dir: str = field(init=False)

    @property
    def tune_vision(self) -> bool:
        """Auto-determine if vision encoder should be trained based on learning rate."""
        return self.vision_lr > 0

    @property
    def tune_mlp(self) -> bool:
        """Auto-determine if MLP connector should be trained based on learning rate."""
        return self.merger_lr > 0

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
        lrs = [self.vision_lr, self.merger_lr, self.llm_lr, self.detection_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1


# Global singleton instance
config: Optional[DirectConfig] = None


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
        config_dict = yaml.safe_load(f)

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
        config = DirectConfig(**converted_dict)
    except TypeError as e:
        raise ValueError(f"Configuration error: Missing or extra keys in YAML. {e}")

    # --- Automatically derive and set paths ---
    if not config.run_name:
        raise ValueError("`run_name` must be defined in the configuration.")

    # 1. Main output directory for the run
    config.run_output_dir = str(Path(config.output_dir) / config.run_name)

    # 2. TensorBoard directory
    config.tensorboard_dir = str(Path(config.tb_dir) / config.run_name)

    # 3. Log file directory
    config.log_file_dir = str(Path(config.run_output_dir) / "logs")

    # Create directories
    Path(config.run_output_dir).mkdir(parents=True, exist_ok=True)

    # Validate the final configuration
    _validate_config(config)

    # ------------------------------------------------------------------
    # Automatic learning-rate scaling based on effective global batch size
    # and collator type
    # ------------------------------------------------------------------
    _apply_auto_lr_scaling(config)
    if config.auto_scale_lr:
        _apply_collator_lr_scaling(config)

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
    """Validate configuration values."""
    if not cfg.model_path:
        raise ValueError("model_path cannot be empty")
    if not cfg.train_data_path or not cfg.val_data_path:
        raise ValueError("train_data_path and val_data_path must be specified")

    if cfg.language not in ["english", "chinese"]:
        raise ValueError(
            f"language must be 'english' or 'chinese' in YAML config, but got: '{cfg.language}'"
        )
    if cfg.num_teacher_samples < 0:
        raise ValueError("num_teacher_samples must be non-negative")
    # Add more validation rules as needed...


def _apply_auto_lr_scaling(cfg: DirectConfig) -> None:
    """Linearly scale all learning-rate fields according to effective
    batch size.

    The scale factor is computed as:
        scale = (per_device_batch * grad_accum_steps * world_size) / reference_bs

    * ``world_size`` is read from the *launcher-set* env var ``BBU_NUM_GPUS`` –
      falling back to 1 when undefined.
    * ``reference_bs`` comes from the YAML key ``lr_reference_batch_size`` or the
      env var ``LR_REFERENCE_BATCH_SIZE``.  If neither is provided, scaling is
      disabled (factor = 1).
    """
    import os

    # ------------------------------------------------------------------
    # Resolve reference (baseline) batch size
    # ------------------------------------------------------------------
    ref_bs_env = os.getenv("LR_REFERENCE_BATCH_SIZE")
    reference_bs = (
        int(ref_bs_env) if ref_bs_env is not None else cfg.lr_reference_batch_size
    )

    if reference_bs is None or reference_bs <= 0:
        # No baseline → skip scaling entirely
        return

    # ------------------------------------------------------------------
    # Compute current *effective* batch size
    # ------------------------------------------------------------------
    world_size_env = os.getenv("BBU_NUM_GPUS", "1")
    try:
        world_size = int(world_size_env)
    except ValueError:
        world_size = 1

    effective_bs = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size
    )

    scale = effective_bs / reference_bs
    if abs(scale - 1.0) < 1e-6:
        # No change needed
        return

    lr_fields = [
        "learning_rate",
        "adapter_lr",
        "vision_lr",
        "merger_lr",
        "llm_lr",
        "detection_lr",
    ]

    for field_name in lr_fields:
        current_lr = getattr(cfg, field_name, None)
        if current_lr is not None and current_lr > 0:
            setattr(cfg, field_name, current_lr * scale)

    # Inform the user once – this runs on every process but is cheap
    print(
        f"🔄 Auto LR scaling: effective_bs={effective_bs}, reference_bs={reference_bs}, "
        f"scale={scale:.2f}.  Learning rates updated accordingly."
    )


def _apply_collator_lr_scaling(cfg: DirectConfig) -> None:
    """
    Apply token-length-aware learning rate scaling to ensure equivalent training
    dynamics between different collator types.

    This function automatically adjusts learning rates based on the collator type
    to compensate for token length inconsistency between standard (padded) and
    packed collators.
    """
    import os

    # Check if collator-aware scaling is disabled
    disable_collator_scaling = (
        os.getenv("DISABLE_COLLATOR_LR_SCALING", "false").lower() == "true"
    )
    if disable_collator_scaling:
        return

    # Create token-length-aware learning rate scaler (import here to avoid circular imports)
    try:
        from src.lr_scaling import create_token_length_scaler
    except ImportError:
        raise RuntimeError("lr_scaling module required for auto_scale_lr=True but not available")

    scaler = create_token_length_scaler(
        auto_scale_lr=cfg.auto_scale_lr,
        base_collator_type="standard",  # Treat standard as reference
    )

    # Convert config to dictionary for processing
    config_dict = {
        "collator_type": cfg.collator_type,
        "learning_rate": cfg.learning_rate,
        "llm_lr": cfg.llm_lr,
        "adapter_lr": cfg.adapter_lr,
        "vision_lr": cfg.vision_lr,
        "merger_lr": cfg.merger_lr,
        "detection_lr": cfg.detection_lr,
    }

    # Apply token-length-aware scaling
    scaled_config = scaler.scale_learning_rates(
        config=config_dict,
        collator_type=cfg.collator_type,
    )

    # Update the original config with scaled values
    if "learning_rate" in scaled_config:
        cfg.learning_rate = scaled_config["learning_rate"]
    if "llm_lr" in scaled_config:
        cfg.llm_lr = scaled_config["llm_lr"]
    if "adapter_lr" in scaled_config:
        cfg.adapter_lr = scaled_config["adapter_lr"]
    if "vision_lr" in scaled_config:
        cfg.vision_lr = scaled_config["vision_lr"]
    if "merger_lr" in scaled_config:
        cfg.merger_lr = scaled_config["merger_lr"]
    if "detection_lr" in scaled_config:
        cfg.detection_lr = scaled_config["detection_lr"]
