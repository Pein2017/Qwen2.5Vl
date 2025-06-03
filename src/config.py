"""
Simplified Configuration System for Qwen2.5-VL BBU Training.

This replaces the complex config/ directory with a single file containing:
- Sensible defaults for 90% of parameters
- Model presets for 3B/7B configurations
- Simple YAML loading without inheritance
- Minimal required parameters (8 core settings)

Key simplifications:
- 60+ parameters → 8 core parameters in YAML
- No complex validation (use Python defaults)
- No inheritance/composition (flat structure)
- Model presets instead of manual tuning
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """
    Simplified configuration with sensible defaults.

    Only 8 core parameters need to be specified in YAML:
    - model_path, train_data, val_data, learning_rate
    - epochs, batch_size, max_length, model_size

    Everything else has sensible defaults.
    """

    # ========================================================================
    # CORE PARAMETERS (must be specified in YAML)
    # ========================================================================
    model_path: str = ""
    train_data: str = ""
    val_data: str = ""
    learning_rate: float = 5e-7
    epochs: int = 10
    batch_size: int = 2
    max_length: int = 8192
    model_size: str = "3B"  # "3B" | "7B"

    # ========================================================================
    # SENSIBLE DEFAULTS (rarely need to change)
    # ========================================================================

    # Data settings
    data_root: str = "./"
    max_pixels: int = 1003520
    min_pixels: int = 784

    # Model settings
    cache_dir: str = "/data4/swift/model_cache"
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"

    # Learning rates (auto-set based on model_size)
    vision_lr: float = 0.0  # Will be set by model preset
    mlp_lr: float = 0.0  # Will be set by model preset
    llm_lr: float = 0.0  # Will be set by model preset

    # Training optimization
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.5
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 20
    save_strategy: str = "steps"
    save_steps: int = 20
    save_total_limit: int = 2

    # Logging
    logging_steps: int = 10
    log_level: str = "INFO"
    report_to: str = "tensorboard"
    verbose: bool = True

    # Loss configuration
    loss_type: str = "object_detection"
    loss_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.6, 0.4, 0.3]
    )  # [lm, bbox, giou, class]
    hungarian_matching: bool = True
    detection_mode: str = "inference"
    inference_frequency: int = 5
    max_generation_length: int = 1024
    use_semantic_similarity: bool = True

    # Performance
    dataloader_num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    data_flatten: bool = False
    remove_unused_columns: bool = False

    # DeepSpeed
    deepspeed_enabled: bool = True
    deepspeed_config: str = "scripts/zero2.json"

    # Output
    output_dir: str = "output"
    run_name: Optional[str] = None

    # Stability (simplified)
    max_consecutive_nan: int = 5
    max_nan_ratio: float = 0.3
    allow_occasional_nan: bool = True

    # Early training mode
    early_training_epochs: int = 3  # Use lenient parsing for first N epochs

    # Debug
    debug_mode: bool = False
