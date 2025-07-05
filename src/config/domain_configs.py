"""
Domain-Specific Configuration Classes for Qwen2.5-VL Training

This module breaks down the monolithic DirectConfig into focused, domain-specific
configuration classes for better organization and maintainability.

Each domain config handles a specific aspect of the training pipeline:
- ModelConfig: Model architecture and inference settings
- TrainingConfig: Training parameters and optimization settings  
- DataConfig: Data processing and teacher-student learning
- DetectionConfig: Object detection specific settings
- InfrastructureConfig: Logging, checkpointing, and stability
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture and inference configuration."""
    
    # Core model settings
    model_path: str
    model_size: str
    model_max_length: int
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    use_cache: bool = False
    
    # Model architecture details (match official Qwen2.5-VL)
    model_hidden_size: int = 3584
    model_num_layers: int = 28
    model_num_attention_heads: int = 28
    model_vocab_size: int = 152064
    
    # Vision processing parameters
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2
    
    # Training configuration
    use_flash_attention: bool = True
    mixed_precision: str = "bf16"
    
    @property
    def is_flash_attention_enabled(self) -> bool:
        """Check if Flash Attention 2 is properly configured."""
        return (
            self.use_flash_attention 
            and self.attn_implementation == "flash_attention_2"
        )


@dataclass 
class TrainingConfig:
    """Training parameters and optimization configuration."""
    
    # Basic training settings
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    
    # Learning rates (differential training support)
    learning_rate: float
    vision_lr: float = 0.0
    merger_lr: float = 0.0
    llm_lr: float = 0.0
    detection_lr: float = 0.0
    adapter_lr: float = 0.0
    
    # Optimization settings
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # Training optimizations
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    
    # Learning rate scaling
    lr_reference_batch_size: int = 0
    auto_scale_lr: bool = True
    
    # Performance settings  
    dataloader_num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    batching_strategy: str = "standard"
    remove_unused_columns: bool = False
    
    @property
    def tune_vision(self) -> bool:
        """Auto-determine if vision encoder should be trained."""
        return self.vision_lr > 0
    
    @property
    def tune_mlp(self) -> bool:
        """Auto-determine if MLP connector should be trained."""
        return self.merger_lr > 0
    
    @property
    def tune_llm(self) -> bool:
        """Auto-determine if LLM should be trained."""
        return self.llm_lr > 0
    
    @property
    def tune_detection(self) -> bool:
        """Auto-determine if detection head should be trained."""
        return self.detection_lr > 0
    
    @property
    def use_differential_lr(self) -> bool:
        """Check if differential learning rates are configured."""
        lrs = [self.vision_lr, self.merger_lr, self.llm_lr, self.detection_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size for learning rate scaling."""
        import os
        world_size = int(os.getenv("BBU_NUM_GPUS", "1"))
        return self.per_device_train_batch_size * self.gradient_accumulation_steps * world_size


@dataclass
class DataConfig:
    """Data processing and teacher-student learning configuration."""
    
    # Data paths
    train_data_path: str
    val_data_path: str
    data_root: str
    
    # Processing settings
    max_total_length: int = 120000
    collator_type: str = "standard"  # "standard" or "packed"
    language: str = "chinese"
    
    # Teacher-student learning
    teacher_pool_file: str = ""
    num_teacher_samples: int = 2
    teacher_ratio: float = 0.7
    max_examples: int = 5
    
    # Optional features
    use_candidates: bool = False
    candidates_file: str = ""
    
    def validate(self):
        """Validate data configuration."""
        if self.language not in ["english", "chinese"]:
            raise ValueError(f"language must be 'english' or 'chinese', got: '{self.language}'")
        
        if self.num_teacher_samples < 0:
            raise ValueError("num_teacher_samples must be non-negative")
        
        if not (0.0 <= self.teacher_ratio <= 1.0):
            raise ValueError("teacher_ratio must be between 0.0 and 1.0")
        
        if self.collator_type not in ["standard", "packed"]:
            raise ValueError(f"collator_type must be 'standard' or 'packed', got: '{self.collator_type}'")


@dataclass
class DetectionConfig:
    """Object detection specific configuration."""
    
    # Detection system control
    detection_enabled: bool = True
    detection_num_queries: int = 100
    detection_max_caption_length: int = 32
    
    # Detection head architecture
    detection_decoder_dim_feedforward_factor: float = 4.0
    detection_decoder_num_layers: int = 3
    detection_caption_decoder_dim_feedforward_factor: float = 4.0
    detection_caption_decoder_num_layers: int = 2
    detection_head_dropout: float = 0.1
    
    # Adapter hyperparameters
    detection_adapter_bottleneck_ratio: int = 4
    detection_adapter_num_layers: int = 2
    
    # Loss component weights
    detection_bbox_weight: float = 5.0
    detection_giou_weight: float = 2.0
    detection_objectness_weight: float = 2.0
    detection_caption_weight: float = 1.0
    
    # Focal loss parameters
    detection_focal_loss_gamma: float = 2.0
    detection_focal_loss_alpha: float = 0.25
    
    # Training schedule
    detection_freeze_epochs: int = 0
    
    def validate(self):
        """Validate detection configuration."""
        if self.detection_num_queries <= 0:
            raise ValueError("detection_num_queries must be positive")
        
        if self.detection_max_caption_length <= 0:
            raise ValueError("detection_max_caption_length must be positive")
        
        # Validate loss weights are non-negative
        weights = [
            self.detection_bbox_weight,
            self.detection_giou_weight, 
            self.detection_objectness_weight,
            self.detection_caption_weight
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All detection loss weights must be non-negative")


@dataclass
class InfrastructureConfig:
    """Infrastructure settings for logging, checkpointing, and stability."""
    
    # Output and run settings
    output_dir: str
    run_name: Optional[str] = None
    tb_dir: str = "tensorboard"
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps" 
    save_steps: int = 100
    save_total_limit: int = 2
    
    # Logging settings
    logging_steps: int = 10
    logging_dir: Optional[str] = None
    log_level: str = "INFO"
    report_to: str = "tensorboard"
    verbose: bool = False
    disable_tqdm: bool = False
    
    # Stability monitoring
    max_consecutive_nan: int = 10
    max_consecutive_zero: int = 5
    max_nan_ratio: float = 0.1
    nan_monitoring_window: int = 100
    allow_occasional_nan: bool = True
    nan_recovery_enabled: bool = True
    learning_rate_reduction_factor: float = 0.5
    gradient_clip_reduction_factor: float = 0.5
    
    # Debug settings
    test_samples: int = 0
    test_forward_pass: bool = False
    
    # --- Derived paths (set automatically) ---
    run_output_dir: str = field(init=False)
    tensorboard_dir: str = field(init=False)
    log_file_dir: str = field(init=False)
    
    def __post_init__(self):
        """Set derived paths after initialization."""
        from pathlib import Path
        
        if not self.run_name:
            raise ValueError("run_name must be specified")
        
        # Set derived paths
        self.run_output_dir = str(Path(self.output_dir) / self.run_name)
        self.tensorboard_dir = str(Path(self.tb_dir) / self.run_name)
        self.log_file_dir = str(Path(self.run_output_dir) / "logs")
        
        # Create directories
        Path(self.run_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_file_dir).mkdir(parents=True, exist_ok=True)
    
    def validate(self):
        """Validate infrastructure configuration."""
        if not self.run_name:
            raise ValueError("run_name cannot be empty")
        
        if self.eval_steps <= 0 or self.save_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("eval_steps, save_steps, and logging_steps must be positive")
        
        if self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be positive")