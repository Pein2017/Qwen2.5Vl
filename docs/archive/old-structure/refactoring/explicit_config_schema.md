# Explicit Configuration Schema Design

## Overview

This document defines the new explicit configuration schema that eliminates all implicit defaults and requires every parameter to be explicitly defined in YAML configuration files.

## Design Principles

1. **No Implicit Defaults**: Every parameter must be explicitly defined in YAML
2. **Fail-Fast Validation**: Missing parameters cause immediate, clear errors
3. **Type Safety**: All parameters have strict type annotations
4. **Comprehensive Coverage**: All 200+ fallback patterns replaced with explicit config
5. **Clear Error Messages**: Validation failures provide actionable information

## New DirectConfig Schema

### Core Structure Changes

```python
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
    
    # === COORDINATE TOKEN CONFIGURATION (ALL REQUIRED) ===
    coordinate_config_enable_coordinate_tokens: bool
    coordinate_config_max_coord_value: int
    coordinate_config_coord_token_init_std: float
    coordinate_config_coordinate_loss_weight: float
    coordinate_config_regular_loss_weight: float
    coordinate_config_soft_expectation_temperature: float
    coordinate_config_use_official_box_tokens: bool
    coordinate_config_enable_multi_geometry: bool
    coordinate_config_max_line_coordinates: int
    coordinate_config_box_start_id: int
    coordinate_config_box_end_id: int
    coordinate_config_square_start_id: int
    coordinate_config_square_end_id: int
    coordinate_config_line_start_id: int
    coordinate_config_line_end_id: int
    coordinate_config_enable_validation: bool
    coordinate_config_enable_caching: bool
    coordinate_config_batch_processing: bool
    coordinate_config_strict_coordinate_validation: bool
    coordinate_config_log_raw_text_on_error: bool
    coordinate_config_required_loss_components: List[str]
    
    # === TRAINING STABILITY SETTINGS (ALL REQUIRED) ===
    stability_max_consecutive_nan: int
    stability_max_consecutive_zero: int
    stability_nan_monitoring_window: int
    stability_max_nan_ratio: float
    stability_nan_recovery_enabled: bool
    stability_learning_rate_reduction_factor: float
    stability_gradient_clip_reduction_factor: float
    
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
    
    # === CHAT PROCESSOR SETTINGS (ALL REQUIRED) ===
    chat_processor_max_coord_value: int
    
    # === RUNTIME PROPERTIES (COMPUTED, NOT IN YAML) ===
    run_output_dir: str = field(init=False)
    tensorboard_dir: str = field(init=False)
    log_file_dir: str = field(init=False)
    
    def __post_init__(self):
        """Compute runtime properties and validate configuration."""
        # Compute derived paths
        self.run_output_dir = str(Path(self.output_dir) / self.run_name)
        self.tensorboard_dir = str(Path(self.tb_dir) / self.run_name)
        self.log_file_dir = str(Path(self.run_output_dir) / "logs")
        
        # Create directories
        Path(self.run_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_all_parameters()
    
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
                errors.append(f"Learning rate '{field_name}' cannot be negative: {lr_value}")
            elif lr_value > 1.0:
                errors.append(f"Learning rate '{field_name}' seems too high: {lr_value}")
        
        # Validate coordinate configuration
        if self.coordinate_config_enable_coordinate_tokens:
            if self.coordinate_config_max_coord_value <= 0:
                errors.append("coordinate_config_max_coord_value must be positive when coordinate tokens are enabled")
            if self.coordinate_config_coordinate_loss_weight < 0:
                errors.append("coordinate_config_coordinate_loss_weight cannot be negative")
            if self.coordinate_config_regular_loss_weight < 0:
                errors.append("coordinate_config_regular_loss_weight cannot be negative")
        
        # Validate model path exists
        if not Path(self.model_path).exists():
            errors.append(f"Model path does not exist: {self.model_path}")
        
        # Validate data paths exist
        if not Path(self.train_data_path).exists():
            errors.append(f"Training data path does not exist: {self.train_data_path}")
        if not Path(self.val_data_path).exists():
            errors.append(f"Validation data path does not exist: {self.val_data_path}")
        
        # Validate enum values
        valid_torch_dtypes = ["bfloat16", "float16", "float32"]
        if self.torch_dtype not in valid_torch_dtypes:
            errors.append(f"Invalid torch_dtype: {self.torch_dtype}. Must be one of {valid_torch_dtypes}")
        
        valid_mixed_precision = ["bf16", "fp16", "no"]
        if self.mixed_precision not in valid_mixed_precision:
            errors.append(f"Invalid mixed_precision: {self.mixed_precision}. Must be one of {valid_mixed_precision}")
        
        valid_collator_types = ["standard", "packed"]
        if self.collator_type not in valid_collator_types:
            errors.append(f"Invalid collator_type: {self.collator_type}. Must be one of {valid_collator_types}")
        
        valid_languages = ["chinese", "english"]
        if self.language not in valid_languages:
            errors.append(f"Invalid language: {self.language}. Must be one of {valid_languages}")
        
        # Fail fast if any errors found
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
```

## Migration Strategy

### 1. Configuration Loading Changes

```python
def load_explicit_config(yaml_path: str) -> ExplicitDirectConfig:
    """Load configuration with strict validation - no defaults allowed."""
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    try:
        # This will fail immediately if any required field is missing
        config = ExplicitDirectConfig(**yaml_data)
        return config
    except TypeError as e:
        # Convert TypeError to more helpful message
        missing_fields = _extract_missing_fields(e)
        raise ValueError(
            f"Missing required configuration fields: {missing_fields}\n"
            f"All parameters must be explicitly defined in {yaml_path}"
        ) from e

def _extract_missing_fields(type_error: TypeError) -> List[str]:
    """Extract missing field names from TypeError message."""
    # Parse TypeError message to extract missing fields
    error_str = str(type_error)
    # Implementation to parse "missing required argument" messages
    return []  # Simplified for brevity
```

### 2. Fallback Pattern Elimination

All getattr(), hasattr(), and dict.get() patterns will be replaced with direct config access:

```python
# OLD (with fallback):
max_coord_value = getattr(self.coordinate_config, "max_coord_value", 0)

# NEW (explicit):
max_coord_value = config.coordinate_config_max_coord_value
```

### 3. YAML Template

A complete YAML template will be provided with all required fields and their expected types/values.

## Benefits

1. **Immediate Error Detection**: Missing config causes startup failure, not runtime errors
2. **Clear Documentation**: YAML file serves as complete parameter documentation
3. **Type Safety**: All parameters have explicit types and validation
4. **Reduced Complexity**: Eliminates 200+ defensive programming patterns
5. **Better Debugging**: No hidden defaults to confuse troubleshooting
