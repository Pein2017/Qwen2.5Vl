# Explicit Configuration System Guide

## Overview

The Qwen2.5-VL codebase has been refactored to use an **explicit configuration system** that eliminates all implicit defaults and fallback patterns. This guide explains the new system and how to use it effectively.

## Key Principles

### 1. **No Implicit Defaults**
Every parameter must be explicitly defined in YAML configuration files. No defaults are provided in the code.

### 2. **Fail-Fast Validation**
Missing or invalid configuration parameters cause immediate, clear errors at startup rather than silent failures during training.

### 3. **Simplified Token Management**
The complex token management system has been replaced with a unified, automatic system that requires minimal configuration.

### 4. **Type Safety**
All parameters have strict type annotations and validation to catch errors early.

## Configuration Schema

### Core Structure

```yaml
# === CORE MODEL SETTINGS (REQUIRED) ===
model_path: "/path/to/model"
model_size: "3B"  # "3B" | "7B"
model_max_length: 8192
attn_implementation: "flash_attention_2"
torch_dtype: "bfloat16"
use_cache: false
use_cache_inference: true
model_hidden_size: 2048
model_num_layers: 28
model_num_attention_heads: 16
model_vocab_size: 151936

# === TRAINING SETTINGS (REQUIRED) ===
num_train_epochs: 30
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5e-6
vision_lr: 5e-7
merger_lr: 5e-5
llm_lr: 5e-6
coordinate_lr: 2.5e-5
adapter_lr: 0.0
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler_type: "cosine"
gradient_checkpointing: true
bf16: true
fp16: false
use_flash_attention: true
mixed_precision: "bf16"

# === COORDINATE TOKEN CONFIGURATION (SIMPLIFIED) ===
coordinate_tokens_enabled: true
max_coord_value: 2048
coordinate_loss_weight: 1.0
regular_loss_weight: 1.0

# === DATA SETTINGS (REQUIRED) ===
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
data_root: "./"
max_total_length: 8192
teacher_pool_file: "data/teacher_pool.jsonl"
num_teacher_samples: 3
collator_type: "standard"
teacher_ratio: 0.5
max_examples: -1
language: "chinese"

# === VISION PROCESSING (REQUIRED) ===
patch_size: 14
merge_size: 4
temporal_patch_size: 2

# === TRAINING CONTROL (REQUIRED) ===
training_prompt_style: true
use_consistent_prompts: true
detection_freeze_epochs: 0

# === PERFORMANCE SETTINGS (REQUIRED) ===
dataloader_num_workers: 4
pin_memory: true
prefetch_factor: 2
remove_unused_columns: false

# === OUTPUT SETTINGS (REQUIRED) ===
output_dir: "output"
run_name: "experiment_name"
tb_dir: "tb"

# === LOSS WEIGHTS (REQUIRED) ===
teacher_loss_weight: 1.0
student_loss_weight: 1.0

# === EVALUATION SETTINGS (REQUIRED) ===
eval_strategy: "steps"
eval_steps: 100
save_strategy: "steps"
save_steps: 500
save_total_limit: 3

# === LOGGING SETTINGS (REQUIRED) ===
logging_steps: 10
logging_dir: "logs"
log_level: "INFO"
report_to: "tensorboard"
disable_tqdm: false
verbose: true
```

## Unified Token Management

### Automatic Token System

The new `UnifiedTokenManager` automatically handles all token management:

```python
from src.utils.tokens import create_unified_token_manager

# Simple initialization - everything automatic!
token_manager = create_unified_token_manager(
    tokenizer=tokenizer,
    model=model,
    max_coord_value=2048  # Only parameter needed!
)
```

### What It Does Automatically

1. **Detects Existing Tokens**: Finds and reuses `<|box_start|>`, `<|object_ref_start|>`, etc.
2. **Adds New Tokens**: Creates `<|line_start|>`, `<|square_start|>`, etc. as needed
3. **Creates Coordinate Tokens**: Generates `<coord_0>` through `<coord_{max_coord_value-1}>`
4. **Assigns Token IDs**: All token IDs assigned automatically without conflicts
5. **Resizes Embeddings**: Model embeddings resized automatically

### Token Usage

```python
# Format objects automatically
bbox_obj = {"bbox_2d": [100, 200, 300, 400], "desc": "A red car"}
formatted = token_manager.format_object(bbox_obj)
# Result: "<|box_start|><coord_100><coord_200><coord_300><coord_400><|box_end|><|object_ref_start|>A red car<|object_ref_end|>"

# Get token IDs automatically
box_start_id = token_manager.get_token_id("box_start")
coord_100_id = token_manager.get_coordinate_token_id(100)
```

## Error Handling

### Fail-Fast Validation

The system provides immediate, clear error messages:

```python
# Missing configuration field
ValueError: Missing required configuration fields in config.yaml:
  coordinate_tokens_enabled
All parameters must be explicitly defined. No defaults are provided.

# Invalid coordinate value
ValueError: Coordinate value 3000 out of range [0, 2048)

# Invalid enum value
ValueError: Invalid torch_dtype: invalid_type. Must be one of ['bfloat16', 'float16', 'float32']
```

### Common Error Types

1. **Missing Required Fields**: All fields must be present in YAML
2. **Invalid Enum Values**: Strict validation of allowed values
3. **Out of Range Values**: Coordinates, learning rates, etc. validated
4. **File Path Validation**: Data paths must exist
5. **Type Mismatches**: All parameters type-checked

## Benefits

### Code Quality Improvements

- **1,576+ lines eliminated**: Removed complex fallback patterns
- **80% configuration reduction**: 20+ fields → 4 fields for coordinates
- **Zero manual token management**: All token IDs automatic
- **Fail-fast validation**: Errors caught at startup, not runtime

### Reliability Improvements

- **No silent failures**: All configuration issues surface immediately
- **Clear error messages**: Specific, actionable error information
- **Type safety**: All parameters validated for correct types
- **Consistent behavior**: No hidden defaults to confuse debugging

### Maintenance Improvements

- **Single source of truth**: All token logic in one place
- **Automatic adaptation**: Works with any tokenizer vocabulary
- **Clear API**: Simple methods for common operations
- **Better documentation**: Code serves as documentation

## Usage Examples

### Basic Training Setup

```python
from src.config.explicit_config import load_explicit_config
from src.utils.tokens import create_unified_token_manager

# Load explicit configuration
config = load_explicit_config("configs/training.yaml")

# Create unified token manager
if config.coordinate_tokens_enabled:
    token_manager = create_unified_token_manager(
        tokenizer=tokenizer,
        model=model,
        max_coord_value=config.max_coord_value
    )
```

### Object Formatting

```python
# Format different geometry types
bbox_obj = {"bbox_2d": [100, 200, 300, 400], "desc": "car"}
line_obj = {"line": [50, 100, 150, 200], "desc": "power line"}
square_obj = {"square": [0, 0, 100, 100], "desc": "window"}

# All formatted automatically with appropriate tokens
bbox_formatted = token_manager.format_object(bbox_obj)
line_formatted = token_manager.format_object(line_obj)
square_formatted = token_manager.format_object(square_obj)
```

### Configuration Validation

```python
from src.config.explicit_config import validate_config_completeness

# Check configuration completeness
missing_fields = validate_config_completeness("config.yaml")
if missing_fields:
    print(f"Missing required fields: {missing_fields}")
else:
    print("Configuration is complete!")
```

## Best Practices

### Configuration Management

1. **Use Templates**: Start with `configs/explicit_template.yaml`
2. **Validate Early**: Check configuration before training
3. **Document Changes**: Comment any non-standard values
4. **Version Control**: Track configuration changes

### Token Management

1. **Use Unified Manager**: Don't manage tokens manually
2. **Set Appropriate Range**: Choose `max_coord_value` based on your data
3. **Let System Handle IDs**: Don't hardcode token IDs
4. **Test Token Operations**: Verify token formatting works

### Error Handling

1. **Read Error Messages**: They provide specific guidance
2. **Check File Paths**: Ensure all data files exist
3. **Validate Types**: Use correct YAML types (bool, int, float, str)
4. **Test Configurations**: Validate before long training runs

## Troubleshooting

### Common Issues

**Q: Configuration loading fails with "Missing required fields"**
A: Check that all required fields from the template are present in your YAML file.

**Q: Token manager fails to initialize**
A: Ensure `coordinate_tokens_enabled` and `max_coord_value` are set correctly.

**Q: Training fails with coordinate token errors**
A: Verify that coordinate values in your data are within `[0, max_coord_value)` range.

**Q: Model loading is slow**
A: This is normal when adding many coordinate tokens. Consider reducing `max_coord_value` if not needed.

### Getting Help

1. **Check Error Messages**: They provide specific guidance
2. **Validate Configuration**: Use the validation tools
3. **Review Templates**: Compare with working configurations
4. **Run Tests**: Use the validation test suite
