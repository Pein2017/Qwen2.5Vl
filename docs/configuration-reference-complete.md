# Complete Configuration Reference

**Comprehensive reference for all configuration parameters in the BBU detection system**

This document provides detailed documentation for every configuration parameter, including validation rules, interdependencies, and usage examples.

---

## 📋 Configuration Overview

The system uses a flattened YAML configuration structure with domain-specific sections automatically validated and type-checked.

### Loading and Access
```python
# Initialize once at startup
from src.config.global_config import init_config
config = init_config("configs/base_flat_det.yaml")

# Access anywhere in codebase
from src.config.global_config import config
learning_rate = config.learning_rate
model_path = config.model_path
```

---

## 🤖 Model Configuration

### Core Model Settings
```yaml
# Model path and variant
model_path: "/path/to/Qwen2.5-VL-3B-Instruct"  # REQUIRED
model_size: "3B"                               # REQUIRED: Model size identifier
model_max_length: 32768                        # REQUIRED: Maximum sequence length
attn_implementation: "flash_attention_2"       # REQUIRED: "flash_attention_2" | "eager"
torch_dtype: "bfloat16"                        # REQUIRED: "bfloat16" | "float16" | "float32"
use_cache: false                               # REQUIRED: Enable/disable KV cache
```

### Model Architecture Parameters
```yaml
# Architecture specifications (must match actual model)
model_hidden_size: 2048                       # REQUIRED: Hidden dimension size
model_num_layers: 27                          # REQUIRED: Number of transformer layers  
model_num_attention_heads: 16                 # REQUIRED: Number of attention heads
model_vocab_size: 151936                      # REQUIRED: Vocabulary size (before extension)
```

**Validation Rules:**
- `model_path` must exist and contain valid model files
- `torch_dtype` must be supported by your hardware
- `attn_implementation: "flash_attention_2"` requires compatible GPU and CUDA version
- Architecture parameters must match the actual model being loaded

---

## 🎯 Training Configuration

### Basic Training Parameters
```yaml
# Training epochs and batch sizes
num_train_epochs: 3                           # REQUIRED: Number of training epochs
per_device_train_batch_size: 2                # REQUIRED: Batch size per GPU/device
per_device_eval_batch_size: 1                 # REQUIRED: Evaluation batch size
gradient_accumulation_steps: 8                # REQUIRED: Steps to accumulate gradients
```

### Learning Rates (Differential Learning Rate System)
```yaml
# Component-specific learning rates
learning_rate: 1e-5                           # REQUIRED: Default learning rate
vision_lr: 1e-6                               # REQUIRED: Vision encoder learning rate
merger_lr: 1e-5                               # REQUIRED: Vision-language merger LR  
llm_lr: 1e-6                                  # REQUIRED: Language model learning rate
coordinate_lr: 1e-4                           # REQUIRED: Coordinate token learning rate
adapter_lr: 1e-4                              # REQUIRED: Adapter learning rate
detection_lr: 1e-4                            # REQUIRED: Legacy detection head LR (unused)
```

**Learning Rate Auto-Properties:**
- `config.tune_vision` = `True` if `vision_lr > 0`
- `config.tune_llm` = `True` if `llm_lr > 0`  
- `config.tune_coordinate_tokens` = `True` if `coordinate_lr > 0`
- `config.use_differential_lr` = `True` if learning rates differ

### Optimization Parameters
```yaml
# Optimizer configuration
warmup_ratio: 0.1                             # REQUIRED: Warmup steps ratio
weight_decay: 0.01                            # REQUIRED: Weight decay for regularization
max_grad_norm: 1.0                            # REQUIRED: Gradient clipping norm
lr_scheduler_type: "cosine"                   # REQUIRED: "linear" | "cosine" | "constant"
```

### Training Optimizations
```yaml
# Memory and performance optimizations
gradient_checkpointing: true                  # REQUIRED: Enable gradient checkpointing
bf16: true                                    # REQUIRED: Use bfloat16 training
fp16: false                                   # REQUIRED: Use float16 training (conflicts with bf16)
use_flash_attention: true                     # REQUIRED: Enable Flash Attention optimizations
mixed_precision: "bf16"                       # REQUIRED: Mixed precision strategy
```

**Validation Rules:**
- `bf16` and `fp16` cannot both be `true`
- `gradient_checkpointing: true` recommended for large models
- `use_flash_attention: true` requires `attn_implementation: "flash_attention_2"`

---

## 📊 Data Configuration

### Dataset Paths
```yaml
# Data file locations
train_data_path: "data/train.jsonl"          # REQUIRED: Training data JSONL file
val_data_path: "data/val.jsonl"              # REQUIRED: Validation data JSONL file  
data_root: "data/"                           # REQUIRED: Root directory for data files
teacher_pool_file: "data/teacher.jsonl"     # REQUIRED: High-quality teacher samples
```

### Data Processing Parameters
```yaml
# Sequence and processing settings
max_total_length: 8192                       # REQUIRED: Maximum total sequence length
use_candidates: false                        # REQUIRED: Use candidate response system
num_teacher_samples: 2                       # REQUIRED: Number of teacher examples per sample
teacher_ratio: 0.7                          # REQUIRED: Ratio of samples using teachers (0.0-1.0)
max_examples: -1                             # REQUIRED: Max training examples (-1 = unlimited)
language: "chinese"                          # REQUIRED: "chinese" | "english"
```

### Data Collation
```yaml
# Data collation strategy
collator_type: "standard"                    # REQUIRED: "standard" | "packed"
```

**Collator Types:**
- **`"standard"`**: Traditional padding-based collation
- **`"packed"`**: Memory-efficient variable-length sequences (experimental)

### DataLoader Performance
```yaml
# DataLoader optimization
dataloader_num_workers: 8                    # REQUIRED: Number of worker processes
pin_memory: true                             # REQUIRED: Pin memory for GPU transfer
prefetch_factor: 2                           # REQUIRED: Prefetch factor per worker
```

**Validation Rules:**
- `teacher_ratio` must be between 0.0 and 1.0
- `dataloader_num_workers` should not exceed CPU cores
- `max_total_length` should not exceed `model_max_length`

---

## 🎯 Coordinate Token System Configuration

### Core Coordinate Token Settings
```yaml
# Primary coordinate token configuration
coordinate_tokens_enabled: true              # REQUIRED: Enable coordinate token system
coordinate_config_enable_coordinate_tokens: true  # REQUIRED: Enable in coordinate config
coordinate_config_max_coord_value: 2048      # REQUIRED: Maximum coordinate value (token range)
coordinate_config_coord_token_init_std: 0.02 # REQUIRED: Initialization std for coordinate embeddings
```

### Loss Configuration  
```yaml
# Coordinate loss weights and parameters
coordinate_config_coordinate_loss_weight: 1.0     # REQUIRED: Weight for coordinate loss component
coordinate_config_regular_loss_weight: 1.0        # REQUIRED: Weight for regular LM loss component
coordinate_config_soft_expectation_temperature: 1.0  # REQUIRED: Temperature for soft expectation
```

### Enhanced Loss Components
```yaml
# Multi-component coordinate loss
coordinate_config_focal_loss_alpha: 0.25     # REQUIRED: Focal loss alpha parameter (0.0-1.0)
coordinate_config_focal_loss_gamma: 2.0      # REQUIRED: Focal loss gamma parameter (>=0.0)
```

### Coordinate Token Format
```yaml
# Token format configuration
coordinate_config_use_official_box_tokens: true   # REQUIRED: Use <|box_start|> <|box_end|> tokens
chat_processor_enable_coordinate_tokens: true     # REQUIRED: Enable in chat processor
chat_processor_max_coord_value: 2048             # REQUIRED: Max coord value in chat processor
chat_processor_use_official_box_tokens: true     # REQUIRED: Use official box tokens in chat
```

**Coordinate Token Vocabulary:**
- **Box Tokens**: `<|box_start|>` (151648), `<|box_end|>` (151649)
- **Coordinate Tokens**: `<coord_0>` through `<coord_2047>` (151665-153713)
- **Total Extension**: 2048 coordinate tokens + 2 box tokens = 2050 new tokens

**Validation Rules:**
- `coordinate_config_max_coord_value` must be positive and consistent across config sections
- `coordinate_config_focal_loss_alpha` must be between 0.0 and 1.0
- `coordinate_config_focal_loss_gamma` must be non-negative
- `coordinate_config_soft_expectation_temperature` must be positive

---

## 📈 Evaluation and Checkpointing

### Evaluation Strategy
```yaml
# Evaluation configuration
eval_strategy: "steps"                       # REQUIRED: "steps" | "epoch" | "no"
eval_steps: 100                             # REQUIRED: Evaluation frequency (if eval_strategy="steps")
```

### Model Saving
```yaml
# Checkpoint configuration
save_strategy: "steps"                      # REQUIRED: "steps" | "epoch" | "no"
save_steps: 500                            # REQUIRED: Save frequency (if save_strategy="steps")
save_total_limit: 3                        # REQUIRED: Maximum number of checkpoints to keep
```

**Validation Rules:**
- `eval_steps` and `save_steps` must be positive integers
- `save_total_limit` should be ≥ 1 to prevent losing all checkpoints

---

## 📝 Logging and Monitoring

### Basic Logging
```yaml
# Logging configuration
logging_steps: 10                           # REQUIRED: Log metrics every N steps
logging_dir: "logs/"                        # REQUIRED: Directory for training logs
log_level: "INFO"                           # REQUIRED: "DEBUG" | "INFO" | "WARNING" | "ERROR"
report_to: "tensorboard"                    # REQUIRED: "tensorboard" | "wandb" | "none"
disable_tqdm: false                         # REQUIRED: Disable progress bars
verbose: true                               # REQUIRED: Enable verbose logging
```

### Output Directories
```yaml
# Output and run configuration
output_dir: "checkpoints/"                  # REQUIRED: Base directory for model checkpoints
run_name: "bbu_det_run_001"                # REQUIRED: Unique run identifier
tb_dir: "tensorboard/"                     # REQUIRED: TensorBoard log directory
```

**Auto-Generated Paths:**
- `config.run_output_dir` = `{output_dir}/{run_name}`
- `config.tensorboard_dir` = `{tb_dir}/{run_name}`
- `config.log_file_dir` = `{run_output_dir}/logs`

---

## 🎛️ Advanced Configuration

### Teacher-Student Learning
```yaml
# Teacher-student loss weights
teacher_loss_weight: 1.0                    # REQUIRED: Weight for teacher loss component
student_loss_weight: 1.0                    # REQUIRED: Weight for student loss component
```

### Learning Rate Auto-Scaling
```yaml
# Automatic learning rate scaling
auto_scale_lr: false                        # REQUIRED: Enable automatic LR scaling
lr_reference_batch_size: 32                 # REQUIRED: Reference batch size for scaling
```

**Auto-Scaling Formula:**
```
scaled_lr = base_lr * (effective_batch_size / lr_reference_batch_size)
effective_batch_size = per_device_batch_size * num_devices * gradient_accumulation_steps
```

### Vision Processing
```yaml
# Vision transformer parameters
patch_size: 14                             # REQUIRED: Vision patch size
merge_size: 2                              # REQUIRED: Patch merging factor
temporal_patch_size: 2                     # REQUIRED: Temporal dimension patch size
```

### System Configuration
```yaml
# System-level settings
remove_unused_columns: false               # REQUIRED: Remove unused data columns
test_samples: 10                           # REQUIRED: Number of samples for testing
test_forward_pass: true                    # REQUIRED: Test forward pass during setup
```

---

## 🔧 Configuration Templates

### Production Training Configuration
```yaml
# High-performance production setup
model_path: "/data/models/Qwen2.5-VL-3B-Instruct"
model_size: "3B"
model_max_length: 32768
attn_implementation: "flash_attention_2"
torch_dtype: "bfloat16"

# Optimized training
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1e-5
coordinate_lr: 1e-4

# Performance optimizations
gradient_checkpointing: true
bf16: true
use_flash_attention: true
dataloader_num_workers: 8

# Coordinate tokens enabled
coordinate_tokens_enabled: true
coordinate_config_max_coord_value: 2048
coordinate_config_coordinate_loss_weight: 1.0
```

### Development/Testing Configuration
```yaml
# Fast development setup
model_path: "/data/models/Qwen2.5-VL-3B-Instruct"
model_size: "3B"
model_max_length: 8192
attn_implementation: "eager"
torch_dtype: "float32"

# Fast training
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1e-4

# Minimal optimizations
gradient_checkpointing: false
bf16: false
use_flash_attention: false
dataloader_num_workers: 2

# Testing setup
test_samples: 5
max_examples: 100
```

---

## ⚠️ Configuration Validation

### Automatic Validation Rules

**Model Configuration:**
- Model path must exist and be readable
- Architecture parameters must match actual model
- `torch_dtype` must be supported by hardware

**Training Configuration:**
- Batch sizes must be positive integers
- Learning rates must be non-negative
- `bf16` and `fp16` cannot both be enabled

**Coordinate Tokens:**
- `coordinate_config_max_coord_value` must be positive
- Focal loss parameters must be within valid ranges
- Temperature must be positive

**Data Configuration:**
- Data paths must exist and be readable
- `teacher_ratio` must be between 0.0 and 1.0
- Sequence lengths must not exceed model capacity

### Manual Validation Commands
```bash
# Validate configuration file
/root/miniconda3/envs/ms/bin/python -c "
from src.config.global_config import init_config
config = init_config('configs/base_flat_det.yaml')
print('✅ Configuration valid')
"

# Test configuration loading
/root/miniconda3/envs/ms/bin/python scripts/validate_config.py --config configs/base_flat_det.yaml
```

---

## 🚀 Usage Examples

### Basic Usage
```python
# Load configuration
from src.config.global_config import init_config, config

# Initialize once at startup
init_config("configs/base_flat_det.yaml")

# Access configuration anywhere
print(f"Training with {config.num_train_epochs} epochs")
print(f"Coordinate tokens: {config.coordinate_tokens_enabled}")
print(f"Model: {config.model_path}")

# Use auto-properties
if config.tune_coordinate_tokens:
    print(f"Coordinate token LR: {config.coordinate_lr}")

if config.use_differential_lr:
    print("Using differential learning rates")
```

### Dynamic Configuration Updates
```python
# Configuration is immutable by design
# To change configuration, create new instance
from src.config.global_config import reset_config, init_config

reset_config()  # Clear existing config
config = init_config("configs/new_config.yaml")
```

---

## 🔗 Related Documentation

### Quick Reference
- **[Configuration Templates →](config-templates.md)** - Ready-to-use configurations
- **[API Reference →](api-core-components.md)** - Configuration-related APIs
- **[Troubleshooting →](../user-journeys/troubleshooter-quickstart.md)** - Configuration issues

### Deep Dive
- **[Architecture Overview →](../architecture-overview.md)** - How configuration fits in system
- **[Training System →](../architecture-appendix-a-components.md#a4-training-system-components)** - Training component configuration
- **[Coordinate Token Guide →](../coordinate-token-system-complete-guide.md)** - Coordinate token configuration details

---

**💡 Configuration Principle**: The system uses a single, flat configuration file that's validated once at startup. All components access the same global configuration instance, ensuring consistency across the entire system.

**Navigation:**
- **[← Back to Main Documentation](../)**
- **[Quick Reference Templates →](config-templates.md)**
- **[Component APIs →](api-core-components.md)**