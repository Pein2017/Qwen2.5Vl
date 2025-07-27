# Configuration System Guide (2025 Architecture)

Comprehensive configuration reference for the Qwen2.5-VL BBU fine-tuning system with the current modular architecture.

## 📋 Configuration Overview (Current System)

The system uses a unified configuration approach with the DirectConfig system for backward compatibility and ease of use.

### Current Configuration Architecture

#### DirectConfig System (`src/config/global_config.py`)
**Primary approach** - Flat configuration with 149+ parameters for comprehensive control.

```python
# Initialize once at startup (automatic in training scripts)
from src.config import get_config
config = get_config()  # Returns DirectConfig instance

# Access anywhere in codebase
learning_rate = config.learning_rate
model_path = config.model_path
detection_enabled = config.detection_enabled
```

**Key Features**:
- **Flat parameter access**: All parameters accessible as `config.parameter_name`
- **Type validation**: Automatic type checking and conversion
- **Backward compatibility**: Works with existing configuration files
- **Global access**: Available throughout the codebase via `get_config()`

---

## 🎯 Quick Start Templates

### 1. Detection Training (Current Recommended)
```yaml
# configs/detection_training.yaml
model_path: "/path/to/qwen2.5-vl-7b-instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"

# Detection and Token Settings
detection_enabled: true
coordinate_tokens_enabled: true
max_coord_value: 2048

# Training Settings
learning_rate: 1e-5
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
warmup_ratio: 0.1
lr_scheduler_type: "cosine"

# Teacher-Student Learning
teacher_ratio: 0.3
teacher_loss_weight: 0.3
student_loss_weight: 1.0

# Model Settings
model_max_length: 120000
attn_implementation: "flash_attention_2"
torch_dtype: "bfloat16"

# Model Settings
model_max_length: 8192
torch_dtype: "bfloat16"
attn_implementation: "flash_attention_2"

# Optimization
weight_decay: 0.01
dataloader_num_workers: 4
fp16: false
bf16: true

# Checkpointing
save_strategy: "steps"
save_steps: 500
eval_strategy: "steps"
eval_steps: 500
save_total_limit: 3
```

### 2. Standard Detection Training
```yaml
# configs/standard_detection.yaml
model_path: "/path/to/qwen2.5-vl-7b-instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"

# Detection Settings
detection_enabled: true
num_queries: 100
detection_lr: 1e-5

# Training Settings
learning_rate: 5e-6
num_train_epochs: 30
per_device_train_batch_size: 4
gradient_accumulation_steps: 2

# Model Settings
model_max_length: 12000
torch_dtype: "bfloat16"
attn_implementation: "flash_attention_2"
```

### 3. Memory-Optimized Training
```yaml
# configs/memory_optimized.yaml
# For systems with limited GPU memory
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
model_max_length: 4096
torch_dtype: "float16"
gradient_checkpointing: true
dataloader_num_workers: 2
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
- `attn_implementation` must be compatible with hardware
- `torch_dtype` affects memory usage and training stability

---

## 🎓 Training Configuration

### Learning Rate Settings
```yaml
# Base learning rates
learning_rate: 5e-6                           # REQUIRED: Base learning rate
vision_lr: 5e-7                               # REQUIRED: Vision encoder learning rate
llm_lr: 5e-6                                  # REQUIRED: Language model learning rate
detection_lr: 1e-5                            # REQUIRED: Detection head learning rate
coordinate_lr: 1e-4                           # OPTIONAL: Coordinate token learning rate

# Learning rate scheduling
lr_scheduler_type: "cosine"                   # REQUIRED: "linear" | "cosine" | "constant"
warmup_ratio: 0.1                             # REQUIRED: Warmup ratio (0.0-1.0)
warmup_steps: 0                               # OPTIONAL: Alternative to warmup_ratio
```

### Training Parameters
```yaml
# Training duration
num_train_epochs: 30                          # REQUIRED: Number of training epochs
max_steps: -1                                 # OPTIONAL: Max steps (overrides epochs)

# Batch settings
per_device_train_batch_size: 4                # REQUIRED: Batch size per device
per_device_eval_batch_size: 4                 # REQUIRED: Eval batch size per device
gradient_accumulation_steps: 1                # REQUIRED: Gradient accumulation steps

# Optimization
weight_decay: 0.01                            # REQUIRED: Weight decay coefficient
max_grad_norm: 1.0                            # REQUIRED: Gradient clipping norm
adam_beta1: 0.9                               # OPTIONAL: Adam beta1
adam_beta2: 0.999                             # OPTIONAL: Adam beta2
adam_epsilon: 1e-8                            # OPTIONAL: Adam epsilon
```

### Mixed Precision Settings
```yaml
# Precision configuration
fp16: false                                   # REQUIRED: Enable FP16 training
bf16: true                                    # REQUIRED: Enable BF16 training (recommended)
fp16_opt_level: "O1"                          # OPTIONAL: FP16 optimization level
fp16_full_eval: false                         # OPTIONAL: Use FP16 for evaluation
```

**Validation Rules:**
- Only one of `fp16` or `bf16` should be true
- `gradient_accumulation_steps` must be positive integer
- Learning rates must be positive floats

---

## 📊 Data Configuration

### Data Paths
```yaml
# Dataset paths
train_data_path: "data/train.jsonl"          # REQUIRED: Training data path
val_data_path: "data/val.jsonl"              # REQUIRED: Validation data path
teacher_data_path: "data/teacher.jsonl"      # OPTIONAL: Teacher examples path

# Data processing
max_train_samples: null                       # OPTIONAL: Limit training samples
max_eval_samples: null                        # OPTIONAL: Limit evaluation samples
```

### Data Loading Settings
```yaml
# DataLoader configuration
dataloader_num_workers: 4                    # REQUIRED: Number of data loading workers
dataloader_pin_memory: true                  # REQUIRED: Pin memory for GPU transfer
remove_unused_columns: false                 # REQUIRED: Keep all data columns

# Collation settings
collator_type: "packed"                      # REQUIRED: "packed" | "standard"
teacher_ratio: 0.7                           # REQUIRED: Fraction of teacher batches (0.0-1.0)
language: "chinese"                          # REQUIRED: "chinese" | "english"
```

**Validation Rules:**
- Data paths must exist and be readable
- `teacher_ratio` must be between 0.0 and 1.0
- `dataloader_num_workers` should match CPU cores

---

## 🎯 Coordinate Token Configuration

⚠️ **CRITICAL:** The coordinate token system has two distinct modes. Choose the appropriate mode for your use case.

### Standard Mode (Recommended for Most Users)
```yaml
# === COORDINATE TOKEN SETTINGS ===
coordinate_tokens_enabled: false             # Standard mode: coordinates as integers
max_coord_value: 2048                        # Coordinate bounds [0, 2047]

# === MODEL SETTINGS ===
model_path: "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
```

**Features:**
- ✅ Minimal vocabulary extension (+4 geometry tokens)
- ✅ Uses integer coordinates: `[150,10,211,35]`
- ✅ Format: `"<|object_ref_start|>desc:xxxxx<|object_ref_end|>,<|box_start|>[150,10,211,35]<|box_end|>"`
- ✅ Compatible with pretrained model weights
- ✅ **Production ready** - Fully stable and tested

### Coordinate Mode (Advanced/Research)
```yaml
# === COORDINATE TOKEN SETTINGS ===
coordinate_tokens_enabled: true              # Coordinate mode: coordinates as tokens
max_coord_value: 2048                        # Coordinate token range [0, 2047]

# === COORDINATE-SPECIFIC SETTINGS ===
coordinate_loss_weight: 1.0                  # Weight for coordinate token loss
regular_loss_weight: 1.0                     # Weight for regular token loss
soft_expectation_temperature: 1.0            # Temperature for coordinate prediction

# === MODEL SETTINGS ===
model_path: "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"

# === OPTIONAL LOSS SETTINGS ===
focal_loss_weight: 0.1                       # OPTIONAL: Weight for focal loss
l1_loss_weight: 0.1                          # OPTIONAL: Weight for L1 loss
giou_loss_weight: 0.1                        # OPTIONAL: Weight for GIoU loss
focal_loss_alpha: 0.25                       # OPTIONAL: Focal loss alpha
focal_loss_gamma: 2.0                        # OPTIONAL: Focal loss gamma
```

**Features:**
- ✅ Extended vocabulary (+2052 tokens: 4 geometry + 2048 coordinate)
- ✅ Uses coordinate tokens: `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]`
- ✅ Format: `"<|object_ref_start|>desc:xxxxx<|object_ref_end|>,<|box_start|>[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]<|box_end|>"`
- ✅ Sequence-based coordinate prediction
- ⚠️ **Known limitation:** HuggingFace trainer compatibility issue

### Required Parameters for Both Modes

**Mandatory:**
- `coordinate_tokens_enabled`: Boolean flag for mode selection
- `max_coord_value`: Maximum coordinate value (typically 2048)
- `model_path`: Path to Qwen2.5-VL model

**Additional for Coordinate Mode:**
- `coordinate_loss_weight`: Loss weighting for coordinate tokens
- `regular_loss_weight`: Loss weighting for non-coordinate tokens
- `soft_expectation_temperature`: Temperature parameter

**Validation Rules:**
- `coordinate_config_max_coord_value` must be power of 2
- All loss weights must be non-negative
- `soft_expectation_temperature` must be positive

---

## 🔍 Detection Configuration

### Detection Head Settings
```yaml
# Detection system
detection_enabled: true                       # REQUIRED: Enable detection head
num_queries: 100                             # REQUIRED: Number of detection queries
max_caption_length: 50                       # REQUIRED: Maximum caption length

# Detection loss weights
detection_loss_weight: 1.0                   # REQUIRED: Overall detection loss weight
bbox_loss_weight: 5.0                        # REQUIRED: Bounding box loss weight
objectness_loss_weight: 1.0                  # REQUIRED: Objectness loss weight
caption_loss_weight: 1.0                     # REQUIRED: Caption loss weight

# Hungarian matching
hungarian_cost_class: 1.0                    # REQUIRED: Classification cost
hungarian_cost_bbox: 5.0                     # REQUIRED: Bounding box cost
hungarian_cost_giou: 2.0                     # REQUIRED: GIoU cost
```

**Validation Rules:**
- `num_queries` must be positive integer
- All loss weights must be non-negative
- Hungarian costs must be positive

---

## 💾 Checkpointing & Logging

### Checkpoint Settings
```yaml
# Checkpointing strategy
save_strategy: "steps"                       # REQUIRED: "steps" | "epoch" | "no"
save_steps: 500                              # REQUIRED: Save every N steps
save_total_limit: 3                          # REQUIRED: Maximum checkpoints to keep
load_best_model_at_end: true                 # REQUIRED: Load best model after training

# Evaluation strategy
eval_strategy: "steps"                       # REQUIRED: "steps" | "epoch" | "no"
eval_steps: 500                              # REQUIRED: Evaluate every N steps
evaluation_strategy: "steps"                 # DEPRECATED: Use eval_strategy
metric_for_best_model: "eval_loss"           # REQUIRED: Metric for best model selection
greater_is_better: false                     # REQUIRED: Whether higher metric is better
```

### Logging Configuration
```yaml
# Logging settings
logging_dir: "logs"                          # REQUIRED: Logging directory
logging_strategy: "steps"                    # REQUIRED: "steps" | "epoch"
logging_steps: 50                            # REQUIRED: Log every N steps
log_level: "info"                            # REQUIRED: "debug" | "info" | "warning" | "error"

# Reporting
report_to: ["tensorboard"]                   # OPTIONAL: ["tensorboard", "wandb", "none"]
run_name: null                               # OPTIONAL: Run name for logging
```

**Validation Rules:**
- Save and eval strategies must be compatible
- Logging steps should be less than save steps
- Output directories must be writable

---

## ⚙️ System Configuration

### Hardware Settings
```yaml
# GPU configuration
local_rank: -1                               # REQUIRED: Local rank for distributed training
device: "auto"                               # REQUIRED: Device selection
no_cuda: false                               # REQUIRED: Disable CUDA

# Memory management
gradient_checkpointing: false                # REQUIRED: Enable gradient checkpointing
dataloader_drop_last: false                  # REQUIRED: Drop last incomplete batch
group_by_length: false                       # REQUIRED: Group samples by length
```

### Performance Settings
```yaml
# Optimization flags
tf32: true                                   # REQUIRED: Enable TF32 on Ampere GPUs
jit_mode_eval: false                         # OPTIONAL: JIT compilation for eval
use_legacy_prediction_loop: false            # OPTIONAL: Use legacy prediction loop

# Distributed training
ddp_backend: "nccl"                          # OPTIONAL: DDP backend
ddp_bucket_cap_mb: 25                        # OPTIONAL: DDP bucket size
ddp_find_unused_parameters: false            # OPTIONAL: Find unused parameters
```

**Validation Rules:**
- Hardware settings must match available resources
- Distributed settings must be consistent across nodes

---

## ✅ Configuration Validation

### Validation Checklist
```python
# Use this checklist to validate your configuration
def validate_configuration(config_path):
    """Comprehensive configuration validation"""
    
    # 1. File existence
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    
    # 2. YAML syntax
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 3. Required parameters
    required_params = [
        'model_path', 'train_data_path', 'val_data_path',
        'learning_rate', 'num_train_epochs', 'per_device_train_batch_size'
    ]
    for param in required_params:
        assert param in config, f"Missing required parameter: {param}"
    
    # 4. Parameter types and ranges
    assert isinstance(config['learning_rate'], float), "learning_rate must be float"
    assert 0 < config['learning_rate'] < 1, "learning_rate must be in (0, 1)"
    
    # 5. Path validation
    assert os.path.exists(config['model_path']), f"Model path not found: {config['model_path']}"
    assert os.path.exists(config['train_data_path']), f"Train data not found: {config['train_data_path']}"
    
    # 6. Hardware compatibility
    if config.get('attn_implementation') == 'flash_attention_2':
        assert torch.cuda.is_available(), "Flash Attention requires CUDA"
    
    print("✅ Configuration validation passed")
```

### Common Configuration Errors
| **Error** | **Cause** | **Fix** |
|-----------|-----------|---------|
| `FileNotFoundError: model_path` | Invalid model path | Check path exists and is accessible |
| `ValueError: learning_rate` | Learning rate out of range | Use values between 1e-6 and 1e-3 |
| `TypeError: per_device_train_batch_size` | Batch size not integer | Use integer values only |
| `ConfigValidationError: coordinate_tokens` | Missing coordinate config | Add coordinate token parameters |
| `MemoryError: batch_size too large` | Insufficient GPU memory | Reduce batch size or enable gradient checkpointing |

### Configuration Testing
```bash
# Test configuration before training
/root/miniconda3/envs/ms/bin/python -c "
from src.config.global_config import init_config
config = init_config('configs/your_config.yaml')
print('Configuration loaded successfully')
print(f'Model path: {config.model_path}')
print(f'Learning rate: {config.learning_rate}')
print(f'Coordinate tokens: {config.coordinate_tokens_enabled}')
"
```

---

## 🔧 Advanced Configuration

### Parameter Group Management
The system automatically manages parameter groups with different learning rates:

```python
# Parameter groups are created automatically based on parameter names
parameter_groups = {
    'visual': config.vision_lr,           # Vision encoder parameters
    'llm': config.llm_lr,                # Language model parameters  
    'detection_head': config.detection_lr, # Detection head parameters
    'coordinate_tokens': config.coordinate_lr # Coordinate token embeddings
}
```

### Environment Variable Integration
```yaml
# Use environment variables in configuration
model_path: "${HF_HOME}/qwen2.5-vl-7b-instruct"
output_dir: "${EXPERIMENT_DIR}/checkpoints"
logging_dir: "${EXPERIMENT_DIR}/logs"
```

### Configuration Inheritance
```yaml
# Base configuration (base_config.yaml)
_base_: "base_config.yaml"

# Override specific parameters
learning_rate: 1e-4
coordinate_tokens_enabled: true
```

---

## 📚 Configuration Examples

### Development Configuration
```yaml
# configs/development.yaml - Fast iteration
max_train_samples: 100
max_eval_samples: 50
num_train_epochs: 1
save_steps: 10
eval_steps: 10
logging_steps: 5
per_device_train_batch_size: 1
```

### Production Configuration
```yaml
# configs/production.yaml - Full training
num_train_epochs: 30
save_steps: 1000
eval_steps: 1000
logging_steps: 100
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
save_total_limit: 5
load_best_model_at_end: true
```

### Debugging Configuration
```yaml
# configs/debug.yaml - Debugging issues
log_level: "debug"
logging_steps: 1
max_train_samples: 10
max_eval_samples: 5
num_train_epochs: 1
gradient_checkpointing: false
dataloader_num_workers: 0
```

---

**Related Files:**
- `src/config/config_manager.py` - Configuration loading and validation
- `src/config/domain_configs.py` - Domain-specific configuration classes
- `src/training/parameter_manager.py` - Parameter group management
- `configs/base_flat_v2.yaml` - Main configuration template