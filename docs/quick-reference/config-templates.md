# Configuration Templates & Validation

Quick-start configuration templates and validation checklists for different training scenarios.

## 🎯 Training Scenarios

### 1. Coordinate Token Training (Recommended)
```yaml
# configs/coordinate_training.yaml
model_path: "/path/to/qwen2.5-vl-7b-instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"

# Coordinate Token Settings
coordinate_tokens_enabled: true
coordinate_config_max_coord_value: 2048
coordinate_lr: 1e-4
coordinate_loss_weight: 1.0
regular_loss_weight: 1.0
soft_expectation_temperature: 1.0

# Focal Loss (for coordinate tokens)
focal_loss_alpha: 0.25
focal_loss_gamma: 2.0

# Training Settings
learning_rate: 1e-5
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
warmup_ratio: 0.1
lr_scheduler_type: "cosine"

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
load_best_model_at_end: true
metric_for_best_model: "eval_loss"
```

### 2. Standard LLM Training (No Coordinate Tokens)
```yaml
# configs/standard_llm.yaml
model_path: "/path/to/qwen2.5-vl-7b-instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"

# Disable Coordinate Tokens
coordinate_tokens_enabled: false

# Standard LLM Settings
learning_rate: 2e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
warmup_ratio: 0.1

# Model Settings
model_max_length: 8192
torch_dtype: "bfloat16"
attn_implementation: "flash_attention_2"

# Rest same as coordinate training...
```

### 3. Teacher-Student Learning
```yaml
# configs/teacher_student.yaml
# Base coordinate token config + these additions:

# Teacher-Student Settings
teacher_ratio: 0.3
teacher_pool_file: "data/teacher.jsonl"
detection_freeze_epochs: 1

# Enhanced Loss Weighting
coordinate_loss_weight: 1.5
regular_loss_weight: 1.0

# Longer training for complexity
num_train_epochs: 5
learning_rate: 8e-6
```

### 4. Development/Debug Config
```yaml
# configs/debug.yaml
# Minimal config for fast iteration

model_path: "/path/to/qwen2.5-vl-7b-instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"

# Quick settings
coordinate_tokens_enabled: true
max_train_samples: 100
max_eval_samples: 50
num_train_epochs: 1
per_device_train_batch_size: 1
eval_strategy: "steps"
eval_steps: 10
save_steps: 10
logging_steps: 5

# Fast model settings
model_max_length: 2048
torch_dtype: "float16"
attn_implementation: "eager"
```

## ✅ Configuration Validation Checklist

### Pre-Training Validation
```bash
# 1. Path Validation
□ Model path exists and contains required files (config.json, model files)
□ Data paths exist (train.jsonl, val.jsonl)
□ Output directory is writable
□ Teacher pool file exists (if using teacher-student)

# Commands to check:
ls -la /path/to/model/config.json
ls -la data/train.jsonl data/val.jsonl
mkdir -p checkpoints/test && rmdir checkpoints/test
```

### Configuration Consistency
```bash
# 2. Coordinate Token Settings
□ coordinate_tokens_enabled matches your training goal
□ coordinate_config_max_coord_value >= max coordinate in data
□ coordinate_lr is reasonable (1e-5 to 1e-3)
□ Loss weights sum to reasonable total

# Validation command:
/root/miniconda3/envs/ms/bin/python -c "
from src.config.coordinate_validator import CoordinateValidator
validator = CoordinateValidator()
result = validator.validate_config()
print('Valid:', result.is_valid)
if not result.is_valid:
    for error in result.errors:
        print('Error:', error)
"
```

### Resource Requirements
```bash
# 3. Hardware & Memory
□ CUDA devices available: nvidia-smi
□ Memory requirements met (estimate: batch_size * seq_len * 8GB for bfloat16)
□ Disk space for checkpoints (estimate: model_size * save_total_limit * 2)

# Memory estimation:
echo "Estimated GPU memory: $((per_device_train_batch_size * model_max_length * 8 / 1024))MB per GPU"
```

### Data Validation
```bash
# 4. Data Format & Quality
□ JSONL format valid
□ Required fields present (image, conversations)
□ Image paths exist and accessible
□ Coordinate data format correct (if using coordinate tokens)

# Data validation command:
/root/miniconda3/envs/ms/bin/python -c "
import json
import os

def validate_jsonl(file_path, max_samples=10):
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                data = json.loads(line)
                assert 'image' in data, f'Missing image field in line {i+1}'
                assert 'conversations' in data, f'Missing conversations field in line {i+1}'
                assert os.path.exists(data['image']), f'Image not found: {data[\"image\"]}'
                print(f'Line {i+1}: OK')
            except Exception as e:
                print(f'Line {i+1}: ERROR - {e}')
                return False
    return True

print('Train data validation:')
validate_jsonl('data/train.jsonl')
print('Val data validation:')
validate_jsonl('data/val.jsonl')
"
```

## 🔧 Configuration Utilities

### Quick Config Generator
```python
# Generate config for your scenario
def generate_config(scenario="coordinate_training", **overrides):
    base_configs = {
        "coordinate_training": {
            "coordinate_tokens_enabled": True,
            "coordinate_lr": 1e-4,
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 2,
            "num_train_epochs": 3
        },
        "standard_llm": {
            "coordinate_tokens_enabled": False,
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 4,
            "num_train_epochs": 3
        },
        "debug": {
            "coordinate_tokens_enabled": True,
            "max_train_samples": 100,
            "max_eval_samples": 50,
            "num_train_epochs": 1,
            "eval_steps": 10
        }
    }
    
    config = base_configs[scenario].copy()
    config.update(overrides)
    return config

# Usage:
# config = generate_config("coordinate_training", learning_rate=5e-6)
```

### Configuration Migration
```bash
# Convert old config to new format
/root/miniconda3/envs/ms/bin/python -c "
from src.config.config_manager import ConfigManager
manager = ConfigManager()
new_config = manager.migrate_legacy_config('old_config.yaml')
manager.save_config(new_config, 'new_config.yaml')
"
```

## 🚨 Common Configuration Issues

### Issue: Out of Memory
```yaml
# Solutions:
per_device_train_batch_size: 1  # Reduce batch size
gradient_accumulation_steps: 8  # Increase to maintain effective batch size
model_max_length: 4096         # Reduce sequence length
torch_dtype: "float16"         # Use lower precision
```

### Issue: Slow Training
```yaml
# Solutions:
attn_implementation: "flash_attention_2"  # Use Flash Attention
dataloader_num_workers: 4                 # Increase data loading workers
pin_memory: true                          # Pin memory for faster transfer
bf16: true                                # Use bfloat16 for speed
```

### Issue: Poor Coordinate Token Performance
```yaml
# Solutions:
coordinate_lr: 1e-3              # Increase coordinate learning rate
coordinate_loss_weight: 2.0      # Increase coordinate loss weight
soft_expectation_temperature: 0.5 # Lower temperature for sharper predictions
focal_loss_gamma: 2.0            # Tune focal loss for hard examples
```

### Issue: Model Not Learning
```yaml
# Debugging:
learning_rate: 1e-4              # Increase learning rate
warmup_ratio: 0.2                # Longer warmup
lr_scheduler_type: "linear"      # Try different scheduler
weight_decay: 0.001              # Reduce regularization
```

---
**💡 Quick Start Commands:**
```bash
# 1. Copy template
cp docs/quick-reference/config-templates.md configs/my_config.yaml

# 2. Edit paths and settings
nano configs/my_config.yaml

# 3. Validate config
/root/miniconda3/envs/ms/bin/python -c "from src.config.coordinate_validator import CoordinateValidator; print(CoordinateValidator().validate_config())"

# 4. Start training
/root/miniconda3/envs/ms/bin/python scripts/train.py --config configs/my_config.yaml
```

## 📚 Related Documentation

### Quick Reference
- **[API Core Components →](api-core-components.md)** - Configuration APIs and usage
- **[Problem Solution Lookup →](problem-solution-lookup.md)** - Configuration troubleshooting

### Deep Dive Documentation
- **[Configuration Reference →](../configuration-reference-complete.md)** - Complete parameter documentation
- **[Configuration Guide →](../configuration.md)** - Configuration usage patterns
- **[Architecture Overview →](../architecture-overview.md)** - How configuration fits in system

### User Guides
- **[New Developer Onboarding →](../user-journeys/new-developer-onboarding.md)** - Setup with configuration
- **[Troubleshooter Guide →](../user-journeys/troubleshooter-quickstart.md)** - Configuration issue fixing

---

**Navigation:**
- **[← Back to Quick Reference](./)**
- **[Complete Config Reference →](../configuration-reference-complete.md)**
- **[API Documentation →](api-core-components.md)**