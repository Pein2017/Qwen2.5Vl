# Configuration System Components

**Detailed documentation for the configuration management system (`src/config/`)**

## Overview

The configuration system provides unified access to all system parameters through the DirectConfig approach. It offers type validation, parameter checking, and global access patterns for the entire BBU Detection System.

## Component Architecture

```
src/config/
├── global_config.py    # DirectConfig system implementation
└── __init__.py         # Configuration access methods
```

## DirectConfig System (`global_config.py`)

### Component Contract

**Input**:
- YAML configuration file path
- Environment variables (optional)
- Command-line overrides (optional)

**Output**:
- DirectConfig instance with all parameters
- Type-validated configuration values
- Global configuration access

**Dependencies**:
- PyYAML for configuration file parsing
- Type checking utilities
- Parameter validation functions

**Side Effects**:
- Sets global configuration state
- Validates all parameters on initialization
- Provides global access throughout codebase

### Key Features

#### Flat Parameter Access
```python
# All parameters accessible as attributes
from src.config import get_config
config = get_config()

# Direct parameter access
learning_rate = config.learning_rate
model_path = config.model_path
detection_enabled = config.detection_enabled
coordinate_tokens_enabled = config.coordinate_tokens_enabled
```

#### Type Validation and Conversion
```python
# Automatic type checking and conversion
config.per_device_train_batch_size  # Ensures integer
config.learning_rate                # Ensures float
config.detection_enabled           # Ensures boolean
config.model_path                  # Ensures string
```

#### Parameter Categories
The DirectConfig system manages 149+ parameters across these categories:

| Category | Parameters | Examples |
|----------|------------|----------|
| **Model** | 15+ | `model_path`, `model_max_length`, `torch_dtype` |
| **Training** | 25+ | `learning_rate`, `num_train_epochs`, `per_device_train_batch_size` |
| **Data** | 20+ | `train_data_path`, `val_data_path`, `teacher_ratio` |
| **Detection** | 10+ | `detection_enabled`, `coordinate_tokens_enabled`, `max_coord_value` |
| **Optimization** | 15+ | `gradient_checkpointing`, `attn_implementation`, `warmup_ratio` |
| **Logging** | 10+ | `logging_steps`, `save_steps`, `output_dir` |
| **System** | 20+ | `dataloader_num_workers`, `remove_unused_columns` |

### Usage Example

```python
# Initialize configuration (done automatically in training scripts)
from src.config import get_config
config = get_config()

# Access any parameter directly
print(f"Model path: {config.model_path}")
print(f"Learning rate: {config.learning_rate}")
print(f"Detection enabled: {config.detection_enabled}")

# Use in components
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path=config.model_path,
    for_inference=False,
    attn_implementation=config.attn_implementation
)
```

## Configuration Access Patterns

### Global Access (Recommended)
```python
# Import and use anywhere in codebase
from src.config import get_config
config = get_config()

# All parameters available
batch_size = config.per_device_train_batch_size
```

### Component-Specific Access
```python
# Pass config to components that need it
trainer = BBUTrainer(
    model=model,
    args=training_args,
    cfg=config,  # Pass entire config
    # ... other arguments
)

# Components can access any parameter
class BBUTrainer(Trainer):
    def __init__(self, cfg, ...):
        self.config = cfg
        self.teacher_ratio = cfg.teacher_ratio
        self.coordinate_tokens_enabled = cfg.coordinate_tokens_enabled
```

### Factory Pattern Integration
```python
# Factories use global config automatically
from src.training.trainer_factory import create_trainer_with_coordinator

# Factory accesses global config internally
trainer = create_trainer_with_coordinator(training_args)
```

## Configuration File Structure

### Main Configuration Template (`configs/base_flat_v2.yaml`)

```yaml
# Model Configuration
model_path: "/path/to/qwen2.5-vl-7b-instruct"
model_max_length: 120000
torch_dtype: "bfloat16"
attn_implementation: "flash_attention_2"

# Detection and Token Settings
detection_enabled: true
coordinate_tokens_enabled: true
max_coord_value: 2048

# Training Configuration
learning_rate: 1e-5
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
warmup_ratio: 0.1
lr_scheduler_type: "cosine"

# Data Configuration
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_ratio: 0.3

# Teacher-Student Learning
teacher_loss_weight: 0.3
student_loss_weight: 1.0

# Output and Logging
output_dir: "./output"
logging_steps: 10
save_steps: 500
eval_steps: 500
save_total_limit: 3

# Memory Optimization
gradient_checkpointing: true
dataloader_num_workers: 4
remove_unused_columns: false
```

### Configuration Validation

#### Required Parameters
```python
# These parameters must be set
required_params = [
    'model_path',
    'train_data_path',
    'val_data_path',
    'output_dir'
]

# Validation happens automatically on config load
config = get_config()  # Will fail if required params missing
```

#### Parameter Constraints
```python
# Automatic constraint checking
assert config.per_device_train_batch_size > 0
assert config.learning_rate > 0
assert 0 <= config.teacher_ratio <= 1
assert config.max_coord_value > 0
```

#### Type Validation
```python
# Automatic type conversion and validation
config.detection_enabled          # bool
config.coordinate_tokens_enabled  # bool
config.per_device_train_batch_size # int
config.learning_rate              # float
config.model_path                 # str
```

## Configuration Customization

### Creating Custom Configurations
```bash
# Copy base template
cp configs/base_flat_v2.yaml configs/my_experiment.yaml

# Edit parameters
vim configs/my_experiment.yaml

# Use custom config
python scripts/train.py --config configs/my_experiment.yaml
```

### Environment Variable Overrides
```bash
# Override specific parameters via environment
export MODEL_PATH="/custom/model/path"
export LEARNING_RATE="2e-5"
export PER_DEVICE_TRAIN_BATCH_SIZE="1"

# Config system will use environment values
python scripts/train.py --config configs/base_flat_v2.yaml
```

### Command-Line Overrides
```bash
# Override parameters via command line
python scripts/train.py \
    --config configs/base_flat_v2.yaml \
    --model_path "/custom/model/path" \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1
```

## Configuration Presets

### Detection Training (Recommended)
```yaml
# configs/detection_training.yaml
detection_enabled: true
coordinate_tokens_enabled: true
teacher_ratio: 0.3
learning_rate: 1e-5
per_device_train_batch_size: 2
```

### Memory-Constrained Training
```yaml
# configs/low_memory.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
model_max_length: 60000
dataloader_num_workers: 2
```

### Fast Experimentation
```yaml
# configs/fast_experiment.yaml
num_train_epochs: 1
per_device_train_batch_size: 4
logging_steps: 5
save_steps: 100
eval_steps: 100
```

### Production Training
```yaml
# configs/production.yaml
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
save_steps: 1000
eval_steps: 1000
save_total_limit: 5
```

## Integration with Components

### Training System Integration
```python
# BBUTrainer uses config directly
class BBUTrainer(Trainer):
    def __init__(self, cfg, ...):
        self.teacher_ratio = cfg.teacher_ratio
        self.coordinate_tokens_enabled = cfg.coordinate_tokens_enabled
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Use config parameters in loss computation
        if self.coordinate_tokens_enabled:
            # Apply coordinate token loss
            pass
```

### Model System Integration
```python
# ModelLoader uses config for model loading
def load_model_and_processor_unified(model_path=None, ...):
    config = get_config()
    
    # Use config defaults if not specified
    model_path = model_path or config.model_path
    torch_dtype = config.torch_dtype
    attn_implementation = config.attn_implementation
```

### Data Pipeline Integration
```python
# DataProcessor uses config for dataset creation
class DataProcessor:
    def __init__(self, tokenizer, image_processor, model):
        config = get_config()
        self.train_data_path = config.train_data_path
        self.val_data_path = config.val_data_path
        self.teacher_ratio = config.teacher_ratio
```

## Debugging and Validation

### Configuration Health Checks
```python
# Verify configuration loading
from src.config import get_config
config = get_config()

# Check critical parameters
assert config.model_path is not None
assert os.path.exists(config.model_path)
assert config.detection_enabled is True
assert config.coordinate_tokens_enabled is True

print("✅ Configuration validation passed")
```

### Parameter Inspection
```python
# List all configuration parameters
config = get_config()
for attr in dir(config):
    if not attr.startswith('_'):
        value = getattr(config, attr)
        print(f"{attr}: {value}")
```

### Configuration Comparison
```python
# Compare configurations
def compare_configs(config1_path, config2_path):
    import yaml
    
    with open(config1_path) as f1, open(config2_path) as f2:
        c1 = yaml.safe_load(f1)
        c2 = yaml.safe_load(f2)
    
    # Find differences
    for key in set(c1.keys()) | set(c2.keys()):
        if c1.get(key) != c2.get(key):
            print(f"{key}: {c1.get(key)} -> {c2.get(key)}")
```

## Common Configuration Patterns

### Development vs Production
```python
# Development configuration
if config.debug_mode:
    config.logging_steps = 1
    config.save_steps = 10
    config.num_train_epochs = 1

# Production configuration
else:
    config.logging_steps = 100
    config.save_steps = 1000
    config.num_train_epochs = 5
```

### Conditional Configuration
```python
# Memory-based configuration
if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:  # < 24GB
    config.per_device_train_batch_size = 1
    config.gradient_checkpointing = True
else:
    config.per_device_train_batch_size = 4
    config.gradient_checkpointing = False
```

### Configuration Inheritance
```python
# Base configuration with overrides
base_config = load_config("configs/base_flat_v2.yaml")
experiment_config = load_config("configs/experiment.yaml")

# Merge configurations (experiment overrides base)
final_config = {**base_config, **experiment_config}
```

## Error Handling and Validation

### Common Configuration Errors
```python
# Missing required parameters
try:
    config = get_config()
except ValueError as e:
    print(f"Configuration error: {e}")
    # Handle missing parameters

# Invalid parameter values
if config.per_device_train_batch_size <= 0:
    raise ValueError("Batch size must be positive")

# File path validation
if not os.path.exists(config.model_path):
    raise FileNotFoundError(f"Model not found: {config.model_path}")
```

### Configuration Recovery
```python
# Fallback to defaults
def get_config_with_fallback():
    try:
        return get_config()
    except Exception:
        # Load default configuration
        return load_config("configs/base_flat_v2.yaml")
```

---

**Next Steps**:
- **Training System**: [training-system.md](training-system.md)
- **Model System**: [model-system.md](model-system.md)
- **Data Pipeline**: [data-pipeline.md](data-pipeline.md)
