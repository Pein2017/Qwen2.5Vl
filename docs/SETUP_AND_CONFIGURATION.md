# Setup and Configuration Guide

Note: Quick config keys and behaviors are summarized in `AI_ASSISTANT_KB.md`. This file covers full setup and examples.

**Complete guide for setting up and configuring the BBU training pipeline with coordinate token system**

## 🚀 **Quick Setup (15 minutes)**

### **Prerequisites**
- Linux environment with CUDA GPUs
- Python 3.10+ with PyTorch
- Access to project directory: `/data3/Qwen2.5-VL-main`
- Basic familiarity with command line

### **Step 1: Environment Setup**
```bash
# Navigate to project
cd /data3/Qwen2.5-VL-main

# Verify Python environment
python --version  # Should be Python 3.10+

# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Quick system health check
python -c "from src_new.training.bbu_trainer import BBUTrainer; print('✅ System OK')"
```

### **Step 2: Install Dependencies**
```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import transformers, torch; print('✅ Dependencies OK')"

# Optional: Install FlashAttention v2 for performance
pip install flash-attn --no-build-isolation
```

### **Step 3: Data Preparation**

#### **Object-Oriented Data Conversion**
The data conversion pipeline supports object-oriented training with flexible object type combinations:

```bash
# Configure object-oriented training in data_conversion/convert_dataset.sh
# Edit these essential variables:
INPUT_DIR="ds_v2"                    # Your V2 data directory
OUTPUT_DIR="data"                    # Base output directory
DATASET_NAME="ds_v2_full"           # Dataset identifier
OBJECT_TYPES="full"                 # Object types: "bbu", "fiber wire", "full", etc.
VAL_RATIO="0.1"                     # 10% validation split
MAX_TEACHERS="10"                   # Teacher samples for few-shot learning
RESIZE="true"                       # Enable smart image resizing
SEED="17"                          # Reproducible random seed

# Run the conversion pipeline
bash data_conversion/convert_dataset.sh
```

#### **Object Type Training Options**
```bash
# Equipment-focused training
OBJECT_TYPES="bbu bbu_shield"       # BBU equipment detection

# Cable system training
OBJECT_TYPES="fiber wire"           # Cable and fiber detection

# Hardware components
OBJECT_TYPES="connect_point label"  # Connection points and labels

# Complete system training
OBJECT_TYPES="full"                 # All object types
```

#### **Verify Output**
```bash
# Check generated files
ls -la data/ds_v2_full/  # Should see train.jsonl, val.jsonl, teacher.jsonl, images/
head -1 data/ds_v2_full/train.jsonl | python -m json.tool  # Check format

# Validate object-oriented structure
python -c "
import json
sample = json.loads(open('data/ds_v2_full/train.jsonl').readline())
print(f'Objects: {len(sample[\"objects\"])}')
print(f'Geometries: {[list(obj.keys())[0] for obj in sample[\"objects\"] if obj]}')
"
```

**Expected output**: JSONL files with multi-geometry objects (bbox_2d, line, square) and hierarchical descriptions.

#### **Advanced Data Conversion Features**

##### **Object-Oriented Training System**
The data conversion pipeline supports sophisticated object-oriented training with flexible combinations:

```bash
# data_conversion/convert_dataset.sh key features:

# 1. Smart Image Resizing
RESIZE="true"                    # Intelligent resizing to ≤1024×1024
RESIZE_QUALITY="85"             # JPEG quality for resized images

# 2. Validation Split Control
VAL_RATIO="0.1"                 # 10% validation split
SEED="17"                       # Reproducible random seed

# 3. Teacher Pool Management
MAX_TEACHERS="10"               # Maximum teacher samples for few-shot learning
TEACHER_SELECTION="diverse"     # Selection strategy: "diverse", "random", "best"

# 4. Object Type Filtering
OBJECT_TYPES="bbu fiber"        # Space-separated object types
INCLUDE_EMPTY="false"           # Exclude images without target objects
```

##### **Data Conversion Pipeline Components**
```python
# data_conversion/ directory structure:
├── convert_dataset.sh          # Main conversion script
├── process_images.py           # Image processing and resizing
├── filter_objects.py           # Object type filtering
├── create_splits.py            # Train/validation splitting
├── generate_teachers.py        # Teacher pool creation
└── validate_output.py          # Output validation and statistics
```

##### **Quality Control and Validation**
```bash
# Validate conversion results
cd data_conversion
python validate_output.py --dataset_dir ../data/ds_v2_full

# Check object distribution
python -c "
import json
from collections import Counter

# Load and analyze training data
objects = []
with open('data/ds_v2_full/train.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        for obj in sample['objects']:
            if obj:  # Skip empty objects
                geometry_type = list(obj.keys())[0]
                objects.append(geometry_type)

print('Object geometry distribution:')
for geom, count in Counter(objects).items():
    print(f'  {geom}: {count}')
"
```

##### **Performance Optimization**
```bash
# Parallel processing for large datasets
export NUM_WORKERS=8            # Parallel image processing
export BATCH_SIZE=100           # Batch size for processing

# Memory optimization
export MAX_IMAGE_SIZE=1024      # Maximum image dimension
export JPEG_QUALITY=85          # Balance quality vs file size
```

### **Step 4: First Training Run**
```bash
# src_new/ implementation (recommended)
python scripts/train_new.py --config bbu_v2 --max_steps 100

# Monitor training
tail -f checkpoints/*/training.log
```

## 📊 **Configuration System**

### **Unified Configuration System**

The `src_new/` implementation uses a single unified configuration file with comprehensive validation:

```python
# Load configuration with automatic validation
from src_new.config.config import load_config
config = load_config('configs/bbu_v2.yaml')
```

### **Configuration Structure (bbu_v2.yaml)**
```yaml
# === REQUIRED FIELDS ===
# Model settings
model_path: "/path/to/Qwen2.5-VL-3B-Instruct"
model_size: "3B"
model_max_length: 32000
attn_implementation: "flash_attention_2"
torch_dtype: "bfloat16"

# Training settings
num_train_epochs: 20
per_device_train_batch_size: 1
learning_rate: 5e-6
vision_lr: 5e-7      # Vision encoder learning rate
merger_lr: 1e-5      # Vision-language merger learning rate
llm_lr: 5e-6         # Language model learning rate

# Data paths
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_pool_file: "data/teacher_pool.jsonl"

# === OPTIONAL FIELDS (with defaults) ===
# Coordinate Token System
coordinate_tokens_enabled: false
max_coord_value: 1024
coordinate_loss_weight: 0.05
# Preferred key for soft-expectation temperature (legacy: coordinate_loss_temperature)
coordinate_temperature: 0.5

# Teacher-Student Training
teacher_ratio: 0.5
num_teacher_samples: 1

# Performance Optimization
use_flash_attention: true
mixed_precision: "bf16"
save_safetensors: true       # Use SafeTensors format for 4-6x faster loading
```

# Performance Settings
fp16: true
dataloader_num_workers: 4
dataloader_pin_memory: true
```

### **Coordinate Mode Configuration (Advanced)**
```yaml
# configs/bbu_coordinate.yaml
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_data_path: "data/teacher_pool.jsonl"

# Coordinate Token Settings
coordinate_tokens_enabled: true         # Use coordinate tokens
remove_unused_columns: false           # REQUIRED for coordinate mode
max_coord_value: 2048
coordinate_loss_weight: 0.05
coordinate_lr: 5e-6

# Training Settings
learning_rate: 1e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
max_steps: 1000

# Output Settings
output_dir: "checkpoints/run_coordinate_001"
logging_steps: 10
save_steps: 100
eval_steps: 100

# Performance Settings
fp16: true
attn_implementation: "flash_attention_2"  # Enable FlashAttention v2
model_max_length: 32000
```

## 🔧 **Core Configuration Parameters**

### **Model and Data Settings**
```yaml
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"  # Model location
train_data_path: "data/train.jsonl"                    # Training data
val_data_path: "data/val.jsonl"                        # Validation data
teacher_data_path: "data/teacher_pool.jsonl"           # Teacher forcing data
```

### **Coordinate Token System**
```yaml
coordinate_tokens_enabled: false       # Enable coordinate token mode
remove_unused_columns: false          # CRITICAL: Prevents data loss
max_coord_value: 2048                 # Maximum coordinate value
coordinate_loss_weight: 0.05          # Weight for coordinate loss (coordinate mode only)
coordinate_lr: 5e-6                   # Learning rate for coordinate tokens
```

### **Training Parameters**
```yaml
learning_rate: 1e-5                   # Base learning rate
num_train_epochs: 3                   # Number of training epochs
max_steps: 1000                       # Maximum training steps (overrides epochs)
per_device_train_batch_size: 4        # Batch size per GPU
gradient_accumulation_steps: 1        # Gradient accumulation
warmup_steps: 100                     # Learning rate warmup
weight_decay: 0.01                    # Weight decay for regularization
```

### **Performance and Hardware**
```yaml
fp16: true                           # Mixed precision training
gradient_checkpointing: false        # Memory optimization (slower training)
dataloader_num_workers: 4            # Data loading workers
dataloader_pin_memory: true          # Pin memory for faster data transfer
attn_implementation: "flash_attention_2"  # FlashAttention v2 for performance
model_max_length: 32000              # Maximum sequence length
```

### **Output and Logging**
```yaml
output_dir: "checkpoints/run_001"    # Checkpoint directory
logging_steps: 10                    # Log every N steps
save_steps: 100                      # Save checkpoint every N steps
eval_steps: 100                      # Evaluate every N steps
report_to: "none"                    # Disable wandb/tensorboard
```

### **Dataset and Debugging**
```yaml
max_dataset_size: -1                 # Full dataset (-1) or limit for debugging
eval_dataset_size: -1               # Full eval dataset or limit
```

## 🎯 **Configuration Modes**

### **Production Mode**
- **File**: `configs/bbu_v2.yaml`
- **Features**: Standard integer coordinates, stable performance
- **Use Case**: Production training, reliable results
- **Memory**: ~13GB GPU memory per device

### **Advanced Mode**
- **File**: `configs/bbu_coordinate.yaml`
- **Features**: Coordinate tokens, advanced sequence prediction
- **Use Case**: Research, advanced coordinate understanding
- **Memory**: ~15GB GPU memory per device

### **Debug Mode**
- **File**: `configs/bbu_v2_debug.yaml`
- **Features**: Limited dataset size, faster iteration
- **Use Case**: Development, testing, debugging
- **Memory**: ~8GB GPU memory per device

## 🔍 **Environment Variables**

### **Required Environment Variables**
```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model cache directory
export HF_HOME="/data3/Qwen2.5-VL-main/model_cache"

# FlashAttention optimization
export TRITON_CACHE_DIR="/tmp/triton_cache"
export TORCH_COMPILE_DISABLE=1
export FLASH_ATTENTION_FORCE_CUDNN=0
```

### **Optional Environment Variables**
```bash
# Debugging
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=1

# Performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## ⚠️ **Common Configuration Issues**

### **Issue 1: CUDA Out of Memory**
**Solution**: Reduce batch size or enable gradient checkpointing
```yaml
per_device_train_batch_size: 2        # Reduce from 4
gradient_checkpointing: true          # Enable memory optimization
```

### **Issue 2: Coordinate Mode Failures**
**Solution**: Ensure `remove_unused_columns: false`
```yaml
coordinate_tokens_enabled: true
remove_unused_columns: false         # CRITICAL for coordinate mode
```

### **Issue 3: Slow Training**
**Solution**: Enable FlashAttention v2 and optimize data loading
```yaml
attn_implementation: "flash_attention_2"
dataloader_num_workers: 8
dataloader_pin_memory: true
```

### **Issue 4: Data Loading Errors**
**Solution**: Verify data paths and format
```bash
# Check data files exist
ls -la data/train.jsonl data/val.jsonl data/teacher_pool.jsonl

# Validate JSON format
head -1 data/train.jsonl | python -m json.tool
```

## 🚀 **Performance Optimization**

### **Memory Optimization**
```yaml
# Reduce memory usage
per_device_train_batch_size: 2
gradient_checkpointing: true
fp16: true
dataloader_pin_memory: false
```

### **Speed Optimization**
```yaml
# Maximize training speed
attn_implementation: "flash_attention_2"
dataloader_num_workers: 8
dataloader_pin_memory: true
gradient_checkpointing: false
```

### **Debugging Optimization**
```yaml
# Fast iteration for debugging
max_dataset_size: 10
eval_dataset_size: 5
max_steps: 50
logging_steps: 5
save_steps: 25
```

## 🔧 **Advanced Configuration Features**

### **Configuration Validation System**
The `src_new/config/` system provides comprehensive validation with fail-fast error handling:

```python
from src_new.config.config import Config, load_config

# Load configuration with automatic validation
config = load_config('configs/bbu_v2.yaml')

# Configuration sections are automatically validated
print(f"Model: {config.model_path}")           # Path validation
print(f"Batch size: {config.per_device_train_batch_size}")  # Range validation
print(f"Learning rate: {config.learning_rate}")  # Type validation
```

### **Environment Variable Support**
```yaml
# configs/bbu_v2.yaml with environment variables
model_path: "${HF_HOME}/Qwen/Qwen2.5-VL-3B-Instruct"
output_dir: "${TRAINING_OUTPUT_DIR}/checkpoints"
logging_dir: "${TRAINING_OUTPUT_DIR}/logs"

# Set environment variables before training
export HF_HOME="/data3/Qwen2.5-VL-main/model_cache"
export TRAINING_OUTPUT_DIR="/data3/Qwen2.5-VL-main/output"
```

### **Path Management Integration**
The configuration system integrates with PathManager for unified path resolution:

```python
from src_new.utils.path_manager import PathManager

# Automatic path resolution and validation
path_manager = PathManager(config)
resolved_paths = path_manager.resolve_all_paths()

# Validate all paths exist
path_manager.validate_paths()
```

### **Configuration Debugging**
```bash
# Validate configuration without training
python -c "
from src_new.config.config import load_config
config = load_config('configs/bbu_v2.yaml')
print('✅ Configuration valid')
print(f'Model: {config.model_path}')
print(f'Output: {config.output_dir}')
"

# Debug path resolution
python -c "
from src_new.utils.path_manager import PathManager
from src_new.config.config import load_config
config = load_config('configs/bbu_v2.yaml')
pm = PathManager(config)
pm.validate_paths()
print('✅ All paths valid')
"
```

## ✅ Best-practice configuration notes

- Set `coordinate_temperature` to 0.3–0.7 to sharpen the soft expectation over coordinate logits. The loader keeps
  legacy `coordinate_loss_temperature` in sync for backward compatibility.
- Use `max_grad_norm: 1.0` (or 2.0) for effective L1 learning. Values like `0.1` can over-clip large early coordinate
  errors and slow down regression.
- Keep CE and coordinate losses decoupled by masking: CE counts only text tokens within assistant spans; L1 counts only
  coordinate tokens (also within assistant spans). This is handled automatically by `LossManager`.

---

**Next Steps**: After configuration, proceed to **[TRAINING_AND_IMPLEMENTATION.md](TRAINING_AND_IMPLEMENTATION.md)** for training system details.
