# Command Cheat Sheet

Quick reference for common operations and commands.

## 🚀 Training Commands

### Basic Training
```bash
# Full training pipeline
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/base_flat_det.yaml \
    --output_dir checkpoints/run_001

# Resume from checkpoint
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/base_flat_det.yaml \
    --output_dir checkpoints/run_001 \
    --resume_from_checkpoint checkpoints/run_001/checkpoint-1000
```

### Training with Custom Settings
```bash
# Coordinate tokens only
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/base_flat_det.yaml \
    --coordinate_tokens_enabled true \
    --coordinate_lr 1e-4

# Disable coordinate tokens (standard LLM)
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/base_flat_det.yaml \
    --coordinate_tokens_enabled false
```

## 📊 Data Processing

### Dataset Conversion
```bash
# Quick start - process default ds/ directory
source activate ms
bash data_conversion/convert_dataset.sh

# Custom directories
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_output" bash data_conversion/convert_dataset.sh

# Language selection
LANGUAGE="english" bash data_conversion/convert_dataset.sh
LANGUAGE="chinese" bash data_conversion/convert_dataset.sh

# Disable image resizing for testing
RESIZE="false" bash data_conversion/convert_dataset.sh
```

### Manual Data Steps
```bash
# Step-by-step processing
cd data_conversion

# 1. Clean raw JSON
/root/miniconda3/envs/ms/bin/python clean_raw_json.py --input_dir ../ds --output_dir ../ds_output

# 2. Copy images
cp -r ../ds/images ../ds_output/

# 3. Process data
/root/miniconda3/envs/ms/bin/python processor.py
```

## 🔧 Model Operations

### Model Loading & Inference
```bash
# Basic inference
/root/miniconda3/envs/ms/bin/python -c "
from src.inference import load_model_and_inference
model, tokenizer = load_model_and_inference('/path/to/checkpoint')
"

# Test coordinate token conversion
/root/miniconda3/envs/ms/bin/python -c "
from src.utils.coordinate_token_manager import create_coordinate_token_manager
# ... test conversion code
"
```

### Model Validation
```bash
# Validate checkpoint
/root/miniconda3/envs/ms/bin/python -c "
from src.core.checkpoint_manager import CheckpointManager
manager = CheckpointManager()
print(manager.validate_checkpoint('/path/to/checkpoint'))
"
```

## 🧪 Testing & Debugging

### Run Tests
```bash
# Run all tests (if test suite exists)
/root/miniconda3/envs/ms/bin/python -m pytest temporal/

# Quick functionality test
/root/miniconda3/envs/ms/bin/python temporal/test_coordinate_tokens.py
```

### Debug Common Issues
```bash
# Check CUDA availability
/root/miniconda3/envs/ms/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check model loading
/root/miniconda3/envs/ms/bin/python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/path/to/model')
print(f'Vocab size: {len(tokenizer)}')
"

# Memory usage check
nvidia-smi
```

## ⚙️ Configuration Management

### Quick Config Updates
```bash
# View current config
/root/miniconda3/envs/ms/bin/python -c "
from src.config.global_config import config
print(config.model_path)
print(config.coordinate_tokens_enabled)
"

# Test config validation
/root/miniconda3/envs/ms/bin/python -c "
from src.config.coordinate_validator import CoordinateValidator
validator = CoordinateValidator()
print(validator.validate_config())
"
```

## 🔍 Monitoring & Logging

### Training Monitoring
```bash
# Watch training logs (if using wandb)
wandb login
# Check wandb dashboard

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check training progress
tail -f checkpoints/run_001/trainer_state.json
```

### Loss Analysis
```bash
# Extract loss components from logs
grep "coordinate_loss" checkpoints/run_001/training.log | tail -10
grep "regular_loss" checkpoints/run_001/training.log | tail -10
```

## 🗂️ File Operations

### Checkpoint Management
```bash
# List checkpoints
ls -la checkpoints/run_001/checkpoint-*/

# Copy best checkpoint
cp -r checkpoints/run_001/checkpoint-1000 final_models/

# Clean up intermediate checkpoints
find checkpoints/run_001 -name "checkpoint-*" -type d | head -n -3 | xargs rm -rf
```

### Data Directory Management
```bash
# Check data structure
tree data/ -L 2
tree ds_output/ -L 2

# Verify image-JSON alignment
/root/miniconda3/envs/ms/bin/python -c "
import json
with open('data/train.jsonl') as f:
    sample = json.loads(f.readline())
    print(f'Image path: {sample[\"image\"]}')
    print(f'Exists: {os.path.exists(sample[\"image\"])}')
"
```

## 🚨 Emergency Recovery

### Training Recovery
```bash
# Find latest checkpoint
find checkpoints/ -name "checkpoint-*" -type d | sort -V | tail -1

# Resume from specific step
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/base_flat_det.yaml \
    --resume_from_checkpoint $(find checkpoints/ -name "checkpoint-*" -type d | sort -V | tail -1)
```

### Environment Recovery
```bash
# Reset conda environment
conda activate ms
pip install -r requirements.txt  # if requirements.txt exists

# Verify critical packages
/root/miniconda3/envs/ms/bin/python -c "
import torch, transformers, datasets
print('Environment OK')
"
```

## 📝 Common Variable Patterns

```bash
# Environment variables
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/data3/Qwen2.5-VL-main/model_cache
export PYTHONPATH=/data3/Qwen2.5-VL-main:$PYTHONPATH

# Common paths
MODEL_PATH="/path/to/qwen2.5-vl-7b-instruct"
DATA_PATH="/data3/Qwen2.5-VL-main/data"
OUTPUT_PATH="/data3/Qwen2.5-VL-main/checkpoints"
CONFIG_PATH="/data3/Qwen2.5-VL-main/configs/base_flat_det.yaml"
```

---
**💡 Pro Tips:**
- Always use `/root/miniconda3/envs/ms/bin/python` for consistency
- Set CUDA_VISIBLE_DEVICES before training for GPU selection
- Use absolute paths in configs to avoid path issues
- Monitor GPU memory with `nvidia-smi` during training
- Keep backup configs before major changes