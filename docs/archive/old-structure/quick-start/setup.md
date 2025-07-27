# Environment Setup

**Detailed setup instructions for the BBU Detection System**

## Prerequisites

### System Requirements
- **OS**: Linux (tested on Ubuntu/CentOS)
- **GPU**: CUDA-compatible GPUs (recommended: 4x GPUs with 24GB+ VRAM each)
- **Python**: 3.10+ (via conda environment `ms`)
- **Storage**: 50GB+ free space for models and data

### Network Requirements
- **Location**: Designed for use in China
- **Restrictions**: Cannot access GitHub, Google, HuggingFace directly
- **Solution**: Uses local mirrors and cached models

## Step 1: Verify Environment

### Check Python Environment
```bash
# Navigate to project directory
cd /data3/Qwen2.5-VL-main

# Use full Python path (recommended approach)
/root/miniconda3/envs/ms/bin/python --version
# Expected: Python 3.10.x

# Alternative: Activate conda environment (if needed)
conda activate ms
python --version
```

**Why use full path?** Avoids conda activation inconsistencies mentioned in project guidelines.

### Verify CUDA Setup
```bash
# Check CUDA availability
/root/miniconda3/envs/ms/bin/python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
"
```

**Expected output**:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA devices: 4
  Device 0: NVIDIA A100-SXM4-40GB
  Device 1: NVIDIA A100-SXM4-40GB
  ...
```

### Test Core Imports
```bash
# Test training system
/root/miniconda3/envs/ms/bin/python -c "
from src.training.trainer import BBUTrainer
from src.training.training_coordinator import TrainingCoordinator
from src.training.loss_manager import LossManager
print('✅ Training system imports OK')
"

# Test model system
/root/miniconda3/envs/ms/bin/python -c "
from src.models.model_loader import load_model_and_processor_unified
from src.models.wrapper import Qwen25VLWithDetection
print('✅ Model system imports OK')
"

# Test data processing
/root/miniconda3/envs/ms/bin/python -c "
from src.core.data_processor import DataProcessor
from src.chat_processor import ChatProcessor
print('✅ Data processing imports OK')
"

# Test configuration
/root/miniconda3/envs/ms/bin/python -c "
from src.config import get_config
print('✅ Configuration system imports OK')
"
```

## Step 2: Environment Variables

### Set Required Variables
```bash
# GPU configuration (adjust based on your setup)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model cache directory (adjust path as needed)
export HF_HOME=/data3/Qwen2.5-VL-main/model_cache

# Optional: Add to ~/.bashrc for persistence
echo 'export CUDA_VISIBLE_DEVICES=0,1,2,3' >> ~/.bashrc
echo 'export HF_HOME=/data3/Qwen2.5-VL-main/model_cache' >> ~/.bashrc
```

### Verify Environment Variables
```bash
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
echo "PWD: $PWD"  # Should be /data3/Qwen2.5-VL-main
```

## Step 3: Validate Project Structure

### Check Key Directories
```bash
# Verify project structure
ls -la | grep -E "(src|data_conversion|scripts|configs)"
# Expected: src/, data_conversion/, scripts/, configs/

# Check source code structure
ls -la src/ | grep -E "(training|models|core|config)"
# Expected: training/, models/, core/, config/

# Check data conversion pipeline
ls -la data_conversion/ | grep -E "(convert_dataset.sh|pipeline_manager.py)"
# Expected: convert_dataset.sh, pipeline_manager.py
```

### Verify Configuration Files
```bash
# Check main config template
ls -la configs/base_flat_v2.yaml
# Expected: -rw-r--r-- ... configs/base_flat_v2.yaml

# Validate config syntax
/root/miniconda3/envs/ms/bin/python -c "
import yaml
with open('configs/base_flat_v2.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Config file syntax OK')
print(f'Model path: {config.get(\"model_path\", \"NOT_SET\")}')
"
```

## Step 4: Test Data Pipeline

### Check Raw Data
```bash
# Verify raw data directory exists
ls -la ds_v2/ 2>/dev/null || echo "⚠️ Raw data directory ds_v2/ not found"

# If raw data exists, check structure
if [ -d "ds_v2" ]; then
    echo "Raw data structure:"
    find ds_v2/ -name "*.json" | head -5
    find ds_v2/ -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | head -5
fi
```

### Test Data Processing (Dry Run)
```bash
# Test data conversion pipeline (without actually processing)
/root/miniconda3/envs/ms/bin/python -c "
from data_conversion.pipeline_manager import PipelineManager
from data_conversion.config import DataConversionConfig
print('✅ Data pipeline imports OK')
"
```

## Step 5: Model Path Configuration

### Check Model Availability
```bash
# Check if model path exists (update path as needed)
MODEL_PATH="/path/to/qwen2.5-vl-7b-instruct"  # Update this
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Model found at $MODEL_PATH"
    ls -la "$MODEL_PATH" | head -5
else
    echo "⚠️ Model not found at $MODEL_PATH"
    echo "Please update model_path in configs/base_flat_v2.yaml"
fi
```

### Update Configuration
```bash
# Create your own config file
cp configs/base_flat_v2.yaml configs/my_config.yaml

# Update model path (replace with your actual path)
sed -i 's|model_path: ".*"|model_path: "/your/actual/model/path"|' configs/my_config.yaml

# Verify the change
grep "model_path:" configs/my_config.yaml
```

## Troubleshooting Common Issues

### Import Errors
```bash
# If you get "ModuleNotFoundError: No module named 'src'"
# Make sure you're in the project root:
pwd  # Should be /data3/Qwen2.5-VL-main

# Check PYTHONPATH
echo $PYTHONPATH

# If needed, add project root to PYTHONPATH
export PYTHONPATH=/data3/Qwen2.5-VL-main:$PYTHONPATH
```

### CUDA Issues
```bash
# If CUDA is not available:
# 1. Check NVIDIA driver
nvidia-smi

# 2. Check CUDA installation
nvcc --version

# 3. Check PyTorch CUDA version
/root/miniconda3/envs/ms/bin/python -c "import torch; print(torch.version.cuda)"
```

### Memory Issues
```bash
# If you get CUDA out of memory:
# 1. Check GPU memory usage
nvidia-smi

# 2. Reduce batch size in config
sed -i 's/per_device_train_batch_size: [0-9]*/per_device_train_batch_size: 1/' configs/my_config.yaml

# 3. Enable gradient checkpointing
sed -i 's/gradient_checkpointing: false/gradient_checkpointing: true/' configs/my_config.yaml
```

### Permission Issues
```bash
# If you get permission errors:
# Check file permissions
ls -la configs/base_flat_v2.yaml
ls -la data_conversion/convert_dataset.sh

# Make scripts executable
chmod +x data_conversion/convert_dataset.sh
chmod +x scripts/run_train.sh
```

## Validation Checklist

Before proceeding to training, ensure all these checks pass:

- [ ] Python 3.10+ available via `/root/miniconda3/envs/ms/bin/python`
- [ ] CUDA available with 4+ GPUs
- [ ] All core imports work without errors
- [ ] Project structure is complete
- [ ] Configuration file syntax is valid
- [ ] Model path is accessible
- [ ] Environment variables are set
- [ ] Data pipeline imports work

## Next Steps

Once setup is complete:
1. **Quick test**: [README.md](README.md) - 5-minute overview
2. **First training**: [first-training.md](first-training.md) - Complete training setup
3. **Common issues**: [common-issues.md](common-issues.md) - If problems arise

---

**Need help?** Check [../reference/troubleshooting.md](../reference/troubleshooting.md) for comprehensive troubleshooting.
