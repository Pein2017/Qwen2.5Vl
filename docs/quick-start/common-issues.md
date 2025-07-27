# Common Issues - Top 5 Problems + Quick Fixes

**Fast solutions for the most common problems**

## 🚨 Emergency Quick Reference

| **Symptom** | **Quick Fix** | **Time** |
|-------------|---------------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `cd /data3/Qwen2.5-VL-main` | 30s |
| `CUDA out of memory` | Reduce batch size: `per_device_train_batch_size: 1` | 1min |
| `AttributeError: 'DirectConfig' object has no attribute 'X'` | Check parameter name in `src/config/global_config.py` | 2min |
| `TrainingCoordinator setup failed` | Check model/tokenizer initialization | 3min |
| `PipelineManager stage X failed` | Check `data_conversion/` logs | 5min |

## Issue #1: Module Import Errors

### Symptoms
```bash
ModuleNotFoundError: No module named 'src'
ModuleNotFoundError: No module named 'src.training'
ImportError: cannot import name 'BBUTrainer'
```

### Root Cause
Running from wrong directory or PYTHONPATH issues.

### Quick Fix (30 seconds)
```bash
# Make sure you're in the project root
cd /data3/Qwen2.5-VL-main
pwd  # Should show /data3/Qwen2.5-VL-main

# Test import
/root/miniconda3/envs/ms/bin/python -c "from src.training.trainer import BBUTrainer; print('✅ Fixed')"
```

### Complete Fix
```bash
# Add project root to PYTHONPATH if needed
export PYTHONPATH=/data3/Qwen2.5-VL-main:$PYTHONPATH

# Verify project structure
ls -la | grep -E "(src|data_conversion|scripts|configs)"
# Should see: src/, data_conversion/, scripts/, configs/
```

## Issue #2: CUDA Out of Memory

### Symptoms
```bash
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
torch.cuda.OutOfMemoryError: CUDA out of memory
```

### Root Cause
Batch size too large for available GPU memory.

### Quick Fix (1 minute)
```bash
# Edit your config file
sed -i 's/per_device_train_batch_size: [0-9]*/per_device_train_batch_size: 1/' configs/my_training.yaml

# Increase gradient accumulation to maintain effective batch size
sed -i 's/gradient_accumulation_steps: [0-9]*/gradient_accumulation_steps: 16/' configs/my_training.yaml

# Enable gradient checkpointing
sed -i 's/gradient_checkpointing: false/gradient_checkpointing: true/' configs/my_training.yaml
```

### Memory Optimization
```bash
# Check GPU memory
nvidia-smi

# For extreme memory constraints
cat >> configs/my_training.yaml << 'EOF'
# Memory optimization
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
gradient_checkpointing: true
model_max_length: 60000  # Reduce from 120000
dataloader_num_workers: 2  # Reduce from 4
EOF
```

## Issue #3: Configuration Errors

### Symptoms
```bash
AttributeError: 'DirectConfig' object has no attribute 'some_parameter'
KeyError: 'model_path'
yaml.scanner.ScannerError: mapping values are not allowed here
```

### Root Cause
- Typo in parameter names
- YAML syntax errors
- Missing required parameters

### Quick Fix (2 minutes)
```bash
# Check YAML syntax
/root/miniconda3/envs/ms/bin/python -c "
import yaml
with open('configs/my_training.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ YAML syntax OK')
"

# Check parameter names against DirectConfig
grep -n "some_parameter" src/config/global_config.py
# If not found, check the correct parameter name

# Common parameter name fixes
sed -i 's/model_name_or_path/model_path/' configs/my_training.yaml
sed -i 's/train_file/train_data_path/' configs/my_training.yaml
sed -i 's/validation_file/val_data_path/' configs/my_training.yaml
```

### Validate Configuration
```bash
# Test configuration loading
/root/miniconda3/envs/ms/bin/python -c "
from src.config import get_config
import os
os.environ['CONFIG_PATH'] = 'configs/my_training.yaml'
config = get_config()
print('✅ Configuration loaded successfully')
print(f'Model path: {config.model_path}')
"
```

## Issue #4: Training Coordinator Setup Failed

### Symptoms
```bash
TrainingCoordinator setup failed
RuntimeError: Model and tokenizer initialization failed
AttributeError: 'NoneType' object has no attribute 'tokenizer'
```

### Root Cause
Model loading or tokenizer initialization issues.

### Quick Fix (3 minutes)
```bash
# Test model loading separately
/root/miniconda3/envs/ms/bin/python -c "
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path='/path/to/your/model',  # Update this
    for_inference=False
)
print('✅ Model loading OK')
print(f'Tokenizer vocab size: {len(tokenizer.get_vocab())}')
"

# Check model path exists
MODEL_PATH=$(grep "model_path:" configs/my_training.yaml | cut -d'"' -f2)
ls -la "$MODEL_PATH" || echo "❌ Model path not found: $MODEL_PATH"

# Update model path if needed
sed -i 's|model_path: ".*"|model_path: "/correct/path/to/model"|' configs/my_training.yaml
```

### Debug Training Coordinator
```bash
# Test coordinator creation
/root/miniconda3/envs/ms/bin/python -c "
from src.training.training_coordinator import TrainingCoordinator
from src.models.model_loader import load_model_and_processor_unified
from src.config import get_config

config = get_config()
model, tokenizer, _ = load_model_and_processor_unified(config.model_path, for_inference=False)
coordinator = TrainingCoordinator(model=model, tokenizer=tokenizer, config_obj=config)
print('✅ TrainingCoordinator setup OK')
"
```

## Issue #5: Data Pipeline Failures

### Symptoms
```bash
PipelineManager stage 1 failed
FileNotFoundError: [Errno 2] No such file or directory: 'ds_v2/'
ValueError: No valid samples found
```

### Root Cause
- Missing raw data directory
- Invalid data format
- Processing pipeline errors

### Quick Fix (5 minutes)
```bash
# Check raw data directory
ls -la ds_v2/ || echo "❌ Raw data directory not found"

# If raw data exists, check structure
if [ -d "ds_v2" ]; then
    echo "Raw data files:"
    find ds_v2/ -name "*.json" | head -3
    find ds_v2/ -name "*.jpg" -o -name "*.jpeg" | head -3
fi

# Check data conversion logs
tail -20 data_conversion/pipeline.log 2>/dev/null || echo "No pipeline log found"

# Test data conversion components
/root/miniconda3/envs/ms/bin/python -c "
from data_conversion.pipeline_manager import PipelineManager
from data_conversion.config import DataConversionConfig
print('✅ Data conversion imports OK')
"
```

### Manual Data Processing
```bash
# Run data processing step by step
cd data_conversion

# Step 1: Clean raw JSON
/root/miniconda3/envs/ms/bin/python clean_raw_json.py ../ds_v2 ../ds_v2_clean --lang zh

# Step 2: Process samples
/root/miniconda3/envs/ms/bin/python pipeline_manager.py \
    --input_dir ../ds_v2_clean \
    --output_dir ../data \
    --object_types "bbu label" \
    --resize true \
    --val_ratio 0.1

# Check output
ls -la ../data/
```

## General Debugging Strategy

### 1. Component Health Check (1 minute)
```bash
# Test all core components
/root/miniconda3/envs/ms/bin/python -c "
try:
    from src.training.trainer import BBUTrainer
    print('✅ Training system')
except Exception as e:
    print(f'❌ Training system: {e}')

try:
    from src.models.model_loader import load_model_and_processor_unified
    print('✅ Model system')
except Exception as e:
    print(f'❌ Model system: {e}')

try:
    from src.config import get_config
    print('✅ Config system')
except Exception as e:
    print(f'❌ Config system: {e}')

try:
    import torch
    print(f'✅ CUDA: {torch.cuda.is_available()}')
except Exception as e:
    print(f'❌ CUDA: {e}')
"
```

### 2. Environment Check (30 seconds)
```bash
# Check environment
echo "PWD: $PWD"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(/root/miniconda3/envs/ms/bin/python --version)"

# Check GPU
nvidia-smi | head -10
```

### 3. Log Analysis (2 minutes)
```bash
# Check recent logs
find . -name "*.log" -mtime -1 -exec echo "=== {} ===" \; -exec tail -10 {} \;

# Check for errors
find . -name "*.log" -exec grep -l -i "error\|exception\|failed" {} \;
```

## When to Seek Help

### Check These First:
1. **Environment**: Python path, CUDA, project directory
2. **Configuration**: YAML syntax, parameter names, file paths
3. **Data**: Raw data exists, processing completed
4. **Resources**: GPU memory, disk space, permissions

### Get Help:
- **Full troubleshooting**: [../reference/troubleshooting.md](../reference/troubleshooting.md)
- **Component details**: [../components/](../components/)
- **Architecture understanding**: [../MENTAL_MODEL.md](../MENTAL_MODEL.md)

## Prevention Tips

### Before Training:
```bash
# Run this checklist
echo "Pre-training checklist:"
echo "1. Project directory: $(pwd)"
echo "2. Python version: $(/root/miniconda3/envs/ms/bin/python --version)"
echo "3. CUDA available: $(/root/miniconda3/envs/ms/bin/python -c 'import torch; print(torch.cuda.is_available())')"
echo "4. Data files: $(ls data/*.jsonl 2>/dev/null | wc -l) files"
echo "5. Config syntax: $(/root/miniconda3/envs/ms/bin/python -c 'import yaml; yaml.safe_load(open(\"configs/my_training.yaml\")); print(\"OK\")' 2>/dev/null || echo 'ERROR')"
```

### During Training:
- Monitor GPU memory with `watch nvidia-smi`
- Check logs regularly with `tail -f output/*/training.log`
- Validate checkpoints are being saved

---

**Still having issues?** Check the comprehensive troubleshooting guide: [../reference/troubleshooting.md](../reference/troubleshooting.md)
