# Troubleshooter's Quick Start (Problem → Solution <5 Minutes)

Emergency guide for when training is broken and you need answers fast.

## 🚨 Emergency Triage (30 seconds)

**What's broken?** Pick the category that matches your problem:

1. **🔥 [Training Crashed](#training-crashed)** - Training stopped with errors
2. **💾 [Out of Memory](#out-of-memory)** - CUDA/memory errors  
3. **📊 [Bad Loss/Metrics](#bad-lossmetrics)** - Loss not decreasing, poor performance
4. **📁 [Data Issues](#data-issues)** - Can't load data, missing files
5. **⚙️ [Config Problems](#config-problems)** - Configuration errors
6. **🔧 [Environment Issues](#environment-issues)** - Import errors, package issues
7. **🤖 [Model Loading](#model-loading)** - Model won't load or save

---

## 🔥 Training Crashed

### Symptoms → Quick Fixes

**"CUDA out of memory"**
```bash
# Immediate fix - reduce batch size
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/your_config.yaml \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

**"Training script crashed with KeyError"**
```bash
# Check data format
head -1 data/train.jsonl | python -m json.tool

# Common fix - missing required fields
# Ensure your JSONL has: image, conversations
```

**"Checkpoint loading failed"**
```bash
# Find latest valid checkpoint
find checkpoints/ -name "checkpoint-*" -type d | sort -V | tail -3

# Resume from earlier checkpoint
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/your_config.yaml \
    --resume_from_checkpoint checkpoints/run/checkpoint-500  # Earlier checkpoint
```

**"Loss became NaN"**
```bash
# Immediate fix - lower learning rate
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/your_config.yaml \
    --learning_rate 1e-6 \
    --coordinate_lr 1e-5
```

---

## 💾 Out of Memory

### Quick Memory Fixes (Choose one that fits)

**GPU Memory Issues**
```bash
# Option 1: Smallest batch size
--per_device_train_batch_size 1 --gradient_accumulation_steps 16

# Option 2: Shorter sequences  
--model_max_length 4096

# Option 3: Lower precision
--torch_dtype "float16" --attn_implementation "eager"

# Option 4: Check available GPUs
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

**System Memory Issues**  
```bash
# Reduce data workers
--dataloader_num_workers 2

# Limit data samples for testing
--max_train_samples 100 --max_eval_samples 50
```

### Memory Diagnostic
```bash
# Check current usage
/root/miniconda3/envs/ms/bin/python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print(f'Currently allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB')
else:
    print('No CUDA available')
"
```

---

## 📊 Bad Loss/Metrics

### Quick Diagnosis & Fixes

**Loss not decreasing**
```bash
# Check if learning rate too low
grep "learning_rate" configs/your_config.yaml

# Quick fix - increase learning rate
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/your_config.yaml \
    --learning_rate 1e-4 \
    --coordinate_lr 1e-3
```

**coordinate_loss is 0**
```bash
# Check if coordinate tokens enabled
grep "coordinate_tokens_enabled" configs/your_config.yaml

# Verify data has bbox info
head -1 data/train.jsonl | python -c "
import json, sys
data = json.load(sys.stdin)
print('Has bbox_2d:', any('bbox_2d' in str(data) for _ in [data]))
"

# Enable coordinate tokens
--coordinate_tokens_enabled true
```

**Training unstable (loss jumping)**
```bash
# Add gradient clipping
--max_grad_norm 1.0

# Reduce learning rate
--learning_rate 1e-6

# Add more warmup
--warmup_ratio 0.2
```

**Model not learning coordinate predictions**
```bash
# Increase coordinate learning rate
--coordinate_lr 1e-3

# Increase coordinate loss weight
--coordinate_loss_weight 2.0

# Check coordinate token detection
/root/miniconda3/envs/ms/bin/python -c "
from src.utils.coordinate_token_manager import create_coordinate_token_manager
# Test coordinate token detection
"
```

---

## 📁 Data Issues

### Quick Data Fixes

**"FileNotFoundError: image not found"**
```bash
# Check image paths in data
head -1 data/train.jsonl | python -c "
import json, sys, os
data = json.load(sys.stdin)
print(f'Image path: {data[\"image\"]}')
print(f'Exists: {os.path.exists(data[\"image\"])}')
"

# Common fix - wrong relative paths
# Re-run data conversion:
bash data_conversion/convert_dataset.sh
```

**"JSONL parsing error"**
```bash
# Validate JSONL format
python -c "
import json
with open('data/train.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except Exception as e:
            print(f'Error on line {i+1}: {e}')
            break
    else:
        print('JSONL format OK')
"
```

**"'list' object has no attribute 'get'"**
```bash
# Clean raw JSON first
cd data_conversion
/root/miniconda3/envs/ms/bin/python clean_raw_json.py --input_dir ../ds --output_dir ../ds_output
```

**"Empty dataset after processing"**
```bash
# Check data paths in config
grep -E "(train_data_path|val_data_path)" configs/your_config.yaml

# Verify files exist
ls -la data/train.jsonl data/val.jsonl

# Check file content
wc -l data/train.jsonl data/val.jsonl
```

---

## ⚙️ Config Problems

### Config Quick Fixes

**"Config validation failed"**
```bash
# Validate config
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

**"Model path not found"**
```bash
# Check model path
ls -la /path/to/your/model/config.json

# Use correct path in config:
# model_path: "/data3/Qwen2.5-VL-main/model_cache/qwen2.5-vl-7b-instruct"
```

**"Coordinate config errors"**
```bash
# Use working coordinate config:
cat > temp_coord_config.yaml << EOF
coordinate_tokens_enabled: true
coordinate_config_max_coord_value: 2048
coordinate_lr: 1e-4
coordinate_loss_weight: 1.0
regular_loss_weight: 1.0
EOF

# Test with minimal config
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config temp_coord_config.yaml \
    --max_train_samples 10
```

---

## 🔧 Environment Issues

### Environment Quick Fixes

**"ModuleNotFoundError: src"**
```bash
# Fix Python path
export PYTHONPATH=/data3/Qwen2.5-VL-main:$PYTHONPATH

# Or add in script:
/root/miniconda3/envs/ms/bin/python -c "
import sys
sys.path.append('/data3/Qwen2.5-VL-main')
import src.core.model_factory  # Should work now
"
```

**"ImportError: transformers"**
```bash
# Use correct Python
/root/miniconda3/envs/ms/bin/python --version

# Check packages
/root/miniconda3/envs/ms/bin/python -c "
import torch, transformers, datasets
print('All packages OK')
"
```

**"CUDA not available"**
```bash
# Check CUDA setup
nvidia-smi
export CUDA_VISIBLE_DEVICES=0

# Test CUDA
/root/miniconda3/envs/ms/bin/python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
"
```

---

## 🤖 Model Loading

### Model Loading Quick Fixes

**"Flash Attention not available"**
```bash
# Fallback to eager attention
--attn_implementation "eager"

# Or install flash attention
pip install flash-attn --no-build-isolation
```

**"Vocabulary size mismatch"**
```bash
# Clear tokenizer cache
rm -rf ~/.cache/huggingface/transformers/

# Force tokenizer recreation
/root/miniconda3/envs/ms/bin/python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('model_path', force_download=True)
"
```

**"Model loading timeout"**
```bash
# Check disk space
df -h

# Check model files
ls -la /path/to/model/

# Use local model path
# Avoid network downloads during training
```

---

## 🎯 5-Minute Complete Diagnostic

Run this when you're not sure what's wrong:

```bash
#!/bin/bash
echo "=== BBU Training System Diagnostic ==="

echo "1. Environment Check:"
/root/miniconda3/envs/ms/bin/python --version
/root/miniconda3/envs/ms/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo "2. Data Check:"
ls -la data/ | head -5
wc -l data/*.jsonl

echo "3. Config Check:"
grep -E "(model_path|coordinate_tokens_enabled|learning_rate)" configs/*.yaml | head -3

echo "4. GPU Check:"
nvidia-smi | grep -E "(python|MiB)"

echo "5. Recent Training Check:"
ls -la checkpoints/ | tail -3

echo "6. Process Check:"
ps aux | grep python | grep train.py

echo "=== Diagnostic Complete ==="
```

---

## 📞 Emergency Escalation

**If this guide doesn't solve your problem:**

1. **Check the comprehensive guides:**
   - [Critical Fixes Database](../critical_fixes.md) - Known issues & solutions
   - [Troubleshooting Guide](../troubleshooting.md) - Detailed debugging
   - [Configuration Guide](../configuration.md) - Config deep dive

2. **Create a minimal reproduction:**
```bash
# Minimal test run
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/minimal.yaml \
    --max_train_samples 5 \
    --max_eval_samples 2 \
    --num_train_epochs 1 \
    --output_dir debug_run
```

3. **Gather diagnostic info:**
```bash
# Save full diagnostic
bash diagnostic_script.sh > debug_info.txt 2>&1
```

4. **Check architecture documentation:**
   - [System Architecture](../architecture.md) - Understanding the system
   - [Implementation Summary](../implementation_summary.md) - Current state

---

**⚡ Remember:** 90% of issues are solved by:
1. Reducing batch size (memory issues)
2. Checking data paths (data issues)  
3. Using correct Python environment (import issues)
4. Validating configuration (config issues)

**🎯 Target:** Problem identified and fix applied within 5 minutes.