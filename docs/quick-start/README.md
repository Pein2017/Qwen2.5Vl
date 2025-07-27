# Quick Start - 5 Minute Overview

**Get up and running with the BBU Detection System in 5 minutes**

## What You'll Accomplish

By the end of this guide, you'll have:
- ✅ Environment set up and validated
- ✅ Data processed and ready for training
- ✅ First training run started
- ✅ Understanding of the system basics

## Prerequisites (30 seconds)

- Linux environment with CUDA GPUs
- Located in China (uses local mirrors)
- Access to `/data3/Qwen2.5-VL-main` directory

## Step 1: Environment Setup (1 minute)

```bash
# Navigate to project
cd /data3/Qwen2.5-VL-main

# Use full Python path (avoid conda activation issues)
/root/miniconda3/envs/ms/bin/python --version  # Should be Python 3.10+

# Verify CUDA
/root/miniconda3/envs/ms/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Quick health check
/root/miniconda3/envs/ms/bin/python -c "from src.training.trainer import BBUTrainer; print('✅ System OK')"
```

**If any step fails**: See [setup.md](setup.md) for detailed troubleshooting.

## Step 2: Process Data (2 minutes)

```bash
# Process your raw data (assumes data in ds_v2/ directory)
bash data_conversion/convert_dataset.sh

# Verify output
ls -la data/  # Should see train.jsonl, val.jsonl, teacher.jsonl
head -1 data/train.jsonl | /root/miniconda3/envs/ms/bin/python -m json.tool  # Check format
```

**Expected output**: 3 JSONL files with coordinate token conversations.

## Step 3: Configure Training (1 minute)

```bash
# Copy working configuration
cp configs/base_flat_v2.yaml configs/my_training.yaml

# Verify key settings (should already be correct):
grep -E "(detection_enabled|coordinate_tokens_enabled|model_path)" configs/my_training.yaml
```

**Key settings to verify**:
- `detection_enabled: true`
- `coordinate_tokens_enabled: true`
- `model_path: "/path/to/qwen2.5-vl-7b-instruct"`

## Step 4: Start Training (1 minute)

```bash
# Start training with the modular system
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/my_training.yaml \
    --output_dir ./output/quick_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --logging_steps 10

# You should see logs like:
# ✅ Training system OK
# ✅ Model loading OK  
# ✅ Data pipeline OK
# 🏋️ Training started...
```

## What Just Happened?

### Data Processing
- **PipelineManager** ran 5 stages: clean → map → process → validate → summarize
- **UnifiedProcessor** converted raw BBU annotations to coordinate token format
- **Object-oriented training** filtered for specific equipment types

### Training System
- **ModelLoader** loaded Qwen2.5-VL with coordinate token support
- **BBUTrainer** started with multi-component loss tracking
- **TrainingCoordinator** orchestrated the training process

### Key Innovation
Your model is learning to predict coordinates as text tokens:
```
Input: Image of BBU equipment
Output: "BBU设备: <coord_264><coord_144><coord_326><coord_201> 显示完整,符合要求"
```

## Next Steps

### If everything worked:
- **Continue training**: See [first-training.md](first-training.md) for full training setup
- **Understand the system**: Read [../MENTAL_MODEL.md](../MENTAL_MODEL.md)
- **Customize training**: Check [../components/](../components/)

### If something failed:
- **Common issues**: See [common-issues.md](common-issues.md)
- **Environment problems**: See [setup.md](setup.md)
- **Full troubleshooting**: See [../reference/troubleshooting.md](../reference/troubleshooting.md)

## Quick Reference

### Health Checks
```bash
# System components
/root/miniconda3/envs/ms/bin/python -c "from src.training.trainer import BBUTrainer; print('✅ Training OK')"

# Data pipeline
ls data/ | grep -E "(train|val).jsonl" && echo "✅ Data OK"

# Configuration
/root/miniconda3/envs/ms/bin/python -c "from src.config import get_config; print('✅ Config OK')"
```

### Key Commands
```bash
# Process data
bash data_conversion/convert_dataset.sh

# Train model  
/root/miniconda3/envs/ms/bin/python scripts/train.py --config configs/my_training.yaml

# Check training progress
tail -f output/quick_test/run.log
```

### Key Files
- `data_conversion/convert_dataset.sh` - Data processing
- `scripts/train.py` - Training entry point
- `configs/base_flat_v2.yaml` - Configuration template
- `src/training/trainer.py` - Enhanced trainer

---

**🎉 Congratulations!** You now have a basic understanding of the BBU Detection System. 

**What's next?** Choose your path:
- 👨‍💻 **Developer**: [first-training.md](first-training.md) → [../components/](../components/)
- 🔬 **Researcher**: [../MENTAL_MODEL.md](../MENTAL_MODEL.md) → [../ARCHITECTURE.md](../ARCHITECTURE.md)
- 🚨 **Troubleshooter**: [common-issues.md](common-issues.md) → [../reference/troubleshooting.md](../reference/troubleshooting.md)
