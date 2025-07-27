# New Developer Onboarding (30 Minutes to Productive)

Fast-track guide to get new developers productive with the BBU detection system in 30 minutes.

## 🎯 Goal: Train a Working Model in 30 Minutes

**What you'll achieve:**
- Understand the system architecture
- Set up the environment 
- Process data and start training
- Understand how to debug issues

---

## ⏱️ Phase 1: Quick Start (5 minutes)

### 1. Environment Setup
```bash
# Navigate to project
cd /data3/Qwen2.5-VL-main

# Check environment
/root/miniconda3/envs/ms/bin/python --version
# Should show Python 3.10.x

# Quick system check
/root/miniconda3/envs/ms/bin/python -c "
import torch, transformers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}') 
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### 2. Understand What You're Building
**This system trains Qwen2.5-VL to:**
- Detect BBU equipment in images
- Generate natural language descriptions
- Output structured coordinates using "coordinate tokens"

**Key Innovation:** Coordinate tokens convert bbox coordinates like `[10, 20, 100, 200]` into special tokens `<coord_10><coord_20><coord_100><coord_200>` that the model learns to predict.

---

## ⏱️ Phase 2: System Overview (10 minutes)

### Core Architecture (2 minutes)
```
Raw Data → Data Processing → Model Training → Trained Model
    ↓           ↓              ↓               ↓
  Images +   Coordinate    Enhanced        BBU Detection
  JSON       Token         Qwen2.5-VL     + Description
            Conversion
```

### Key Components (8 minutes)
**Read these in order for maximum understanding:**

1. **📚 [API Quick Reference](../quick-reference/api-core-components.md)** (3 min)
   - Essential APIs you'll use daily
   - Copy-paste code examples

2. **⚙️ [Configuration Templates](../quick-reference/config-templates.md)** (3 min)
   - Ready-to-use training configs
   - Common scenarios covered

3. **🚨 [Problem-Solution Lookup](../quick-reference/problem-solution-lookup.md)** (2 min)
   - Bookmark this for when things break
   - Symptom → solution format

---

## ⏱️ Phase 3: Hands-On Training (10 minutes)

### 1. Data Processing (3 minutes)
```bash
# Process your data (assumes raw data in ds/ directory)
bash data_conversion/convert_dataset.sh

# Verify output
ls -la data/
# Should see: train.jsonl, val.jsonl
# Should see: ds_output/ with images

# Quick data check
head -1 data/train.jsonl | python -m json.tool
```

### 2. Configure Training (2 minutes)
```bash
# Copy template config
cp docs/quick-reference/config-templates.md configs/my_first_training.yaml

# Edit the config (change paths to match your setup)
nano configs/my_first_training.yaml

# Key settings to verify:
# - model_path: "/path/to/qwen2.5-vl-7b-instruct"
# - train_data_path: "data/train.jsonl"
# - val_data_path: "data/val.jsonl"
```

### 3. Start Training (5 minutes)
```bash
# Start training with debug config for fast iteration
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/my_first_training.yaml \
    --output_dir checkpoints/my_first_run \
    --coordinate_tokens_enabled true \
    --max_train_samples 50 \
    --max_eval_samples 20 \
    --num_train_epochs 1 \
    --eval_steps 5 \
    --save_steps 5

# Monitor training
tail -f checkpoints/my_first_run/training.log
```

---

## ⏱️ Phase 4: Validation & Next Steps (5 minutes)

### 1. Verify Training Works (2 minutes)
```bash
# Check if training produced checkpoints
ls -la checkpoints/my_first_run/

# Look for coordinate loss in logs
grep "coordinate_loss" checkpoints/my_first_run/training.log

# Should see coordinate_loss decreasing over steps
```

### 2. Quick Model Test (2 minutes)
```bash
# Test model loading
/root/miniconda3/envs/ms/bin/python -c "
from src.core.checkpoint_manager import CheckpointManager
manager = CheckpointManager()
checkpoint_path = 'checkpoints/my_first_run/checkpoint-5'  # Adjust number
print('Checkpoint valid:', manager.validate_checkpoint(checkpoint_path))
"
```

### 3. Next Steps (1 minute)
Now you're ready for real training! Check these guides:

**Immediate Next Steps:**
- 📖 [Full training guide](../runbook.md) - Production training
- 🏗️ [Architecture deep dive](../architecture.md) - Understand the system  
- 🔧 [Advanced configuration](../configuration.md) - Optimize your training

**When Things Break:**
- 🚨 [Problem-solution lookup](../quick-reference/problem-solution-lookup.md)
- 🛠️ [Critical fixes](../critical_fixes.md)
- 📞 [Troubleshooting guide](../troubleshooting.md)

---

## 🎓 What You Just Learned

In 30 minutes, you:
✅ Set up the environment and verified it works  
✅ Understood the coordinate token innovation  
✅ Processed data into training format  
✅ Configured and started training  
✅ Verified the system is working  
✅ Know where to go for advanced topics  

## 🚀 Graduation Test

Can you answer these questions?

1. **What makes this system special?**
   - Answer: Coordinate tokens that convert bbox coordinates to learnable tokens

2. **What's the main training script?**
   - Answer: `scripts/train.py`

3. **Where do you find quick solutions to problems?**
   - Answer: `docs/quick-reference/problem-solution-lookup.md`

4. **What's the key config for coordinate token training?**
   - Answer: `coordinate_tokens_enabled: true`

5. **How do you resume training from a checkpoint?**
   - Answer: Add `--resume_from_checkpoint path/to/checkpoint-N`

**If you can answer these, you're ready for production development!**

---

## 📚 Reference Card (Bookmark This)

```bash
# Essential Commands
# Start training:
/root/miniconda3/envs/ms/bin/python scripts/train.py --config configs/my_config.yaml

# Process data:
bash data_conversion/convert_dataset.sh

# Monitor training:
tail -f checkpoints/run_name/training.log

# Check GPU:
nvidia-smi

# Quick system check:
/root/miniconda3/envs/ms/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Key Files to Bookmark:**
- `docs/quick-reference/` - All quick references
- `docs/architecture.md` - System understanding
- `docs/critical_fixes.md` - When things break
- `CLAUDE.md` - Project overview

**Configuration Essentials:**
- `coordinate_tokens_enabled: true` - Enable coordinate system
- `model_path: "/path/to/qwen2.5-vl"` - Base model
- `train_data_path: "data/train.jsonl"` - Training data

---
**🎉 Congratulations! You're now a productive BBU detection system developer.**

Next stop: [Advanced Developer Journey](advanced-developer-deepdive.md) for production-grade development.