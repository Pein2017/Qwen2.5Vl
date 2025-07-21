# BBU Detection System - Quick Start (15 Minutes)

Get up and running with the BBU detection system in 15 minutes.

## 🎯 What You'll Build

A vision-language model that detects BBU equipment in images and generates natural language descriptions with precise coordinates.

**Key Innovation**: Uses "coordinate tokens" to predict bounding boxes as part of the text sequence instead of separate regression heads.

## ⚡ Prerequisites (2 minutes)

```bash
# Check environment
cd /data3/Qwen2.5-VL-main
/root/miniconda3/envs/ms/bin/python --version  # Should be Python 3.10+

# Verify CUDA
/root/miniconda3/envs/ms/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Step 1: Process Data (5 minutes)

```bash
# Process your raw data (assumes data in ds/ directory)
bash data_conversion/convert_dataset.sh

# Verify output
ls -la data/  # Should see train.jsonl, val.jsonl
head -1 data/train.jsonl | python -m json.tool  # Check format
```

## 🤖 Step 2: Configure Training (3 minutes)

```bash
# Copy working configuration
cp configs/base_flat_v2.yaml configs/my_training.yaml

# Key settings to verify:
# - coordinate_tokens_enabled: true
# - model_path: "/path/to/qwen2.5-vl-7b-instruct"
# - train_data_path: "data/train.jsonl"
```

## 🏃 Step 3: Start Training (5 minutes)

```bash
# Quick training test
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/my_training.yaml \
    --output_dir checkpoints/quick_test \
    --max_train_samples 20 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1

# Monitor progress
tail -f checkpoints/quick_test/training.log
```

## ✅ Verify Success

Look for these indicators:
- `coordinate_loss` appears in logs and decreases
- No CUDA out of memory errors
- Checkpoint files created in output directory

## 🎓 Next Steps

**For Production Training**: See [Developer Guide](guides/developer-guide.md)
**For Understanding the System**: See [Architecture](core/architecture.md)
**When Things Break**: See [Troubleshooting](guides/troubleshooting.md)

## 🆘 Quick Fixes

**CUDA out of memory**: Add `--per_device_train_batch_size 1`
**Data not found**: Check paths in config file
**Import errors**: Verify Python environment with `/root/miniconda3/envs/ms/bin/python`

---
**🎉 Congratulations!** You now have a working BBU detection system. Time to explore the full capabilities!