# First Training - Complete Setup

**Step-by-step guide to your first successful training run**

## Prerequisites

- ✅ Environment setup complete ([setup.md](setup.md))
- ✅ Quick start overview done ([README.md](README.md))
- ✅ Raw data available in `ds_v2/` directory

## Step 1: Data Processing (5-10 minutes)

### Process Raw Data
```bash
# Navigate to project root
cd /data3/Qwen2.5-VL-main

# Run the complete data processing pipeline
bash data_conversion/convert_dataset.sh

# Monitor progress (in another terminal)
tail -f data_conversion/pipeline.log
```

### Verify Data Processing
```bash
# Check output files
ls -la data/
# Expected: train.jsonl, val.jsonl, teacher.jsonl

# Check file sizes (should be substantial)
wc -l data/*.jsonl
# Expected: thousands of lines each

# Inspect data format
head -1 data/train.jsonl | /root/miniconda3/envs/ms/bin/python -m json.tool
```

**Expected data format**:
```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"
    }
  ],
  "width": 532,
  "height": 728
}
```

### Customize Data Processing (Optional)
```bash
# For specific object types only
export OBJECT_TYPES="bbu label"  # Equipment + text recognition
bash data_conversion/convert_dataset.sh

# For cable system training
export OBJECT_TYPES="fiber wire"  # Fiber + wire cables
bash data_conversion/convert_dataset.sh

# For complete system
export OBJECT_TYPES="full"  # All object types
bash data_conversion/convert_dataset.sh
```

## Step 2: Configuration Setup (2-3 minutes)

### Create Training Configuration
```bash
# Copy base configuration
cp configs/base_flat_v2.yaml configs/first_training.yaml

# Update key settings for first training
cat > configs/first_training.yaml << 'EOF'
# BBU Detection Training Configuration
model_path: "/path/to/qwen2.5-vl-7b-instruct"  # UPDATE THIS PATH
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"

# Detection and Token Settings
detection_enabled: true
coordinate_tokens_enabled: true
max_coord_value: 2048

# Training Settings (Conservative for first run)
learning_rate: 1e-5
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
warmup_ratio: 0.1
lr_scheduler_type: "cosine"

# Teacher-Student Learning
teacher_ratio: 0.3
teacher_loss_weight: 0.3
student_loss_weight: 1.0

# Model Settings
model_max_length: 120000
attn_implementation: "flash_attention_2"
torch_dtype: "bfloat16"

# Output and Logging
output_dir: "./output/first_training"
logging_steps: 10
save_steps: 500
eval_steps: 500
save_total_limit: 3

# Memory Optimization
gradient_checkpointing: true
dataloader_num_workers: 4
remove_unused_columns: false
EOF
```

### Update Model Path
```bash
# Find your model path (update as needed)
MODEL_PATH="/data3/models/qwen2.5-vl-7b-instruct"  # Example path

# Update configuration
sed -i "s|model_path: \".*\"|model_path: \"$MODEL_PATH\"|" configs/first_training.yaml

# Verify the change
grep "model_path:" configs/first_training.yaml
```

### Validate Configuration
```bash
# Test configuration loading
/root/miniconda3/envs/ms/bin/python -c "
import yaml
with open('configs/first_training.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Configuration syntax OK')
print(f'Model path: {config[\"model_path\"]}')
print(f'Batch size: {config[\"per_device_train_batch_size\"]}')
print(f'Learning rate: {config[\"learning_rate\"]}')
"
```

## Step 3: Pre-Training Validation (2 minutes)

### System Health Check
```bash
# Test all core components
/root/miniconda3/envs/ms/bin/python -c "
# Test imports
from src.training.trainer import BBUTrainer
from src.training.training_coordinator import TrainingCoordinator
from src.models.model_loader import load_model_and_processor_unified
from src.config import get_config
print('✅ All imports successful')

# Test configuration
import yaml
with open('configs/first_training.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Configuration loaded')

# Test data files
import os
assert os.path.exists('data/train.jsonl'), 'train.jsonl not found'
assert os.path.exists('data/val.jsonl'), 'val.jsonl not found'
print('✅ Data files exist')

print('🎉 Pre-training validation passed!')
"
```

### Memory Estimation
```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Estimate memory requirements
echo "Memory estimation for batch_size=1:"
echo "- Model: ~14GB (Qwen2.5-VL-7B)"
echo "- Gradients: ~14GB"
echo "- Optimizer: ~28GB (AdamW)"
echo "- Activations: ~4-8GB"
echo "- Total: ~60-64GB"
echo ""
echo "Recommended: 4x A100 (40GB each) or 2x A100 (80GB each)"
```

## Step 4: Start Training (30+ minutes)

### Launch Training
```bash
# Create output directory
mkdir -p output/first_training

# Start training with full logging
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/first_training.yaml \
    2>&1 | tee output/first_training/training.log

# Alternative: Run in background
nohup /root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/first_training.yaml \
    > output/first_training/training.log 2>&1 &
```

### Monitor Training Progress
```bash
# Watch training logs
tail -f output/first_training/training.log

# Check GPU utilization
watch -n 1 nvidia-smi

# Monitor training metrics (if using wandb)
# Check wandb dashboard or logs
```

### Expected Training Output
```
🏭 Creating trainer with new coordinator system...
📄 Using unified configuration system
✅ Model loading with patches applied
✅ Training system OK
✅ Data pipeline OK
✅ Configuration OK

🎯 Training started...
Step 10: loss=2.345, llm_loss=1.234, coordinate_l1_loss=0.567
Step 20: loss=2.123, llm_loss=1.098, coordinate_l1_loss=0.432
...
```

## Step 5: Monitor and Validate (During Training)

### Key Metrics to Watch
1. **Total Loss**: Should decrease steadily
2. **LLM Loss**: Language modeling component
3. **Coordinate L1 Loss**: Coordinate prediction accuracy
4. **GPU Memory**: Should be stable (not increasing)
5. **Training Speed**: Steps per second

### Health Checks During Training
```bash
# Check if training is progressing
grep "Step" output/first_training/training.log | tail -5

# Check for errors
grep -i "error\|exception\|failed" output/first_training/training.log

# Check GPU memory usage
nvidia-smi | grep -A 1 "GPU Memory Usage"

# Check checkpoint creation
ls -la output/first_training/checkpoint-*
```

### Common Issues and Solutions

#### CUDA Out of Memory
```bash
# Reduce batch size
sed -i 's/per_device_train_batch_size: [0-9]*/per_device_train_batch_size: 1/' configs/first_training.yaml

# Increase gradient accumulation
sed -i 's/gradient_accumulation_steps: [0-9]*/gradient_accumulation_steps: 16/' configs/first_training.yaml

# Enable gradient checkpointing
sed -i 's/gradient_checkpointing: false/gradient_checkpointing: true/' configs/first_training.yaml
```

#### Training Too Slow
```bash
# Reduce sequence length
sed -i 's/model_max_length: [0-9]*/model_max_length: 60000/' configs/first_training.yaml

# Increase batch size (if memory allows)
sed -i 's/per_device_train_batch_size: [0-9]*/per_device_train_batch_size: 2/' configs/first_training.yaml
```

#### Loss Not Decreasing
```bash
# Check learning rate
grep "learning_rate:" configs/first_training.yaml

# Check data quality
head -5 data/train.jsonl | /root/miniconda3/envs/ms/bin/python -m json.tool

# Check coordinate token setup
grep -i "coordinate" output/first_training/training.log
```

## Step 6: Evaluate Results (After Training)

### Check Training Completion
```bash
# Look for completion message
tail -20 output/first_training/training.log | grep -i "training completed\|finished"

# Check final checkpoints
ls -la output/first_training/checkpoint-*

# Check final metrics
grep "eval_loss" output/first_training/training.log | tail -5
```

### Quick Inference Test
```bash
# Test the trained model
/root/miniconda3/envs/ms/bin/python src/inference.py \
    --model_path output/first_training/checkpoint-1000 \
    --image_path "path/to/test/image.jpg" \
    --output_file test_results.json

# Check inference results
cat test_results.json | /root/miniconda3/envs/ms/bin/python -m json.tool
```

## Success Criteria

Your first training is successful if:
- ✅ Training completed without crashes
- ✅ Loss decreased over time
- ✅ Checkpoints were saved
- ✅ Model can run inference
- ✅ Coordinate tokens are generated in outputs

## Next Steps

### If Training Succeeded:
1. **Experiment with hyperparameters**: Adjust learning rate, batch size
2. **Try different object types**: Use `OBJECT_TYPES` for specialized training
3. **Longer training**: Increase `num_train_epochs` for better results
4. **Advanced features**: Explore [../components/](../components/) documentation

### If Training Failed:
1. **Check common issues**: [common-issues.md](common-issues.md)
2. **Full troubleshooting**: [../reference/troubleshooting.md](../reference/troubleshooting.md)
3. **Component debugging**: [../components/](../components/)

### Understanding the System:
1. **Architecture**: [../MENTAL_MODEL.md](../MENTAL_MODEL.md)
2. **Components**: [../components/](../components/)
3. **Workflows**: [../workflows/](../workflows/)

---

**🎉 Congratulations!** You've completed your first training run with the BBU Detection System.

**Questions?** Check [common-issues.md](common-issues.md) or [../reference/troubleshooting.md](../reference/troubleshooting.md)
