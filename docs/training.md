# BBU Training Guide

**Complete guide for training the BBU pipeline with coordinate token system**

## 🚀 **Quick Start Training**

### **Standard Mode Training (Recommended)**
```bash
# Start training with integer coordinates
python scripts/train.py --config configs/bbu_v2.yaml

# Monitor progress
tail -f checkpoints/run_*/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### **Coordinate Mode Training (Advanced)**
```bash
# Start training with coordinate tokens
python scripts/train.py --config configs/bbu_coordinate.yaml

# Monitor coordinate-specific metrics
grep "coordinate_loss" checkpoints/run_*/training.log | tail -10
```

## 📊 **Training Workflows**

### **1. Data Preparation**
```bash
# Process raw data
bash data_conversion/convert_dataset.sh

# Verify data format
head -1 data/train.jsonl | python -m json.tool
wc -l data/*.jsonl  # Check file sizes

# Expected output:
# data/train.jsonl    - Training samples
# data/val.jsonl      - Validation samples  
# data/teacher.jsonl  - Teacher forcing samples
```

### **2. Configuration Setup**
```bash
# Copy and customize configuration
cp configs/bbu_v2.yaml configs/my_training.yaml

# Edit configuration
vim configs/my_training.yaml

# Validate configuration
python -c "from src.config import load_config; load_config('configs/my_training.yaml'); print('✅ Config OK')"
```

### **3. Training Execution**
```bash
# Full training run
python scripts/train.py --config configs/my_training.yaml

# Training with custom parameters
python scripts/train.py \
    --config configs/my_training.yaml \
    --output_dir checkpoints/experiment_001 \
    --max_steps 2000 \
    --per_device_train_batch_size 2

# Resume from checkpoint
python scripts/train.py \
    --config configs/my_training.yaml \
    --resume_from_checkpoint checkpoints/run_001/checkpoint-500
```

## 📈 **Monitoring Training**

### **Training Logs**
```bash
# Real-time monitoring
tail -f checkpoints/run_*/training.log

# Extract specific metrics
grep "loss=" checkpoints/run_*/training.log | tail -20
grep "eval_loss" checkpoints/run_*/training.log

# Loss component analysis
grep "lm_loss" checkpoints/run_*/training.log | tail -10
grep "coordinate_loss" checkpoints/run_*/training.log | tail -10
```

### **Key Metrics to Monitor**

#### **Standard Mode Metrics**
```
Step 100: loss=2.345, lm_loss=2.345, eval_loss=2.123
```
- **`loss`**: Combined training loss
- **`lm_loss`**: Language modeling loss
- **`eval_loss`**: Validation loss

#### **Coordinate Mode Metrics**
```
Step 100: loss=2.345, lm_loss=1.890, coordinate_loss=0.455, eval_loss=2.123
```
- **`loss`**: Combined training loss
- **`lm_loss`**: Language modeling loss
- **`coordinate_loss`**: Coordinate prediction loss
- **`eval_loss`**: Validation loss

### **Performance Monitoring**
```bash
# GPU utilization
watch -n 1 nvidia-smi

# System resources
htop

# Training speed
grep "train_samples_per_second" checkpoints/run_*/training.log | tail -5
```

## 🔧 **Training Optimization**

### **Memory Optimization**
```yaml
# For limited GPU memory
per_device_train_batch_size: 2        # Reduce batch size
gradient_accumulation_steps: 2        # Maintain effective batch size
gradient_checkpointing: true          # Enable gradient checkpointing
fp16: true                           # Mixed precision training
```

### **Speed Optimization**
```yaml
# For faster training
dataloader_num_workers: 8            # More data loading workers
dataloader_pin_memory: true          # Pin memory for faster transfer
fp16: true                           # Mixed precision
remove_unused_columns: false         # Required but affects speed
```

### **Multi-GPU Training**
```bash
# Single node, multiple GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/my_training.yaml

# Check GPU usage across devices
nvidia-smi
```

## 🎯 **Training Strategies**

### **Progressive Training**
```bash
# Stage 1: Quick validation (100 steps)
python scripts/train.py --config configs/my_training.yaml --max_steps 100

# Stage 2: Medium run (500 steps)
python scripts/train.py --config configs/my_training.yaml --max_steps 500

# Stage 3: Full training (2000+ steps)
python scripts/train.py --config configs/my_training.yaml --max_steps 2000
```

### **Hyperparameter Tuning**
```bash
# Learning rate sweep
for lr in 1e-5 5e-6 2e-5; do
    python scripts/train.py \
        --config configs/my_training.yaml \
        --learning_rate $lr \
        --output_dir checkpoints/lr_${lr} \
        --max_steps 500
done

# Batch size sweep
for bs in 2 4 8; do
    python scripts/train.py \
        --config configs/my_training.yaml \
        --per_device_train_batch_size $bs \
        --output_dir checkpoints/bs_${bs} \
        --max_steps 500
done
```

### **Coordinate Mode Tuning**
```bash
# Coordinate loss weight sweep
for weight in 0.01 0.05 0.1; do
    python scripts/train.py \
        --config configs/bbu_coordinate.yaml \
        --coordinate_loss_weight $weight \
        --output_dir checkpoints/coord_weight_${weight} \
        --max_steps 500
done
```

## 🧪 **Training Validation**

### **Quick Validation Tests**
```bash
# Test training setup
python scripts/train.py --config configs/my_training.yaml --max_steps 5

# Test both modes
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 5
python scripts/train.py --config configs/bbu_coordinate.yaml --max_steps 5

# Validate data loading
python -c "
from src.core.data_processor import DataProcessor
from src.config import load_config
config = load_config('configs/my_training.yaml')
processor = DataProcessor(None, None, None, config=config)
train_dataset, eval_dataset = processor.create_datasets()
print(f'Train samples: {len(train_dataset)}')
print(f'Eval samples: {len(eval_dataset)}')
"
```

### **Model Validation**
```python
# Test model loading and forward pass
from src.models.model_loader import load_model_and_processor_unified
from src.config import load_config

config = load_config('configs/my_training.yaml')
model, tokenizer, processor = load_model_and_processor_unified(
    config.model_path, for_inference=False
)

print(f"Model loaded: {model.__class__.__name__}")
print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
print(f"Coordinate tokens enabled: {config.coordinate_tokens_enabled}")
```

## 📁 **Output Management**

### **Checkpoint Structure**
```
checkpoints/run_001/
├── checkpoint-100/              # Model checkpoint at step 100
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer.pt
│   └── scheduler.pt
├── checkpoint-200/              # Model checkpoint at step 200
├── training.log                 # Training logs
├── config.yaml                  # Training configuration
└── trainer_state.json          # Training state
```

### **Checkpoint Management**
```bash
# List checkpoints
ls -la checkpoints/run_001/checkpoint-*/

# Copy best checkpoint
cp -r checkpoints/run_001/checkpoint-1000 final_models/best_model

# Clean up old checkpoints (keep last 3)
find checkpoints/run_001 -name "checkpoint-*" -type d | head -n -3 | xargs rm -rf

# Check checkpoint sizes
du -sh checkpoints/run_001/checkpoint-*
```

## 🔍 **Troubleshooting Training Issues**

### **Common Training Problems**

#### **Training Stops/Crashes**
```bash
# Check logs for errors
grep -i "error\|exception\|failed" checkpoints/run_*/training.log

# Check GPU memory
nvidia-smi

# Restart with smaller batch size
python scripts/train.py --config configs/my_training.yaml --per_device_train_batch_size 2
```

#### **Slow Training**
```bash
# Check data loading bottleneck
grep "train_samples_per_second" checkpoints/run_*/training.log

# Increase data workers
# In config: dataloader_num_workers: 8

# Enable mixed precision
# In config: fp16: true
```

#### **Loss Not Decreasing**
```bash
# Check learning rate
grep "learning_rate" checkpoints/run_*/training.log

# Try different learning rate
python scripts/train.py --config configs/my_training.yaml --learning_rate 5e-6

# Check data quality
head -5 data/train.jsonl | python -m json.tool
```

## 📚 **Related Documentation**

- [**Getting Started**](getting-started.md) - Quick setup and first run
- [**Configuration**](configuration.md) - Complete configuration guide
- [**Coordinate Tokens**](coordinate-tokens.md) - Coordinate token system
- [**Troubleshooting**](troubleshooting.md) - Training issue solutions

---

**Need help with training?** Check [Troubleshooting](troubleshooting.md) for solutions to common training issues.
