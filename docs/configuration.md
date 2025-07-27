# BBU Training Pipeline Configuration

**Complete configuration guide for the BBU training pipeline with coordinate token system**

## 🎯 **Quick Start Configurations**

### **Standard Mode (Recommended for Production)**
```yaml
# configs/bbu_v2.yaml
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_data_path: "data/teacher.jsonl"

# Coordinate Token Settings
coordinate_tokens_enabled: false        # Use integer coordinates
remove_unused_columns: false           # Recommended for consistency
max_coord_value: 2048

# Training Settings
learning_rate: 1e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
max_steps: 1000

# Output Settings
output_dir: "checkpoints/run_001"
logging_steps: 10
save_steps: 100
eval_steps: 100
```

### **Coordinate Mode (Advanced Features)**
```yaml
# configs/bbu_coordinate.yaml
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_data_path: "data/teacher.jsonl"

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
```

## 📊 **Configuration Parameters**

### **Core Settings**

#### **Model and Data**
```yaml
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"  # Model location
train_data_path: "data/train.jsonl"                    # Training data
val_data_path: "data/val.jsonl"                        # Validation data
teacher_data_path: "data/teacher.jsonl"                # Teacher forcing data
```

#### **Coordinate Token System**
```yaml
coordinate_tokens_enabled: false       # Enable coordinate token mode
remove_unused_columns: false          # CRITICAL: Prevents data loss
max_coord_value: 2048                 # Maximum coordinate value
coordinate_loss_weight: 0.05          # Weight for coordinate loss (coordinate mode only)
coordinate_lr: 5e-6                   # Learning rate for coordinate tokens
```

#### **Training Parameters**
```yaml
learning_rate: 1e-5                   # Base learning rate
num_train_epochs: 3                   # Number of training epochs
max_steps: 1000                       # Maximum training steps (overrides epochs)
per_device_train_batch_size: 4        # Batch size per GPU
gradient_accumulation_steps: 1        # Gradient accumulation
warmup_steps: 100                     # Learning rate warmup
weight_decay: 0.01                    # Weight decay for regularization
```

#### **Hardware and Performance**
```yaml
dataloader_num_workers: 4             # Data loading workers
fp16: true                           # Mixed precision training
gradient_checkpointing: false        # Memory optimization (slower training)
dataloader_pin_memory: true          # Pin memory for faster data transfer
```

#### **Logging and Checkpoints**
```yaml
output_dir: "checkpoints/run_001"     # Output directory
logging_steps: 10                    # Log every N steps
save_steps: 100                      # Save checkpoint every N steps
eval_steps: 100                      # Evaluate every N steps
save_total_limit: 3                  # Keep only N latest checkpoints
load_best_model_at_end: true         # Load best model after training
```

## 🔧 **Advanced Configuration**

### **Memory Optimization**
```yaml
# For limited GPU memory
per_device_train_batch_size: 2        # Reduce batch size
gradient_accumulation_steps: 2        # Maintain effective batch size
gradient_checkpointing: true          # Enable gradient checkpointing
fp16: true                           # Use mixed precision
dataloader_num_workers: 2            # Reduce workers
```

### **Multi-GPU Training**
```yaml
# For multiple GPUs
per_device_train_batch_size: 4        # Per-GPU batch size
gradient_accumulation_steps: 1        # Adjust based on total GPUs
dataloader_num_workers: 8            # More workers for data loading
```

### **Debugging Configuration**
```yaml
# For debugging and testing
max_steps: 10                        # Quick test run
logging_steps: 1                     # Frequent logging
save_steps: 5                        # Frequent saves
eval_steps: 5                        # Frequent evaluation
per_device_train_batch_size: 1       # Small batch for debugging
```

## 🧪 **Configuration Validation**

### **Validate Configuration**
```python
# Test configuration loading
from src.config import load_config

try:
    config = load_config('configs/bbu_v2.yaml')
    print("✅ Configuration loaded successfully")
    print(f"Model path: {config.model_path}")
    print(f"Coordinate tokens: {config.coordinate_tokens_enabled}")
    print(f"Remove unused columns: {config.remove_unused_columns}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### **Check Required Settings**
```python
# Validate critical settings for coordinate mode
def validate_coordinate_config(config_path):
    config = load_config(config_path)
    
    if config.coordinate_tokens_enabled and config.remove_unused_columns:
        print("❌ ERROR: coordinate mode requires remove_unused_columns: false")
        return False
    
    if not os.path.exists(config.model_path):
        print(f"❌ ERROR: Model path not found: {config.model_path}")
        return False
    
    required_data_files = [config.train_data_path, config.val_data_path]
    for file_path in required_data_files:
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Data file not found: {file_path}")
            return False
    
    print("✅ Configuration validation passed")
    return True

# Test your configuration
validate_coordinate_config('configs/bbu_coordinate.yaml')
```

## 🎯 **Configuration Best Practices**

### **For Production Training**
1. **Use Standard Mode** for most production scenarios
2. **Set appropriate batch sizes** based on GPU memory
3. **Enable mixed precision** (`fp16: true`) for faster training
4. **Configure proper checkpointing** to avoid losing progress
5. **Set reasonable logging intervals** for monitoring

### **For Coordinate Mode**
1. **Always set `remove_unused_columns: false`** - Critical requirement
2. **Start with lower coordinate loss weight** (0.05) and adjust
3. **Monitor gradient norms** for coordinate tokens
4. **Test with small max_steps** before full training

### **For Development/Testing**
1. **Use small max_steps** for quick iteration
2. **Increase logging frequency** for debugging
3. **Reduce batch sizes** to fit on smaller GPUs
4. **Enable gradient checkpointing** if memory is limited

## 🔍 **Configuration Troubleshooting**

### **Common Configuration Errors**

#### **"Data loading pipeline is clearing data"**
```yaml
# Fix: Ensure remove_unused_columns is false
remove_unused_columns: false  # REQUIRED for coordinate mode
```

#### **"Model path not found"**
```yaml
# Fix: Use correct model path
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"  # Correct path
# Not: "Qwen/Qwen2.5-VL-3B-Instruct"  # Wrong - missing model_cache/
```

#### **"CUDA out of memory"**
```yaml
# Fix: Reduce memory usage
per_device_train_batch_size: 2        # Reduce from 4
gradient_checkpointing: true          # Enable memory optimization
fp16: true                           # Use mixed precision
```

### **Configuration Testing**
```bash
# Test configuration with dry run
python scripts/train.py --config configs/bbu_v2.yaml --dry-run

# Test specific configuration values
python -c "
from src.config import load_config
config = load_config('configs/bbu_v2.yaml')
print(f'Coordinate tokens: {config.coordinate_tokens_enabled}')
print(f'Remove unused columns: {config.remove_unused_columns}')
print(f'Batch size: {config.per_device_train_batch_size}')
"
```

## 📚 **Related Documentation**

- [**Getting Started**](getting-started.md) - Quick setup and first run
- [**Coordinate Tokens**](coordinate-tokens.md) - Coordinate token system details
- [**Training**](training.md) - Training workflows and monitoring
- [**Troubleshooting**](troubleshooting.md) - Configuration issue solutions

---

**Need help with configuration?** Check [Troubleshooting](troubleshooting.md) for solutions to common configuration issues.
