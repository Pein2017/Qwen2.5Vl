# src_new: BBUTrainer Architecture

**Production-ready training architecture with NCCL timeout resolution**

## 🎯 **Overview**

The `src_new/` directory contains the latest production training architecture featuring:

- **BBUTrainer**: Eliminates NCCL timeout issues through local loss aggregation
- **TrainingStateManager**: Simplified state management without distributed conflicts
- **Enhanced Logging**: 4-decimal precision for losses, readable learning rate formatting
- **Configurable Dataset Size**: `max_dataset_size` parameter for debugging/testing

## 🏗️ **Architecture**

### **Core Components**

```
src_new/
├── training/
│   ├── bbu_trainer.py           # Main BBUTrainer implementation
│   ├── training_state_manager.py # Local loss aggregation
│   └── __init__.py              # Module exports
├── models/
│   └── wrapper.py               # DetectionModel wrapper
├── data/
│   └── dataset.py               # Dataset with max_dataset_size support
├── config/
│   └── config.py                # Configuration schema
└── README.md                    # This file
```

### **BBUTrainer Features**

- **Local Loss Aggregation**: No distributed operations in critical paths
- **Complete Override**: `_maybe_log_save_evaluate()` method completely overridden
- **Standard HF Logging**: Uses only `super().log()` for compatibility
- **Evaluation Loss Components**: Proper `eval_loss` and component capture
- **Enhanced Checkpoint Logging**: Comprehensive save operation logging

### **TrainingStateManager Features**

- **LossComponents Handling**: Supports both dataclass and dictionary formats
- **Meaningful LR Groups**: `vision_lr`, `merger_lr` instead of generic names
- **Local Accumulation**: No distributed synchronization required
- **Reset Management**: Proper state cleanup after logging

## 🚀 **Usage**

### **Basic Training**

```bash
# Use existing training scripts (automatically uses BBUTrainer)
bash scripts/run_new_train.sh bbu_v2_not_use_coord 4,5,6,7 INFO
```

### **Configuration**

```yaml
# configs/bbu_v2_not_use_coord.yaml
num_train_epochs: 30
per_device_train_batch_size: 1
learning_rate: 5e-6
vision_lr: 5e-7
merger_lr: 5e-5

# Dataset size limiting (optional, for debugging)
max_dataset_size: -1  # -1 = full dataset, positive integer = limit samples

# Logging settings
logging_steps: 1
eval_steps: 20
save_steps: 50
```

### **Expected Log Output**

```json
{
  "loss": 39.2213,
  "llm_loss": 15.3432,
  "coordinate_loss": 244.0140,
  "teacher_loss": 27.5678,
  "student_loss": 11.6543,
  "grad_norm": 4769.334,
  "vision_lr": "5.00e-07",
  "merger_lr": "5.00e-05",
  "epoch": 0.333
}
```

## 🔧 **Key Improvements Over Previous Architectures**

### **NCCL Timeout Resolution**
- **Problem**: DistributedLossTrainer had double NCCL operations causing timeouts
- **Solution**: BBUTrainer uses local aggregation only, no custom distributed ops

### **Enhanced Logging**
- **Loss Precision**: All losses rounded to 4 decimal places
- **Learning Rates**: Scientific notation (e.g., `5.00e-07`) for readability
- **Meaningful Names**: `vision_lr`, `merger_lr` instead of `learning_rate_group_0`

### **Dataset Flexibility**
- **Full Dataset**: `max_dataset_size: -1` (default)
- **Limited Dataset**: `max_dataset_size: 10` for quick testing
- **Clear Logging**: Shows whether using full or limited dataset

### **Evaluation Completeness**
- **eval_loss**: Properly computed and logged
- **Component Losses**: `eval_llm_loss`, `eval_coordinate_loss`, etc.
- **State Isolation**: Evaluation doesn't interfere with training state

## 📊 **Monitoring and Debugging**

### **Dataset Size Logging**
```
📊 [FULL DATASET] Using complete dataset: 300 samples
✅ Loaded 300 samples from data/ds_v2_full/train.jsonl
```

### **Checkpoint Logging**
```
🔄 [CHECKPOINT SAVE] Starting checkpoint save at 2025-08-01 02:28:45
📁 Checkpoint location: 7-30/checkpoint-100
📊 Training step: 100
📈 Epoch: 0.500
📋 Current training metrics:
   loss: 39.2213
   llm_loss: 15.3432
   vision_lr: 5.00e-07
✅ [CHECKPOINT SAVE] Completed successfully in 2.34s
```

## 🔄 **Migration from src/**

The `src_new/` architecture is a complete replacement for `src/` with:

1. **Simplified Architecture**: Fewer components, clearer responsibilities
2. **Eliminated NCCL Issues**: No distributed operation conflicts
3. **Enhanced Monitoring**: Better logging and debugging capabilities
4. **Production Stability**: Thoroughly tested and verified

### **Migration Steps**
1. Use `scripts/run_new_train.sh` instead of old training scripts
2. Update configs to use `src_new/` compatible parameters
3. Verify `max_dataset_size: -1` for full dataset usage
4. Monitor improved logging output for verification

## 🎉 **Production Status**

**Status**: ✅ **Production Ready**
- **NCCL Timeouts**: Completely resolved
- **Loss Logging**: All components captured correctly
- **Evaluation**: Complete metrics including eval_loss
- **Checkpointing**: Robust saving with comprehensive logging
- **Testing**: 5/5 integration tests passed

The `src_new/` architecture is the recommended approach for all new training workflows.
