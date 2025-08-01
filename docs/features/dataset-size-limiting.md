# Dataset Size Limiting Feature

**Configurable dataset size limiting for debugging and testing**

## 🎯 **Overview**

The `max_dataset_size` parameter provides flexible dataset size limiting primarily for debugging and testing purposes. This feature allows you to quickly test training workflows with smaller datasets while preserving the ability to use the full dataset for production training.

## 🔧 **Configuration**

### **Parameter: max_dataset_size**

```yaml
# Full dataset (default, production use)
max_dataset_size: -1

# Limited dataset (debugging/testing)
max_dataset_size: 10    # Use only first 10 samples
max_dataset_size: 100   # Use only first 100 samples
```

### **Behavior**

| Value | Behavior | Use Case |
|-------|----------|----------|
| `-1` | Use full dataset | **Production training** |
| `0` | Use full dataset | Same as -1 |
| `> 0` | Limit to N samples | **Debugging/testing** |

## 📊 **Logging Output**

### **Full Dataset Usage**
```
📊 [FULL DATASET] Using complete dataset: 300 samples
✅ Loaded 300 samples from data/ds_v2_full/train.jsonl
```

### **Limited Dataset Usage**
```
🔬 [DATASET LIMITING] Using limited dataset: 10/300 samples
📊 Dataset size limited to 10 samples for debugging/testing
✅ Loaded 10 samples from data/ds_v2_full/train.jsonl
```

### **Edge Case: Limit Larger Than Dataset**
```
📊 [FULL DATASET] Using complete dataset: 300 samples
💡 max_dataset_size=500 >= dataset size, using all available samples
✅ Loaded 300 samples from data/ds_v2_full/train.jsonl
```

## 🚀 **Usage Examples**

### **Quick Testing (10 samples)**
```yaml
# configs/debug.yaml
max_dataset_size: 10
num_train_epochs: 2
eval_steps: 2
save_steps: 5
```

### **Medium Testing (100 samples)**
```yaml
# configs/test.yaml
max_dataset_size: 100
num_train_epochs: 5
eval_steps: 10
save_steps: 20
```

### **Production Training**
```yaml
# configs/production.yaml
max_dataset_size: -1  # Full dataset
num_train_epochs: 30
eval_steps: 100
save_steps: 500
```

## 🔍 **Implementation Details**

### **Dataset Loading Logic**
```python
# In src_new/data/dataset.py
original_size = len(raw_data)
max_dataset_size = getattr(self.config, "max_dataset_size", -1)

# Handle None case (fallback to -1 for full dataset)
if max_dataset_size is None:
    max_dataset_size = -1

if max_dataset_size > 0 and max_dataset_size < original_size:
    # Limit dataset size for debugging/testing
    raw_data = raw_data[:max_dataset_size]
    logger.info(f"🔬 [DATASET LIMITING] Using limited dataset: {len(raw_data)}/{original_size} samples")
else:
    # Use full dataset
    logger.info(f"📊 [FULL DATASET] Using complete dataset: {len(raw_data)} samples")
```

### **Configuration Schema**
```python
# In src_new/config/config.py
@dataclass
class DirectConfig:
    # Dataset size limiting (optional, primarily for debugging/testing)
    # -1 = use full dataset (default), positive integer = limit to N samples
    max_dataset_size: int = -1
```

## 🎯 **Use Cases**

### **1. Quick Debugging**
- **Problem**: Training crashes after 1 hour
- **Solution**: Set `max_dataset_size: 5` to reproduce issue quickly
- **Benefit**: Fast iteration on bug fixes

### **2. Configuration Testing**
- **Problem**: Testing new hyperparameters
- **Solution**: Set `max_dataset_size: 50` for rapid experimentation
- **Benefit**: Quick validation of config changes

### **3. Development Workflow**
- **Problem**: Developing new features
- **Solution**: Set `max_dataset_size: 20` during development
- **Benefit**: Fast feedback loop for code changes

### **4. CI/CD Testing**
- **Problem**: Automated testing needs to be fast
- **Solution**: Set `max_dataset_size: 3` in test configs
- **Benefit**: Quick validation in CI pipelines

## ⚠️ **Important Notes**

### **Production Usage**
- **Always use `max_dataset_size: -1`** for production training
- **Never limit dataset size** for final model training
- **Document any testing** that uses limited datasets

### **Epoch Calculation Impact**
- **Limited datasets affect epoch calculation**
- **Example**: 10 samples with batch size 4 = 2.5 steps per epoch
- **Monitor epoch values** to ensure they make sense

### **Reproducibility**
- **Same `max_dataset_size`** produces same training subset
- **Samples are taken from the beginning** of the dataset (deterministic)
- **Use same value** for consistent debugging sessions

## 🔄 **Migration from Legacy Systems**

### **From max_examples**
```yaml
# OLD (deprecated)
max_examples: 10

# NEW (current)
max_dataset_size: 10
```

### **From Hard-coded Limits**
```python
# OLD (hard-coded in dataset.py)
if DEBUG_MODE:
    raw_data = raw_data[:10]

# NEW (configurable)
max_dataset_size: 10  # In YAML config
```

## 📈 **Performance Impact**

### **Memory Usage**
- **Limited datasets use less memory** (proportional to dataset size)
- **Useful for memory-constrained environments**

### **Training Speed**
- **Faster epochs** with smaller datasets
- **Quicker iteration** for debugging
- **Reduced I/O** for data loading

### **Evaluation Speed**
- **Faster evaluation** with smaller validation sets
- **Quicker feedback** during development

## 🎉 **Summary**

The `max_dataset_size` parameter provides:

- **✅ Flexible dataset limiting** for debugging and testing
- **✅ Clear logging** of dataset usage
- **✅ Production-safe defaults** (full dataset by default)
- **✅ Deterministic behavior** for reproducible debugging
- **✅ Easy configuration** via YAML files

This feature significantly improves the development and debugging experience while maintaining production reliability.
