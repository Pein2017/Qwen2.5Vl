# Migration Guide

**Guide for upgrading to the latest BBU training pipeline with coordinate token system**

## 🎯 **Migration Overview**

This guide helps you migrate from older versions of the BBU training pipeline to the current unified system with coordinate token support.

### **What's New in Current Version**
- ✅ **Dual-mode coordinate system** (Standard + Coordinate modes)
- ✅ **Simplified configuration** with unified YAML format
- ✅ **Enhanced testing framework** with 30+ comprehensive tests
- ✅ **Improved documentation** with consolidated structure
- ✅ **Production-ready coordinate tokens** with proper trainer compatibility

## 📋 **Migration Checklist**

### **Before Migration**
- [ ] Backup existing configuration files
- [ ] Backup trained models and checkpoints
- [ ] Document current training parameters
- [ ] Test current system functionality

### **After Migration**
- [ ] Update configuration format
- [ ] Validate coordinate token functionality
- [ ] Run system tests
- [ ] Verify training pipeline
- [ ] Update deployment scripts

## 🔧 **Configuration Migration**

### **Old Configuration Format**
```yaml
# Old format (deprecated)
model_name: "Qwen/Qwen2.5-VL-3B-Instruct"
data_dir: "data/"
coordinate_mode: true
use_coordinate_tokens: true
max_coordinate_value: 2048
```

### **New Configuration Format**
```yaml
# New unified format
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_data_path: "data/teacher.jsonl"

# Coordinate token settings
coordinate_tokens_enabled: true
remove_unused_columns: false  # REQUIRED for coordinate mode
max_coord_value: 2048
coordinate_loss_weight: 0.05
coordinate_lr: 5e-6

# Training settings
learning_rate: 1e-5
per_device_train_batch_size: 4
num_train_epochs: 3
output_dir: "checkpoints/run_001"
```

### **Configuration Migration Script**
```python
# migrate_config.py
import yaml
from pathlib import Path

def migrate_config(old_config_path, new_config_path):
    """Migrate old configuration to new format."""
    
    with open(old_config_path, 'r') as f:
        old_config = yaml.safe_load(f)
    
    # Create new configuration structure
    new_config = {
        # Model settings
        'model_path': old_config.get('model_name', 'model_cache/Qwen/Qwen2.5-VL-3B-Instruct'),
        
        # Data settings
        'train_data_path': f"{old_config.get('data_dir', 'data/')}/train.jsonl",
        'val_data_path': f"{old_config.get('data_dir', 'data/')}/val.jsonl",
        'teacher_data_path': f"{old_config.get('data_dir', 'data/')}/teacher.jsonl",
        
        # Coordinate token settings
        'coordinate_tokens_enabled': old_config.get('use_coordinate_tokens', False),
        'remove_unused_columns': False,  # Required for coordinate mode
        'max_coord_value': old_config.get('max_coordinate_value', 2048),
        'coordinate_loss_weight': 0.05,
        'coordinate_lr': 5e-6,
        
        # Training settings
        'learning_rate': old_config.get('learning_rate', 1e-5),
        'per_device_train_batch_size': old_config.get('batch_size', 4),
        'num_train_epochs': old_config.get('epochs', 3),
        'output_dir': old_config.get('output_dir', 'checkpoints/run_001'),
        
        # Required settings
        'logging_steps': 10,
        'save_steps': 100,
        'eval_steps': 100,
    }
    
    # Write new configuration
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    print(f"✅ Configuration migrated from {old_config_path} to {new_config_path}")

# Usage
migrate_config('configs/old_config.yaml', 'configs/migrated_config.yaml')
```

## 🔄 **Data Format Migration**

### **Old Data Format**
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Describe the image"
    },
    {
      "from": "gpt", 
      "value": "I see a device at coordinates [150,10,211,35]"
    }
  ],
  "images": ["image.jpg"]
}
```

### **New Data Format**
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Describe the image"
    },
    {
      "from": "gpt",
      "value": "<|object_ref_start|>desc:device<|object_ref_end|>,<|box_start|>[150,10,211,35]<|box_end|>"
    }
  ],
  "images": ["image.jpg"]
}
```

### **Data Migration Script**
```python
# migrate_data.py
import json
import re
from pathlib import Path

def migrate_data_format(input_file, output_file):
    """Migrate data to new coordinate token format."""
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            sample = json.loads(line.strip())
            
            # Process conversations
            for conv in sample.get('conversations', []):
                if conv.get('from') == 'gpt':
                    content = conv['value']
                    
                    # Convert coordinate patterns
                    # Pattern: "device at coordinates [150,10,211,35]"
                    # To: "<|object_ref_start|>desc:device<|object_ref_end|>,<|box_start|>[150,10,211,35]<|box_end|>"
                    
                    pattern = r'(\w+) at coordinates \[([0-9,]+)\]'
                    def replace_coords(match):
                        object_desc = match.group(1)
                        coords = match.group(2)
                        return f"<|object_ref_start|>desc:{object_desc}<|object_ref_end|>,<|box_start|>[{coords}]<|box_end|>"
                    
                    conv['value'] = re.sub(pattern, replace_coords, content)
            
            # Write migrated sample
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Data migrated from {input_file} to {output_file}")

# Usage
migrate_data_format('data/old_train.jsonl', 'data/train.jsonl')
```

## 🧪 **Testing Migration**

### **Validate Migration**
```bash
# Test configuration loading
python -c "
from src.config import load_config
config = load_config('configs/migrated_config.yaml')
print('✅ Configuration loads successfully')
print(f'Coordinate tokens: {config.coordinate_tokens_enabled}')
print(f'Remove unused columns: {config.remove_unused_columns}')
"

# Test data loading
python -c "
from src.core.data_processor import DataProcessor
from src.config import load_config
config = load_config('configs/migrated_config.yaml')
processor = DataProcessor(None, None, None, config=config)
train_dataset, eval_dataset = processor.create_datasets()
print(f'✅ Data loads successfully: {len(train_dataset)} train, {len(eval_dataset)} eval')
"

# Test coordinate token processing
python -c "
from src.core.coordinate_manager import SimpleCoordinateManager
from src.utils.tokens.special_tokens import UnifiedTokenManager

# Test standard mode
manager_std = SimpleCoordinateManager(coordinate_tokens_enabled=False)
result_std = manager_std.format_object('device', 'box', [150,10,211,35])
print(f'Standard mode: {result_std}')

# Test coordinate mode
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()
manager_coord = SimpleCoordinateManager(coordinate_tokens_enabled=True, tokenizer=tokenizer)
result_coord = manager_coord.format_object('device', 'box', [150,10,211,35])
print(f'Coordinate mode: {result_coord}')
print('✅ Coordinate token processing works')
"
```

### **Run System Tests**
```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Test specific migration-related components
python -m pytest tests/test_coordinate_tokens.py -v
python -m pytest tests/test_data_pipeline.py -v
python -m pytest tests/test_training_components.py -v

# Quick training test
python scripts/train.py --config configs/migrated_config.yaml --max_steps 5
```

## 🔧 **Common Migration Issues**

### **1. Configuration Issues**

#### **Problem**: `remove_unused_columns` not set
```
Error: TRAINER CONFIGURATION ISSUE: data loading pipeline is clearing data
```

#### **Solution**:
```yaml
# Add to configuration
remove_unused_columns: false  # REQUIRED for coordinate mode
```

### **2. Model Path Issues**

#### **Problem**: Model not found
```
Error: Model path not found: Qwen/Qwen2.5-VL-3B-Instruct
```

#### **Solution**:
```yaml
# Update model path
model_path: "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"  # Correct path
```

### **3. Data Format Issues**

#### **Problem**: Coordinate tokens not recognized
```
Error: Coordinate token <coord_150> not found in vocabulary
```

#### **Solution**:
```python
# Ensure correct token format in data
# Wrong: <coord_150>
# Correct: <|coord_150|>

# Use migration script to fix data format
migrate_data_format('data/old_train.jsonl', 'data/train.jsonl')
```

### **4. Vocabulary Size Issues**

#### **Problem**: Token ID exceeding vocabulary size
```
Error: Found token ID exceeding vocabulary size
```

#### **Solution**:
```python
# Ensure proper model loading with embedding resize
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, processor = load_model_and_processor_unified(
    model_path, for_inference=False  # This triggers embedding resize
)
```

## 📚 **Post-Migration Steps**

### **1. Update Training Scripts**
```bash
# Update training command
# Old: python train.py --config old_config.yaml
# New: python scripts/train.py --config configs/migrated_config.yaml
```

### **2. Update Monitoring**
```bash
# Update log monitoring
# Old: tail -f logs/training.log
# New: tail -f checkpoints/run_*/training.log
```

### **3. Update Deployment**
```bash
# Update deployment scripts to use new configuration format
# Update model paths and data paths
# Ensure remove_unused_columns: false in production configs
```

## 🎯 **Migration Validation Checklist**

- [ ] Configuration loads without errors
- [ ] Data loads and processes correctly
- [ ] Coordinate token system works in both modes
- [ ] Training starts and runs successfully
- [ ] All tests pass
- [ ] Model checkpoints save correctly
- [ ] Logging and monitoring work
- [ ] Production deployment updated

## 📚 **Related Documentation**

- [**Getting Started**](getting-started.md) - Quick setup with new system
- [**Configuration**](configuration.md) - Complete configuration guide
- [**Coordinate Tokens**](coordinate-tokens.md) - Coordinate token system details
- [**Troubleshooting**](troubleshooting.md) - Migration issue solutions

---

**Need migration help?** Check [Troubleshooting](troubleshooting.md) for solutions to common migration issues.
