# BBU Training Pipeline Troubleshooting

**Comprehensive troubleshooting guide for the BBU training pipeline and coordinate token system**

## 🚨 **Quick Fixes for Common Issues**

### **Training Won't Start**
```bash
# Check environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from src.training.trainer import BBUTrainer; print('✅ System OK')"

# Check data
ls -la data/  # Should see train.jsonl, val.jsonl, teacher.jsonl
head -1 data/train.jsonl | python -m json.tool  # Verify format

# Check configuration
python -c "from src.config import load_config; config = load_config('configs/bbu_v2.yaml'); print('✅ Config OK')"
```

### **Coordinate Normalization Errors**
```bash
# "Degenerate box with zero area" error during training
# Error: RuntimeError: Degenerate box with zero area detected
# Fix: Ensure coordinate normalization is applied

# Test coordinate normalization
python -c "
from data_conversion.coordinate_manager import CoordinateManager
# Test problematic horizontal line
result = CoordinateManager.normalize_line_coordinates([174, 304, 10, 304], 420, 896)
print(f'Normalized: {result}')  # Should be [10, 303, 174, 305]
"

# Run coordinate validation tests
python -m pytest tests/test_coordinate_normalization.py -v

# Check if normalization is integrated in data pipeline
grep -n "normalize_object_coordinates" data_conversion/unified_processor.py
```

### **Coordinate Token Errors**
```bash
# Wrong token format error
# Error: "Coordinate token <coord_X> not found in vocabulary"
# Fix: Use correct format <|coord_X|> with pipe characters

# Check token format
python -c "
from src.utils.tokens.special_tokens import UnifiedTokenManager
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()
print('<|coord_150|>' in tokenizer.get_vocab())  # Should be True
print('<coord_150>' in tokenizer.get_vocab())    # Should be False
"
```

### **Empty Data Dictionaries**
```yaml
# Error: "TRAINER CONFIGURATION ISSUE: data loading pipeline is clearing data"
# Fix: Set remove_unused_columns to false in configuration
remove_unused_columns: false  # REQUIRED for coordinate mode
```

## 🔧 **Coordinate Token System Issues**

### **1. Token Format Problems**

#### **Symptoms**
- `Coordinate token <coord_X> not found in vocabulary`
- `KeyError: '<coord_150>'`

#### **Diagnosis**
```python
# Check current token format in your data
with open('data/train.jsonl', 'r') as f:
    sample = json.loads(f.readline())
    content = sample['conversations'][0]['value']
    print("Current format:", content)
    
    # Look for coordinate patterns
    import re
    coord_patterns = re.findall(r'<\|?coord_\d+\|?>', content)
    print("Found patterns:", coord_patterns)
```

#### **Solution**
```python
# Ensure correct format throughout pipeline
# Correct: <|coord_150|>
# Wrong: <coord_150>

# Fix in data processing
from src.core.coordinate_manager import SimpleCoordinateManager
manager = SimpleCoordinateManager(coordinate_tokens_enabled=True, tokenizer=tokenizer)
result = manager.format_object("device", "box", [150,10,211,35])
print(f"Correct format: {result}")
```

### **2. Vocabulary Size Mismatch**

#### **Symptoms**
- `Found token ID exceeding vocabulary size`
- `RuntimeError: CUDA error: device-side assert triggered`
- Model embedding size errors

#### **Diagnosis**
```python
# Check vocabulary sizes
from src.utils.tokens.special_tokens import UnifiedTokenManager

# Standard mode
tokenizer_std = UnifiedTokenManager.create_base_tokenizer()
print(f"Standard mode vocab: {len(tokenizer_std.get_vocab())}")  # ~151,669

# Coordinate mode
tokenizer_coord = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()
print(f"Coordinate mode vocab: {len(tokenizer_coord.get_vocab())}")  # ~153,717

# Check model embedding size
from src.models.model_loader import load_model_and_processor_unified
model, _, _ = load_model_and_processor_unified('model_cache/Qwen/Qwen2.5-VL-3B-Instruct')
print(f"Model embedding size: {model.get_input_embeddings().num_embeddings}")
```

#### **Solution**
```python
# Ensure proper model loading with embedding resize
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, processor = load_model_and_processor_unified(
    model_path='model_cache/Qwen/Qwen2.5-VL-3B-Instruct',
    for_inference=False  # This triggers embedding resize
)
```

### **3. Configuration Issues**

#### **Symptoms**
- Empty data dictionaries in coordinate mode
- Training fails with "data loading pipeline is clearing data"

#### **Diagnosis**
```python
# Check critical configuration settings
from src.config import load_config
config = load_config('configs/bbu_coordinate.yaml')

print(f"coordinate_tokens_enabled: {config.coordinate_tokens_enabled}")
print(f"remove_unused_columns: {config.remove_unused_columns}")  # Must be False

# Verify data processor creation
from src.core.data_processor import DataProcessor
processor = DataProcessor(tokenizer, processor, model, config=config)
print("✅ Data processor created successfully")
```

#### **Solution**
```yaml
# Required configuration for coordinate mode
coordinate_tokens_enabled: true
remove_unused_columns: false  # CRITICAL - prevents data loss
max_coord_value: 2048
coordinate_loss_weight: 0.05
```

## 🏃 **Training Issues**

### **1. Training Crashes**

#### **Memory Issues**
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size
# In config: per_device_train_batch_size: 2  # Reduce from 4

# Enable gradient checkpointing
# In config: gradient_checkpointing: true
```

#### **CUDA Errors**
```bash
# Check CUDA compatibility
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### **2. Loss Issues**

#### **NaN Losses**
```python
# Check for NaN in data
import torch
import json

def check_data_for_nan():
    with open('data/train.jsonl', 'r') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            # Check for invalid coordinates
            if 'conversations' in sample:
                content = sample['conversations'][0]['value']
                # Look for coordinate values
                import re
                coords = re.findall(r'\[([0-9,\s]+)\]', content)
                for coord_str in coords:
                    try:
                        coords_list = [int(x.strip()) for x in coord_str.split(',')]
                        if any(c < 0 or c > 2048 for c in coords_list):
                            print(f"Invalid coordinates in sample {i}: {coords_list}")
                    except ValueError:
                        print(f"Invalid coordinate format in sample {i}: {coord_str}")

check_data_for_nan()
```

#### **High Gradient Norms**
```python
# Monitor gradient norms
# Add to training script:
def log_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm}")
    return total_norm
```

## 🧪 **Testing and Validation**

### **System Health Check**
```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_coordinate_tokens.py -v
python -m pytest tests/test_data_pipeline.py -v
python -m pytest tests/test_training_components.py -v

# Quick validation
python -c "
from src.core.coordinate_manager import SimpleCoordinateManager
from src.utils.tokens.special_tokens import UnifiedTokenManager

# Test standard mode
manager_std = SimpleCoordinateManager(coordinate_tokens_enabled=False)
result_std = manager_std.format_object('device', 'box', [150,10,211,35])
print(f'Standard: {result_std}')

# Test coordinate mode
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()
manager_coord = SimpleCoordinateManager(coordinate_tokens_enabled=True, tokenizer=tokenizer)
result_coord = manager_coord.format_object('device', 'box', [150,10,211,35])
print(f'Coordinate: {result_coord}')
"
```

### **Data Validation**
```python
# Validate training data format
def validate_training_data():
    import json
    from pathlib import Path
    
    data_files = ['data/train.jsonl', 'data/val.jsonl', 'data/teacher.jsonl']
    
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"\nValidating {file_path}:")
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Check first 3 samples
                        break
                    try:
                        sample = json.loads(line)
                        print(f"  Sample {i}: ✅ Valid JSON")
                        
                        # Check required fields
                        required_fields = ['conversations', 'images']
                        for field in required_fields:
                            if field in sample:
                                print(f"    {field}: ✅ Present")
                            else:
                                print(f"    {field}: ❌ Missing")
                                
                    except json.JSONDecodeError as e:
                        print(f"  Sample {i}: ❌ Invalid JSON - {e}")
        else:
            print(f"❌ {file_path} not found")

validate_training_data()
```

## 📚 **Getting Help**

### **Log Analysis**
```bash
# Check training logs
tail -f checkpoints/run_*/training.log

# Look for specific errors
grep -i "error\|exception\|failed" checkpoints/run_*/training.log

# Check loss progression
grep "loss=" checkpoints/run_*/training.log | tail -20
```

### **Debug Mode**
```bash
# Run with debug logging
export PYTHONPATH=/data3/Qwen2.5-VL-main
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 10 --logging_level DEBUG
```

### **Community Resources**
- [**Getting Started**](../getting-started.md) - Basic setup and first run
- [**Configuration**](../implementation/configuration.md) - Complete configuration guide
- [**Coordinate Tokens**](../features/coordinate-tokens.md) - Coordinate token system details
- [**API Reference**](../guides/api-reference.md) - Technical documentation

---

**Still having issues?** Create a detailed issue report with:
1. Error message (full traceback)
2. Configuration file used
3. System information (GPU, Python version)
4. Steps to reproduce
5. Expected vs actual behavior
