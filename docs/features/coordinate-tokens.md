# Coordinate Token System

**Status:** ✅ PRODUCTION READY | **Both Standard and Coordinate modes fully operational**

The coordinate token system provides two distinct approaches for handling object coordinates in the BBU training pipeline, each optimized for different use cases.

## 🎯 **System Overview**

### **Two Operational Modes**

#### **Standard Mode** (Recommended for Production)
- **Coordinates**: Integer format `[150,10,211,35]`
- **Vocabulary**: Minimal extension (+4 geometry tokens)
- **Use Case**: Production training with stable performance
- **Status**: ✅ Production ready

#### **Coordinate Mode** (Advanced Features)
- **Coordinates**: Token format `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]`
- **Vocabulary**: Extended (+2052 coordinate tokens)
- **Use Case**: Advanced sequence-based coordinate prediction
- **Status**: ✅ Production ready (requires `remove_unused_columns: false`)

## 📊 **Mode Comparison**

| Feature | Standard Mode | Coordinate Mode |
|---------|---------------|-----------------|
| **Coordinate Format** | `[150,10,211,35]` | `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]` |
| **Vocabulary Size** | +4 tokens | +2052 tokens |
| **Training Stability** | High | High (with proper config) |
| **Memory Usage** | Low | Higher |
| **Production Ready** | ✅ Yes | ✅ Yes |
| **Configuration Requirement** | Standard | `remove_unused_columns: false` |

## 🔧 **Configuration**

### **Standard Mode Setup**
```yaml
# configs/bbu_v2.yaml
coordinate_tokens_enabled: false
remove_unused_columns: false  # Recommended for consistency
max_coord_value: 2048
```

### **Coordinate Mode Setup**
```yaml
# configs/bbu_coordinate.yaml
coordinate_tokens_enabled: true
remove_unused_columns: false  # REQUIRED for coordinate mode
max_coord_value: 2048
coordinate_loss_weight: 0.05
coordinate_lr: 5e-6
```

## 📝 **Output Format Examples**

### **Standard Mode Output**
```
Object Reference + Geometry + Integer Coordinates:
"<|object_ref_start|>desc:BBU设备<|object_ref_end|>,<|box_start|>[150,10,211,35]<|box_end|>"

Multi-geometry Support:
"<|object_ref_start|>desc:光纤<|object_ref_end|>,<|line_start|>[100,200,150,250,200,300]<|line_end|>"
"<|object_ref_start|>desc:标签<|object_ref_end|>,<|square_start|>[50,60,80,65,85,95,55,90]<|square_end|>"
```

### **Coordinate Mode Output**
```
Object Reference + Geometry + Coordinate Tokens:
"<|object_ref_start|>desc:BBU设备<|object_ref_end|>,<|box_start|>[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]<|box_end|>"

Multi-geometry Support:
"<|object_ref_start|>desc:光纤<|object_ref_end|>,<|line_start|>[<|coord_100|>,<|coord_200|>,<|coord_150|>,<|coord_250|>,<|coord_200|>,<|coord_300|>]<|line_end|>"
```

## 🚀 **Getting Started**

### **Quick Test - Standard Mode**
```bash
# Test standard mode
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 10

# Verify output format
python -c "
from src.core.coordinate_manager import SimpleCoordinateManager
manager = SimpleCoordinateManager(coordinate_tokens_enabled=False)
result = manager.format_object('BBU设备', 'box', [150,10,211,35])
print(f'Standard mode: {result}')
"
```

### **Quick Test - Coordinate Mode**
```bash
# Test coordinate mode
python scripts/train.py --config configs/bbu_coordinate.yaml --max_steps 10

# Verify output format
python -c "
from src.core.coordinate_manager import SimpleCoordinateManager
from src.utils.tokens.special_tokens import UnifiedTokenManager
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()
manager = SimpleCoordinateManager(coordinate_tokens_enabled=True, tokenizer=tokenizer)
result = manager.format_object('BBU设备', 'box', [150,10,211,35])
print(f'Coordinate mode: {result}')
"
```

## 🔍 **Technical Implementation**

### **Token Management**
```python
# Automatic token detection and management
from src.utils.tokens.special_tokens import UnifiedTokenManager

# Create tokenizer with coordinate tokens (if needed)
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()

# Check coordinate token availability
has_coord_tokens = UnifiedTokenManager.has_coordinate_tokens(tokenizer)
print(f"Coordinate tokens available: {has_coord_tokens}")
```

### **Coordinate Processing**
```python
# Coordinate conversion between modes
from src.core.coordinate_manager import SimpleCoordinateManager

# Standard mode
manager_std = SimpleCoordinateManager(coordinate_tokens_enabled=False)
std_output = manager_std.format_object("device", "box", [150,10,211,35])

# Coordinate mode
manager_coord = SimpleCoordinateManager(coordinate_tokens_enabled=True, tokenizer=tokenizer)
coord_output = manager_coord.format_object("device", "box", [150,10,211,35])
```

## 🧪 **Testing and Validation**

### **System Tests**
```bash
# Test coordinate token system
python -m pytest tests/test_coordinate_tokens.py -v

# Test both modes
python -m pytest tests/test_data_pipeline.py::TestDataPipeline::test_bbu_dataset_loading_standard_mode -v
python -m pytest tests/test_data_pipeline.py::TestDataPipeline::test_bbu_dataset_loading_coordinate_mode -v

# Test training components
python -m pytest tests/test_training_components.py -v
```

### **Manual Validation**
```python
# Validate coordinate token format
def validate_coordinate_tokens(tokenizer):
    """Validate coordinate token format and availability."""
    vocab = tokenizer.get_vocab()
    
    # Check for coordinate tokens
    coord_tokens = [token for token in vocab.keys() if token.startswith('<|coord_') and token.endswith('|>')]
    print(f"Found {len(coord_tokens)} coordinate tokens")
    
    # Validate format
    for i in range(min(10, len(coord_tokens))):
        expected = f"<|coord_{i}|>"
        if expected in vocab:
            print(f"✅ {expected} found")
        else:
            print(f"❌ {expected} missing")
```

## ⚠️ **Configuration Requirements**

### **Critical Setting for Coordinate Mode**
```yaml
# REQUIRED for coordinate mode to work properly
remove_unused_columns: false
```

**Why this is required**: HuggingFace Trainer's default `remove_unused_columns=True` removes essential data columns in coordinate mode, causing empty data dictionaries.

### **Troubleshooting Configuration Issues**
```bash
# Verify configuration
python -c "
from src.config import load_config
config = load_config('configs/bbu_coordinate.yaml')
print(f'coordinate_tokens_enabled: {config.coordinate_tokens_enabled}')
print(f'remove_unused_columns: {config.remove_unused_columns}')
"

# Test data loading
python -c "
from src.core.data_processor import DataProcessor
from src.config import load_config
config = load_config('configs/bbu_coordinate.yaml')
processor = DataProcessor(None, None, None, config=config)
print('✅ Data processor created successfully')
"
```

## 📚 **Related Documentation**

- [**Getting Started**](getting-started.md) - Quick setup and first run
- [**Configuration**](configuration.md) - Complete configuration guide
- [**Training**](training.md) - Training workflows and monitoring
- [**Troubleshooting**](troubleshooting.md) - Common issues and solutions
- [**API Reference**](api-reference.md) - Technical API documentation

## 🎯 **Best Practices**

### **For Production Use**
1. **Use Standard Mode** for most production scenarios
2. **Set `remove_unused_columns: false`** in both modes for consistency
3. **Test thoroughly** before deploying coordinate mode
4. **Monitor vocabulary size** and memory usage

### **For Research/Experimentation**
1. **Use Coordinate Mode** for sequence-based coordinate prediction research
2. **Ensure proper configuration** with `remove_unused_columns: false`
3. **Compare both modes** to understand performance differences
4. **Validate token formats** before training

## 🔧 **Quick Troubleshooting**

### **Common Issues**

#### **"Coordinate token not found in vocabulary"**
```bash
# Check token format - must use <|coord_X|> format
python -c "
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()
print('<|coord_150|>' in tokenizer.get_vocab())  # Should be True
print('<coord_150>' in tokenizer.get_vocab())    # Should be False
"
```

#### **"Empty dictionaries in data collator"**
```yaml
# Fix: Set remove_unused_columns to false
remove_unused_columns: false  # REQUIRED for coordinate mode
```

#### **"Vocabulary size mismatch"**
```python
# Check vocabulary extension
from src.utils.tokens.special_tokens import UnifiedTokenManager
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()
print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
# Standard mode: ~151,669 tokens
# Coordinate mode: ~153,717 tokens
```

---

**Need more help?** Check [Troubleshooting](troubleshooting.md) for comprehensive solutions.
