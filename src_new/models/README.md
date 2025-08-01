# Models Module - New Qwen2.5-VL Architecture

> **📚 Main Documentation**: For complete documentation, see [**docs/**](../../docs/) directory.

This module implements the Models integration for the new simplified Qwen2.5-VL architecture. For implementation guidance, see [**docs/IMPLEMENTATION_GUIDE.md**](../../docs/IMPLEMENTATION_GUIDE.md).

## 📁 Module Structure

```
src_new/models/
├── __init__.py          # Module exports and API
├── wrapper.py          # DetectionModel composition-based wrapper
├── loss_manager.py     # Multi-component loss computation
├── coordinate_loss.py  # Soft expectation + L1 coordinate loss
├── patches.py          # Qwen2.5-VL compatibility fixes (100% reused)

└── README.md          # This documentation
```

## 🏗️ Architecture Overview

### Composition Over Inheritance
The new architecture uses **composition instead of inheritance** for cleaner separation of concerns:

- **DetectionModel**: Main model wrapper using composition
- **CoordinateProcessor**: Handles coordinate token operations
- **LossManager**: Manages dual-loss computation (LLM + coordinate)
- **SoftExpectationCoordinateLoss**: L1 regression with probability distributions
- **Patches**: Direct reuse of existing compatibility fixes

### Key Data Structures

#### LossComponents
```python
@dataclass
class LossComponents:
    loss: torch.Tensor  # Main loss for HuggingFace Trainer
    llm_loss: torch.Tensor
    coordinate_loss: Optional[torch.Tensor]
    teacher_loss: Optional[torch.Tensor]
    student_loss: Optional[torch.Tensor]
```

#### ModelOutput
```python
@dataclass
class ModelOutput:
    loss: torch.Tensor
    logits: torch.Tensor  # [batch_size, seq_len, vocab_size]
    loss_components: LossComponents
    hidden_states: Optional[torch.Tensor]
```

## 🔧 Key Features

### 1. HuggingFace Trainer Compatibility
- Required `loss` field in LossComponents for trainer integration
- Structured loss tracking with component breakdown
- Compatible with standard training loops

### 2. Coordinate Token Support
- 2049 coordinate tokens (`<|coord_0|>` to `<|coord_2048|>`)
- Soft expectation + L1 loss for coordinate regression
- Quad-based line token initialization for better transfer learning

### 3. Dual-Loss Training Architecture
- **LLM Loss**: Standard cross-entropy for language modeling
- **Coordinate Loss**: Soft expectation + L1 for coordinate regression
- **Dual-Mask System**: Separate masks for LLM vs coordinate tokens
- **Structured Loss Reporting**: Component-wise loss tracking and monitoring

### 4. Migration Support
- Compatibility utilities for gradual migration
- Helper functions to adapt old configurations
- Validation tools for migration consistency

## 🚀 Usage Examples

### Basic Model Loading
```python
from src_new.models import DetectionModel

# Load model with config
model = DetectionModel.from_pretrained(
    model_path="/path/to/model",
    config=config,
    tokenizer=tokenizer
)

# Enable coordinate mode
model.enable_coordinate_mode()
```

### Forward Pass with Loss Components
```python
# Training forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

# Access structured loss components
loss_components = outputs.loss_components
print(f"Total Loss: {loss_components.loss.item()}")
print(f"LLM Loss: {loss_components.llm_loss.item()}")
if loss_components.coordinate_loss is not None:
    print(f"Coordinate Loss: {loss_components.coordinate_loss.item()}")
```

### Direct Model Creation
```python
from src_new.models import DetectionModel

# Create model directly
model = DetectionModel.from_pretrained(
    model_path=model_path,
    config=config,
    tokenizer=tokenizer
)
```

## 📊 Implementation Details

### Code Reuse Percentages
- **patches.py**: 100% direct reuse from `src/models/patches.py`
- **wrapper.py**: 70% logic reuse with composition refactoring
- **loss_manager.py**: Complete rewrite with structured approach


### Configuration Requirements
- Requires properly configured `bbu_v2.yaml` with all required fields
- All configuration fields must be explicitly defined
- No fallback values or default handling

### Performance Characteristics
- Minimal overhead over base model
- Efficient coordinate token processing
- Structured loss computation without performance penalty

## 🧪 Testing

The module includes comprehensive integration tests that verify:
- ✅ LossComponents dataclass functionality
- ✅ CoordinateProcessor coordinate masking
- ✅ LossManager multi-component loss computation
- ✅ DetectionModel composition and mode switching
- ✅ ModelOutput structured data handling

## 🚀 Usage

### Direct Implementation
1. **Configure properly**: Ensure all required fields are in configuration
2. **Create model**: Use `DetectionModel.from_pretrained()` directly
3. **Initialize components**: All components require explicit initialization
4. **Train model**: Use the new training pipeline

### Key Changes
- Replace inheritance with composition
- Use structured loss components instead of global storage
- Enable/disable coordinate mode explicitly
- Access loss components through structured output

## 🎯 Benefits of New Architecture

1. **Simplified Maintenance**: Clear separation of concerns
2. **Better Testing**: Modular components easy to unit test
3. **Enhanced Monitoring**: Structured loss component tracking
4. **Migration Friendly**: Gradual migration with compatibility layer
5. **Future Extensible**: Clean interfaces for new features

## 📈 Integration with Training System

The models module integrates seamlessly with the new training system:
- HuggingFace Trainer compatible loss structure
- Component-wise loss tracking for monitoring
- Dynamic coordinate mode switching during training
- Structured output format for enhanced debugging

---

## 📚 **Additional Resources**

### **Documentation**
- **[Model System Documentation](../../docs/implementation/model-system.md)** - Complete model system guide
- **[Training System Documentation](../../docs/implementation/training-system.md)** - Training framework details
- **[API Reference](../../docs/api-reference/src-new-api.md)** - Complete src_new/ API

### **Technical Details**
- **[Coordinate Loss System](COORDINATE_LOSS.md)** - Soft expectation + L1 loss implementation
- **[Architecture Overview](../ARCHITECTURE.md)** - Complete src_new/ architecture
- **[Troubleshooting](../../docs/troubleshooting/common-issues.md)** - Common issues and solutions

This implementation successfully achieves the goals outlined in `new_src.md` while maintaining full compatibility with existing configurations and providing a clear migration path from the old architecture.