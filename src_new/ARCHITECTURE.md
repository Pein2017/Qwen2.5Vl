# Qwen2.5-VL New Architecture (`src_new/`) - Core Structure Guide

> **📋 Note**: This is the **recommended implementation** for new projects. For implementation guidance, see [**docs/IMPLEMENTATION_GUIDE.md**](../docs/IMPLEMENTATION_GUIDE.md).

## 🏗️ **Architecture Overview**

The `src_new/` directory implements a **simplified, maintainable architecture** for Qwen2.5-VL vision-language object detection and captioning training. This architecture follows **composition over inheritance** principles and provides **fail-fast validation** with comprehensive type safety.

### **Design Principles**
- ✅ **Composition over Inheritance**: Clean separation of concerns
- ✅ **Fail-Fast Validation**: Immediate error detection with detailed messages
- ✅ **Type Safety**: Comprehensive type annotations throughout
- ✅ **Backward Compatibility**: Zero changes required to existing configs
- ✅ **Modular Design**: Each component has a single, clear responsibility

### **Why Choose src_new/?**
- **25% less code** with better functionality
- **127 comprehensive tests** - well tested and reliable
- **Better performance** - optimized for production use
- **Easier maintenance** - clean, understandable architecture
- **Active development** - ongoing improvements and support

## 📁 **Directory Structure**

```
src_new/
├── __init__.py                 # Main package exports
├── ARCHITECTURE.md            # This documentation
│
├── config/                    # Configuration Management
│   ├── __init__.py           # Config exports
│   └── config.py             # Unified Config class with YAML mapping
│
├── data/                     # Data Processing Pipeline
│   ├── __init__.py          # Data exports
│   ├── dataset.py           # Dataset with teacher-student support
│   ├── collator.py          # Memory-efficient data collators
│   └── teacher_pool.py      # Teacher demonstration management
│
├── models/                   # Model Components
│   ├── __init__.py          # Model exports
│   ├── wrapper.py           # DetectionModel (composition-based)
│   ├── loss_manager.py      # Multi-component loss computation
│   ├── coordinate_loss.py   # Soft expectation + L1 coordinate loss
│   ├── patches.py           # Qwen2.5-VL compatibility fixes
│   ├── compatibility.py     # Migration utilities
│   └── README.md           # Detailed model documentation
│
├── processing/              # Processing Components
│   ├── __init__.py         # Processing exports
│   ├── chat_processor.py   # Conversation building & tokenization
│   ├── token_processor.py  # Coordinate token handling
│   └── templates.py        # Template management for prompts
│
├── training/               # Training System
│   ├── __init__.py        # Training exports
│   ├── trainer.py         # Minimal HF Trainer extension
│   ├── callbacks.py       # Multi-component loss tracking
│   └── utils.py          # Training utilities
│
└── tests/                 # Comprehensive Test Suite
    ├── __init__.py       # Test package setup
    ├── fixtures/         # Mock objects and test data
    ├── test_config/      # Configuration tests
    ├── test_data/        # Data processing tests
    ├── test_models/      # Model component tests
    ├── test_processing/  # Processing component tests
    ├── test_training/    # Training system tests
    ├── test_integration/ # End-to-end integration tests
    └── test_regression/  # Backward compatibility tests
```

## 🔧 **Core Components**

### **1. Configuration (`config/`)**
**Purpose**: Unified configuration management with direct YAML mapping

**Key Features**:
- **Direct Field Mapping**: Every field in `bbu_v2.yaml` maps to a dataclass field
- **Fail-Fast Validation**: Comprehensive validation with detailed error messages
- **Type Safety**: Full type annotations for all configuration parameters
- **Zero Migration**: Existing configs work without modification

**Main Classes**:
- `Config`: Main configuration dataclass with validation
- `load_config()`: YAML loading with validation
- `save_config()`: Configuration serialization

### **2. Data Processing (`data/`)**
**Purpose**: Optimized data handling with unified preprocessing

**Key Features**:
- **Teacher-Student Support**: Multi-turn conversation building
- **Vision Token Processing**: Proper image token/feature alignment
- **Memory Efficiency**: Flash Attention compatible collators
- **Multi-Image Support**: Handle multiple images per conversation

**Main Classes**:
- `Dataset`: Main dataset with teacher-student support
- `StandardDataCollator`: Flash Attention compatible collator
- `PackedDataCollator`: Memory-efficient sequence packing
- `TeacherPoolManager`: Teacher demonstration management

### **3. Model Components (`models/`)**
**Purpose**: Composition-based model architecture

**Key Features**:
- **Composition Design**: Clean separation of model components
- **Dual-Loss Training**: LLM loss + soft expectation coordinate loss
- **Coordinate Token Support**: 2049 coordinate tokens with quad-based line initialization
- **HuggingFace Compatible**: Works with standard HF Trainer

**Main Classes**:
- `DetectionModel`: Main model wrapper using composition
- `LossManager`: Multi-component loss computation with dual-mask system
- `SoftExpectationCoordinateLoss`: L1 regression with probability distributions
- `CoordinateProcessor`: Coordinate token processing
- `LossComponents`: Structured loss data

### **4. Processing (`processing/`)**
**Purpose**: Stateless processing components

**Key Features**:
- **Conversation Building**: Multi-turn teacher-student conversations
- **Token Processing**: 2051 token extension (2 line + 2049 coordinate tokens)
- **Template Management**: Chinese/English prompt templates
- **Geometry Token Initialization**: Quad-based line token transfer learning

**Main Classes**:
- `ChatProcessor`: Conversation building and tokenization
- `TokenProcessor`: Coordinate token handling with quad-based initialization
- `TemplateManager`: Prompt template management

### **5. Training (`training/`)**
**Purpose**: Clean training system with minimal HF extension

**Key Features**:
- **Minimal Extension**: Composition over inheritance for HF Trainer
- **Dual-Loss Tracking**: Track LLM loss + soft expectation coordinate loss
- **Memory Optimization**: Gradient scaling and memory management
- **Callback System**: Clean loss logging and monitoring

**Main Classes**:
- `DistributedLossTrainer`: HF Trainer with built-in distributed synchronization
- `Trainer`: Legacy composition-based trainer
- `LossTracker`: Component-wise loss averaging

## 🔄 **Data Flow Architecture**

### **Training Pipeline**
```
1. Configuration Loading
   └── config/config.py → Config dataclass

2. Data Processing
   └── data/dataset.py → Raw samples → Teacher-student pairs
   └── processing/chat_processor.py → Conversations
   └── data/collator.py → Batched tensors

3. Model Processing  
   └── models/wrapper.py → DetectionModel forward pass
   └── models/loss_manager.py → Multi-component loss

4. Training Loop
   └── training/trainer.py → HF Trainer integration
   └── training/callbacks.py → Loss tracking & logging
```

### **Image Processing Pipeline**
```
Raw Sample → Load Images → Create Conversation → Expand Vision Tokens → 
Tokenize → Create Pixel Values → Collate Batch → Model Forward Pass
```

## 🎯 **Key Architectural Benefits**

### **1. Maintainability**
- **Single Responsibility**: Each module has one clear purpose
- **Composition Design**: Easy to test and modify individual components
- **Type Safety**: Comprehensive type annotations prevent runtime errors
- **Clear Interfaces**: Well-defined APIs between components

### **2. Reliability** 
- **Fail-Fast Validation**: Errors caught immediately with detailed messages
- **Comprehensive Testing**: 127 tests covering all components
- **Backward Compatibility**: Existing configs and data work unchanged
- **Production Ready**: Validated with real training scenarios

### **3. Performance**
- **Memory Efficiency**: Flash Attention compatible data collators
- **Vision Processing**: Proper image token/feature alignment
- **Multi-GPU Support**: DeepSpeed integration and distributed training
- **Optimized Batching**: Packed sequences and efficient padding

### **4. Extensibility**
- **Modular Design**: Easy to add new components
- **Clean Interfaces**: Well-defined extension points
- **Migration Support**: Gradual migration from old architecture
- **Future Proof**: Designed for long-term maintainability

## 🚀 **Usage Examples**

### **Basic Training**
```python
from src_new import Config, Dataset, create_data_collator
from src_new.models import DetectionModel
from src_new.training import Trainer

# Load configuration
config = Config.from_yaml("configs/bbu_v2.yaml")

# Create dataset and collator
dataset = Dataset(config.train_data_path, config=config)
collator = create_data_collator("standard", tokenizer)

# Create model and trainer
model = DetectionModel.from_pretrained(config.model_path, config=config)
trainer = Trainer(model=model, train_dataset=dataset, data_collator=collator)

# Start training
trainer.train()
```

### **Component Testing**
```python
# Test individual components
from src_new.models import LossManager, CoordinateProcessor
from src_new.processing import ChatProcessor

# Test loss computation
loss_manager = LossManager(config)
loss_components = loss_manager.compute_loss(outputs, labels)

# Test coordinate processing  
coord_processor = CoordinateProcessor(config)
coord_tokens = coord_processor.process_coordinates(bbox_data)

# Test conversation building
chat_processor = ChatProcessor(tokenizer, config)
conversation = chat_processor.build_conversation(sample_data)
```

This architecture provides a **robust, maintainable, and production-ready** foundation for Qwen2.5-VL training with clear separation of concerns and comprehensive validation at every level.

## 🔍 **Technical Deep Dive**

### **Configuration System Details**

The configuration system uses a **dataclass-based approach** with direct YAML mapping:

```python
@dataclass
class Config:
    # Required fields (no defaults) - must come first
    model_path: str
    model_size: str
    model_max_length: int

    # Optional fields with defaults
    coordinate_tokens_enabled: bool = True
    max_coord_value: int = 2048

    def __post_init__(self):
        # Fail-fast validation
        self._validate_model_settings()
        self._validate_training_settings()
```

**Key Features**:
- **Direct Mapping**: Every YAML field maps to a dataclass field
- **Type Validation**: Automatic type checking and conversion
- **Path Validation**: File/directory existence checking
- **Range Validation**: Numeric range and constraint checking

### **Data Processing Pipeline Details**

The data processing follows a **unified preprocessing approach**:

```python
# 1. Sample Structure Creation
raw_sample → structured_sample (teachers + student)

# 2. Conversation Building
structured_sample → conversation_text + images

# 3. Vision Token Expansion
conversation_text → expanded_conversation (with <|image_pad|> tokens)

# 4. Tokenization
expanded_conversation + images → input_ids + pixel_values + labels

# 5. Collation
individual_samples → batched_tensors (with proper padding)
```

**Vision Token Processing**:
- **Token Calculation**: Based on image grid dimensions and merge_size
- **Token Expansion**: Replace `<image>` with correct number of `<|image_pad|>` tokens
- **Feature Alignment**: Ensure token count matches pixel_values features

### **Model Architecture Details**

The model uses **composition over inheritance**:

```python
class DetectionModel(nn.Module):
    def __init__(self, base_model, config):
        self.base_model = base_model  # Qwen2.5-VL model
        self.loss_manager = LossManager(config)
        self.coord_processor = CoordinateProcessor(config)

    def forward(self, **inputs):
        # 1. Base model forward pass
        outputs = self.base_model(**inputs)

        # 2. Multi-component loss computation
        loss_components = self.loss_manager.compute_loss(outputs, inputs)

        # 3. Return structured output
        return ModelOutput(
            loss=loss_components.total_loss,
            logits=outputs.logits,
            loss_components=loss_components
        )
```

**Benefits**:
- **Clean Separation**: Each component has a single responsibility
- **Easy Testing**: Components can be tested independently
- **Flexible Configuration**: Enable/disable features dynamically
- **HF Compatibility**: Works seamlessly with HuggingFace Trainer

### **Training System Details**

The training system extends HuggingFace Trainer **minimally**:

```python
class DistributedLossTrainer(HFTrainer):
    def __init__(self, model, tokenizer, training_args, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, args=training_args, **kwargs)
        self.loss_tracker = LossTracker()

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        # Call parent for standard loss logging
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate)

        # Add distributed loss component synchronization
        self._log_distributed_loss_components()

    def _log_distributed_loss_components(self):
        # Use HuggingFace's proven _nested_gather() for synchronization
        gathered_tensor = self._nested_gather(local_tensor)
        global_average = gathered_tensor.mean().item()
```

**Key Features**:
- **Minimal Extension**: Only adds necessary functionality
- **Component Tracking**: Track coordinate, teacher, student losses separately
- **Memory Optimization**: Gradient scaling and efficient batching
- **Monitoring**: Comprehensive loss and performance tracking

## 📊 **Performance Characteristics**

### **Memory Efficiency**
- **Flash Attention**: LEFT padding for optimal attention computation
- **Packed Sequences**: Remove padding for memory savings
- **Vision Processing**: Efficient image token/feature alignment
- **Gradient Scaling**: Automatic mixed precision support

### **Training Speed**
- **Multi-GPU**: DeepSpeed integration for distributed training
- **Efficient Collation**: Optimized batching and padding strategies
- **Lazy Loading**: On-demand data loading and processing
- **Caching**: Teacher pool and tokenization caching

### **Scalability**
- **Large Sequences**: Support for 120K+ token sequences
- **Multi-Image**: Handle multiple images per conversation
- **Batch Optimization**: Dynamic batching based on sequence length
- **Resource Management**: Automatic memory and compute optimization

## 🧪 **Testing Strategy**

### **Test Coverage**
- **Unit Tests**: Individual component validation (85 tests)
- **Integration Tests**: End-to-end pipeline validation (25 tests)
- **Regression Tests**: Backward compatibility validation (17 tests)
- **Performance Tests**: Speed and memory benchmarking

### **Test Categories**
```
src_new/tests/
├── test_config/      # Configuration validation tests
├── test_data/        # Data processing pipeline tests
├── test_models/      # Model component tests
├── test_processing/  # Processing component tests
├── test_training/    # Training system tests
├── test_integration/ # End-to-end integration tests
└── test_regression/  # Backward compatibility tests
```

### **Validation Approach**
- **Fail-Fast Testing**: Immediate error detection
- **Mock Objects**: Isolated component testing
- **Real Data Testing**: Validation with actual training data
- **Performance Benchmarking**: Speed and memory usage validation

This comprehensive architecture ensures **reliability, maintainability, and performance** for production Qwen2.5-VL training scenarios.

## 🔄 **Migration from Old Architecture**

### **Key Differences**

| Aspect | Old Architecture (`src/`) | New Architecture (`src_new/`) |
|--------|---------------------------|-------------------------------|
| **Design Pattern** | Inheritance-heavy | Composition-based |
| **Configuration** | Multiple config files | Single unified Config class |
| **Error Handling** | Silent failures | Fail-fast validation |
| **Testing** | Limited test coverage | Comprehensive test suite (127 tests) |
| **Type Safety** | Minimal type hints | Full type annotations |
| **Modularity** | Tightly coupled | Loosely coupled components |
| **Maintenance** | Complex inheritance chains | Clear component boundaries |

### **Migration Path**

#### **1. Configuration Migration**
```python
# Old approach (multiple configs)
from src.config.global_config import DirectConfig
from src.config.training_config import TrainingConfig

# New approach (unified config)
from src_new import Config
config = Config.from_yaml("configs/bbu_v2.yaml")  # Same YAML file!
```

#### **2. Dataset Migration**
```python
# Old approach
from src.data import BBUDataset

# New approach
from src_new import Dataset
dataset = Dataset(data_path, config=config)  # Simplified interface
```

#### **3. Model Migration**
```python
# Old approach (inheritance)
from src.models.qwen2_5_vl_wrapper import Qwen25VLWrapper

# New approach (composition)
from src_new.models import DetectionModel
model = DetectionModel.from_pretrained(model_path, config=config)
```

#### **4. Training Migration**
```python
# Old approach
from src.training.training_coordinator import TrainingCoordinator

# New approach
from src_new.training import Trainer
trainer = Trainer(model=model, train_dataset=dataset, ...)
```

### **Direct Usage**

The new architecture provides a clean, direct interface:

```python
from src_new.models import DetectionModel

# Create model directly
model = DetectionModel.from_pretrained(model_path, config=config)
```

### **Configuration Requirements**

1. **Phase 1**: Test new architecture with existing configs
2. **Phase 2**: Migrate data processing pipeline
3. **Phase 3**: Migrate model components
4. **Phase 4**: Migrate training system
5. **Phase 5**: Full cutover to new architecture

## 🎯 **Production Deployment**

### **Entry Points**

#### **Training Script**
```bash
# New training script with full src_new support
python scripts/train_new.py --config bbu_v2 --log_level INFO

# Multi-GPU training with DeepSpeed
BBU_DEEPSPEED_ENABLED=true ./scripts/run_train_new.sh
```

#### **Validation and Testing**
```bash
# Configuration validation
python scripts/train_new.py --config bbu_v2 --validate-only

# Trainer creation test
python scripts/train_new.py --config bbu_v2 --test-trainer

# Full test suite
python -m pytest src_new/tests/ -v
```

### **Monitoring and Debugging**

#### **Loss Component Tracking**
```python
# Access detailed loss information
outputs = model(**batch)
loss_components = outputs.loss_components

print(f"Total Loss: {loss_components.total_loss}")
print(f"Coordinate Loss: {loss_components.coordinate_loss}")
print(f"Teacher Loss: {loss_components.teacher_loss}")
print(f"Student Loss: {loss_components.student_loss}")
```

#### **Debug Mode**
```bash
# Enable debug logging for detailed information
python scripts/train_new.py --config bbu_v2 --log_level DEBUG
```

### **Performance Optimization**

#### **Memory Optimization**
- Use `PackedDataCollator` for memory-efficient training
- Enable gradient checkpointing for large models
- Use DeepSpeed ZeRO for distributed training

#### **Speed Optimization**
- **FlashAttention v2**: 5x faster on long sequences (22,876 vs 4,589 tok/s)
  - See [FlashAttention v2 Setup Guide](../docs/implementation/flashattention-v2-setup.md)
  - Automatic PyTorch 2.5.1 compatibility patches applied
- **Mixed Precision**: Enable automatic mixed precision training
- **DataLoader**: Optimize workers and prefetch factor

### **Production Checklist**

- ✅ **Configuration Validated**: All required fields present and valid
- ✅ **Data Pipeline Tested**: Dataset loading and processing working
- ✅ **Model Loading Verified**: Model loads and creates trainer successfully
- ✅ **Loss Computation Working**: Multi-component loss calculation correct
- ✅ **Memory Usage Acceptable**: Training fits within available GPU memory
- ✅ **Performance Benchmarked**: Training speed meets requirements
- ✅ **Monitoring Enabled**: Loss tracking and logging configured
- ✅ **Error Handling Tested**: Fail-fast validation catches issues early

## 📚 **Additional Resources**

### **Documentation**
- `src_new/models/README.md`: Detailed model architecture documentation
- `src_new/tests/`: Comprehensive test examples and usage patterns
- `configs/bbu_v2.yaml`: Reference configuration with all parameters

### **Examples**
- `scripts/train_new.py`: Production training script
- `src_new/tests/test_integration/`: End-to-end usage examples
- `src_new/tests/fixtures/`: Mock objects for testing

### **Support**
- **Issue Tracking**: Use fail-fast validation error messages for debugging
- **Performance Monitoring**: Component-wise loss tracking and memory usage
- **Testing**: Run full test suite to validate changes

This architecture provides a **complete, production-ready solution** for Qwen2.5-VL training with comprehensive documentation, testing, and migration support.
