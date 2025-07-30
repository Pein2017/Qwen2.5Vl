# 📁 BBU Training Source Code Structure

*Updated for production-ready refactored architecture (2025)*

## 🏗️ Directory Organization

### Configuration (`config/`)
Unified configuration system with validation:
- **`__init__.py`** - Configuration loading and global access
- **`config.py`** - Main configuration schema and validation

### Core Components (`core/`)
Central processing and management components:
- **`data_processor.py`** - Unified data processing and dataset creation
- **`checkpoint_manager.py`** - Model saving/loading and checkpoint management

### Training System (`training/`)
Modular training components for multi-task learning:
- **`loss_manager.py`** - Multi-component loss computation (LLM + coordinate losses)
- **`training_coordinator.py`** - Training orchestration and state management
- **`training_state_manager.py`** - Training state and metrics management
- **`base_manager.py`** - Common manager functionality
- **`trainer.py`** - Enhanced BBUTrainer with coordinate support
- **`trainer_factory.py`** - Factory functions for trainer creation
- **`callbacks.py`** - Training callbacks and monitoring
- **`stability.py`** - Training stability utilities

### Models (`models/`)
Model architecture and coordinate token integration:
- **`model_loader.py`** - Unified model loader for training-inference consistency
- **`wrapper.py`** - Qwen2.5-VL wrapper with coordinate token support
- **`coordinate_handler.py`** - Coordinate token handling and management
- **`detection_integration.py`** - Detection capabilities integration
- **`model_adapter.py`** - Model adaptation utilities
- **`loss_manager.py`** - Model-level loss computation
- **`patches.py`** - Qwen2.5-VL compatibility patches

### Utilities (`utils/`)
Support utilities and helper functions:
- **`utils.py`** - General utilities (JSONL, tensor debugging, etc.)
- **`data_utils.py`** - Data processing utilities and response parser
- **`model_utils.py`** - Model utilities and helpers
- **`training_utils.py`** - Training utilities and helpers
- **`prompt.py`** - Prompt templates and conversation formatting
- **`response_parser.py`** - Output parsing and validation (compatibility shim)
- **`schema.py`** - Type definitions and validation schemas
- **`tokens/`** - Unified token management system

### Reference (`reference/`)
Official reference implementations (unchanged):
- **`official_huggingface_qwen2_5_vl/`** - Official HF Qwen2.5-VL code
- **`qwen2_5vl_collator.py`** - Reference collator implementation

### Root Level
Core data and processing modules:
- **`data.py`** - BBUDataset with complete multi-geometry support (bbox_2d, square, line)
- **`chat_processor.py`** - Conversation building with coordinate token integration
- **`teacher_pool.py`** - Teacher demonstration management
- **`inference.py`** - Production inference with multi-geometry support
- **`logger_utils.py`** - Advanced logging and monitoring

## 🎯 Current Architecture (2025)

### Production-Ready Features
- ✅ **Unified Configuration System** with proper validation
- ✅ **Coordinate Token Support** (Standard + Coordinate modes)
- ✅ **Multi-Geometry Processing** (bbox_2d, square, line)
- ✅ **Teacher-Student Learning** with span-based loss distribution
- ✅ **Multi-GPU Training** with synchronized custom losses
- ✅ **Training-Inference Consistency** through unified model loader

### Data Format - Multi-Geometry Support
```json
{
  "images": ["images/example.jpg"],
  "objects": [
    {"bbox_2d": [100, 200, 300, 400], "desc": "BBU基带处理单元"},
    {"square": [150, 10, 211, 35, 218, 16, 166, 0], "desc": "标签/5G-BBU"},
    {"line": [260, 52, 219, 31, 173, 6], "desc": "光纤连接"}
  ],
  "width": 532,
  "height": 728
}
```

### Architecture Overview
```
src/
├── config/          # Unified configuration management
├── core/            # Central processing components
├── training/        # Modular training system
├── models/          # Model loading and coordinate integration
├── utils/           # Utilities and token management
├── reference/       # Official reference implementations
└── [Root modules]   # Core data and processing
```

## 🔧 Coordinate Token System

### Two Operational Modes

**Standard Mode** (Recommended for Production):
- **Coordinates**: Integer format `[150,10,211,35]`
- **Vocabulary**: Minimal extension (+8 geometry tokens)
- **Configuration**: `coordinate_tokens_enabled: false`
- **Use Case**: Production training with stable performance

**Coordinate Mode** (Advanced Features):
- **Coordinates**: Token format `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]`
- **Vocabulary**: Extended (+2056 coordinate tokens)
- **Configuration**: `coordinate_tokens_enabled: true`
- **Use Case**: Advanced sequence-based coordinate prediction

### Token Format Examples

**Standard Mode:**
```
"图像中有一个BBU基带处理单元，位置为<|box_start|>100,200,300,400<|box_end|>。"
"方形设备位置为<|square_start|>100,200,300,200,300,400,100,400<|square_end|>。"
"线性连接路径为<|line_start|>100,200,150,250,200,300<|line_end|>。"
```

**Coordinate Mode:**
```
"图像中有一个BBU基带处理单元，位置为<|box_start|><|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>。"
```

### Supported Geometry Types
- **bbox_2d**: `[x1, y1, x2, y2]` - Rectangular bounding boxes
- **square**: `[x1, y1, x2, y2, x3, y3, x4, y4]` - Quadrilateral shapes
- **line**: `[x1, y1, x2, y2, ..., xN, yN]` - Multi-point line segments

### Object Categories
- **Equipment**: BBU设备, 挡风板 → `bbox_2d`/`square` geometry
- **Hardware**: 螺丝、光纤插头, 标签 → `bbox_2d`/`square` geometry
- **Cables**: 光纤, 电线 → `line` geometry

## 🚀 Usage Patterns

### Model Loading

**Unified Model Loader:**
```python
from src.models.model_loader import load_model_and_processor_unified

model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path=config.model_path,
    for_inference=False,
    config=config
)
```

**Benefits:**
- Training-inference consistency
- Automatic coordinate token initialization
- Detection vs non-detection model handling
- Vocabulary extension management

### Training Setup

**Creating Training Components:**
```python
from src.training.trainer_factory import create_trainer_with_coordinator

trainer = create_trainer_with_coordinator(
    model=model,
    tokenizer=tokenizer,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
```

### Data Processing

**ChatProcessor Usage:**
```python
from src.chat_processor import ChatProcessor

chat_processor = ChatProcessor(
    tokenizer=tokenizer,
    image_processor=image_processor,
    config=config
)

# Process training sample
result = chat_processor.process_sample(sample_data)
```

### Configuration Access

**Global Configuration:**
```python
from src.config import load_config, get_config

# Load configuration
config = load_config('configs/bbu_v2.yaml')

# Access global config
config = get_config()
learning_rate = config.learning_rate
model_path = config.model_path
coordinate_tokens_enabled = config.coordinate_tokens_enabled
```

## 🎯 Benefits of Refactored Architecture

### 1. **Production Readiness**
- Unified configuration system with validation
- Consolidated coordinate token management
- Correct teacher-student loss weighting
- Multi-GPU training support with synchronized losses

### 2. **Maintainability**
- Clear separation of concerns
- Modular components instead of monolithic classes
- Single responsibility principle throughout
- Comprehensive test coverage

### 3. **Flexibility**
- Support for both Standard and Coordinate token modes
- Multi-geometry processing (bbox_2d, square, line)
- Training-inference consistency
- Easy configuration switching

### 4. **Performance**
- Optimized loss computation
- Efficient coordinate token handling
- Memory-efficient data processing
- Flash Attention 2 support

## 🔧 Development Guidelines

### Adding New Components
1. Place in appropriate domain folder (`core/`, `training/`, `models/`, etc.)
2. Follow the established patterns for managers and factories
3. Add comprehensive logging and error handling
4. Include proper type annotations and validation

### Import Guidelines
- Use absolute imports: `from src.training import LossManager`
- Organize imports by domain
- Prefer factory methods over direct instantiation
- Keep circular dependencies minimal

### Testing
- Test each component in isolation
- Use comprehensive test coverage in `tests/` directory
- Mock external dependencies appropriately
- Validate configuration systems thoroughly

## 📊 Monitoring and Debugging

The refactored system provides enhanced monitoring:
- **Loss Tracking**: Component-wise loss computation and tracking
- **Training Coordination**: Multi-task training state management
- **Configuration Validation**: Comprehensive config validation and reporting
- **Token Management**: Coordinate token consistency validation
- **Performance Metrics**: Training performance and stability monitoring

This production-ready architecture provides a solid foundation for BBU training with coordinate token support while maintaining clean, maintainable code.