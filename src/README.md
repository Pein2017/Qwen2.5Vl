# 📁 BBU Training Source Code Structure

*Updated for refactored modular architecture*

## 🏗️ Directory Organization

### Core Components (`core/`)
Central factory classes and managers for the training system:
- **`model_factory.py`** - Centralized model creation and configuration
- **`data_processor.py`** - Unified data processing and dataset creation  
- **`checkpoint_manager.py`** - Model saving/loading and checkpoint management

### Configuration (`config/`)
Dual configuration system supporting both legacy and new approaches:
- **`global_config.py`** - Legacy DirectConfig system (flat 149+ parameters)
- **`domain_configs.py`** - **NEW** Domain-specific config classes
- **`config_manager.py`** - **NEW** Config validation and cross-domain dependencies

### Training System (`training/`)
Modular training components extracted from monolithic trainer:
- **`trainer.py`** - Enhanced BBUTrainer with optional coordinator integration
- **`training_coordinator.py`** - **NEW** Training orchestration and state management
- **`loss_manager.py`** - **NEW** Multi-task loss computation (LM + detection)
- **`parameter_manager.py`** - **NEW** Parameter grouping for differential learning rates
- **`trainer_factory.py`** - **NEW** Factory functions for trainer creation
- **`callbacks.py`** - Training callbacks and monitoring
- **`stability.py`** - Training stability utilities

### Object Detection (`detection/`)
DETR-style object detection components:
- **`detection_head.py`** - DETR decoder with dual-stream processing
- **`detection_loss.py`** - Hungarian matching and multi-task loss computation
- **`detection_adapter.py`** - Vision and language adapters

### Models (`models/`)
Model architecture and integration:
- **`model_loader.py`** - **NEW** Unified model loader for training-inference consistency
- **`wrapper.py`** - Legacy Qwen2.5-VL wrapper with coordinate tokens (deprecated)
- **`patches.py`** - Model patches and optimizations

### Utilities (`utils/`)
Support utilities and helper functions:
- **`utils.py`** - General utilities (JSONL, tensor debugging, etc.)
- **`prompt.py`** - Prompt templates and conversation formatting
- **`response_parser.py`** - Output parsing and validation
- **`schema.py`** - Type definitions and validation schemas
- **`simple_token_manager.py`** - **NEW** Simple token manager (ms-swift approach)
- **`coordinate_token_manager.py`** - Legacy coordinate token system (deprecated)
- **`tokens/`** - Special token definitions and handling

### Legacy (`legacy/`)
Preserved old implementations for reference:
- **`losses_old.py`** - Original loss implementation (reference only)
- **`lr_scaling.py`** - Token-length-aware LR scaling (reference only)
- **`rope2d.py`** - RoPE 2D position encoding (moved to patches)
- **`attention_backup.py`** - Flash attention backup utilities

### Reference (`reference/`)
Official reference implementations (unchanged):
- **`official_huggingface_qwen2_5_vl/`** - Official HF Qwen2.5-VL code
- **`qwen2_5vl_collator.py`** - Reference collator implementation

### Root Level
Core data and processing modules:
- **`data.py`** - **V2 MIGRATED** BBUDataset with complete multi-geometry support (bbox_2d, square, line)
- **`chat_processor.py`** - **V2 MIGRATED** Conversation building with simple token integration and hierarchical descriptions
- **`teacher_pool.py`** - Teacher demonstration management
- **`inference.py`** - Production inference with Flash Attention 2
- **`logger_utils.py`** - Advanced logging and monitoring

## 🔄 Migration Status: V1 → V2 **[COMPLETE]**

### V1 Legacy Data Format (Deprecated)  
```json
{
  "teachers": [...],
  "student": {
    "objects": [{"bbox_2d": [...], "description": "..."}]
  }
}
```

### V2 Current Data Format **[MIGRATED]**
```json
{
  "images": ["images/..."],
  "objects": [
    {"bbox_2d": [...], "desc": "hierarchical/description,format"},
    {"square": [...], "desc": "..."},
    {"line": [...], "desc": "..."}
  ],
  "width": 532, "height": 728
}
```

### Architecture Evolution
```
src/
├── core/ (🆕 Central factories and managers)
├── config/ (Enhanced with domain-specific configs)
├── training/ (Modular components extracted)
├── utils/ (🆕 V2 simple token system, coordinate managers)
├── legacy/ (🆕 Preserved old implementations)
└── [V2-compatible root modules]
```

## 🎯 Current Training Architecture (2025)

### Token System - Simple Token Approach (ms-swift inspired)

**NEW Simple Token Manager:**
- **Philosophy**: Lightweight token addition using standard HuggingFace infrastructure
- **Approach**: `tokenizer.add_special_tokens()` + `model.resize_token_embeddings()`
- **New Tokens**: `<|line_start|>`, `<|line_end|>`, `<|square_start|>`, `<|square_end|>`
- **Reused Tokens**: `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`

```python
# Configuration
simple_tokens_enabled: true      # Enable simple token system
coordinate_tokens_enabled: false # Disable legacy system

# Expected Training Format
"<|box_start|>100, 200, 300, 400<|box_end|> <|object_ref_start|>BBU设备<|object_ref_end|>"
"<|square_start|>150, 10, 211, 35, 218, 16, 166, 0<|square_end|> <|object_ref_start|>标签<|object_ref_end|>"
"<|line_start|>579, 1385, 679, 1451, 764, 1444<|line_end|> <|object_ref_start|>光纤<|object_ref_end|>"
```

### Data Format - V2 Multi-Geometry Support **[MIGRATED]**

**V2 Multi-Geometry Objects with Object-Oriented Training:**
```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {"bbox_2d": [264, 144, 326, 201], "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"},
    {"square": [704, 487, 670, 554, 973, 644, 993, 590], "desc": "标签/4G-RRU3-光纤"},
    {"line": [614, 1271, 498, 1179, 419, 1216, 280, 1280, 117, 1456, 3, 1721], "desc": "光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管"}
  ],
  "width": 532,
  "height": 728
}
```

**Object Types Supported (6 total):**
- **Equipment**: `bbu` (BBU设备), `bbu_shield` (挡风板) → `bbox_2d`/`square` geometry
- **Hardware**: `connect_point` (螺丝、光纤插头), `label` (标签) → `bbox_2d`/`square` geometry  
- **Cables**: `fiber` (光纤), `wire` (电线) → `line` geometry

**Validation Strategy**: Fail-fast with detailed error messages - no silent handling of data issues

### Model Loading - Unified Loader

**Unified Model Loader:**
```python
from src.models.model_loader import load_model_and_processor_unified

model, tokenizer, image_processor = load_model_and_processor_unified(
    config,
    for_inference=False,
    deepspeed_enabled=True
)
```

**Benefits:**
- Training-inference consistency
- Automatic simple token initialization
- Detection vs non-detection model handling
- DeepSpeed compatibility

## 🚀 Usage Patterns

### Creating Training Components

**Current System (2025):**
```python
from src.training.trainer_factory import create_trainer_with_coordinator
trainer = create_trainer_with_coordinator(training_args)
# Uses unified model loader + simple tokens automatically
```

### Simple Token Manager Usage

**Initialization (automatic in training):**
```python
# Automatically initialized by unified model loader
from src.utils.simple_token_manager import create_simple_token_manager
token_manager = create_simple_token_manager(tokenizer, model)
```

**Token Wrapping:**
```python
# Automatic wrapping in ChatProcessor
manager.wrap_coordinates([150, 10, 211, 35], "bbox_2d")
# → "<|box_start|>150.0, 10.0, 211.0, 35.0<|box_end|>"

manager.wrap_description("BBU设备/华为,显示完整")  
# → "<|object_ref_start|>BBU设备/华为,显示完整<|object_ref_end|>"
```

### Data Processing

**Current System:**
```python
from src.core import DataProcessor
processor = DataProcessor(tokenizer, image_processor, model)
train_dataset, eval_dataset = processor.create_datasets()
data_collator = processor.create_data_collator()
```

### Configuration Access

**Current System:**
```python
from src.config import get_config
config = get_config()
learning_rate = config.llm_lr
model_path = config.model_path
simple_tokens_enabled = config.simple_tokens_enabled
```

## 🎯 Benefits of New Structure

### 1. **Maintainability**
- Clear separation of concerns
- Modular components instead of monolithic classes
- Single responsibility principle throughout

### 2. **Testability**
- Isolated components easy to unit test
- Clean interfaces and dependencies
- Mockable factory methods

### 3. **Extensibility**
- Easy to add new components
- Clean plugin architecture
- Backward compatible design

### 4. **Organization**
- Logical grouping of related functionality
- Reduced import complexity
- Clear dependency hierarchy

### 5. **Configuration Management**
- Domain-specific validation
- Cross-config dependency checking
- Better error messages and fail-fast validation

## 🔧 Development Guidelines

### Adding New Components
1. Place in appropriate domain folder (`core/`, `training/`, `detection/`, etc.)
2. Use factory pattern for complex object creation
3. Support both legacy and new config systems during transition
4. Add comprehensive logging and error handling

### Import Guidelines
- Use absolute imports: `from src.core import ModelFactory`
- Organize imports by domain
- Prefer factory methods over direct instantiation
- Keep circular dependencies minimal

### Testing
- Test each component in isolation
- Use factory methods for test object creation
- Mock external dependencies
- Validate both legacy and new config systems

## 📊 Metrics and Monitoring

The refactored system provides enhanced monitoring:
- Component-wise parameter statistics
- Training coordinator status summaries
- Loss manager component tracking
- Configuration validation reports
- Checkpoint integrity validation

This modular architecture provides a solid foundation for continued development while maintaining full backward compatibility with existing workflows.