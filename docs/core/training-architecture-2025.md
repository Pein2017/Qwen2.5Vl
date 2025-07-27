# BBU Training Architecture - 2025 Modular Edition

## Overview

The BBU training system has been completely refactored into a modular architecture with clear separation of concerns. This document outlines the current (2025) training architecture with its modular components.

## 🎯 Key Principles (2025 Refactoring)

### 1. **Modular Architecture**
- Separated training components into `src/training/` module
- Clear component boundaries and responsibilities
- Independent testing and debugging capabilities

### 2. **Unified Model Loading**
- Single model loader (`src/models/model_loader.py`) for training and inference consistency
- Automatic token initialization and embedding resizing
- Proper DeepSpeed integration with detection capabilities

### 3. **Multi-Geometry Support**
- Native support for `bbox_2d`, `square`, and `line` geometries
- Object-oriented training with flexible equipment type combinations
- Robust validation with clear error messages

### 4. **Simplified Token Management**
- Lightweight token addition using ms-swift inspired approach
- Standard HuggingFace methods instead of complex custom implementations
- Fail-fast validation instead of silent error handling

## 🏗️ Modular Architecture Components

### Training System (`src/training/`)

#### BBUTrainer (`trainer.py`)
**Enhanced HuggingFace Trainer** with multi-component loss logging and robust validation.

```python
from src.training.trainer import BBUTrainer

trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    cfg=config,
    image_processor=image_processor,
    training_coordinator=coordinator  # Optional
)
```

**Key Features**:
- Multi-component loss tracking (LLM + coordinate losses)
- Teacher-student learning support
- Robust logging with per-step component breakdown
- Integration with TrainingCoordinator for advanced orchestration

#### TrainingCoordinator (`training_coordinator.py`)
**Training orchestration** for multi-task learning coordination.

```python
from src.training.training_coordinator import TrainingCoordinator

coordinator = TrainingCoordinator(
    model=model,
    tokenizer=tokenizer,
    config_obj=config
)
coordinator.setup_training()
```

**Responsibilities**:
- Multi-task training coordination
- Training state management and recovery
- Component-wise training control and monitoring
- Integration with domain-specific configurations

#### LossManager (`loss_manager.py`)
**Multi-task loss computation** with clean separation of LLM and coordinate losses.

```python
from src.training.loss_manager import LossManager

loss_manager = LossManager(
    tokenizer=tokenizer,
    model=model,
    teacher_loss_weight=0.3,
    student_loss_weight=1.0
)
```

**Loss Components**:
- **LLM Loss**: Standard language modeling loss
- **Coordinate L1 Loss**: L1 loss for coordinate accuracy
- **Teacher-Student Splitting**: Proportional loss allocation based on token counts

**New Tokens Added**:
- `<|line_start|>`, `<|line_end|>` - For line geometries (fiber, wire)
- `<|square_start|>`, `<|square_end|>` - For square geometries (labels, equipment)

**Existing Tokens Reused**:
- `<|object_ref_start|>`, `<|object_ref_end|>` - For description wrapping
- `<|box_start|>`, `<|box_end|>` - For bbox_2d geometries

**Token Wrapping Examples**:
```
<|box_start|>100, 200, 300, 400<|box_end|> <|object_ref_start|>BBU设备/华为,显示完整<|object_ref_end|>
<|square_start|>150, 10, 211, 35, 218, 16, 166, 0<|square_end|> <|object_ref_start|>标签/5G-BBU<|object_ref_end|>
<|line_start|>579, 1385, 679, 1451, 764, 1444<|line_end|> <|object_ref_start|>光纤/有保护措施<|object_ref_end|>
```

### Data Processing (V2 Multi-Geometry)

**Primary Components**:
- `src/data.py` - Dataset with multi-geometry validation
- `src/chat_processor.py` - Conversation building with token wrapping

**Supported Geometries**:
1. **bbox_2d**: `[x1, y1, x2, y2]` - Rectangular bounding boxes
2. **square**: `[x1, y1, x2, y2, x3, y3, x4, y4]` - Quadrilateral shapes
3. **line**: `[x1, y1, x2, y2, ..., xn, yn]` - Multi-point lines

**Data Format**:
```json
{
  "images": ["image.jpg"],
  "objects": [
    {"bbox_2d": [237, 100, 348, 673], "desc": "BBU设备/华为,显示完整,无需安装"},
    {"square": [283, 198, 300, 539, 532, 545, 523, 182], "desc": "标签/5G-BBU-（接地线）"},
    {"line": [579, 1385, 679, 1451, 764, 1444], "desc": "光纤/有遮挡,有保护措施"}
  ],
  "width": 532,
  "height": 728
}
```

### Complete Training Pipeline (2025)

**Entry Point**: `scripts/train.py` → `src/training/trainer_factory.py`

**Training Flow**:
```python
# 1. Configuration and model setup
config = get_config()
model, tokenizer, image_processor = load_model_and_processor_unified(...)

# 2. Dataset creation via DataProcessor
data_processor = DataProcessor(tokenizer, image_processor, model)
train_dataset, eval_dataset = data_processor.create_datasets()
data_collator = data_processor.create_data_collator()

# 3. Training coordinator setup
coordinator = TrainingCoordinator(model=model, tokenizer=tokenizer, config_obj=config)
coordinator.setup_training()

# 4. Trainer creation and training
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    cfg=config,
    image_processor=image_processor,
    training_coordinator=coordinator
)
trainer.train()
```

### Configuration System (`src/config/`)

**Unified Configuration Access**:
```python
from src.config import get_config
config = get_config()  # Returns DirectConfig instance
```

**Key Settings**:
```yaml
# Detection and Token System
detection_enabled: true
coordinate_tokens_enabled: true

# Model Configuration
model_path: "/path/to/Qwen2.5-VL-7B-Instruct"
model_max_length: 120000
attn_implementation: "flash_attention_2"

# Training Configuration
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_ratio: 0.3
```

## 🔧 Implementation Details

### Simple Token Integration

**Model Level**:
1. Unified loader calls `create_simple_token_manager(tokenizer, model)`
2. Adds 4 new special tokens to tokenizer
3. Resizes model embeddings with `model.resize_token_embeddings()`
4. Stores token manager on model for later use

**Chat Processor Level**:
1. DataProcessor passes model to ChatProcessor
2. ChatProcessor calls `initialize_simple_tokens(model)` 
3. Automatic token wrapping in `_format_objects_response()`
4. Semantic tokens based on geometry type

### Multi-Geometry Handling

**Validation Strategy**:
- Fail-fast on malformed data with detailed error messages
- No silent handling or normalization of bad coordinates
- Clear attribution of which object/sample caused issues

**Ground Truth Conversion**:
- All geometries converted to bounding boxes for loss computation
- Original coordinates preserved in training format with semantic tokens
- Degenerate cases (zero-width lines) handled with 1-pixel minimum

### Error Handling Philosophy

**No Silent Failures**:
```python
# Old approach (bad)
try:
    process_coordinates(coords)
except:
    coords = default_coords  # Silent failure

# New approach (good) 
if not validate_coordinates(coords):
    raise ValueError(f"Invalid coordinates {coords} in object {obj}. This indicates a data preprocessing issue.")
```

## 🚀 Training Execution

### Launch Training

```bash
./scripts/run_train.sh
```

**Expected Log Output**:
```
INFO - training - 🤖 Loading model...
INFO - training - ✅ Simple Token Manager initialized successfully!
INFO - training - 🎯 CHAT PROCESSOR CONFIG: simple_enabled=True
INFO - training - ✅ Chat processor initialized with SIMPLE tokens
INFO - training - 📊 Creating datasets...
INFO - training - ✅ All tests passed! Found N training objects
```

### Configuration Check

Verify your config has:
```yaml
simple_tokens_enabled: true
coordinate_tokens_enabled: false  
```

### Data Validation

The system will fail immediately on:
- Missing geometry fields (`bbox_2d`, `square`, or `line`)
- Missing description fields (`desc`)
- Invalid coordinate formats
- Out-of-bounds coordinates
- Multiple geometry types in same object

## 📊 Benefits of Current Architecture

### 1. **Reliability**
- Fail-fast validation catches data issues immediately  
- No silent handling of malformed inputs
- Training-inference consistency via unified loader

### 2. **Simplicity** 
- 100 lines vs 1000+ lines for token management
- Standard HuggingFace infrastructure
- Clear separation of concerns

### 3. **Extensibility**
- Easy to add new geometry types
- Simple token addition process
- Modular component design

### 4. **Performance**
- Semantic tokens enable better geometry understanding
- Multi-geometry support without performance penalty
- Efficient token wrapping and validation

## 🔄 Migration Notes

### From Legacy System

**Old (Deprecated)**:
- Complex coordinate token system with soft expectation regression
- Monolithic trainer with embedded token management
- Silent error handling and coordinate normalization
- Single geometry type (bbox_2d only)

**New (Current)**:
- Simple token manager with semantic wrapping
- Unified model loader with automatic token initialization  
- Fail-fast validation with detailed error messages
- Multi-geometry support (bbox_2d, square, line)

### Breaking Changes

1. **Configuration**: Must set `simple_tokens_enabled: true`
2. **Data Format**: Must provide proper geometry fields for each object
3. **Error Handling**: Data issues will cause immediate failures instead of silent handling
4. **Model Loading**: Must use unified loader instead of wrapper

The current architecture provides a robust, maintainable foundation for BBU object detection training with multi-geometry support and reliable token handling.