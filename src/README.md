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
- **`wrapper.py`** - Qwen2.5-VL wrapper with detection capabilities
- **`patches.py`** - Model patches and optimizations

### Utilities (`utils/`)
Support utilities and helper functions:
- **`utils.py`** - General utilities (JSONL, tensor debugging, etc.)
- **`prompt.py`** - Prompt templates and conversation formatting
- **`response_parser.py`** - Output parsing and validation
- **`schema.py`** - Type definitions and validation schemas
- **`tokens/`** - Special token definitions and handling

### Legacy (`legacy/`)
Preserved old implementations for reference:
- **`losses_old.py`** - Original loss implementation (reference only)
- **`lr_scaling.py`** - Token-length-aware LR scaling (deprecated)
- **`rope2d.py`** - RoPE 2D position encoding (moved to patches)
- **`attention_backup.py`** - Flash attention backup utilities

### Reference (`reference/`)
Official reference implementations (unchanged):
- **`official_huggingface_qwen2_5_vl/`** - Official HF Qwen2.5-VL code
- **`qwen2_5vl_collator.py`** - Reference collator implementation

### Root Level
Core data and processing modules:
- **`data.py`** - BBUDataset and data collators
- **`chat_processor.py`** - Conversation building and tokenization
- **`teacher_pool.py`** - Teacher demonstration management
- **`inference.py`** - Production inference with Flash Attention 2
- **`logger_utils.py`** - Advanced logging and monitoring

## 🔄 Migration from Legacy Structure

### Before (Legacy)
```
src/
├── config/global_config.py (2100+ line monolithic config)
├── training/trainer.py (2100+ line monolithic trainer)
├── detection_loss.py
├── models/detection_head.py
├── utils.py (mixed utilities)
└── [various scattered files]
```

### After (Refactored)
```
src/
├── core/ (🆕 Central factories and managers)
├── config/ (Enhanced with domain-specific configs)
├── training/ (Modular components extracted)
├── detection/ (🆕 Organized detection components)
├── utils/ (🆕 Organized utility modules)
├── legacy/ (🆕 Preserved old implementations)
└── [clean root-level modules]
```

## 🚀 Usage Patterns

### Creating Training Components

**Legacy System:**
```python
from src.training.trainer import create_trainer
trainer = create_trainer(training_args)
```

**New System:**
```python
from src.training.trainer_factory import create_trainer_with_coordinator
trainer = create_trainer_with_coordinator(training_args, use_new_config=True)
```

### Factory Pattern Usage

**Model Creation:**
```python
from src.core import ModelFactory
factory = ModelFactory(use_new_config=True)
model = factory.create_model()
tokenizer, image_processor = factory.create_tokenizer_and_processor()
```

**Data Processing:**
```python
from src.core import DataProcessor
processor = DataProcessor(tokenizer, image_processor, use_new_config=True)
train_dataset, eval_dataset = processor.create_datasets()
data_collator = processor.create_data_collator()
```

**Checkpoint Management:**
```python
from src.core import CheckpointManager
checkpoint_manager = CheckpointManager(use_new_config=True)
success = checkpoint_manager.save_model_safely(trainer, output_dir)
```

### Configuration Access

**Legacy:**
```python
from src.config import config
learning_rate = config.llm_lr
model_path = config.model_path
```

**New Domain-Specific:**
```python
from src.config import get_config_manager
config_manager = get_config_manager()
learning_rate = config_manager.training.llm_lr
model_path = config_manager.model.model_path
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