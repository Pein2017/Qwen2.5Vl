# Modular Architecture

> **NEW in 2025**: Refactored codebase with improved component organization and separation of concerns

---

## Overview

The BBU training system has been completely refactored from a monolithic design to a modular architecture. This redesign improves code organization, maintainability, testability, and extensibility while maintaining full backward compatibility with existing workflows.

## Key Benefits

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

---

## Directory Structure

```
src/
├── core/ (Central factories and managers)
│   ├── data_processor.py     # Data processing factory
│   └── checkpoint_manager.py # Model saving/loading
├── config/ (Enhanced configuration)
│   ├── global_config.py      # Configuration management
│   └── __init__.py           # Config access methods
├── training/ (Modular training components)
│   ├── trainer.py            # Enhanced BBUTrainer
│   ├── training_coordinator.py # Training orchestration
│   ├── loss_manager.py       # Multi-task loss computation
│   ├── parameter_manager.py  # Parameter grouping
│   ├── trainer_factory.py    # Factory methods
│   ├── callbacks.py          # Training callbacks
│   └── stability.py          # Training stability utilities
├── models/ (Model architecture)
│   ├── model_loader.py       # Unified model loading
│   ├── wrapper.py            # Model wrapper with token handling
│   └── patches.py            # Model patches and optimizations
├── utils/ (Support utilities)
│   ├── simple_token_manager.py     # NEW token system
│   ├── coordinate_token_manager.py # Legacy tokens (deprecated)
│   ├── utils.py              # General utilities
│   ├── prompt.py             # Prompt templating
│   ├── response_parser.py    # Output parsing
│   ├── schema.py             # Type definitions
│   └── tokens/               # Special token definitions
└── [Root modules]
    ├── data.py               # BBUDataset with V2 support
    ├── chat_processor.py     # Conversation building
    ├── teacher_pool.py       # Teacher demonstration management
    ├── inference.py          # Production inference
    └── logger_utils.py       # Advanced logging
```

---

## Component Responsibilities

### Core Components (`core/`)

Central factory classes and managers for the training system:

- **`data_processor.py`** - Unified data processing and dataset creation
  - Creates and configures datasets
  - Manages data collators
  - Handles data validation

- **`checkpoint_manager.py`** - Model saving/loading and checkpoint management
  - Saves and loads checkpoints
  - Handles model versioning
  - Validates model integrity

### Configuration (`config/`)

Dual configuration system supporting both legacy and new approaches:

- **`global_config.py`** - Legacy DirectConfig system (flat parameters)
  - Backward compatibility
  - Global parameter access
  - Parameter validation

### Training System (`training/`)

Modular training components extracted from monolithic trainer:

- **`trainer.py`** - Enhanced BBUTrainer with optional coordinator integration
  - Executes training loop
  - Manages evaluation
  - Handles model saving

- **`training_coordinator.py`** - Training orchestration and state management
  - Coordinates training phases
  - Manages training state
  - Handles distributed training

- **`loss_manager.py`** - Multi-task loss computation (LM + detection)
  - Computes and scales losses
  - Tracks loss components
  - Handles gradient propagation

- **`parameter_manager.py`** - Parameter grouping for differential learning rates
  - Groups parameters by layer
  - Applies learning rate scheduling
  - Optimizes training performance

- **`trainer_factory.py`** - Factory functions for trainer creation
  - Creates configured trainers
  - Handles dependency injection
  - Streamlines setup

### Models (`models/`)

Model architecture and integration:

- **`model_loader.py`** - Unified model loader for training-inference consistency
  - Loads models for both training and inference
  - Handles token initialization
  - Manages model configuration

- **`wrapper.py`** - Qwen2.5-VL wrapper with token support
  - Wraps HuggingFace model
  - Handles token validation
  - Manages forward pass

- **`patches.py`** - Model patches and optimizations
  - Applies performance optimizations
  - Fixes model bugs
  - Enhances functionality

### Utilities (`utils/`)

Support utilities and helper functions:

- **`simple_token_manager.py`** - Simple token manager (ms-swift approach)
  - Adds special tokens
  - Handles token wrapping
  - Validates token usage

- **`coordinate_token_manager.py`** - Legacy coordinate token system (deprecated)
  - Backward compatibility
  - Legacy coordinate support
  - Gradual migration support

- **`tokens/`** - Special token definitions and handling
  - Centralizes token constants
  - Manages token formatting
  - Validates token configuration

### Root Level

Core data and processing modules:

- **`data.py`** - BBUDataset with complete multi-geometry support
  - Loads and processes data
  - Handles multi-geometry objects
  - Manages teacher-student integration

- **`chat_processor.py`** - Conversation building with simple token integration
  - Formats conversations
  - Processes images
  - Handles token validation

---

## Factory Pattern Usage

The refactored architecture extensively uses the factory pattern to create complex objects with proper configuration:

```python
# Using the trainer factory
from src.training.trainer_factory import create_trainer_with_coordinator
trainer = create_trainer_with_coordinator(training_args)

# Using the model loader factory
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, processor = load_model_and_processor_unified(config)

# Using the token manager factory
from src.utils.simple_token_manager import create_simple_token_manager
token_manager = create_simple_token_manager(tokenizer, model)

# Using the data processor factory
from src.core.data_processor import DataProcessor
processor = DataProcessor(tokenizer, processor, model)
train_dataset, eval_dataset = processor.create_datasets()
```

---

## Configuration Management

The system supports both legacy and new configuration approaches:

```python
# Global config access
from src.config import get_config
config = get_config()

# Key parameters
model_path = config.model_path
learning_rate = config.llm_lr
simple_tokens_enabled = config.simple_tokens_enabled
```

---

## Development Guidelines

### Adding New Components

1. Place in appropriate domain folder (`core/`, `training/`, `models/`, etc.)
2. Use factory pattern for complex object creation
3. Support both legacy and new config systems during transition
4. Add comprehensive logging and error handling

### Import Guidelines

- Use absolute imports: `from src.core import DataProcessor`
- Organize imports by domain
- Prefer factory methods over direct instantiation
- Keep circular dependencies minimal

### Testing

- Test each component in isolation
- Use factory methods for test object creation
- Mock external dependencies
- Validate both legacy and new config systems

---

## Metrics and Monitoring

The refactored system provides enhanced monitoring:

- Component-wise parameter statistics
- Training coordinator status summaries
- Loss manager component tracking
- Configuration validation reports
- Checkpoint integrity validation

---

## Migration Path

The refactored architecture maintains full backward compatibility:

1. **Existing scripts** continue to work unchanged
2. **New components** are accessible through factory methods
3. **Legacy components** are preserved but marked as deprecated
4. **Gradual migration** is supported for incremental adoption

---

**Status**: Production Ready ✅  
**Backward Compatible**: Yes ✅  
**Future-Proof**: Designed for extension ✅ 