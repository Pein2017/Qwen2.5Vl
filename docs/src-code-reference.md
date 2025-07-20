# Source Code Reference Guide

**Quick navigation and reference for the src/ codebase structure**

This document provides a fast overview of the current source code organization, key files, and their purposes for developers working with the codebase.

---

## 📁 Current Source Code Structure

```
src/
├── __init__.py                    # Package initialization
├── README.md                      # Source code overview
│
├── config/                        # Configuration Management
│   ├── __init__.py
│   ├── global_config.py          # ✅ Direct global configuration system
│   ├── config_manager.py.bak     # 🗄️ Legacy domain-specific config (backed up)
│   ├── coordinate_validator.py.bak # 🗄️ Legacy validation (backed up)  
│   └── domain_configs.py.bak     # 🗄️ Legacy domain configs (backed up)
│
├── core/                          # Central Factories & Processors
│   ├── __init__.py
│   ├── checkpoint_manager.py     # ✅ Model checkpointing and recovery
│   ├── data_processor.py         # ✅ Unified data processing pipeline
│   └── model_factory.py.bak      # 🗄️ Legacy model factory (backed up)
│
├── models/                        # Model Management
│   ├── __init__.py
│   ├── model_loader.py           # ✅ Unified model loading with patches
│   ├── patches.py                # ✅ Critical model patches (mRoPE, Flash Attention)
│   └── wrapper.py                # ✅ Qwen25VLWithDetection main model wrapper
│
├── training/                      # Training System
│   ├── __init__.py
│   ├── callbacks.py              # ✅ Training callbacks and monitoring
│   ├── loss_manager.py           # ✅ Multi-component loss computation
│   ├── parameter_manager.py      # ✅ Differential learning rate management
│   ├── stability.py              # ✅ Training stability utilities
│   ├── trainer.py                # ✅ Enhanced HuggingFace Trainer
│   ├── trainer_factory.py        # ✅ Trainer creation factory
│   ├── training_coordinator.py   # ✅ Training orchestration
│   └── loss_manager.py.bak       # 🗄️ Legacy loss manager (backed up)
│
├── utils/                         # Utilities & Support
│   ├── __init__.py
│   ├── coordinate_token_manager.py # ✅ Core coordinate token management
│   ├── prompt.py                 # ✅ BBU-specific prompt engineering
│   ├── response_parser.py        # ✅ Robust response parsing
│   ├── schema.py                 # ✅ Data validation and tensor shapes
│   ├── utils.py                  # ✅ General utilities
│   ├── coordinate_loss_computer.py.bak # 🗄️ Legacy coordinate loss (backed up)
│   ├── coordinate_processor.py.bak      # 🗄️ Legacy coordinate processor (backed up)
│   └── tokens/                   # Token Management
│       ├── __init__.py
│       └── special_tokens.py     # ✅ Coordinate tokens and vocabulary
│
├── reference/                     # Reference Implementation
│   ├── offical_huggingface_qwen2_5_vl/ # 🔍 Official HuggingFace reference
│   │   ├── __init__.py
│   │   ├── configuration_qwen2_5_vl.py
│   │   ├── modeling_qwen2_5_vl.py
│   │   ├── modular_qwen2_5_vl.py
│   │   └── processing_qwen2_5_vl.py
│   └── qwen2_5vl_collator.py     # 🔍 Reference collator implementation
│
├── legacy/                        # Legacy Components
│   └── trainer_unified.py        # 🗄️ Legacy unified trainer
│
├── chat_processor.py              # ✅ BBU annotations to chat format conversion
├── data.py                       # ✅ Data loading and collation
├── inference.py                  # ✅ Standalone inference engine
├── logger_utils.py               # ✅ Logging utilities
└── teacher_pool.py               # ✅ Teacher sample management
```

**Legend:**
- ✅ Active, current implementation
- 🗄️ Backed up legacy file (preserved for reference)
- 🔍 Reference implementation for comparison

---

## 🎯 Key Active Files by Category

### Configuration System
- **`config/global_config.py`** - Main configuration system
  - Direct access configuration with flat YAML structure
  - Type validation and auto-property generation
  - Usage: `from src.config.global_config import config`

### Model Management
- **`models/wrapper.py`** - Core model wrapper
  - `Qwen25VLWithDetection` class combining VLM + coordinate tokens
  - Non-destructive vocabulary extension
  - Enhanced forward pass with coordinate token support

- **`models/model_loader.py`** - Unified model loading
  - Consistent loading for training and inference
  - Automatic patch application
  - Model validation and compatibility checks

- **`models/patches.py`** - Critical model patches
  - mRoPE dimension fix for multi-image training
  - Flash Attention 2 compatibility patches
  - Visual processing enhancements

### Training System
- **`training/training_coordinator.py`** - Training orchestration
  - Coordinates between trainer, loss manager, and parameter manager
  - Setup training pipeline and loss computation delegation

- **`training/loss_manager.py`** - Multi-component loss computation
  - Mode-aware loss computation (coordinate vs standard)
  - Teacher-student loss splitting
  - Multi-component coordinate loss (focal, L1, GIoU)

- **`training/trainer.py`** - Enhanced HF Trainer
  - Integration with training coordinator
  - Robust loss component logging
  - Enhanced error handling and validation

### Data Processing
- **`chat_processor.py`** - Core data conversion
  - BBU annotations → chat format conversion
  - Automatic bbox → coordinate token conversion
  - Multi-language support (English/Chinese)

- **`data.py`** - Data loading and collation
  - Standard and packed data collation
  - Safe dictionary access patterns
  - Teacher-student sample mixing

- **`core/data_processor.py`** - Unified data processing
  - Dataset creation with validation
  - Data statistics and quality checks
  - Integration with coordinate token system

### Coordinate Token System
- **`utils/coordinate_token_manager.py`** - Core coordinate management
  - Automatic JSON ↔ coordinate token conversion
  - Soft expectation regression computation
  - Multi-component loss calculation
  - Bbox span detection and validation

- **`utils/tokens/special_tokens.py`** - Token management
  - Coordinate token vocabulary (`<coord_0>` - `<coord_2047>`)
  - Box tokens (`<|box_start|>`, `<|box_end|>`)
  - Safe vocabulary extension

### Utilities
- **`inference.py`** - Standalone inference
  - Single and batch inference support
  - Automatic coordinate token parsing
  - Response validation and error handling

- **`utils/response_parser.py`** - Robust parsing
  - Multiple parsing strategies with fallbacks
  - JSON, regex, and coordinate token parsing
  - Error recovery mechanisms

---

## 🔧 Component Dependencies

### Training Flow
```
scripts/train.py
    ↓
training/trainer_factory.py → training/trainer.py
    ↓                              ↓
training/training_coordinator.py ←→ training/loss_manager.py
    ↓                              ↓
models/wrapper.py              utils/coordinate_token_manager.py
    ↓
models/model_loader.py + models/patches.py
```

### Data Flow
```
Raw JSONL → chat_processor.py → data.py → core/data_processor.py
    ↓              ↓                ↓             ↓
BBU Format → Chat Format → Collated Batches → Training Ready
```

### Configuration Flow
```
YAML Config → config/global_config.py → Global config instance
    ↓                    ↓                      ↓
Domain configs → Type validation → Accessible everywhere
```

---

## 📝 File Purpose Quick Reference

| File | Primary Purpose | Key Classes/Functions |
|------|----------------|----------------------|
| `models/wrapper.py` | Main model with coordinate tokens | `Qwen25VLWithDetection` |
| `training/trainer.py` | Enhanced HF trainer | `BBUTrainer` |
| `training/loss_manager.py` | Multi-task loss computation | `LossManager.compute_total_loss()` |
| `utils/coordinate_token_manager.py` | Coordinate token core logic | `CoordinateTokenManager` |
| `chat_processor.py` | Data format conversion | `ChatProcessor.process_conversations()` |
| `models/model_loader.py` | Unified model loading | `ModelLoader.load_model_with_patches()` |
| `config/global_config.py` | Configuration system | `init_config()`, `config` |
| `data.py` | Data collation | `StandardDataCollator`, `PackedDataCollator` |
| `inference.py` | Standalone inference | `InferenceEngine` |

---

## 🧭 Navigation Patterns

### Finding Component Implementation
1. **Start with**: `src/README.md` for overview
2. **Model-related**: Check `models/` directory
3. **Training-related**: Check `training/` directory  
4. **Data-related**: Check `chat_processor.py`, `data.py`, `core/data_processor.py`
5. **Configuration**: Always `config/global_config.py`
6. **Utilities**: Check `utils/` directory

### Common Development Tasks
- **Add new model patch**: Edit `models/patches.py`
- **Modify training behavior**: Edit `training/training_coordinator.py` or `training/loss_manager.py`
- **Change data processing**: Edit `chat_processor.py` or `core/data_processor.py`
- **Add configuration parameter**: Edit config YAML and `config/global_config.py`
- **Extend coordinate tokens**: Edit `utils/coordinate_token_manager.py`

### Integration Points
- **Configuration**: All components use `from src.config.global_config import config`
- **Logging**: All components use `logger_utils.py` patterns
- **Model Access**: All training components go through `models/wrapper.py`
- **Data Access**: All data goes through `chat_processor.py` → `data.py` pipeline

---

## 🗂️ Legacy vs Current Files

### Replaced Components
| Legacy File (backed up) | Current Replacement | Reason for Change |
|-------------------------|-------------------|------------------|
| `config/config_manager.py.bak` | `config/global_config.py` | Simplified direct access |
| `config/domain_configs.py.bak` | Integrated into global config | Reduced complexity |
| `utils/coordinate_processor.py.bak` | `utils/coordinate_token_manager.py` | Enhanced functionality |
| `utils/coordinate_loss_computer.py.bak` | Integrated into loss manager | Better integration |
| `core/model_factory.py.bak` | `models/model_loader.py` | More focused responsibility |

### Why Files Were Backed Up
- **Preserve historical knowledge** about implementation decisions
- **Reference for debugging** if issues arise with new implementation
- **Documentation of evolution** showing system development
- **Recovery option** if new implementation has issues

---

## 🚀 Quick Development Commands

### Code Navigation
```bash
# Find implementation of specific function
find src/ -name "*.py" -exec grep -l "function_name" {} \;

# Find all references to a class
rg "ClassName" src/ --type py

# Find configuration usage
rg "config\." src/ --type py -A 2 -B 2
```

### Component Testing
```bash
# Test specific component
/root/miniconda3/envs/ms/bin/python -c "
from src.models.wrapper import Qwen25VLWithDetection
print('Model wrapper import successful')
"

# Test configuration system
/root/miniconda3/envs/ms/bin/python -c "
from src.config.global_config import init_config
config = init_config('configs/base_flat_det.yaml')
print('Configuration loaded successfully')
"
```

### Integration Testing
```bash
# Test complete pipeline
/root/miniconda3/envs/ms/bin/python -c "
from src.chat_processor import ChatProcessor
from src.models.wrapper import Qwen25VLWithDetection
print('Pipeline components import successfully')
"
```

---

## 🔗 Related Documentation

### Architecture
- **[Architecture Overview →](architecture-overview.md)** - High-level system design
- **[Component Details →](architecture-appendix-a-components.md)** - Detailed component specs

### Development
- **[API Reference →](quick-reference/api-core-components.md)** - API documentation
- **[Configuration Reference →](configuration-reference-complete.md)** - Complete config docs
- **[User Journeys →](user-journeys/)** - Development workflows

### Navigation
- **[← Back to Main Documentation]()**
- **[Architecture Deep Dive →](architecture-overview.md)**
- **[API Quick Reference →](quick-reference/api-core-components.md)**

---

**💡 Source Code Principle**: The codebase follows a modular architecture with clear separation of concerns. Each component has a single responsibility and well-defined interfaces. Legacy files are preserved but clearly marked to maintain historical context while enabling clean development.