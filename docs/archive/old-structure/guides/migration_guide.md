# 🔄 Migration Guide: Legacy → 2025 Modular Architecture **[COMPLETE]**

*Modular architecture with enhanced training system and data processing pipeline*

---

## Migration Status: **COMPLETE** ✅

The BBU training system has completed migration to a modular architecture (2025). The current system provides:

- **Modular Training System** (`src/training/`): BBUTrainer, TrainingCoordinator, LossManager
- **Unified Model Loading** (`src/models/`): Consistent training-inference model loading
- **5-Stage Data Pipeline** (`data_conversion/`): PipelineManager with object-oriented training
- **Simplified Token Management** (`src/utils/`): Lightweight token addition approach
- **Multi-geometry support** (`bbox_2d`, `square`, `line`) with object type filtering
- **DirectConfig System** (`src/config/`): Unified configuration management

## 2025 Modular Architecture Overview

### 1. **Legacy vs Current Architecture Comparison**

**Legacy Architecture (Pre-2025)**:
```
Single monolithic trainer (2100+ lines)
Scattered configuration (149+ parameters in one file)
Complex coordinate token system
Difficult to debug and extend
```

**Current Modular Architecture (2025)**:
```
src/
├── training/          # Modular training components
│   ├── trainer.py            # BBUTrainer
│   ├── training_coordinator.py # Training orchestration
│   ├── loss_manager.py       # Multi-task loss computation
│   └── trainer_factory.py    # Factory pattern
├── models/           # Model management
│   ├── model_loader.py       # Unified model loading
│   ├── wrapper.py           # Qwen25VLWithDetection
│   └── patches.py           # Model patches
├── core/             # Central processors
│   ├── data_processor.py     # Data processing
│   └── checkpoint_manager.py # Checkpoint management
└── config/           # Configuration management
    └── global_config.py      # DirectConfig system
```

### 2. **Current Training Pipeline Usage**
```bash
# 5-Stage Data Processing Pipeline
cd /data3/Qwen2.5-VL-main
bash data_conversion/convert_dataset.sh  # Uses PipelineManager

# Or use Python pipeline manager directly
python data_conversion/pipeline_manager.py \
    --input_dir ds_v2 \
    --output_dir data \
    --object_types "bbu label fiber" \
    --resize true

# Modular Training System
python scripts/train.py --config configs/base_flat_v2.yaml
```

### 3. **V2 Training Configuration**
The V2 system uses updated configuration with multi-geometry and simple token support:

**V2 Configuration Options:**
```yaml
# V2 Data Format Support
simple_tokens_enabled: true         # Use simple token system (V2)
coordinate_tokens_enabled: false    # Disable legacy coordinate tokens (V1)

# V2 Multi-Geometry Support  
multi_geometry_enabled: true        # Support bbox_2d, square, line geometries
object_type_filtering: true         # Enable object-oriented training

# V2 Token Configuration
special_tokens:
  - "<|line_start|>"                # For fiber/wire objects
  - "<|line_end|>"
  - "<|square_start|>"              # For rotated equipment/labels  
  - "<|square_end|>"

# Validation and Safety
enable_config_validation: true
enable_cross_config_checks: true
fail_fast_on_errors: true
```

## What's Changed

### Before (Legacy System)
- **Single monolithic trainer** (2100+ lines)
- **Flat configuration** (149 parameters in single namespace)
- **Embedded loss computation** within trainer
- **Manual parameter grouping**

### After (New System)
- **Modular components**:
  - `TrainingCoordinator` - orchestrates training
  - `LossManager` - handles all loss computation
  - `ParameterGroupManager` - manages differential learning rates
  - `ConfigManager` - validates and manages domain-specific configs
- **Domain-specific configuration** with cross-validation
- **Better separation of concerns**
- **Enhanced logging and monitoring**

## Architecture Components

### 1. Configuration System
```python
# Legacy
from src.config import config
learning_rate = config.llm_lr

# New
from src.config import get_config_manager
config_manager = get_config_manager()
learning_rate = config_manager.training.llm_lr
```

### 2. Training System
```python
# Legacy - manual trainer creation
trainer = create_trainer(training_args)

# New - factory with coordinator
trainer = create_trainer_with_coordinator(training_args, use_new_config=True)
```

### 3. Loss Management
```python
# Legacy - embedded in trainer
def compute_loss(self, model, inputs, return_outputs=False):
    # 200+ lines of loss computation in trainer...

# New - extracted manager
loss_manager = LossManager(config_manager)
total_loss = loss_manager.compute_total_loss(model_outputs, inputs)
```

## Validation and Testing

### Test Configuration
```bash
# Validate legacy config
python scripts/train.py --config base_flat --validate-only

# Validate new config system
python scripts/train.py --config base_flat_v2 --use-new-config --validate-only
```

### Print Configuration
```bash
# Print legacy config
python scripts/train.py --config base_flat --print-config

# Print new config
python scripts/train.py --config base_flat_v2 --use-new-config --print-config
```

## Backward Compatibility

✅ **Full backward compatibility maintained**
- Legacy system continues to work unchanged
- Both systems can coexist
- Gradual migration supported
- No breaking changes to existing workflows

## Benefits of New System

### 1. **Maintainability**
- Clear separation of concerns
- Modular components (LossManager, ParameterGroupManager, etc.)
- Reduced complexity in individual components

### 2. **Configuration Management**
- Domain-specific validation (model, training, data, detection, infrastructure)
- Cross-config dependency checking
- Better error messages and fail-fast validation

### 3. **Training Orchestration**
- Centralized training coordination
- Component-wise state management
- Better gradient monitoring and parameter statistics

### 4. **Extensibility**
- Easy to add new components
- Clean interfaces for extending functionality
- Better testing and debugging capabilities

## File Structure Changes

### New Files Added
```
src/config/
├── domain_configs.py          # Domain-specific config classes
├── config_manager.py          # Config validation and management

src/training/
├── training_coordinator.py    # Training orchestration
├── loss_manager.py           # Loss computation logic  
├── parameter_manager.py      # Parameter grouping
└── trainer_factory.py       # Trainer creation functions

configs/
└── base_flat_v2.yaml         # Enhanced configuration
```

### Updated Files
```
src/config/__init__.py         # Support for both systems
src/training/trainer.py        # Optional coordinator integration
scripts/train.py              # --use-new-config flag
scripts/run_train.sh          # USE_NEW_CONFIG variable
```

## Troubleshooting

### Common Issues

1. **"Config not initialized" error**
   - Ensure you use `--use-new-config` flag for new system
   - Check config file path is correct

2. **Import errors**
   - New system imports different modules
   - Use trainer factory functions for consistent setup

3. **Configuration validation errors**
   - New system has stricter validation
   - Check `base_flat_v2.yaml` for required new parameters

### Getting Help

- Check `CLAUDE.md` for updated development commands
- Review `ongoing_task/06_configuration_reference.md` for parameter reference
- Use `--validate-only` flag to test configurations without training

## Recommendation

**For new development**: Use the new system (`base_flat_v2.yaml` + `--use-new-config`)
**For production stability**: Legacy system continues to work reliably
**For gradual migration**: Both systems can coexist during transition