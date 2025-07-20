# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen2.5-VL fine-tuning project for BBU (Base-Band Unit) equipment detection and captioning. The project implements end-to-end training of a vision-language model for dense object detection with natural language descriptions in both English and Chinese.

This implementation has evolved significantly beyond standard Qwen2.5-VL fine-tuning, featuring:
- **Coordinate Token System**: Soft expectation regression with automatic bbox→token conversion
- **Multi-task Training**: Teacher-student learning with span-based loss splitting  
- **Modular Architecture**: Refactored from monolithic to component-based design
- **Enhanced Loss Management**: Mode-aware coordinate vs standard LLM loss computation

## Architecture Overview

### Core System Components

```
src/
├── core/                    # Central factories and processors
│   ├── model_factory.py     # Model creation with coordinate token support
│   ├── data_processor.py    # Unified data processing pipeline
│   └── checkpoint_manager.py # Model checkpointing and recovery
├── models/
│   ├── wrapper.py           # Qwen25VLWithDetection - main model wrapper
│   ├── model_loader.py      # Model loading with patches and validation
│   └── patches.py           # Critical fixes (mRoPE, Flash Attention 2)
├── training/
│   ├── training_coordinator.py  # Modern training orchestration
│   ├── loss_manager.py         # Mode-aware loss computation
│   ├── trainer.py              # Enhanced HF Trainer with robust logging
│   └── parameter_manager.py    # Differential learning rate management
├── utils/
│   ├── coordinate_token_manager.py    # Core coordinate token management
│   ├── coordinate_loss_computer.py    # Multi-component loss computation
│   ├── coordinate_processor.py        # Legacy interface (deprecated)
│   └── tokens/special_tokens.py       # Coordinate token vocabulary
└── config/
    ├── global_config.py        # Direct configuration access
    ├── config_manager.py       # Domain-specific config validation
    └── coordinate_validator.py # Coordinate token config validation
```

### Coordinate Token System

The system automatically converts bbox coordinates to coordinate tokens during training:

**Data Flow:**
```
Raw JSONL: {"bbox_2d": [3, 259, 295, 653], "desc": "bbu基带处理单元"}
     ↓ (ChatProcessor)
Tokens: "bbu基带处理单元: <|box_start|><coord_3><coord_259><coord_295><coord_653><|box_end|>"
     ↓ (Model Forward)
Loss: coordinate_loss + focal_loss + regular_loss + l1_loss + giou_loss
```

**Key Features:**
- **Automatic Conversion**: No manual data preprocessing required
- **Soft Expectation Loss**: Differentiable coordinate prediction  
- **Multi-component Loss**: Enhanced with focal, L1, and GIoU losses
- **Mode-aware Training**: Seamless switching between coordinate and standard modes

## Environment Setup

The project requires:
- Conda environment: `ms`
- CUDA_VISIBLE_DEVICES for GPU selection
- HF_HOME for model cache (typically `/data3/Qwen2.5-VL-main/model_cache`)

**Important Reminders:**
- We need to activate `ms` virtual environment, remember this.
- **NEW**: Directly use `/root/miniconda3/envs/ms/bin/python` to avoid conda activation inconsistencies
- Always use the full path to Python in the `ms` environment to ensure package consistency

## Enhanced Data Processing Pipeline

### Quick Start
```bash
source activate ms
bash data_conversion/convert_dataset.sh
```

### Pipeline Overview
The data conversion pipeline has been **enhanced** with improved JSON cleaning and path management:

```
ds/ (raw JSON + images) → convert_dataset.sh → ① clean_raw_json.py (NEW)
                                             → ② copy images to ds_output
                                             → ③ unified processor.py 
                                             → ④ smart resize + bbox scaling
                                             → data/ (train.jsonl, val.jsonl, teacher.jsonl)
```

### Key Improvements
1. **JSON Cleaning**: `clean_raw_json.py` strips unnecessary metadata while preserving essential structure
2. **Unified Output**: All processed data (JSON + images) goes to `ds_output/` directory  
3. **Path Consistency**: JSONL files reference correct `ds_output/` paths
4. **Enhanced Validation**: Improved bbox scaling accuracy and error checking
5. **Environment Integration**: Automatic conda environment activation

### Configuration Options
```bash
# Custom directories
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" bash data_conversion/convert_dataset.sh

# Language selection (chinese/english)
LANGUAGE="english" bash data_conversion/convert_dataset.sh

# Disable resizing for testing
RESIZE="false" bash data_conversion/convert_dataset.sh
```

### Troubleshooting
- **'list' object has no attribute 'get'**: Run JSON cleaning first
- **FileNotFoundError for images**: Check image copying step
- **Bbox out of bounds**: Verify smart_resize parameters

## Documentation Index

### Architecture and Design
- `docs/architecture.md` - Complete system architecture with tensor flow diagrams
- `docs/coordinate_token_system_update.md` - Current coordinate token implementation status
- `docs/coordinate_regression_guide.md` - Soft expectation regression theory and usage
- `docs/coordinate_loss_visibility_fix.md` - Loss computation debugging and fixes

### Configuration and Setup  
- `docs/configuration.md` - Configuration system and parameter management
- `docs/getting_started.md` - Project setup and quick start guide
- `docs/data_schema.md` - Data format specifications and validation

### Implementation Details
- `docs/implementation_summary.md` - Core implementation patterns and decisions
- `docs/migration_guide.md` - Evolution from monolithic to modular architecture  
- `docs/critical_fixes.md` - Important bug fixes and patches applied
- `docs/lessons_learned.md` - Development insights and best practices

### Advanced Topics
- `docs/advanced/teacher_student.md` - Teacher-student learning implementation
- `docs/advanced/peft_adapter.md` - Parameter-efficient fine-tuning setup
- `docs/advanced/collator_notes.md` - Data collation optimization strategies
- `docs/soft_expectation_coordinate_regression.md` - Mathematical foundation

### Operations
- `docs/runbook.md` - Training execution and monitoring
- `docs/testing.md` - Test suite and validation procedures  
- `docs/troubleshooting.md` - Common issues and solutions

## Key Implementation Patterns

### Modular Design Philosophy
- **No Legacy Support**: Direct override approach, clean refactoring
- **Component Separation**: Training, models, data, config as distinct modules
- **Factory Pattern**: Centralized creation in `src/core/` for consistency
- **Defensive Programming**: Comprehensive validation and error handling

### Coordinate Token Integration
- **Transparent Operation**: Automatic bbox→token conversion in data pipeline
- **Mode-aware Processing**: Configuration-driven coordinate vs standard mode
- **Multi-component Loss**: Enhanced loss computation with validation and debugging
- **Token Management**: Extended vocabulary with proper initialization and validation