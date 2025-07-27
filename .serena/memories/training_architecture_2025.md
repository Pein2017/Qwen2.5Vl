# Training Architecture - 2025 Edition

## Key Principles
1. **Simplicity Over Complexity**: Replaced 1000+ line coordinate token system with simple token manager using standard HuggingFace infrastructure
2. **Unified Model Loading**: Single model loader for training and inference consistency with automatic token initialization
3. **Multi-Geometry Support**: Native support for `bbox_2d`, `square`, and `line` geometries with semantic token wrapping

## Core Components

### Model Loading
- **Primary**: `src/models/model_loader.py` with `load_model_and_processor_unified()`
- Features: Training-inference consistency, automatic simple token initialization, detection handling

### Token System
- **Primary**: `src/utils/simple_token_manager.py`
- Philosophy: Inspired by ms-swift's lightweight approach using standard HuggingFace methods

### Training Pipeline
- **Trainer**: `src/training/trainer.py` - BBUTrainer with teacher-student learning
- **Coordinator**: `src/training/training_coordinator.py` - Multi-task orchestration
- **Loss Management**: `src/training/loss_manager.py` - Mode-aware coordinate vs LLM losses

## Current Status (2025)
- Modernized architecture focused on reliability
- Fail-fast validation instead of silent error handling
- Proper DeepSpeed integration
- Robust multi-geometry validation with clear error messages