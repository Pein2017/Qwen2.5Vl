# Codebase Structure

## Project Layout
```
├── data_conversion/           # Data processing pipeline
│   ├── convert_dataset.sh    # Main pipeline script
│   ├── unified_processor.py  # Core processing engine
│   └── utils/                # Processing utilities
├── src/                      # Training and inference code
│   ├── training/             # Training system
│   ├── models/               # Model management & patches
│   ├── utils/                # Utilities and token management
│   ├── config/               # Configuration system
│   └── core/                 # Core processors
├── configs/                  # Configuration files
├── docs/                     # Documentation
├── scripts/                  # Training and validation scripts
└── eval/                     # Evaluation scripts
```

## Key Components

### Training System (`src/training/`)
- **BBUTrainer**: Main trainer class with teacher-student learning
- **TrainingCoordinator**: Orchestrates multi-task training
- **LossManager**: Handles coordinate vs standard LLM losses
- **StabilityMonitor**: Training stability tracking

### Model System (`src/models/`)
- **Qwen25VLWithDetection**: Wrapper with detection capabilities
- **patches.py**: mRoPE, Flash Attention 2, and compatibility fixes
- **model_loader.py**: Unified model and processor loading

### Configuration (`src/config/`)
- **DirectConfig**: Flat configuration access system
- **global_config.py**: Configuration initialization and validation

### Data Processing (`data_conversion/`)
- **unified_processor.py**: Core 5-stage processing pipeline
- **coordinate_manager.py**: Coordinate transformation system
- **flexible_taxonomy_processor.py**: Multi-format annotation handling

### Utilities (`src/utils/`)
- **coordinate_token_manager.py**: Coordinate token system
- **response_parser.py**: Response parsing and validation
- **prompt.py**: Multi-language prompt templates

## Entry Points
- **Training**: `scripts/train.py` (called via `scripts/run_train.sh`)
- **Data Processing**: `data_conversion/convert_dataset.sh`
- **Inference**: `src/inference.py`
- **Validation**: Various scripts in `scripts/` directory