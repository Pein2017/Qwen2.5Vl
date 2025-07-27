# Project Map - File Purpose at a Glance

**Quick reference for navigating the BBU Detection System codebase**

## 🚀 Entry Points (How to Start)

### Training Entry Points
```bash
scripts/train.py                      # Main training script - START HERE for training
src/training/trainer_factory.py       # Creates trainer with all components
```

### Data Processing Entry Points  
```bash
data_conversion/convert_dataset.sh    # Main data pipeline - START HERE for data processing
data_conversion/pipeline_manager.py   # Python pipeline orchestrator
```

### Inference Entry Points
```bash
src/inference.py                      # Production inference script
```

### Configuration Entry Points
```bash
configs/base_flat_v2.yaml            # Main config template - COPY AND MODIFY THIS
src/config/global_config.py          # Config system implementation
```

## 📁 Core Components (Import These)

### Training System
```python
from src.training.trainer import BBUTrainer                    # Enhanced HF trainer
from src.training.training_coordinator import TrainingCoordinator  # Training orchestration
from src.training.loss_manager import LossManager             # Multi-task loss computation
from src.training.trainer_factory import create_trainer_with_coordinator  # Factory
```

### Model System
```python
from src.models.model_loader import load_model_and_processor_unified  # Unified model loading
from src.models.wrapper import Qwen25VLWithDetection          # Main model wrapper
```

### Data Processing
```python
from src.core.data_processor import DataProcessor             # Dataset creation
from src.chat_processor import ChatProcessor                  # Format conversion
from src.data import BBUDataset, create_data_collator        # Dataset and collator
```

### Configuration
```python
from src.config import get_config                             # Get DirectConfig instance
```

## 🗂️ Directory Structure

### `src/` - Main Source Code
```
src/
├── training/              # Training system components
│   ├── trainer.py            # BBUTrainer (enhanced HF trainer)
│   ├── training_coordinator.py  # Training orchestration
│   ├── loss_manager.py       # Multi-task loss computation
│   ├── trainer_factory.py    # Factory for trainer creation
│   ├── parameter_manager.py  # Learning rate management
│   ├── callbacks.py          # Training callbacks
│   └── stability.py          # Training stability utilities
├── models/                # Model management
│   ├── model_loader.py       # Unified model loading
│   ├── wrapper.py           # Qwen25VLWithDetection wrapper
│   └── patches.py           # Model patches (mRoPE, Flash Attention)
├── core/                  # Central processors
│   ├── data_processor.py     # Data processing factory
│   └── checkpoint_manager.py # Checkpoint management
├── config/                # Configuration management
│   └── global_config.py      # DirectConfig system
├── utils/                 # Utilities
│   ├── simple_token_manager.py   # Token management
│   ├── coordinate_token_manager.py # Legacy coordinate tokens
│   ├── response_parser.py    # Response parsing
│   └── prompt.py            # Prompt templates
├── data.py                # BBUDataset implementation
├── chat_processor.py      # Format conversion
├── teacher_pool.py        # Teacher-student learning
├── inference.py           # Inference script
└── logger_utils.py        # Logging utilities
```

### `data_conversion/` - Data Processing Pipeline
```
data_conversion/
├── convert_dataset.sh        # Main pipeline script
├── pipeline_manager.py       # 5-stage pipeline orchestrator
├── unified_processor.py      # Core processing engine
├── coordinate_manager.py     # Coordinate transformations
├── flexible_taxonomy_processor.py  # Chinese annotation processing
├── data_splitter.py         # Train/val splitting
├── clean_raw_json.py        # JSON cleaning
├── config.py               # Data conversion config
└── utils/                  # Utility functions
```

### `scripts/` - Execution Scripts
```
scripts/
├── train.py               # Main training script
└── run_train.sh          # Training shell script
```

### `configs/` - Configuration Files
```
configs/
├── base_flat_v2.yaml     # Main config template
└── [other config variants]
```

## 🎯 Task-Based Navigation

### I want to understand the system
1. **Start**: `docs/MENTAL_MODEL.md` (5 minutes)
2. **Then**: `docs/ARCHITECTURE.md` (15 minutes)
3. **Deep dive**: `docs/components/` (component-specific)

### I want to train a model
1. **Setup**: `docs/quick-start/setup.md`
2. **First training**: `docs/quick-start/first-training.md`
3. **Full workflow**: `docs/workflows/training.md`

### I want to process data
1. **Quick start**: `bash data_conversion/convert_dataset.sh`
2. **Understanding**: `docs/workflows/data-processing.md`
3. **Customization**: `docs/components/data-pipeline.md`

### I want to run inference
1. **Quick start**: `python src/inference.py --help`
2. **Full workflow**: `docs/workflows/inference.md`

### I want to debug issues
1. **Common issues**: `docs/quick-start/common-issues.md`
2. **Full troubleshooting**: `docs/reference/troubleshooting.md`
3. **Component health checks**: `docs/MENTAL_MODEL.md#quick-health-checks`

### I want to extend the system
1. **Component contracts**: `docs/components/`
2. **Decision trees**: `docs/DECISION_TREES.md`
3. **API reference**: `docs/reference/api.md`

## 🔧 Key Files by Function

### Configuration
- `configs/base_flat_v2.yaml` - Main config template
- `src/config/global_config.py` - Config system implementation

### Data Processing
- `data_conversion/convert_dataset.sh` - Main pipeline
- `data_conversion/pipeline_manager.py` - Pipeline orchestrator
- `data_conversion/unified_processor.py` - Core processor

### Training
- `scripts/train.py` - Main training script
- `src/training/trainer.py` - Enhanced trainer
- `src/training/trainer_factory.py` - Trainer factory

### Model Management
- `src/models/model_loader.py` - Unified model loading
- `src/models/wrapper.py` - Model wrapper
- `src/models/patches.py` - Model patches

### Inference
- `src/inference.py` - Inference script

### Utilities
- `src/logger_utils.py` - Logging
- `src/utils/simple_token_manager.py` - Token management

## 🚨 Emergency Reference

### Quick Health Checks
```bash
# Test core imports
python -c "from src.training.trainer import BBUTrainer; print('✅ Training OK')"
python -c "from src.models.model_loader import load_model_and_processor_unified; print('✅ Model OK')"
python -c "from src.config import get_config; print('✅ Config OK')"

# Check data pipeline
ls data/ | grep -E "(train|val).jsonl" && echo "✅ Data OK"

# Check environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Common Commands
```bash
# Process data
bash data_conversion/convert_dataset.sh

# Train model
python scripts/train.py --config configs/base_flat_v2.yaml

# Run inference
python src/inference.py --model_path /path/to/model --image_path /path/to/image
```

---

**Next Steps**: 
- New to project? → `docs/MENTAL_MODEL.md`
- Ready to code? → `docs/quick-start/README.md`
- Need specific info? → `docs/components/` or `docs/reference/`
