# Project Structure

## Root Directory Organization

```
├── src/                      # Core source code
├── data_conversion/          # Data processing pipeline
├── configs/                  # Training configurations
├── docs/                     # Documentation
├── data/                     # Processed training data
├── ds/                       # Raw dataset (images + JSON)
├── eval/                     # Evaluation and testing
├── demo/                     # Demo and inference scripts
├── vis_tools/                # Visualization utilities
└── scripts/                  # Utility scripts
```

## Core Source Code (`src/`)

### Main Modules
- **`inference.py`** - Standalone inference engine
- **`chat_processor.py`** - Chat/conversation processing
- **`data.py`** - Data loading and collation
- **`teacher_pool.py`** - Teacher-student learning utilities
- **`logger_utils.py`** - Logging utilities and configuration

### Subdirectories
- **`config/`** - Configuration management system
  - `global_config.py` - Global configuration
  - `domain_configs.py` - Domain-specific configs
  - `config_manager.py` - Configuration loading/validation
- **`training/`** - Training system components
  - `trainer.py` - Main training orchestrator
  - `trainer_factory.py` - Training factory for different modes
  - `training_coordinator.py` - Training coordination and orchestration
  - `loss_manager.py` - Loss computation and balancing
  - `parameter_manager.py` - Parameter optimization
  - `callbacks.py` - Training callbacks
  - `stability.py` - Training stability utilities
- **`models/`** - Model management
  - `model_loader.py` - Model loading utilities
  - `patches.py` - Model patches and modifications
  - `wrapper.py` - Model wrapper classes
- **`core/`** - Core system components
  - `model_factory.py` - Model instantiation
  - `data_processor.py` - Data processing utilities
  - `checkpoint_manager.py` - Checkpoint handling
- **`utils/`** - Utility functions
  - `coordinate_processor.py` - Coordinate transformations
  - `response_parser.py` - Response parsing
  - `prompt.py` - Prompt templates
  - `schema.py` - Data schemas
  - `utils.py` - General utilities
  - `tokens/` - Token processing utilities
- **`reference/`** - Reference implementations
  - `qwen2_5vl_collator.py` - Reference collator implementation
  - `offical_huggingface_qwen2_5_vl/` - Official HuggingFace reference
- **`legacy/`** - Legacy code for reference
  - `trainer_unified.py` - Legacy unified trainer

## Data Processing Pipeline (`data_conversion/`)

### Core Components
- **`convert_dataset.sh`** - Main pipeline entry point
- **`pipeline_manager.py`** - Python pipeline orchestrator
- **`unified_processor.py`** - Core processing engine
- **`core_modules.py`** - Processing modules

### Specialized Processors
- **`coordinate_manager.py`** - Coordinate system handling
- **`image_processor.py`** - Image transformations
- **`sample_processor.py`** - Sample-level processing
- **`data_splitter.py`** - Train/validation splitting
- **`teacher_selector.py`** - Teacher pool creation
- **`processor.py`** - Legacy processor interface
- **`data_loader.py`** - Data loading utilities
- **`vision_process.py`** - Vision processing utilities

### Utility Modules
- **`utils/`** - Processing utilities
  - `file_ops.py` - File operations
  - `transformations.py` - Data transformations
  - `validators.py` - Validation utilities

### Configuration Files
- **`label_hierarchy.json`** - Label taxonomy
- **`label_hierarchy_full.json`** - Full label hierarchy
- **`token_map.json`** - Token mappings (English)
- **`token_map_zh.json`** - Token mappings (Chinese)
- **`candidate_phrases.json`** - Phrase candidates
- **`config.py`** - Pipeline configuration

### Validation & Testing
- **`simple_validate.py`** - Pipeline validation
- **`test_pipeline.py`** - Pipeline testing
- **`clean_raw_json.py`** - JSON cleaning utilities
- **`strip_exif_orientation.py`** - EXIF processing

### Legacy Components
- **`legacy_backup/`** - Archived legacy implementations

## Configuration System (`configs/`)

- **`base_flat_v2.yaml`** - Primary training configuration
- **`base_flat_det.yaml`** - Detection-focused configuration

## Documentation (`docs/`)

### Getting Started
- **`getting_started.md`** - New user guide
- **`architecture.md`** - System architecture
- **`runbook.md`** - Operational procedures

### Technical Guides
- **`data_schema.md`** - Data format specifications
- **`configuration.md`** - Configuration guide
- **`coordinate_regression_guide.md`** - Coordinate system details

### Troubleshooting
- **`troubleshooting.md`** - Common issues
- **`critical_fixes.md`** - Bug fixes and solutions
- **`lessons_learned.md`** - Historical knowledge

## Evaluation System (`eval/`)

### Core Evaluation
- **`test_all_evaluations.py`** - Main test suite
- **`eval_dataset.py`** - Dataset evaluation utilities
- **`eval_utils.py`** - Evaluation helper functions
- **`validate_results.py`** - Result validation

### Analysis & Metrics
- **`detailed_analysis.py`** - Comprehensive analysis tools
- **`coco_metrics.py`** - COCO-style metrics computation
- **`compare_experiments.py`** - Experiment comparison utilities

### Visualization
- **`visualization_utils.py`** - Visualization utilities
- **`visualize_samples_pure_json.py`** - Sample visualization

### Scripts & Automation
- **`run_evaluation.sh`** - Evaluation automation
- **`run_experiment.sh`** - Experiment runner
- **`infer_dataset.sh`** - Dataset inference

### Documentation
- **`DETAILED_ANALYSIS_USAGE.md`** - Analysis usage guide
- **`NEW_PIPELINE_USAGE.md`** - Pipeline usage documentation

## Visualization Tools (`vis_tools/`)

### Core Visualization
- **`vis_generation.py`** - Generation visualization
- **`vis_raw.py`** - Raw data visualization
- **`visualize_train.py`** - Training visualization

### Analysis Tools
- **`vis_scaling_comparison.py`** - Scaling comparison visualization

### Documentation
- **`README.md`** - Visualization tools overview
- **`README_scaling_comparison.md`** - Scaling comparison guide

### Output
- **`output/`** - Generated visualization outputs

## Utility Scripts (`scripts/`)

### Training Scripts
- **`run_train.sh`** - Training automation
- **`train.py`** - Training script
- **`zero2.json`** - DeepSpeed ZeRO-2 configuration

### Validation Scripts
- **`validate_config.py`** - Configuration validation
- **`validate_consistency.py`** - Data consistency validation
- **`validate_teacher_ratio.py`** - Teacher ratio validation
- **`validate_teacher_student_loss.py`** - Loss validation

### Utility Scripts
- **`lm_sanity_check.py`** - Language model sanity checks
- **`strip_exif.py`** - EXIF data processing

## Data Directories

### Processed Data (`data/`)
- **`train.jsonl`** - Training dataset
- **`val.jsonl`** - Validation dataset
- **`teacher.jsonl`** - Teacher examples
- **`label_vocabulary.json`** - Label definitions

### Raw Dataset (`ds/`)
- **Image files** - `.jpeg` format with EXIF data
- **Annotation files** - `.json` format with coordinates
- **Naming convention** - `QC-YYYYMMDD-NNNNNNN_IIIIIII.*`

## Key Architectural Patterns

### Modular Design
- **Separation of concerns** - Clear boundaries between components
- **Plugin architecture** - Configurable processing modules
- **Factory patterns** - Dynamic model and trainer creation

### Configuration-Driven
- **YAML-based configs** - Human-readable configuration
- **Environment integration** - Environment variable support
- **Type-safe validation** - Dataclass-based configuration

### Fail-Fast Philosophy
- **Early validation** - Catch errors before processing
- **Comprehensive logging** - Detailed error reporting
- **Graceful degradation** - Fallback mechanisms where appropriate

### Data Flow
1. **Raw data** (`ds/`) → **Data conversion** (`data_conversion/`)
2. **Processed data** (`data/`) → **Training** (`src/training/`)
3. **Trained models** → **Inference** (`src/inference.py`)
4. **Results** → **Evaluation** (`eval/`)

## Critical Documentation References

### Essential Reading
- **#[[file:docs/getting_started.md]]** - New user onboarding guide
- **#[[file:docs/architecture.md]]** - Complete system architecture
- **#[[file:docs/critical_fixes.md]]** - Known issues and solutions
- **#[[file:docs/data_schema.md]]** - Data format specifications
- **#[[file:docs/runbook.md]]** - Operational procedures

### Advanced Topics
- **#[[file:docs/advanced/teacher_student.md]]** - Teacher-student learning methodology
- **#[[file:docs/advanced/collator_notes.md]]** - Packed sequence collation internals
- **#[[file:docs/advanced/peft_adapter.md]]** - Parameter-efficient fine-tuning

### Troubleshooting
- **#[[file:docs/troubleshooting.md]]** - Common issues and solutions
- **#[[file:docs/lessons_learned.md]]** - Historical knowledge and pitfalls
- **#[[file:docs/testing.md]]** - Testing and validation procedures

## Import Conventions

- **Absolute imports** - Always use full module paths
- **First-party modules** - `src`, `data_conversion`, `vis_tools`
- **Type hints** - Comprehensive type annotations
- **Docstrings** - Google-style documentation