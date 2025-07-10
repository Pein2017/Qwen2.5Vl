# Qwen2.5-VL BBU Fine-tuning Project

This project implements end-to-end fine-tuning of Qwen2.5-VL for BBU (Base-Band Unit) equipment detection and captioning. The system features sophisticated multi-task training with teacher-student learning and DETR-style object detection, supporting both English and Chinese annotations.

## 🚀 Quick Start

### Prerequisites
- Conda environment: `ms`
- CUDA-compatible GPU
- Python 3.8+

### Environment Setup
```bash
# Activate the conda environment
conda activate ms

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your GPU setup
export HF_HOME=/data4/swift/model_cache  # Model cache directory
```

### Data Processing Pipeline
```bash
# Process your dataset (from raw JSON + images to training-ready JSONL)
bash data_conversion/convert_dataset.sh

# With custom settings
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" bash data_conversion/convert_dataset.sh
```

### Training
```bash
# Start training with your processed data
python -m src.training.trainer --config configs/base_flat_v2.yaml
```

### Inference
```bash
# Run inference on new images
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg
```

## 📋 Project Structure

```
├── data_conversion/           # Data processing pipeline
│   ├── convert_dataset.sh    # Main pipeline script
│   ├── unified_processor.py  # Core processing engine
│   └── utils/                # Processing utilities
├── src/                      # Training and inference code
│   ├── training/             # Training system
│   ├── detection/            # DETR-style detection
│   ├── models/               # Model management
│   └── inference.py          # Inference engine
├── configs/                  # Configuration files
├── docs/                     # Documentation
└── eval/                     # Evaluation scripts
```

## 🎯 Key Features

### Advanced Data Processing
- **5-Stage Pipeline**: JSON cleaning → Token mapping → Sample processing → Validation → Summary
- **Multi-Format Support**: Handles both `dataList` and `markResult` JSON formats
- **Coordinate Transformation**: 3-stage system with EXIF orientation, dimension rescaling, and smart resize
- **Teacher-Student Selection**: Intelligent teacher pool creation for improved training

### Sophisticated Training System
- **Multi-Task Learning**: Combined VLM training with object detection
- **DETR-Style Detection**: Hungarian matching with dynamic loss weighting
- **Teacher-Student Learning**: Span-based loss splitting for enhanced learning
- **Model Patches**: mRoPE integration, Flash Attention 2, and visual processing fixes

### Production-Ready Features
- **Fail-Fast Validation**: Comprehensive error detection and prevention
- **Unified Configuration**: Type-safe configuration system with environment support
- **Robust Inference**: Standalone inference engine with batch processing
- **Comprehensive Logging**: Detailed progress tracking and error reporting

## 📚 Documentation

### Getting Started
- [Getting Started Guide](docs/getting_started.md) - Step-by-step tutorial for new contributors (includes project overview)
- [Architecture](docs/architecture.md) - Technical architecture and design decisions

### Data Processing
- [Data Schema](docs/data_schema.md) - Input/output formats and pipeline details (includes JSON cleaning)
- [Runbook](docs/runbook.md) - Operational procedures and workflows

### Training & Models
- [Configuration Guide](docs/configuration.md) - Parameter settings and optimization
- [Teacher-Student Learning](docs/advanced/teacher_student.md) - Advanced training techniques
- [Testing Guide](docs/testing.md) - Testing procedures and validation workflows

### Troubleshooting
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions
- [Critical Fixes](docs/critical_fixes.md) - Comprehensive bug fixes and solutions (consolidated)
- [Lessons Learned](docs/lessons_learned.md) - Historical knowledge and pitfalls

## 🛠️ Development

### Code Structure
The project follows a modular architecture with clear separation of concerns:
- **Data Processing**: Centralized in `data_conversion/` with unified interfaces
- **Training**: Modular training system in `src/training/` with configurable components
- **Detection**: DETR-style detection head in `src/detection/` with Hungarian matching
- **Models**: Unified model management in `src/models/` with patching system

### Testing
```bash
# Run the test suite
python -m pytest eval/test_all_evaluations.py

# Validate pipeline output
python data_conversion/simple_validate.py
```

### Configuration
The project uses a sophisticated configuration system supporting:
- Type-safe configuration with dataclasses
- Environment variable integration
- Domain-specific parameter groups
- Automatic validation and defaults

## 🔧 Advanced Usage

### Custom Data Formats
The pipeline supports extending to new annotation formats by:
1. Adding format handlers in `data_conversion/core_modules.py`
2. Updating the unified processor configuration
3. Adding validation rules in `utils/validators.py`

### Model Customization
Extend the model system by:
1. Adding new detection heads in `src/detection/`
2. Implementing custom loss functions in `src/training/loss_manager.py`
3. Adding model patches in `src/models/patches.py`

### Performance Optimization
- **Memory Management**: Gradient checkpointing and mixed precision training
- **Data Loading**: Optimized collation and batch processing
- **Inference**: Batch processing and caching for production workloads

## 🚨 Important Notes

### Environment Requirements
- **Always activate the `ms` conda environment** before running any scripts
- **Set CUDA_VISIBLE_DEVICES** appropriately for your GPU configuration
- **Configure HF_HOME** for model cache management

### Data Pipeline Caveats
- **Coordinate Systems**: The pipeline handles complex coordinate transformations - see [lessons learned](docs/lessons_learned.md)
- **Image Formats**: EXIF orientation handling is critical for annotation accuracy
- **Validation**: Always run pipeline validation before training

### Training Stability
- **Loss Balancing**: Detection and VLM losses are dynamically weighted
- **Gradient Clipping**: Implemented to prevent training instability
- **Checkpointing**: Regular checkpointing with validation metrics

## 📊 Performance

### Benchmarks
- **Data Processing**: ~1000 images/minute on modern hardware
- **Training**: Support for multi-GPU training with gradient accumulation
- **Inference**: Real-time inference on single images, batch processing for datasets

### Resource Requirements
- **GPU Memory**: 24GB+ recommended for training (depends on batch size)
- **CPU**: Multi-core recommended for data processing
- **Storage**: Fast SSD recommended for training data

## 🤝 Contributing

1. Read the [Getting Started Guide](docs/getting_started.md)
2. Check the [Architecture Documentation](docs/architecture.md)
3. Review [Critical Fixes](docs/critical_fixes.md) for known issues
4. Follow the fail-fast development philosophy
5. Update documentation for any changes

## 📄 License

This project is for internal use and research purposes.

## 🙏 Acknowledgments

Built on top of the Qwen2.5-VL foundation model by Alibaba Cloud, with significant enhancements for BBU equipment detection and multi-task learning.

---

For detailed documentation, see the [docs/](docs/) directory. For quick operational procedures, check the [runbook](docs/runbook.md).