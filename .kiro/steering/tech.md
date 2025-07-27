# Technology Stack

## Core Framework
- **Python 3.10+** - Primary development language (3.8+ supported)
- **PyTorch 2.5.1** - Deep learning framework with CUDA support
- **Transformers 4.51.3** - Hugging Face transformers library
- **Flash Attention 2** - Mandatory performance optimization (no fallback)
- **xFormers** - Memory-efficient attention implementations

## Model Architecture
- **Qwen2.5-VL-3B-Instruct** - Base foundation model
- **DETR-style Detection** - Object detection with Hungarian matching
- **mRoPE Integration** - Rotary position embeddings
- **Mixed Precision Training** - BFloat16 for memory efficiency
- **Teacher-Student Learning** - Advanced training methodology

## Critical Architecture Notes
- **Modular Design**: Refactored from monolithic (2100+ line trainer) to component-based architecture
- **Training Coordinator**: `src/training/training_coordinator.py` orchestrates training components
- **Loss Manager**: `src/training/loss_manager.py` handles multi-task loss computation
- **Model Patches**: Critical fixes in `src/models/patches.py` for mRoPE, Flash Attention compatibility
- **Packed Collation**: Custom collator for 100% GPU memory utilization vs ~70% with padding

## Key Dependencies
- **CUDA 12.x** - GPU acceleration (CPU mode not supported)
- **Accelerate 1.6.0** - Distributed training support
- **PEFT 0.15.1** - Parameter-efficient fine-tuning
- **Datasets 3.2.0** - Data loading and processing
- **OpenCV 4.11.0** - Image processing and transformations
- **Pillow 11.1.0** - Image handling and EXIF processing
- **DeepSpeed 0.16.7** - Large-scale model training optimization
- **TRL 0.16.0** - Transformer Reinforcement Learning

## Additional Tools & Libraries

### Evaluation & Metrics
- **COCO Metrics** - Object detection evaluation
- **ROUGE** - Text generation evaluation (Chinese & English)
- **BLEU/SacreBLEU** - Translation quality metrics
- **Sentence Transformers 4.1.0** - Semantic similarity evaluation

### Data Processing & Analysis
- **Pandas 2.2.3** - Data manipulation and analysis
- **NumPy 1.26.4** - Numerical computing
- **Matplotlib 3.10.1** - Plotting and visualization
- **Seaborn 0.13.2** - Statistical visualization
- **Scikit-learn 1.6.1** - Machine learning utilities

### Development & Debugging
- **Jupyter** - Interactive development environment
- **TensorBoard 2.19.0** - Training visualization
- **Gradio 5.23.3** - Web-based demos and interfaces
- **Rich 14.0.0** - Enhanced terminal output

### Language Processing
- **Jieba 0.42.1** - Chinese text segmentation
- **OpenCC 1.1.9** - Chinese text conversion
- **Tiktoken 0.9.0** - Token counting and encoding

## Environment Management
- **Conda Environment**: `ms` (must be activated for all operations)
- **Model Cache**: `/data4/swift/model_cache` (set via HF_HOME)
- **CUDA Devices**: Configure via `CUDA_VISIBLE_DEVICES`

## Common Commands

### Environment Setup
```bash
# Always activate conda environment first
conda activate ms

# Set required environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data4/swift/model_cache
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
```

### Data Processing
```bash
# Run complete data conversion pipeline
bash data_conversion/convert_dataset.sh

# Custom data processing
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" bash data_conversion/convert_dataset.sh
```

### Training
```bash
# Start training with configuration
python -m src.training.trainer --config configs/base_flat_v2.yaml

# Multi-GPU training (automatic detection)
python -m src.training.trainer --config configs/base_flat_v2.yaml

# Alternative training script
bash scripts/run_train.sh
```

### Inference
```bash
# Single image inference
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg

# Batch inference on dataset
python src/inference.py --model_path path/to/checkpoint --dataset_path data.jsonl
```

### Testing & Validation
```bash
# Run test suite
python -m pytest eval/test_all_evaluations.py

# Validate pipeline output
python data_conversion/simple_validate.py

# Check data processing results
python data_conversion/test_pipeline.py

# Run evaluation pipeline
bash eval/run_evaluation.sh

# Validate configuration
python scripts/validate_config.py
```

### Visualization & Analysis
```bash
# Generate training visualizations
python vis_tools/visualize_train.py

# Create scaling comparisons
python vis_tools/vis_scaling_comparison.py

# Detailed analysis
python eval/detailed_analysis.py
```

## Build System
- **No traditional build system** - Python-based with direct execution
- **Configuration-driven** - YAML configs in `configs/` directory
- **Modular architecture** - Import-based module system
- **Ruff 0.11.2** - Fast Python linter and formatter
- **Type checking** - Comprehensive type hints with validation

## Performance Requirements
- **GPU Memory**: 24GB+ recommended for training
- **Flash Attention 2**: Mandatory (Ampere architecture or newer)
- **Fast Storage**: SSD recommended for training data
- **Multi-core CPU**: For data processing pipeline

## Development Tools
- **Ruff Configuration**: Defined in `pyproject.toml`
  - Import organization with known first-party modules
  - Code formatting with double quotes
  - Linting with F, E, W, I rules
- **Jupyter Integration**: Full notebook support for development
- **DeepSpeed Integration**: ZeRO-2 configuration in `scripts/zero2.json`