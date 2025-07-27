# Getting Started Guide (2025 Modular Architecture)

This guide provides step-by-step instructions for new contributors to get up and running with the Qwen2.5-VL BBU fine-tuning project using the current modular architecture.

## Prerequisites

### Network Considerations
- This project is designed for use in China and cannot access foreign websites like GitHub, Google, or HuggingFace directly
- Ensure you have access to local model mirrors and package repositories

## Step 1: Environment Setup

### 1.1 Activate Conda Environment
```bash
# Always activate the ms environment first
conda activate ms
```

**Important**: You must activate the `ms` environment before running any scripts in this project.

### 1.2 Set Environment Variables
```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your available GPUs

# Model cache directory (adjust path as needed)
export HF_HOME=/data4/swift/model_cache

# Optional: Add to your ~/.bashrc for persistence
echo 'export HF_HOME=/data4/swift/model_cache' >> ~/.bashrc
```

### 1.3 Verify Installation
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check required packages
python -c "import transformers, PIL, numpy, torch; print('All packages available')"
```

## Step 2: Understanding the Current Architecture

### 2.1 Project Overview (2025 Edition)
We fine-tune **Qwen-2.5-VL-7B** end-to-end for *simultaneous* multi-geometry object detection **and** captioning in BBU rooms. The system now features a **modular architecture** with clear separation of concerns:

- **Training System** (`src/training/`): BBUTrainer, TrainingCoordinator, LossManager
- **Model System** (`src/models/`): Unified model loading, patches, wrapper
- **Data Processing** (`data_conversion/`): 5-stage pipeline with object-oriented training
- **Configuration** (`src/config/`): Unified DirectConfig system

### 2.2 Canonical Data Schema
The project uses a teacher-student training format with this structure:
```jsonc
{
  "teachers": [
    {"images": ["ds_output/<img>.jpeg"], "objects": [{"bbox_2d": [x1,y1,x2,y2], "description": "螺丝连接点/BBU安装螺丝/连接正确"}]}
  ],
  "student": {
    "images": ["ds_output/<img>.jpeg"],
    "objects": [{"bbox_2d": [x1,y1,x2,y2], "description": "螺丝连接点/BBU安装螺丝/连接正确"}]
  }
}
```
Key facts: absolute pixel boxes, natural-language descriptions, pre-scaled JPEGs, and a teacher-student training format.

### 2.3 End-to-End Execution Flow
```
bash data_conversion/convert_dataset.sh → python -m src.training.trainer --config configs/base_flat_v2.yaml
         ↳ PipelineManager (5-stage processing)      ↳ ConfigManager.load_from_yaml() (Domain-specific configs)
         ↳ unified_processor.py (Core engine)        ↳ TrainingCoordinator (Orchestrates training)
         ↳ CoordinateManager (3-stage transforms)    ↳ LossManager (Multi-task loss computation)
                                                      ↳ BBUTrainer.train() (Enhanced HF trainer)
```

### 2.4 Key Directories
```
Qwen2.5-VL-main/
├── data_conversion/      # Data processing pipeline
│   ├── convert_dataset.sh    # Main entry point
│   ├── unified_processor.py  # Core processing logic
│   └── utils/                # Utility modules
├── src/                 # Training and inference code
│   ├── training/             # Training system
│   ├── detection/            # Object detection components
│   ├── models/               # Model management
│   └── inference.py          # Inference engine
├── configs/             # Configuration files
├── docs/                # Documentation (you're here!)
├── eval/                # Evaluation scripts
└── vis_tools/           # Visualization tools
```

### 2.2 Data Flow Overview
```
Raw Data (ds/) → Data Processing → Processed Data (data/) → Training → Model → Inference
```

## Step 3: Data Preparation

### 3.1 Prepare Your Raw Data
Your raw data should be organized as:
```
ds/
├── annotations.json     # Raw JSON annotations
├── image1.jpg          # Images
├── image2.jpg
└── ...
```

### 3.2 Run the Data Processing Pipeline
```bash
# Basic usage - processes ds/ to data/
bash data_conversion/convert_dataset.sh

# Custom input/output directories
INPUT_DIR="my_dataset" OUTPUT_DIR="my_processed_data" bash data_conversion/convert_dataset.sh

# Process English annotations instead of Chinese
LANGUAGE="english" bash data_conversion/convert_dataset.sh

# Disable image resizing for testing
RESIZE="false" bash data_conversion/convert_dataset.sh
```

### 3.3 Verify Processing Results
```bash
# Check that processing completed successfully
ls data/  # Should see train.jsonl, val.jsonl, teacher.jsonl

# Validate the processed data
python data_conversion/simple_validate.py

# View processing summary
cat data_conversion/processing_summary.json
```

## Step 4: Configuration

### 4.1 Understanding Configuration Files
```bash
# Main configuration file
cat configs/base_flat_v2.yaml

# Configuration documentation
cat docs/configuration.md
```

### 4.2 Key Configuration Parameters
- **Model Parameters**: Model size, learning rate, batch size
- **Detection Parameters**: Detection head configuration, loss weights
- **Training Parameters**: Epochs, validation frequency, checkpointing
- **Data Parameters**: Data paths, preprocessing options

### 4.3 Environment-Specific Configuration
Create a local configuration override:
```bash
# Copy base configuration
cp configs/base_flat_v2.yaml configs/my_config.yaml

# Edit for your specific needs
vim configs/my_config.yaml
```

## Step 5: Training

### 5.1 Start Training
```bash
# Basic training with default configuration
python -m src.training.trainer --config configs/base_flat_v2.yaml

# Training with custom configuration
python -m src.training.trainer --config configs/my_config.yaml

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.training.trainer --config configs/base_flat_v2.yaml
```

### 5.2 Monitor Training Progress
```bash
# Check training logs
tail -f run.log

# Monitor GPU usage
nvidia-smi -l 1
```

### 5.3 Training Checkpoints
Training checkpoints are saved to the directory specified in your configuration file. Look for:
- `checkpoint-best/` - Best performing checkpoint
- `checkpoint-latest/` - Most recent checkpoint
- `training_metrics.json` - Training metrics history

## Step 6: Inference

### 6.1 Single Image Inference
```bash
# Run inference on a single image
python src/inference.py \
    --model_path path/to/checkpoint \
    --image_path path/to/test_image.jpg

# With custom output format
python src/inference.py \
    --model_path path/to/checkpoint \
    --image_path path/to/test_image.jpg \
    --output_format json
```

### 6.2 Batch Inference
```bash
# Process multiple images
python src/inference.py \
    --model_path path/to/checkpoint \
    --image_dir path/to/test_images/ \
    --output_dir path/to/results/
```

## Step 7: Evaluation

### 7.1 Run Evaluation
```bash
# Evaluate on validation set
bash eval/run_evaluation.sh

# Run all evaluation tests
python eval/test_all_evaluations.py

# Validate results
python eval/validate_results.py
```

### 7.2 Visualization
```bash
# Generate visualization samples
python vis_tools/visualize_train.py

# Check generated visualizations
ls vis_tools/output/
```

## Common Workflows

### Development Workflow
1. **Make changes** to code
2. **Test changes** with small dataset
3. **Run validation** to ensure no regressions
4. **Update documentation** if needed
5. **Test full pipeline** before committing

### Training Workflow
1. **Prepare data** using the conversion pipeline
2. **Validate data** quality and format
3. **Configure training** parameters
4. **Start training** with monitoring
5. **Evaluate results** and iterate

### Debugging Workflow
1. **Check environment** setup and activation
2. **Validate data** processing pipeline
3. **Review logs** for error messages
4. **Consult troubleshooting guide** (docs/troubleshooting.md)
5. **Check critical fixes** (docs/critical_fixes.md) for known issues

## Best Practices

### Code Quality
- **Follow the fail-fast philosophy**: Let errors surface rather than hiding them
- **Use explicit configuration**: Never rely on hidden defaults
- **Validate inputs**: Check data format and parameters before processing
- **Update documentation**: Keep docs in sync with code changes

### Data Management
- **Backup original data** before processing
- **Validate processing results** before training
- **Use consistent naming** for datasets and experiments
- **Document data sources** and processing steps

### Training Management
- **Monitor resource usage** during training
- **Save regular checkpoints** for recovery
- **Track experiments** with clear naming and documentation
- **Validate models** before deployment

## Troubleshooting Quick Reference

### Common Issues
- **Environment not activated**: Always run `conda activate ms` first
- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **JSON parsing errors**: Check data format and run JSON cleaning
- **Missing dependencies**: Verify package installation in `ms` environment

### Getting Help
1. **Check the troubleshooting guide**: [docs/troubleshooting.md](troubleshooting.md)
2. **Review critical fixes**: [docs/critical_fixes.md](critical_fixes.md)
3. **Check lessons learned**: [docs/lessons_learned.md](lessons_learned.md)
4. **Review architecture docs**: [docs/architecture.md](architecture.md)

## Next Steps

Once you've completed this getting started guide:

1. **Read the architecture documentation** to understand the system design
2. **Review the data schema** to understand input/output formats
3. **Study the configuration guide** for advanced parameter tuning
4. **Explore the advanced documentation** for specialized topics
5. **Contribute to the project** by fixing issues or adding features

## Quick Reference Commands

```bash
# Environment setup
conda activate ms
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Data processing
bash data_conversion/convert_dataset.sh

# Training
python -m src.training.trainer --config configs/base_flat_v2.yaml

# Inference
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg

# Evaluation
bash eval/run_evaluation.sh

# Validation
python data_conversion/simple_validate.py
```

---

Welcome to the project! This getting started guide should have you up and running quickly. For more detailed information, explore the other documentation files in the `docs/` directory.