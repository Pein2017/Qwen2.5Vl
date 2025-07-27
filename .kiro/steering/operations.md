# Operational Procedures and Best Practices

## Environment Setup (MANDATORY)

### Always Required Before Any Operation
```bash
# CRITICAL: Must activate conda environment first
conda activate ms

# Set required environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data4/swift/model_cache
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
```

## Standard Workflows

### Data Processing Pipeline (5-Stage Process)
```bash
# Complete pipeline - processes raw data in ds/ to training data in data/
bash data_conversion/convert_dataset.sh

# Custom processing with specific directories
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" bash data_conversion/convert_dataset.sh

# Always validate after processing
python data_conversion/simple_validate.py
```

### Training Workflow
```bash
# Primary training command
python -m src.training.trainer --config configs/base_flat_v2.yaml

# Alternative via script
bash scripts/run_train.sh

# Validation before training
python scripts/validate_config.py --config base_flat_v2
python scripts/validate_consistency.py
```

### Inference Operations
```bash
# Single image inference
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg

# Batch inference on dataset
python src/inference.py --model_path path/to/checkpoint --dataset_path data.jsonl
```

## Critical Best Practices

### Fail-Fast Philosophy
- **Never suppress errors** - Let them surface with clear messages
- **Validate early** - Check inputs before processing
- **Use explicit configuration** - No hidden defaults
- **Log comprehensively** - Track all operations

### Data Management
- **Backup original data** before any processing
- **Validate processing results** before training
- **Use consistent naming** for datasets and experiments
- **Check data format** with validation scripts

### Training Safety
- **Monitor GPU memory** during training
- **Save regular checkpoints** for recovery
- **Validate model architecture** before training
- **Check loss convergence** patterns

## Common Validation Commands

### Pre-Training Validation
```bash
# Configuration validation
python scripts/validate_config.py --config base_flat_v2

# Data consistency checks
python scripts/validate_consistency.py
python data_conversion/simple_validate.py

# Teacher-student loss validation
python scripts/validate_teacher_student_loss.py

# Teacher ratio validation
python scripts/validate_teacher_ratio.py
```

### System Health Checks
```bash
# CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Package verification
python -c "import transformers, PIL, numpy, torch; print('All packages available')"

# GPU memory status
nvidia-smi
```

## Emergency Procedures

### Training Issues
```bash
# Kill all training processes
pkill -f "python -m src.training.trainer"

# Clear cache if memory issues
rm -rf ~/.cache/huggingface/
rm -rf /tmp/tmp*
rm -rf ~/.cache/torch_extensions/

# Check disk space
df -h
```

### Data Pipeline Issues
```bash
# Clean and restart pipeline
rm -rf data/
bash data_conversion/convert_dataset.sh

# Validate specific components
python data_conversion/test_pipeline.py
python data_conversion/clean_raw_json.py
```

## Documentation References

### Essential Reading Order
1. **#[[file:docs/getting_started.md]]** - Start here for new users
2. **#[[file:docs/architecture.md]]** - Understand system design
3. **#[[file:docs/data_schema.md]]** - Learn data formats
4. **#[[file:docs/runbook.md]]** - Operational commands

### Troubleshooting Resources
- **#[[file:docs/troubleshooting.md]]** - Common issues and solutions
- **#[[file:docs/critical_fixes.md]]** - Known bugs and fixes
- **#[[file:docs/lessons_learned.md]]** - Historical pitfalls to avoid

### Advanced Topics
- **#[[file:docs/advanced/teacher_student.md]]** - Teacher-student methodology
- **#[[file:docs/advanced/collator_notes.md]]** - Memory optimization details
- **#[[file:docs/testing.md]]** - Comprehensive testing procedures

## Key System Constraints

### Hardware Requirements
- **GPU Memory**: 24GB+ recommended for training
- **Flash Attention 2**: Mandatory (Ampere architecture or newer)
- **CUDA 12.x**: Required for GPU acceleration
- **Fast Storage**: SSD recommended for training data

### Network Considerations
- **China-based deployment** - Cannot access GitHub, Google, HuggingFace directly
- **Local model mirrors** required
- **Package repositories** must be accessible locally

### Critical File Paths
```bash
# Data paths
ds/                     # Raw data input
data/                   # Processed training data  
ds_output/              # Processed images and JSON

# Configuration
configs/base_flat_v2.yaml  # Main training config
scripts/zero2.json         # DeepSpeed config

# Outputs
checkpoints/            # Training checkpoints
vis_tools/output/       # Visualization outputs
run.log                 # Training logs
```

## Performance Optimization

### Memory Efficiency
- **Packed Collation**: Achieves 100% GPU memory utilization vs ~70% with padding
- **Mixed Precision**: BFloat16 for memory efficiency
- **Gradient Accumulation**: For large effective batch sizes
- **DeepSpeed ZeRO-2**: For large-scale model training

### Training Stability
- **Gradient Clipping**: Prevents exploding gradients
- **Loss Monitoring**: Track both teacher and student losses
- **Checkpoint Frequency**: Regular saves for recovery
- **Validation Checks**: Monitor convergence patterns