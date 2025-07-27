# Runbook: Operational Commands and Workflows

> This file provides comprehensive shell commands and Python snippets for all common operations in the project.

---

## Environment Setup

### Activate Environment
```bash
# Always activate the ms environment first
conda activate ms

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data4/swift/model_cache
```

### Verify Environment
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check GPU memory
nvidia-smi

# Verify packages
python -c "import transformers, PIL, numpy; print('All packages available')"
```

---

## Data Processing Pipeline

### Basic Data Conversion
```bash
# Process default dataset (ds/ → data/)
conda activate ms
bash data_conversion/convert_dataset.sh
```

### Custom Data Conversion
```bash
# Custom input/output directories
INPUT_DIR="my_dataset" OUTPUT_DIR="my_output" bash data_conversion/convert_dataset.sh

# Process English annotations
LANGUAGE="english" bash data_conversion/convert_dataset.sh

# Disable image resizing for testing
RESIZE="false" bash data_conversion/convert_dataset.sh

# Custom teacher ratio
TEACHER_RATIO="0.8" bash data_conversion/convert_dataset.sh
```

### Pipeline Validation
```bash
# Validate pipeline output
python data_conversion/simple_validate.py

# View processing summary
cat data_conversion/processing_summary.json

# Check specific validation
python -c "
from data_conversion.utils.validators import DataValidator
validator = DataValidator()
validator.validate_pipeline_output('data/')
"
```

### Individual Pipeline Components
```bash
# Run only JSON cleaning
python data_conversion/clean_raw_json.py

# Test coordinate transformation
python -c "
from data_conversion.coordinate_manager import CoordinateManager
cm = CoordinateManager()
result = cm.transform_coordinates([100, 100, 200, 200], (1920, 1440), (728, 532))
print(f'Transformed: {result}')
"

# Test teacher selection
python -c "
from data_conversion.teacher_selector import TeacherSelector
selector = TeacherSelector()
teachers = selector.select_teachers(samples, config)
print(f'Selected {len(teachers)} teachers')
"
```

---

## Training Operations

### Standard Training
```bash
# Basic training with default configuration
conda activate ms
python -m src.training.trainer --config configs/base_flat_v2.yaml

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.training.trainer --config configs/base_flat_v2.yaml

# Training with custom configuration
python -m src.training.trainer --config configs/my_config.yaml
```

### Advanced Training Options
```bash
# Training with gradient checkpointing
python -m src.training.trainer --config configs/base_flat_v2.yaml --gradient_checkpointing

# Mixed precision training
python -m src.training.trainer --config configs/base_flat_v2.yaml --fp16

# Resume from checkpoint
python -m src.training.trainer --config configs/base_flat_v2.yaml --resume_from_checkpoint path/to/checkpoint

# Validation only
python -m src.training.trainer --config configs/base_flat_v2.yaml --do_train false --do_eval true
```

### Training with DeepSpeed
```bash
# Enable DeepSpeed ZeRO-2
export BBU_DEEPSPEED_ENABLED=true
export BBU_DEEPSPEED_CONFIG=configs/deepspeed_zero2.json
python -m src.training.trainer --config configs/base_flat_v2.yaml

# DeepSpeed with custom configuration
deepspeed --num_gpus=4 src/training/trainer.py --config configs/base_flat_v2.yaml --deepspeed configs/deepspeed_zero3.json
```

### Training Monitoring
```bash
# Monitor training progress
tail -f run.log

# Watch GPU utilization
nvidia-smi -l 1

# Check training metrics
python -c "
import json
with open('training_metrics.json') as f:
    metrics = json.load(f)
    print(f'Latest loss: {metrics[-1][\"loss\"]}')
"
```

---

## Inference Operations

### Basic Inference
```python
# Single image inference
from src.inference import Inference
inference = Inference('path/to/checkpoint')
result = inference.predict_detection('path/to/image.jpg')
print(f'Detected: {result}')
```

### Command Line Inference
```bash
# Single image
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg

# Batch processing
python src/inference.py --model_path path/to/checkpoint --image_dir path/to/images/ --output_dir results/

# Custom prompt
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg --prompt "请描述这张图片中的BBU设备"

# Output format options
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg --output_format json
```

### Advanced Inference
```python
# Batch inference with custom settings
from src.inference import Inference
inference = Inference('path/to/checkpoint', batch_size=8)

# Process multiple images
results = inference.batch_predict([
    'image1.jpg', 'image2.jpg', 'image3.jpg'
], prompts=['Describe BBU equipment'] * 3)

# Custom configuration
inference = Inference('path/to/checkpoint', config={
    'max_length': 512,
    'temperature': 0.7,
    'top_p': 0.9
})
```

---

## Model Management

### Model Loading and Validation
```python
# Load model with validation
from src.models.model_loader import ModelLoader
loader = ModelLoader('path/to/checkpoint')
model = loader.load_model()

# Validate model components
from src.models.wrapper import Qwen25VLWithDetection
model = Qwen25VLWithDetection.from_pretrained('path/to/checkpoint')
print(f'Model loaded: {model.config}')
```

### Model Conversion and Export
```bash
# Convert checkpoint format
python -c "
from src.models.model_loader import ModelLoader
loader = ModelLoader('old_checkpoint')
loader.save_model('new_checkpoint', format='safetensors')
"

# Export model for inference
python src/models/export_model.py --checkpoint path/to/checkpoint --output path/to/exported_model
```

### Model Patching
```python
# Apply model patches
from src.models.patches import apply_model_patches
model = apply_model_patches(model, config)

# Verify patches
from src.models.patches import verify_mrope_fix
verify_mrope_fix(model)
```

---

## Evaluation and Testing

### Evaluation Scripts
```bash
# Run complete evaluation
bash eval/run_evaluation.sh

# Individual evaluation components
python eval/validate_results.py
python eval/test_all_evaluations.py

# Detailed analysis
python eval/detailed_analysis.py --dataset_path data/val.jsonl --checkpoint_path path/to/checkpoint
```

### Testing and Validation
```bash
# Run all tests
python -m pytest eval/test_all_evaluations.py -v

# Test specific components
python -m pytest data_conversion/test_pipeline.py -v
python -m pytest src/training/test_trainer.py -v

# Validate data processing
python data_conversion/simple_validate.py
```

### Performance Testing
```bash
# Benchmark data processing
python -c "
import time
from data_conversion.unified_processor import UnifiedProcessor
start = time.time()
processor = UnifiedProcessor()
processor.process_dataset()
print(f'Processing time: {time.time() - start:.2f}s')
"

# Benchmark inference
python -c "
import time
from src.inference import Inference
inference = Inference('path/to/checkpoint')
start = time.time()
result = inference.predict_detection('test_image.jpg')
print(f'Inference time: {time.time() - start:.2f}s')
"
```

---

## Visualization and Analysis

### Generate Visualizations
```bash
# Generate training visualizations
python vis_tools/visualize_train.py --data_path data/train.jsonl --output_dir vis_tools/output/

# Visualize model outputs
python vis_tools/vis_generation.py --model_path path/to/checkpoint --image_path path/to/image.jpg

# Create sample visualizations
python eval/visualize_samples_pure_json.py --input data/val.jsonl --output vis_tools/output/
```

### Analysis Tools
```bash
# Analyze dataset statistics
python -c "
from data_conversion.utils.validators import DataValidator
validator = DataValidator()
stats = validator.analyze_dataset('data/train.jsonl')
print(stats)
"

# Check label distribution
python -c "
import json
from collections import Counter
with open('data/train.jsonl') as f:
    labels = []
    for line in f:
        sample = json.loads(line)
        for obj in sample.get('objects', []):
            labels.append(obj['object_type'])
    print(Counter(labels))
"
```

---

## Configuration Management

### Configuration Validation
```bash
# Validate configuration file
python -c "
from src.config.config_manager import ConfigManager
config = ConfigManager('configs/base_flat_v2.yaml')
print('Configuration valid')
"

# Check configuration domains
python -c "
from src.config.config_manager import ConfigManager
config = ConfigManager('configs/base_flat_v2.yaml')
print(f'Domains: {list(config.domains.keys())}')
"
```

### Configuration Utilities
```bash
# Generate default configuration
python src/config/generate_default_config.py --output configs/default.yaml

# Merge configurations
python src/config/merge_configs.py --base configs/base_flat_v2.yaml --override configs/custom.yaml --output configs/merged.yaml

# Validate cross-dependencies
python -c "
from src.config.config_manager import ConfigManager
config = ConfigManager('configs/base_flat_v2.yaml')
config.validate_cross_dependencies()
print('Cross-dependencies valid')
"
```

---

## Debugging and Troubleshooting

### Debug Data Processing
```bash
# Debug with verbose output
VERBOSE=true bash data_conversion/convert_dataset.sh

# Check specific sample processing
python -c "
from data_conversion.sample_processor import SampleProcessor
processor = SampleProcessor()
result = processor.process_sample(sample_data)
print(f'Processed: {result}')
"

# Validate coordinate transformations
python -c "
from data_conversion.coordinate_manager import CoordinateManager
cm = CoordinateManager()
cm.debug_transformations = True
result = cm.transform_coordinates([100, 100, 200, 200], (1920, 1440), (728, 532))
"
```

### Debug Training
```bash
# Training with debug output
python -m src.training.trainer --config configs/base_flat_v2.yaml --log_level DEBUG

# Check model components
python -c "
from src.models.wrapper import Qwen25VLWithDetection
model = Qwen25VLWithDetection.from_pretrained('path/to/checkpoint')
for name, param in model.named_parameters():
    print(f'{name}: {param.shape}')
"

# Debug loss computation
python -c "
from src.training.loss_manager import LossManager
loss_manager = LossManager(config)
# Debug with sample data
"
```

### System Diagnostics
```bash
# Check system resources
free -h
df -h
nvidia-smi

# Check Python environment
python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
"

# Check file permissions
find data_conversion/ -name "*.py" -exec ls -la {} \;
```

---

## Maintenance Operations

### Cleanup Operations
```bash
# Clean temporary files
rm -rf data_conversion/temp/
rm -rf vis_tools/output/temp_*

# Clean old checkpoints
find checkpoints/ -name "checkpoint-*" -mtime +30 -delete

# Clean cache
rm -rf ~/.cache/huggingface/
```

### Backup Operations
```bash
# Backup configuration
cp -r configs/ configs_backup_$(date +%Y%m%d)/

# Backup processed data
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# Backup checkpoints
rsync -av checkpoints/ backup_server:/path/to/backup/
```

---

## Quick Reference

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3        # GPU selection
export HF_HOME=/data4/swift/model_cache     # Model cache
export BBU_DEEPSPEED_ENABLED=true           # Enable DeepSpeed
export BBU_DEEPSPEED_CONFIG=configs/zero2.json  # DeepSpeed config
```

### Common File Paths
```bash
# Data paths
ds/                     # Raw data input
data/                   # Processed training data  
ds_output/              # Processed images and JSON

# Configuration
configs/base_flat_v2.yaml  # Main training config
configs/deepspeed_zero2.json  # DeepSpeed config

# Outputs
checkpoints/            # Training checkpoints
vis_tools/output/       # Visualization outputs
run.log                 # Training logs
```

### Emergency Commands
```bash
# Kill all training processes
pkill -f "python -m src.training.trainer"

# Check disk space
df -h

# Check GPU memory
nvidia-smi

# Force cleanup
rm -rf /tmp/tmp* && rm -rf ~/.cache/torch_extensions/
```

---

## System Operations and Diagnostics

### Special Tokens and Validation
The system uses specific tokens that are validated at startup:

| Token | String | Purpose |
|-------|--------|---------|
| IM_START | `<|im_start|>` | Start of chat turn |
| IM_END   | `<|im_end|>`   | End of turn & sequence |
| VISION_START | `<|vision_start|>` | Vision prefix |
| VISION_END   | `<|vision_end|>`   | Vision suffix |
| IMAGE_PAD    | `<|image_pad|>`    | One image patch |

### Fail-Fast Guards
The system implements comprehensive validation before training:
- **Configuration Validation**: ConfigManager validates domain-specific configs and cross-dependencies
- **Path Consistency**: All data paths are checked for existence
- **Parameter Grouping**: Ensures all trainable parameters are assigned to learning rate groups

### Monitoring Hooks
Training metrics are logged by different components:

| Metric | Logged by | Location |
|--------|-----------|----------|
| `lm_loss`, `teacher_lm_loss`, `student_lm_loss` | `LossManager` | `training/loss_manager.py` |
| `bbox_*`, `objectness_loss`, `caption_loss` | `LossManager` | `training/loss_manager.py` |
| Gradient / weight norms | `BBUTrainer` | `training/trainer.py` |

### Common Pitfalls and Quick Fixes
| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| "BOX BOX BOX" string, no coords | Using `model.generate()` instead of detection pipeline | Use `Inference.predict_detection()` |
| Vision token mismatch assertion | Images not processed by ChatProcessor | Check image preprocessing |
| `ConfigValidation-Error` | Missing/invalid YAML parameters | Validate config file |
| `KeyError` during optimizer creation | Parameter not assigned to group | Check ParameterGroupManager |

### Packed Collator Assertions
The `PackedDataCollator` includes safety checks:
1. **Boundary masking**: Labels at sample boundaries set to `-100` to prevent cross-sample supervision
2. **Position-ID reset**: Verifies `position_ids[0, cu_seqlens[:-1]] == 0` for rotary cache reset

---

This runbook provides comprehensive operational commands for all aspects of the project. For detailed explanations of concepts and architecture, see the other documentation files.