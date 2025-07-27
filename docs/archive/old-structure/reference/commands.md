# Commands Reference

**Complete command-line reference for the BBU Detection System**

## 🔄 Data Processing Commands

### Main Data Pipeline
```bash
# Process raw data using shell script (recommended)
bash data_conversion/convert_dataset.sh

# Process with specific object types
export OBJECT_TYPES="bbu label fiber"
bash data_conversion/convert_dataset.sh

# Process with custom settings
export OBJECT_TYPES="full"
export RESIZE_IMAGES="true"
export VAL_RATIO="0.15"
bash data_conversion/convert_dataset.sh
```

### Python Pipeline Manager
```bash
# Basic processing
python data_conversion/pipeline_manager.py \
    --input_dir ds_v2 \
    --output_dir data \
    --object_types "bbu label fiber" \
    --resize true \
    --val_ratio 0.1

# Advanced processing with custom settings
python data_conversion/pipeline_manager.py \
    --input_dir ds_v2 \
    --output_dir data \
    --object_types "full" \
    --resize true \
    --val_ratio 0.1 \
    --teacher_ratio 0.1 \
    --parallel_workers 4 \
    --batch_size 100

# Resume from specific stage
python data_conversion/pipeline_manager.py \
    --resume_from_stage 3 \
    --input_dir ds_v2 \
    --output_dir data
```

### Individual Pipeline Stages
```bash
# Stage 1: Clean raw JSON files
python data_conversion/clean_raw_json.py \
    ds_v2 ds_v2_clean --lang zh

# Stage 3: Process samples (core processing)
python data_conversion/unified_processor.py \
    --input_dir ds_v2_clean \
    --output_dir data_processed \
    --object_types "bbu label"

# Stage 4: Split data
python data_conversion/data_splitter.py \
    --input_file data_processed/samples.jsonl \
    --output_dir data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --teacher_ratio 0.1
```

## 🏋️ Training Commands

### Main Training Script
```bash
# Basic training
python scripts/train.py --config configs/base_flat_v2.yaml

# Training with custom output directory
python scripts/train.py \
    --config configs/base_flat_v2.yaml \
    --output_dir ./output/my_experiment

# Training with parameter overrides
python scripts/train.py \
    --config configs/base_flat_v2.yaml \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 5

# Training with custom config
python scripts/train.py \
    --config configs/my_custom_config.yaml \
    --output_dir ./output/custom_training
```

### Distributed Training
```bash
# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/distributed_training.yaml

# Multi-node training (2 nodes, 4 GPUs each)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    scripts/train.py \
    --config configs/distributed_training.yaml
```

### Training with Shell Script
```bash
# Use the training shell script
bash scripts/run_train.sh

# With custom parameters
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_train.sh
```

## 🤖 Inference Commands

### Single Image Inference
```bash
# Basic inference
python src/inference.py \
    --model_path output/final_model \
    --image_path path/to/test_image.jpg \
    --output_file results.json

# Inference with custom prompt
python src/inference.py \
    --model_path output/final_model \
    --image_path path/to/test_image.jpg \
    --prompt "请详细描述图像中所有的BBU设备。" \
    --output_file detailed_results.json

# Inference with visualization
python src/inference.py \
    --model_path output/final_model \
    --image_path path/to/test_image.jpg \
    --output_file results.json \
    --save_visualization \
    --visualization_path annotated_image.jpg
```

### Batch Inference
```bash
# Process directory of images
python src/inference.py \
    --model_path output/final_model \
    --image_dir path/to/test_images/ \
    --output_dir results/ \
    --batch_size 4

# Batch inference with filtering
python src/inference.py \
    --model_path output/final_model \
    --image_dir path/to/test_images/ \
    --output_dir results/ \
    --batch_size 4 \
    --confidence_threshold 0.8 \
    --max_objects 10
```

### Advanced Inference Options
```bash
# Inference with custom generation parameters
python src/inference.py \
    --model_path output/final_model \
    --image_path path/to/test_image.jpg \
    --output_file results.json \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --do_sample false

# Inference with specific object types
python src/inference.py \
    --model_path output/final_model \
    --image_path path/to/test_image.jpg \
    --output_file results.json \
    --object_types "bbu label" \
    --detailed_descriptions
```

## ⚙️ Configuration Commands

### Configuration Validation
```bash
# Validate configuration file
python -c "
import yaml
from src.config import get_config
import os
os.environ['CONFIG_PATH'] = 'configs/my_config.yaml'
config = get_config()
print('✅ Configuration valid')
"

# Check specific parameters
python -c "
from src.config import get_config
config = get_config()
print(f'Model path: {config.model_path}')
print(f'Batch size: {config.per_device_train_batch_size}')
print(f'Detection enabled: {config.detection_enabled}')
"
```

### Configuration Creation
```bash
# Copy and modify base configuration
cp configs/base_flat_v2.yaml configs/my_experiment.yaml

# Update specific parameters
sed -i 's/learning_rate: 1e-5/learning_rate: 2e-5/' configs/my_experiment.yaml
sed -i 's/num_train_epochs: 3/num_train_epochs: 5/' configs/my_experiment.yaml
sed -i 's/per_device_train_batch_size: 2/per_device_train_batch_size: 1/' configs/my_experiment.yaml

# Validate changes
python -c "
import yaml
with open('configs/my_experiment.yaml') as f:
    config = yaml.safe_load(f)
print('Updated parameters:')
print(f'Learning rate: {config[\"learning_rate\"]}')
print(f'Epochs: {config[\"num_train_epochs\"]}')
print(f'Batch size: {config[\"per_device_train_batch_size\"]}')
"
```

## 🔍 System Validation Commands

### Environment Health Checks
```bash
# Check Python environment
/root/miniconda3/envs/ms/bin/python --version

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Check project structure
ls -la | grep -E "(src|data_conversion|scripts|configs)"
```

### Component Health Checks
```bash
# Test core imports
python -c "
from src.training.trainer import BBUTrainer
from src.models.model_loader import load_model_and_processor_unified
from src.config import get_config
print('✅ All core components importable')
"

# Test data pipeline
python -c "
from data_conversion.pipeline_manager import PipelineManager
from data_conversion.config import DataConversionConfig
print('✅ Data pipeline components OK')
"

# Test model loading (requires model path)
python -c "
from src.models.model_loader import load_model_and_processor_unified
# model, tokenizer, image_processor = load_model_and_processor_unified('/path/to/model', for_inference=False)
print('✅ Model loading components OK')
"
```

### Data Validation Commands
```bash
# Check processed data
ls -la data/
wc -l data/*.jsonl

# Validate data format
head -1 data/train.jsonl | python -m json.tool

# Check coordinate tokens in data
grep -o '<coord_[0-9]*>' data/train.jsonl | head -10

# Validate image files exist
python -c "
import json
import os
with open('data/train.jsonl') as f:
    sample = json.loads(f.readline())
    image_path = sample['images'][0]
    print(f'Image exists: {os.path.exists(image_path)}')
"
```

## 🐛 Debugging Commands

### Training Debugging
```bash
# Monitor training progress
tail -f output/training.log

# Check training metrics
grep "Step" output/training.log | tail -10

# Check for errors
grep -i "error\|exception\|failed" output/training.log

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check checkpoint creation
ls -la output/checkpoint-*
```

### Data Processing Debugging
```bash
# Check pipeline logs
tail -f data_conversion/pipeline.log

# Check processing stages
grep "Stage" data_conversion/pipeline.log

# Check for processing errors
grep -i "error\|failed" data_conversion/pipeline.log

# Validate coordinate transformations
python -c "
from data_conversion.coordinate_manager import CoordinateManager
manager = CoordinateManager()
# Test coordinate transformation
coords = [100, 100, 200, 200]
transformed = manager.transform_coordinates(coords, 'bbox_2d', (1920, 1080))
print(f'Original: {coords}')
print(f'Transformed: {transformed}')
"
```

### Model Debugging
```bash
# Test model loading
python -c "
from src.models.model_loader import load_model_and_processor_unified
try:
    model, tokenizer, image_processor = load_model_and_processor_unified('/path/to/model', for_inference=False)
    print(f'✅ Model loaded: {type(model).__name__}')
    print(f'✅ Vocab size: {len(tokenizer.get_vocab())}')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"

# Check coordinate tokens
python -c "
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, _ = load_model_and_processor_unified('/path/to/model', for_inference=False)
coord_token_id = tokenizer.convert_tokens_to_ids('<coord_100>')
print(f'Coordinate token ID: {coord_token_id}')
print(f'Is valid: {coord_token_id != tokenizer.unk_token_id}')
"
```

## 📊 Performance Monitoring Commands

### Training Performance
```bash
# Monitor training speed
grep "it/s\|s/it" output/training.log | tail -5

# Check loss progression
grep "loss=" output/training.log | awk '{print $NF}' | tail -10

# Monitor memory usage
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1

# Check training stability
python -c "
import re
with open('output/training.log') as f:
    losses = []
    for line in f:
        if 'loss=' in line:
            loss = re.search(r'loss=([0-9.]+)', line)
            if loss:
                losses.append(float(loss.group(1)))
    
    if len(losses) > 10:
        recent_losses = losses[-10:]
        print(f'Recent losses: {recent_losses}')
        print(f'Loss trend: {\"decreasing\" if recent_losses[-1] < recent_losses[0] else \"increasing\"}')
"
```

### Inference Performance
```bash
# Benchmark inference speed
time python src/inference.py \
    --model_path output/final_model \
    --image_path path/to/test_image.jpg \
    --output_file benchmark_results.json

# Batch inference throughput
python -c "
import time
import json
# Run batch inference and measure time
start_time = time.time()
# ... batch inference code ...
end_time = time.time()
throughput = num_images / (end_time - start_time)
print(f'Throughput: {throughput:.2f} images/second')
"
```

## 🔧 Utility Commands

### File Management
```bash
# Clean up old checkpoints
find output/ -name "checkpoint-*" -type d | head -n -3 | xargs rm -rf

# Archive training logs
mkdir -p logs/archive
mv output/*/training.log logs/archive/training_$(date +%Y%m%d_%H%M%S).log

# Check disk space
df -h .
du -sh output/ data/ ds_v2/
```

### Environment Management
```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data3/Qwen2.5-VL-main/model_cache
export PYTHONPATH=/data3/Qwen2.5-VL-main:$PYTHONPATH

# Check environment
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "PWD: $PWD"

# Activate conda environment (if needed)
conda activate ms
```

## 📋 Common Command Combinations

### Complete Training Pipeline
```bash
# 1. Process data
bash data_conversion/convert_dataset.sh

# 2. Validate data
ls -la data/ && wc -l data/*.jsonl

# 3. Train model
python scripts/train.py --config configs/base_flat_v2.yaml

# 4. Test inference
python src/inference.py \
    --model_path output/final_model \
    --image_path path/to/test_image.jpg \
    --output_file test_results.json
```

### Quick Experiment Setup
```bash
# 1. Create experiment config
cp configs/base_flat_v2.yaml configs/experiment.yaml
sed -i 's/num_train_epochs: 3/num_train_epochs: 1/' configs/experiment.yaml
sed -i 's/max_steps: -1/max_steps: 100/' configs/experiment.yaml

# 2. Run quick training
python scripts/train.py --config configs/experiment.yaml --output_dir ./output/quick_test

# 3. Check results
ls -la output/quick_test/
tail -20 output/quick_test/training.log
```

### Production Deployment
```bash
# 1. Validate final model
python -c "
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, image_processor = load_model_and_processor_unified('output/final_model', for_inference=True)
print('✅ Model validation passed')
"

# 2. Run batch inference
python src/inference.py \
    --model_path output/final_model \
    --image_dir production_images/ \
    --output_dir production_results/ \
    --batch_size 8

# 3. Generate summary report
python -c "
import json
import os
results_dir = 'production_results/'
total_images = len([f for f in os.listdir(results_dir) if f.endswith('.json')])
print(f'Processed {total_images} images')
"
```

---

**Next Steps**:
- **API Reference**: [api.md](api.md)
- **Troubleshooting**: [troubleshooting.md](troubleshooting.md)
- **Quick Start**: [../quick-start/README.md](../quick-start/README.md)
