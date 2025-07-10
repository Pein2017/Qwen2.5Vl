# Troubleshooting Guide

This comprehensive guide consolidates all known issues, solutions, and debugging approaches for the Qwen2.5-VL BBU fine-tuning project.

## Quick Reference

### Emergency Checklist
1. **Environment activated?** `conda activate ms`
2. **CUDA visible?** `echo $CUDA_VISIBLE_DEVICES`
3. **Data pipeline completed?** Check `data/` directory exists
4. **Configuration valid?** Check YAML syntax and paths
5. **Logs available?** Check `run.log` for detailed errors

### Common Error Patterns
| Error Pattern | Quick Fix |
|---------------|-----------|
| `'list' object has no attribute 'get'` | Run JSON cleaning: `python data_conversion/clean_raw_json.py` |
| `shape '[0, 4, -1]' is invalid` | Image embed shape issue - check data format |
| `split_with_sizes expects 128 but got 288` | mRoPE dimension mismatch - use patched version |
| `ConfigValidation-Error` | Missing/invalid YAML parameters |
| `BOX BOX BOX` in outputs | Using wrong inference mode - use detection pipeline |

## Environment Issues

### 1. Conda Environment Not Activated
**Symptoms:**
- ImportError for required packages
- Python version mismatch
- Missing CUDA support

**Solution:**
```bash
# Always activate the environment first
conda activate ms

# Verify activation
which python
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. CUDA Configuration Problems
**Symptoms:**
- `CUDA out of memory` errors
- No GPU detected
- Slow training performance

**Diagnosis:**
```bash
# Check GPU availability
nvidia-smi

# Check CUDA in Python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check environment variables
echo $CUDA_VISIBLE_DEVICES
echo $HF_HOME
```

**Solutions:**
```bash
# Set GPU devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Reduce memory usage
# Edit your config file to reduce batch_size
# Enable gradient_checkpointing: true
# Use mixed_precision: true
```

### 3. Network and Access Issues
**Symptoms:**
- Connection timeouts to HuggingFace
- Package installation failures
- Model download errors

**Solutions:**
```bash
# Set local model cache
export HF_HOME=/data4/swift/model_cache

# Use local mirrors for packages
# Configure pip to use local repositories
# Ensure firewall allows required connections
```

## Data Pipeline Issues

### 1. JSON Format Problems
**Symptoms:**
- `'list' object has no attribute 'get'` errors
- Parsing failures during data conversion
- Inconsistent object counts

**Root Cause:**
Raw JSON files contain unnecessary metadata that conflicts with the processing pipeline.

**Solution:**
```bash
# Always run JSON cleaning first
python data_conversion/clean_raw_json.py

# Then run the full pipeline
bash data_conversion/convert_dataset.sh
```

### 2. Label Hierarchy Filtering Issues
**Symptoms:**
- Objects being filtered out incorrectly
- Significant reduction in object count
- Missing annotations in final data

**Root Cause:**
The `label_hierarchy.json` file is missing object types or properties that exist in the data.

**Diagnosis:**
```bash
# Check object filtering stats
python data_conversion/simple_validate.py

# Analyze missing labels
python -c "
import json
with open('data_conversion/processing_summary.json') as f:
    summary = json.load(f)
    print('Missing tokens:', summary.get('missing_tokens', []))
"
```

**Solution:**
1. **Update label hierarchy** with missing tokens
2. **Use token-mapped terms** (not raw terms) in hierarchy
3. **Re-run pipeline** after hierarchy updates

### 3. Coordinate Transformation Issues
**Symptoms:**
- Bounding box coordinates out of bounds
- Massive pixel differences between expected and actual
- Objects appearing in wrong locations

**Root Cause:**
Usually caused by incorrect EXIF orientation handling or dimension mismatches.

**Diagnosis:**
```bash
# Check image dimensions vs annotation dimensions
python -c "
from PIL import Image
img = Image.open('path/to/image.jpg')
print(f'Image dimensions: {img.size}')
print(f'EXIF orientation: {img.getexif().get(274, 1)}')
"
```

**Solutions:**
1. **Verify EXIF handling** is enabled in pipeline
2. **Check smart resize** parameters match model requirements
3. **Validate coordinate scaling** formulas

### 4. Image Processing Problems
**Symptoms:**
- Images not found during processing
- Dimension mismatch errors
- EXIF orientation issues

**Solutions:**
```bash
# Ensure image copying completed
ls ds_output/  # Should contain both JSON and image files

# Check image file permissions
find ds_output/ -name "*.jpg" -exec ls -la {} \;

# Verify image processing
python data_conversion/strip_exif_orientation.py
```

## Training Issues

### 1. Shape Mismatch Errors
**Symptoms:**
```
RuntimeError: shape '[0, 4, -1]' is invalid for input of size 1280
```

**Root Cause:**
Image embedding shape mismatch in loss calculation - typically occurs with masked_scatter operations.

**Solution:**
The fix is already implemented in `src/training/loss_manager.py`:
```python
# Fixed shape handling
if image_embeds.dim() == 2:
    image_embeds_flat = image_embeds.view(-1)
else:
    image_embeds_flat = image_embeds.reshape(-1)

num_mask = image_mask.sum().item()
assert len(image_embeds_flat) >= num_mask, "Not enough image features"
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_flat[:num_mask])
```

### 2. mRoPE Dimension Mismatch
**Symptoms:**
```
split_with_sizes expects 128 but got 288 (dimension mismatch)
```

**Root Cause:**
Original HuggingFace implementation incorrectly doubled `mrope_section` dimensions.

**Solution:**
The fix is implemented in `src/models/patches.py`:
```python
# De-duplicate mrope_section if needed
if len(mrope_section) > 6 and mrope_section[:len(mrope_section)//2] == mrope_section[len(mrope_section)//2:]:
    mrope_section = mrope_section[: len(mrope_section)//2]

# Validate dimensions
expected = sum(mrope_section)
assert expected == cos.size(-1), f"mRoPE dim mismatch: {expected=} {cos.size(-1)=}"
```

### 3. Response Parser Failures
**Symptoms:**
- Training loss becomes NaN
- Poor detection performance
- "BOX BOX BOX" strings in outputs

**Root Cause:**
Response parser cannot handle mixed annotation formats or early training states.

**Solution:**
Updated parser in `src/utils/response_parser.py` handles:
- True JSON format lists
- Unquoted legacy format
- Alternative regex patterns
- Graceful degradation for missing labels

### 4. Configuration Validation Errors
**Symptoms:**
```
ConfigValidation-Error: Missing required parameter 'model_path'
```

**Root Cause:**
Invalid or incomplete configuration file.

**Solution:**
```bash
# Validate configuration
python -c "
from src.config.config_manager import ConfigManager
config = ConfigManager('configs/base_flat_v2.yaml')
print('Configuration valid')
"

# Check for missing parameters
cat configs/base_flat_v2.yaml | grep -E "(null|~|'')"
```

### 5. Memory Issues
**Symptoms:**
- CUDA out of memory errors
- System freezing during training
- Slow training progress

**Solutions:**
```bash
# Reduce batch size in config
batch_size: 4  # or smaller

# Enable gradient checkpointing
gradient_checkpointing: true

# Use mixed precision
mixed_precision: true

# Adjust accumulation steps
gradient_accumulation_steps: 4
```

## Inference Issues

### 1. Model Loading Problems
**Symptoms:**
- Model checkpoint not found
- Incompatible model versions
- Missing model components

**Solutions:**
```bash
# Check checkpoint structure
ls -la path/to/checkpoint/

# Verify model compatibility
python -c "
from src.models.model_loader import ModelLoader
loader = ModelLoader('path/to/checkpoint')
print('Model loaded successfully')
"
```

### 2. Detection Pipeline Issues
**Symptoms:**
- "BOX BOX BOX" strings instead of coordinates
- Missing object detection results
- Incorrect bounding boxes

**Root Cause:**
Using standard generation instead of detection pipeline.

**Solution:**
```python
# Use detection pipeline
from src.inference import Inference
inference = Inference('path/to/checkpoint')
result = inference.predict_detection('path/to/image.jpg')

# NOT: model.generate()
```

### 3. Image Processing Errors
**Symptoms:**
- Images not loading correctly
- Dimension mismatch during inference
- EXIF orientation issues

**Solutions:**
```bash
# Verify image format
python -c "
from PIL import Image
img = Image.open('path/to/image.jpg')
print(f'Mode: {img.mode}, Size: {img.size}')
"

# Check image preprocessing
from src.core.data_processor import DataProcessor
processor = DataProcessor()
processed = processor.process_image('path/to/image.jpg')
```

## Validation and Testing Issues

### 1. Pipeline Validation Failures
**Symptoms:**
- Object count mismatches
- Coordinate validation errors
- Set overlap warnings

**Diagnosis:**
```bash
# Run comprehensive validation
python data_conversion/simple_validate.py

# Check processing summary
cat data_conversion/processing_summary.json
```

### 2. Evaluation Script Failures
**Symptoms:**
- Evaluation scripts crash
- Metric calculation errors
- Visualization generation failures

**Solutions:**
```bash
# Run individual evaluation components
python eval/validate_results.py
python eval/test_all_evaluations.py

# Check evaluation dependencies
python -c "import matplotlib, seaborn, pandas; print('Visualization packages available')"
```

## Performance Issues

### 1. Slow Data Processing
**Symptoms:**
- Pipeline takes hours to complete
- High CPU/memory usage
- Disk I/O bottlenecks

**Solutions:**
```bash
# Use parallel processing
PARALLEL_JOBS=4 bash data_conversion/convert_dataset.sh

# Check disk space
df -h

# Monitor resource usage
htop
```

### 2. Training Performance
**Symptoms:**
- Very slow training progress
- Low GPU utilization
- Memory inefficiency

**Solutions:**
```bash
# Enable Flash Attention 2
flash_attention: true

# Optimize batch size
per_device_train_batch_size: 8
dataloader_num_workers: 4

# Use gradient accumulation
gradient_accumulation_steps: 2
```

## Debugging Strategies

### 1. Systematic Debugging Approach
1. **Check environment** (conda, CUDA, paths)
2. **Validate data** (format, processing, counts)
3. **Examine logs** (run.log, error traces)
4. **Test components** (individual modules)
5. **Verify configuration** (YAML syntax, parameters)

### 2. Log Analysis
```bash
# Check recent errors
tail -50 run.log | grep -i error

# Monitor training progress
tail -f run.log | grep -E "(loss|epoch|step)"

# Check memory usage
tail -f run.log | grep -i memory
```

### 3. Component Testing
```bash
# Test data pipeline
python data_conversion/test_pipeline.py

# Test model components
python -c "
from src.models.model_loader import ModelLoader
from src.detection.detection_head import DetectionHead
print('Components imported successfully')
"

# Test inference
python src/inference.py --model_path path/to/checkpoint --image_path path/to/test.jpg
```

## Advanced Troubleshooting

### 1. Memory Profiling
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Python memory profiling
python -c "
import torch
print(f'GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
print(f'GPU cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB')
"
```

### 2. Distributed Training Issues
**Symptoms:**
- Hanging during multi-GPU training
- Synchronization errors
- Uneven GPU utilization

**Solutions:**
```bash
# Check GPU topology
nvidia-smi topo -m

# Use proper distributed settings
export NCCL_P2P_DISABLE=1  # if P2P issues
export NCCL_IB_DISABLE=1   # if InfiniBand issues
```

### 3. Model Compatibility
**Symptoms:**
- Version mismatch errors
- Missing model components
- Incorrect tensor shapes

**Solutions:**
```bash
# Check model version
python -c "
import torch
checkpoint = torch.load('path/to/checkpoint')
print(f'Model version: {checkpoint.get(\"version\", \"unknown\")}')
"

# Verify model architecture
python -c "
from src.models.wrapper import Qwen25VLWithDetection
model = Qwen25VLWithDetection.from_pretrained('path/to/checkpoint')
print(f'Model components: {list(model.named_modules())[:5]}')
"
```

## Prevention Strategies

### 1. Pre-Training Checklist
- [ ] Conda environment activated
- [ ] CUDA environment variables set
- [ ] Data pipeline completed successfully
- [ ] Configuration file validated
- [ ] Sufficient disk space available
- [ ] Model checkpoints accessible

### 2. Monitoring Setup
```bash
# Create monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
    tail -5 run.log
    echo
    sleep 60
done
EOF
chmod +x monitor_training.sh
```

### 3. Regular Validation
```bash
# Weekly validation routine
python data_conversion/simple_validate.py
python eval/validate_results.py
python -m pytest tests/ -v
```

## Getting Help

### 1. Information to Collect
Before seeking help, gather:
- Complete error message and stack trace
- Environment information (`conda list`, `nvidia-smi`)
- Configuration file contents
- Recent log entries
- System resource usage

### 2. Debugging Commands
```bash
# Complete system information
echo "=== Environment ===" && conda list | grep -E "(torch|transformers|PIL)"
echo "=== CUDA ===" && nvidia-smi
echo "=== Config ===" && cat configs/base_flat_v2.yaml
echo "=== Recent Logs ===" && tail -20 run.log
```

### 3. Documentation Resources
- [Critical Fixes](critical_fixes.md) - Known issues and solutions
- [Lessons Learned](lessons_learned.md) - Historical knowledge
- [Architecture](architecture.md) - System design
- [Configuration](configuration.md) - Parameter reference

---

**Remember:** This project follows a fail-fast philosophy. When errors occur, they are designed to surface quickly with clear messages. Don't suppress errors - investigate and fix the root cause.