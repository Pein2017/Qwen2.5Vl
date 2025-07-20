# Problem → Solution Lookup

Fast troubleshooting reference organized by symptoms. Find your problem, get the solution.

## 🚨 Training Issues

### Memory & GPU Problems

| **Symptom** | **Quick Solution** | **Config Fix** |
|-------------|-------------------|----------------|
| `CUDA out of memory` | Reduce batch size | `per_device_train_batch_size: 1` |
| `RuntimeError: out of memory` | Use gradient accumulation | `gradient_accumulation_steps: 8` |
| `No CUDA devices available` | Check GPU availability | `nvidia-smi && export CUDA_VISIBLE_DEVICES=0` |
| Training very slow | Enable Flash Attention | `attn_implementation: "flash_attention_2"` |
| High memory usage | Use bfloat16 | `torch_dtype: "bfloat16"` |

### Model Loading Errors

| **Symptom** | **Quick Solution** | **Details** |
|-------------|-------------------|-------------|
| `FileNotFoundError: config.json` | Check model path | Verify `/path/to/model/config.json` exists |
| `Vocabulary size mismatch` | Clean coordinate tokens | Remove cached tokenizer files |
| `AttributeError: patches not applied` | Apply model patches | Ensure `src.models.patches` imported |
| `Flash Attention not available` | Fallback to eager | `attn_implementation: "eager"` |
| `Model loading timeout` | Increase timeout | Check network/disk speed |

### Data Processing Errors

| **Symptom** | **Quick Solution** | **Command** |
|-------------|-------------------|-------------|
| `'list' object has no attribute 'get'` | Clean JSON first | `python clean_raw_json.py` |
| `FileNotFoundError: image not found` | Copy images | `cp -r ds/images ds_output/` |
| `JSONL parsing error` | Validate format | Check JSON structure in data files |
| `Bbox out of bounds` | Check smart_resize | Verify image dimensions vs bbox coords |
| `Empty dataset after processing` | Check data paths | Verify `train_data_path` in config |

## 🔧 Configuration Issues

### Coordinate Token Problems

| **Symptom** | **Root Cause** | **Fix** |
|-------------|----------------|---------|
| `coordinate_loss` always 0 | No coordinate data detected | Check bbox format in data |
| Poor coordinate predictions | Low learning rate | `coordinate_lr: 1e-3` |
| Coordinate tokens not recognized | Vocabulary not extended | Verify tokenizer extension |
| `focal_loss` computation error | Invalid loss parameters | Check `focal_loss_alpha`, `focal_loss_gamma` |
| Mode detection fails | Missing bbox spans | Validate coordinate token detection |

### Training Configuration

| **Symptom** | **Root Cause** | **Fix** |
|-------------|----------------|---------|
| Loss not decreasing | Learning rate too low | `learning_rate: 1e-4` |
| Loss exploding | Learning rate too high | `learning_rate: 1e-6` |
| Validation loss increasing | Overfitting | `weight_decay: 0.1` |
| Training stops early | Bad checkpoint | Check `save_total_limit` |
| Inconsistent metrics | Mixed precision issues | `fp16: false, bf16: true` |

## 🗂️ Data & File Issues

### Data Pipeline Problems

| **Symptom** | **Quick Fix** | **Prevention** |
|-------------|---------------|----------------|
| Image paths broken | Use absolute paths | Check `ds_output/` structure |
| Missing teacher data | Create teacher pool | Run teacher generation script |
| Data loading slow | Increase workers | `dataloader_num_workers: 4` |
| Inconsistent image sizes | Enable smart resize | `RESIZE="true"` in conversion |
| Wrong data format | Follow schema | Check `docs/data_schema.md` |

### Checkpoint & Model Issues

| **Symptom** | **Quick Fix** | **Long-term Solution** |
|-------------|---------------|----------------------|
| Checkpoint corrupted | Use previous checkpoint | Enable checkpoint validation |
| Model not saving | Check disk space | Monitor storage usage |
| Loading old checkpoint fails | Update checkpoint format | Use CheckpointManager |
| Gradients not updating | Check parameter requires_grad | Verify parameter groups |
| Loss components missing | Update loss computation | Use enhanced LossManager |

## 🐛 Development & Debug Issues

### Environment Problems

| **Symptom** | **Quick Fix** | **Command** |
|-------------|---------------|-------------|
| `ModuleNotFoundError` | Check Python path | `export PYTHONPATH=/data3/Qwen2.5-VL-main:$PYTHONPATH` |
| Conda environment issues | Use direct Python path | `/root/miniconda3/envs/ms/bin/python` |
| Package version conflicts | Reinstall dependencies | `pip install -r requirements.txt` |
| Import errors from src/ | Add to path | `sys.path.append('/data3/Qwen2.5-VL-main')` |
| Permission denied | Check file permissions | `chmod +x script.py` |

### Code Integration Issues

| **Symptom** | **Root Cause** | **Solution** |
|-------------|----------------|-------------|
| Legacy code conflicts | Mixed old/new APIs | Use migration guide |
| Coordinate system not working | Missing initialization | Check CoordinateTokenManager setup |
| Loss computation errors | Incorrect loss manager | Use new LossManager implementation |
| Training coordinator fails | Wrong parameter setup | Verify TrainingCoordinator config |
| Model wrapper issues | Incompatible base model | Check Qwen2.5-VL compatibility |

## 🎯 Performance Issues

### Speed Optimization

| **Problem** | **Solution** | **Impact** |
|-------------|-------------|------------|
| Slow data loading | `dataloader_num_workers: 8` | 2-3x faster |
| Slow attention | `attn_implementation: "flash_attention_2"` | 20-30% faster |
| Memory inefficient | `torch_dtype: "bfloat16"` | 50% memory reduction |
| IO bottleneck | Use SSD storage | Significant speedup |
| Batch processing slow | `pin_memory: true` | 10-15% faster |

### Quality Optimization

| **Problem** | **Solution** | **Config** |
|-------------|-------------|------------|
| Poor coordinate accuracy | Increase coordinate LR | `coordinate_lr: 1e-3` |
| Model not converging | Longer warmup | `warmup_ratio: 0.2` |
| Overfitting quickly | More regularization | `weight_decay: 0.1` |
| Underfitting | Less regularization | `weight_decay: 0.001` |
| Unstable training | Gradient clipping | `max_grad_norm: 1.0` |

## 🚑 Emergency Recovery

### Critical Failures

| **Emergency** | **Immediate Action** | **Recovery Command** |
|---------------|---------------------|---------------------|
| Training crashed | Find last checkpoint | `find checkpoints/ -name "checkpoint-*" \| sort -V \| tail -1` |
| All checkpoints corrupted | Use model backup | Copy from backup directory |
| Config file deleted | Recreate from template | Use `docs/quick-reference/config-templates.md` |
| Data directory missing | Restore from source | Re-run data conversion pipeline |
| Environment broken | Rebuild conda env | `conda create -n ms_new python=3.10` |

### Quick Recovery Scripts

```bash
# 1. Find and resume from latest checkpoint
LATEST_CHECKPOINT=$(find checkpoints/ -name "checkpoint-*" -type d | sort -V | tail -1)
/root/miniconda3/envs/ms/bin/python scripts/train.py \
    --config configs/base_flat_det.yaml \
    --resume_from_checkpoint "$LATEST_CHECKPOINT"

# 2. Validate system health
/root/miniconda3/envs/ms/bin/python -c "
import torch, transformers, datasets
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('System OK')
"

# 3. Quick data validation
/root/miniconda3/envs/ms/bin/python -c "
import json, os
with open('data/train.jsonl') as f:
    sample = json.loads(f.readline())
    print(f'Data sample: {sample.keys()}')
    print(f'Image exists: {os.path.exists(sample[\"image\"])}')
"
```

## 📋 Diagnostic Commands

### System Check
```bash
# Complete system diagnostic
echo "=== GPU Check ==="
nvidia-smi

echo "=== Environment Check ==="
/root/miniconda3/envs/ms/bin/python --version
/root/miniconda3/envs/ms/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "=== Data Check ==="
ls -la data/
head -1 data/train.jsonl | python -m json.tool

echo "=== Config Check ==="
/root/miniconda3/envs/ms/bin/python -c "
from src.config.global_config import config
print(f'Model path: {config.model_path}')
print(f'Coordinate tokens: {config.coordinate_tokens_enabled}')
"
```

### Training Health Check
```bash
# Monitor training progress
tail -f checkpoints/$(ls checkpoints/ | sort -V | tail -1)/training.log | grep -E "(loss|epoch|step)"

# Check GPU utilization
watch -n 1 "nvidia-smi | grep python"

# Memory usage
/root/miniconda3/envs/ms/bin/python -c "
import torch
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
if torch.cuda.is_available():
    print(f'Memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB')
"
```

---
**🆘 Emergency Contacts (Documentation):**
- Configuration issues → `docs/configuration.md`
- Architecture questions → `docs/architecture.md`  
- Critical bugs → `docs/critical_fixes.md`
- Data problems → `docs/troubleshooting.md`
- Advanced topics → `docs/advanced/`

**💡 Pro Tips for Faster Resolution:**
1. Always check the exact error message first
2. Use diagnostic commands to isolate the issue
3. Check if it's a known issue in `docs/critical_fixes.md`
4. Try the quick fix before deep debugging
5. Keep configs and data backed up for quick recovery