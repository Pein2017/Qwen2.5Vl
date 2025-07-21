# Problem-Solution Quick Reference

This guide provides quick lookups for common problems. For detailed explanations, see the full [Troubleshooting Guide](../guides/troubleshooting.md).

## 🚨 Emergency Quick Reference

### 30-Second Checklist
1. **Environment activated?** `conda activate ms`
2. **CUDA visible?** `echo $CUDA_VISIBLE_DEVICES`
3. **Data pipeline completed?** Check `data/` directory exists
4. **Configuration valid?** Check YAML syntax and paths
5. **Logs available?** Check `run.log` for detailed errors

### Critical Error Patterns → Quick Fixes
| **Symptom** | **Quick Fix** | **Category** |
|-------------|---------------|--------------|
| `AttributeError: module 'torch.library' has no attribute 'wrap_triton'` | Apply Flash Attention patch | [Flash Attention](../guides/troubleshooting.md#flash-attention-issues) |
| `split_with_sizes expects 128 but got 288` | Apply mRoPE dimension fix | [Model Architecture](../guides/troubleshooting.md#model-architecture-issues) |
| `shape '[0, 4, -1]' is invalid for input of size 1280` | Fix image embedding shapes | [Model Architecture](../guides/troubleshooting.md#model-architecture-issues) |
| `'list' object has no attribute 'get'` | Run `python data_conversion/clean_raw_json.py` | [Data Pipeline](../guides/troubleshooting.md#data-pipeline-issues) |
| `coordinate_loss` always 0 | Check bbox format in data | [Training Issues](../guides/troubleshooting.md#training-issues) |
| `CUDA out of memory` | Reduce batch size: `per_device_train_batch_size: 1` | [Memory Issues](../guides/troubleshooting.md#memory-issues) |
| `BOX BOX BOX` in outputs | Use detection pipeline, not `model.generate()` | [Inference Issues](../guides/troubleshooting.md#inference-issues) |
