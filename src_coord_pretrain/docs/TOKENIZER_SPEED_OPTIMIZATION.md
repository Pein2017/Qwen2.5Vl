# Tokenizer Speed Optimization Guide

This document outlines the tokenizer speed optimizations implemented in the coordinate pretraining pipeline to improve training performance.

## Overview

The coordinate pretraining pipeline has been optimized to use HuggingFace's fast tokenizers and the new `processing_class` API for improved tokenization speed and reduced training time.

## Key Optimizations

### 1. Fast Tokenizer Detection and Usage

The pipeline automatically detects and uses fast tokenizers when available:

```python
# In trainer.py
if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "is_fast"):
    if processor.tokenizer.is_fast:
        print("✅ Using fast tokenizer for optimized performance")
    else:
        print("⚠️ Using slow tokenizer - consider using fast tokenizer for better speed")
```

### 2. Processing Class API Migration

Migrated from deprecated `tokenizer` property to `processing_class`:

**Before (Deprecated):**
```python
trainer.tokenizer = processor.tokenizer  # Triggers deprecation warning
```

**After (Optimized):**
```python
trainer.processing_class = processor  # Uses new API, enables optimizations
```

### 3. Dataset Tokenization Optimizations

#### Fast Tokenizer Parameters
- Disabled unnecessary padding in dataset (handled in collator for better batching)
- Disabled truncation at individual sample level
- Optimized tokenizer kwargs for fast tokenizers

#### Optimized Tokenization Call
```python
tokenizer_kwargs = {
    "add_special_tokens": False,
    "return_tensors": "pt",
}

if self._use_fast_tokenizer:
    tokenizer_kwargs.update({
        "padding": False,  # Handle in collator for better batching
        "truncation": False,  # Handle at batch level if needed
    })

tokens = self.tokenizer(text, **tokenizer_kwargs)
```

### 4. Collator Batch Processing Optimizations

#### Optimized Padding
- Uses fast tokenizer's optimized batch padding
- Lets tokenizer decide optimal padding strategy
- Enables efficient attention mask generation

```python
padding_kwargs = {"padding": True, "return_tensors": "pt"}
if self._use_fast_tokenizer:
    padding_kwargs.update({
        "pad_to_multiple_of": None,  # Let tokenizer decide optimal padding
        "return_attention_mask": True,
    })
```

#### Efficient Offset Mapping
- Optimized retokenization for label alignment
- Disabled unnecessary special token masks
- Streamlined offset mapping for fast tokenizers

## Performance Benefits

### Speed Improvements
1. **Fast Tokenizer**: 2-10x faster tokenization compared to slow tokenizers
2. **Batch Processing**: Optimized padding and attention mask generation
3. **Reduced Memory**: Efficient tensor operations and reduced intermediate allocations
4. **API Optimization**: Uses latest HuggingFace optimizations

### Memory Efficiency
1. **Lazy Padding**: Padding handled at batch level, not individual samples
2. **Optimized Tensors**: Reduced tensor copying and reshaping
3. **Efficient Caching**: Fast tokenizers cache vocabulary lookups

## Configuration Requirements

### Fast Tokenizer Availability
Ensure your model uses a fast tokenizer:

```python
from transformers import Qwen2_5_VLProcessor

processor = Qwen2_5_VLProcessor.from_pretrained("model_path")
if hasattr(processor.tokenizer, "is_fast") and processor.tokenizer.is_fast:
    print("Fast tokenizer available")
else:
    print("Consider using a model with fast tokenizer support")
```

### Environment Variables
Set these for optimal performance:

```bash
export TOKENIZERS_PARALLELISM=false  # Avoid threading conflicts in training
export HF_DATASETS_DISABLE_PROGRESS_BARS=1  # Reduce overhead
```

## Monitoring Performance

### Training Logs
The trainer will log tokenizer type at startup:
- `✅ Using fast tokenizer for optimized performance`
- `⚠️ Using slow tokenizer - consider using fast tokenizer for better speed`

### Benchmarking
To measure tokenization speed improvements:

```python
import time
from transformers import Qwen2_5_VLProcessor

processor = Qwen2_5_VLProcessor.from_pretrained("model_path")
text = "Sample text for tokenization"

# Benchmark tokenization speed
start_time = time.time()
for _ in range(1000):
    tokens = processor.tokenizer(text, return_tensors="pt")
end_time = time.time()

print(f"Tokenization speed: {1000 / (end_time - start_time):.2f} samples/sec")
```

## Troubleshooting

### Common Issues

1. **Deprecation Warning**: `Trainer.tokenizer is now deprecated`
   - **Solution**: Code has been updated to use `processing_class`

2. **Slow Tokenization**: Training seems slower than expected
   - **Check**: Verify fast tokenizer is being used
   - **Solution**: Ensure model supports fast tokenizers

3. **Memory Issues**: High memory usage during tokenization
   - **Check**: Padding strategy and batch size
   - **Solution**: Adjust batch size or enable gradient checkpointing

### Performance Verification

Run the test suite to verify optimizations:

```bash
python -m pytest src_coord_pretrain/tests/test_tokenizer_optimization.py -v
```

## Best Practices

1. **Always use fast tokenizers** when available
2. **Handle padding at batch level** for better efficiency
3. **Use processing_class API** instead of deprecated tokenizer property
4. **Monitor tokenization speed** during training
5. **Profile memory usage** to optimize batch sizes

## Future Improvements

1. **Parallel Tokenization**: Explore multi-process tokenization for large datasets
2. **Caching**: Implement tokenization caching for repeated samples
3. **Hardware Optimization**: GPU-accelerated tokenization where available
4. **Batch Size Tuning**: Automatic batch size optimization based on tokenizer speed
