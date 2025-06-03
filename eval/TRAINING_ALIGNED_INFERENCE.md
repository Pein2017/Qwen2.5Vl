# Training-Aligned Inference for Qwen2.5-VL

## Overview

This document explains the training-aligned inference approach implemented to ensure maximum stability and consistency between training and evaluation phases.

## Problem Statement

The original evaluation system was experiencing instability issues:
- **Sampling instability**: Small changes in generation parameters caused large variations in output
- **Repetitive generation**: Models would get stuck in repetitive loops (e.g., "addCriterion" repetitions)
- **Inconsistent results**: Different evaluation runs produced significantly different metrics
- **Training-inference mismatch**: Evaluation used different settings than training

## Solution: Training-Aligned Inference

### Core Principle
**Use EXACTLY the same settings during inference as during training** - no fancy generation parameters, no sampling tricks, just pure deterministic generation that matches the training forward pass.

### Key Alignment Points

#### 1. Model Loading
```python
# Training settings (from src/core.py)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    cache_dir=config.cache_dir,
    attn_implementation="flash_attention_2",  # ✅ SAME
    torch_dtype=torch.bfloat16,              # ✅ SAME
)
model.config.use_cache = False               # ✅ CRITICAL: Same as training
```

#### 2. Tokenizer Configuration
```python
# Training settings (from src/core.py)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=config.cache_dir,
    model_max_length=2048,      # ✅ SAME
    padding_side="right",       # ✅ SAME
    use_fast=False,            # ✅ SAME
)
```

#### 3. Image Processor Settings
```python
# Training settings (from src/core.py)
image_processor.max_pixels = 1003520  # ✅ SAME
image_processor.min_pixels = 784      # ✅ SAME
```

#### 4. Preprocessing Pipeline
- **Vision token replacement**: Exact same logic as training
- **Position ID calculation**: Uses same `get_rope_index_25` function
- **Chat template**: Identical template application
- **Attention mask**: Same format and calculation

#### 5. Generation Parameters
```python
# Training-aligned generation (NO fancy sampling)
output_ids = model.generate(
    **inputs,
    max_new_tokens=2048,        # Reduced to match training context
    do_sample=False,            # ✅ Deterministic like training
    temperature=None,           # ✅ No temperature
    top_p=None,                # ✅ No top_p sampling
    top_k=None,                # ✅ No top_k sampling
    num_beams=1,               # ✅ No beam search
    use_cache=False,           # ✅ CRITICAL: Same as training
    repetition_penalty=None,   # ✅ No repetition penalty
    length_penalty=None,       # ✅ No length penalty
    early_stopping=False,      # ✅ No early stopping
    min_new_tokens=1,          # ✅ Minimal constraint
    no_repeat_ngram_size=None, # ✅ No n-gram blocking
)
```

## Implementation

### New Components

1. **`TrainingAlignedInferenceEngine`** (`eval/infer_dataset.py`)
   - Replaces `SimpleInferenceEngine`
   - Uses training's `ModelWrapper` and `Config` classes
   - Ensures exact alignment with training settings

2. **Training alignment verification** (`eval/test_training_alignment.py`)
   - Tests that all settings match training exactly
   - Verifies generation parameters
   - Checks for problematic patterns

3. **Updated evaluation pipeline** (`eval/eval_dataset.py`)
   - Uses training-aligned inference by default
   - Reduced `max_new_tokens` to 2048 (from 8192)
   - Clear documentation of training alignment

### Usage

#### Quick Test
```bash
python eval/test_training_alignment.py
```

#### Single Sample Inference
```bash
python eval/infer_dataset.py \
    --model_path output/checkpoint-XXX \
    --validation_jsonl 521_qwen_val.jsonl \
    --output_file responses.json \
    --max_new_tokens 2048
```

#### Complete Evaluation
```bash
python eval/eval_dataset.py \
    --model_path output/checkpoint-XXX \
    --validation_jsonl 521_qwen_val.jsonl \
    --output_dir eval_results \
    --max_new_tokens 2048
```

#### Shell Script
```bash
bash eval/eval.sh
```

## Benefits

### 1. **Stability**
- Deterministic generation eliminates sampling variance
- Consistent results across multiple evaluation runs
- No more repetitive generation loops

### 2. **Training Consistency**
- Model behaves exactly as it did during training
- No training-inference mismatch
- Faithful representation of model capabilities

### 3. **Reproducibility**
- Same results every time (given same inputs)
- Easier debugging and analysis
- Reliable benchmarking

### 4. **Simplicity**
- No complex generation parameter tuning
- Clear, understandable approach
- Easier to maintain and debug

## Verification

The training alignment can be verified by checking:

```python
# Model settings
assert model.config.attn_implementation == "flash_attention_2"
assert model.dtype == torch.bfloat16
assert model.config.use_cache == False

# Tokenizer settings
assert tokenizer.model_max_length == 2048
assert tokenizer.padding_side == "right"
assert tokenizer.use_fast == False

# Image processor settings
assert image_processor.max_pixels == 1003520
assert image_processor.min_pixels == 784

# Generation settings
assert generation_config["do_sample"] == False
assert generation_config["use_cache"] == False
assert generation_config["temperature"] == None
```

## Migration from Previous Approach

### Before (Unstable)
```python
# Old approach with fancy sampling
output_ids = model.generate(
    **inputs,
    max_new_tokens=8192,           # Too long
    do_sample=True,                # Non-deterministic
    temperature=0.0,               # Conflicting with do_sample
    repetition_penalty=1.05,       # Artificial constraint
    no_repeat_ngram_size=3,        # More artificial constraints
    use_cache=True,                # Different from training
)
```

### After (Stable)
```python
# New training-aligned approach
output_ids = model.generate(
    **inputs,
    max_new_tokens=2048,           # Matches training context
    do_sample=False,               # Deterministic
    use_cache=False,               # Same as training
    # No other parameters - keep it simple
)
```

## Troubleshooting

### If you still see instability:
1. **Verify training alignment**: Run `eval/test_training_alignment.py`
2. **Check model loading**: Ensure `use_cache=False` and `flash_attention_2`
3. **Verify generation config**: Ensure `do_sample=False` and no sampling parameters
4. **Check preprocessing**: Ensure same vision token replacement and position IDs

### If responses are too short:
- Increase `max_new_tokens` (but keep reasonable, e.g., 2048-4096)
- Check that `min_new_tokens=1` (minimal constraint)

### If responses are truncated:
- Check for early stopping in generation config
- Verify EOS token handling
- Ensure no length penalties

## Conclusion

Training-aligned inference provides a stable, consistent, and faithful evaluation approach by eliminating the training-inference mismatch. This approach prioritizes reliability over generation flexibility, ensuring that evaluation results accurately reflect the model's true capabilities as learned during training. 