# Epoch-Based RL Training Refactor - Summary

## Overview
Converted RL training from step-based to epoch-based control flow, matching SFT's pattern. Removed all dummy/unused configuration fields for a cleaner codebase.

## Major Changes

### 1. Epoch-Based Training
**Before**: Training controlled by `max_steps` directly
**After**: Training controlled by `num_train_epochs` and `dataset_size`

- `max_steps` is now auto-computed: `num_train_epochs × (dataset_size // prompt_batch_size)`
- `dataset_size = -1` auto-detects from `len(train_dataset)`
- Explicit `dataset_size` values limit training samples (useful for debugging)

### 2. Warmup Configuration
**Before**: Choice between `warmup_steps` or `warmup_ratio`
**After**: Only `warmup_ratio` (required field)

- `warmup_steps` removed entirely
- `warmup_ratio` must be specified (e.g., 0.1 for 10% warmup)
- Computed as: `warmup_steps = int(warmup_ratio × max_steps)`

### 3. Beta Annealing Configuration
**Before**: `beta_anneal.steps` (fixed step count)
**After**: `beta_anneal.ratio` (fraction of training)

- `beta_anneal.steps` removed entirely
- `beta_anneal.ratio` specifies fraction of total training for annealing
- Example: ratio=0.67 means anneal over 2/3 of training
- Computed as: `anneal_steps = int(ratio × max_steps)`

### 4. Removed Dummy Fields

#### From TrainingConfig:
- ✗ `per_device_train_batch_size` (hardcoded to 1 in trainer)
- ✗ `per_device_eval_batch_size` (not used)
- ✗ `gradient_accumulation_steps` (auto-computed from sampling)

#### From LearningRatesConfig:
- ✗ `coordinate` (no coordinate params in RL phase)

#### From OptimizerConfig:
- ✗ `adam_beta1` (optional, defaults to AdamW's 0.9)
- ✗ `adam_beta2` (optional, defaults to AdamW's 0.999)
- ✗ `adam_epsilon` (optional, defaults to AdamW's 1e-8)

#### From LoggingConfig:
- ✗ `disable_tqdm` (not used)
- ✗ `report_to` (TensorBoard is hardcoded)
- ✗ `log_predictions` (not used)
- ✗ `log_rewards_breakdown` (not used)
- ✗ `log_on_each_node` (not used)

#### From CheckpointingConfig:
- ✗ `save_strategy` (always "steps")
- ✗ `save_tokenizer` (always true)
- ✗ `save_processor` (always true)

## Configuration Examples

### Production (standard.yaml)
```yaml
training:
  num_train_epochs: 3
  dataset_size: -1           # Auto-detect
  warmup_ratio: 0.1          # 10% warmup
  lr_scheduler_type: "cosine"
  gradient_checkpointing: false
  dataloader_num_workers: 8
  pin_memory: true
  prefetch_factor: 4
  bf16: true
  fp16: false

grpo:
  beta_start: 0.05
  beta_anneal:
    type: "cosine"
    ratio: 0.67  # Anneal over 2/3 of training
```

### Debug (debug.yaml)
```yaml
training:
  num_train_epochs: 1
  dataset_size: 100          # Limit to 100 samples
  warmup_ratio: 0.1
  # ... same as production

grpo:
  beta_start: 0.05
  beta_anneal:
    type: "cosine"
    ratio: 0.5  # Anneal over 1/2 of training
```

## Auto-Computed Values

The trainer automatically computes:

1. **`max_steps`** = `num_train_epochs × (dataset_size // prompt_batch_size)`
2. **`warmup_steps`** = `int(warmup_ratio × max_steps)`
3. **`beta_anneal_steps`** = `int(beta_anneal.ratio × max_steps)`
4. **`gradient_accumulation_steps`** = `local_sample_k × prompt_batch_size`
5. **`per_device_train_batch_size`** = `1` (hardcoded)

### Example Computation
With `num_train_epochs=3`, `dataset_size=1000`, `prompt_batch_size=8`:
- `steps_per_epoch = 1000 // 8 = 125`
- `max_steps = 3 × 125 = 375`
- `warmup_steps = int(0.1 × 375) = 37` (with warmup_ratio=0.1)
- `beta_anneal_steps = int(0.67 × 375) = 251` (with ratio=0.67)

## Migration Guide

### Old Config (Step-Based)
```yaml
training:
  max_steps: 3000
  warmup_steps: 300
  # ...

grpo:
  beta_start: 0.05
  beta_anneal:
    type: "cosine"
    steps: 2000  # Fixed step count
```

### New Config (Epoch-Based)
```yaml
training:
  num_train_epochs: 3
  dataset_size: -1
  warmup_ratio: 0.1
  # ...

grpo:
  beta_start: 0.05
  beta_anneal:
    type: "cosine"
    ratio: 0.67  # Fraction of training
```

## Validation & Error Handling

### Legacy Field Detection
The config will raise clear errors if old fields are detected:

```python
# If "max_steps" in training config:
ConfigValidationError: Legacy field 'max_steps' detected in training config.
RL training is now epoch-based. Please use:
  - num_train_epochs: <int>
  - dataset_size: -1  # or explicit size
  - warmup_ratio: <float>

# If "steps" in beta_anneal config:
ConfigValidationError: Legacy field 'steps' detected in beta_anneal.
Beta annealing is now epoch-based. Please use:
  - ratio: <float>  # e.g., 0.67 for 2/3 of training
```

### Validation Rules
- `num_train_epochs > 0` (required)
- `dataset_size >= -1` (required, -1 for auto-detect)
- `warmup_ratio >= 0.0` (required)
- `beta_anneal.ratio` must be in (0.0, 1.0] (if beta annealing is enabled)

## Files Modified

1. **Config Schema**: `src_new/config/rl_config_v2.py`
   - Updated `TrainingConfig` for epoch-based training
   - Updated `BetaAnnealConfig` to use ratio instead of steps
   - Removed dummy fields from all config classes
   
2. **Trainer**: `src_new/rl/grpo_trainer.py`
   - `_build_manual_cfg`: Computes `max_steps` from epochs and `beta_anneal_steps` from ratio
   - `_build_scheduler`: Uses only `warmup_ratio`
   
3. **Configs**:
   - `configs/dense_rl/standard.yaml`: Updated beta_anneal from steps to ratio
   - `configs/dense_rl/debug.yaml`: Updated beta_anneal from steps to ratio
   - `configs/dense_rl/dense_base.yaml`: Updated comments
   
4. **Documentation**: `configs/dense_rl/README.md`

## Benefits

1. **Consistency**: RL now matches SFT's epoch-based approach; all time-based configs use same pattern (warmup_ratio, beta_anneal.ratio)
2. **Clarity**: Auto-computed values are explicit in logs
3. **Simplicity**: Removed 13 dummy/unused configuration fields
4. **Flexibility**: `dataset_size` allows easy limiting for experiments; ratios adapt to any training length
5. **Maintainability**: Fewer config fields = less confusion

## Testing Checklist

- [ ] Load standard.yaml with beta_anneal.ratio
- [ ] Load debug.yaml with beta_anneal.ratio
- [ ] Verify beta_anneal_steps computed correctly from ratio
- [ ] Confirm beta decay works as expected
- [ ] Legacy "steps" in beta_anneal raises clear error
- [ ] Ratio validation works (must be in 0-1 range)
- [ ] Log messages show computed anneal_steps
- [ ] All epoch-based configs work together (warmup_ratio + beta_anneal.ratio)
