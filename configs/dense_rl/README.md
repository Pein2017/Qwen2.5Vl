# Dense RL Configuration Guide

## Strict Configuration Policy

**⚠️ NO DEFAULTS** - All hyperparameters must be explicitly set in YAML configs.

### Core Principles

1. **Base config = Immutable constants ONLY**
   - `dense_base.yaml` contains values that NEVER change across runs
   - Model dtype, loss ratios, runtime backend
   - NO tunable hyperparameters

2. **Child configs = Explicit everything**
   - Every hyperparameter must be explicitly set
   - No `.get(key, default)` for any tunable parameter
   - Missing keys raise clear errors immediately

3. **Truly optional fields**
   - `paths.ref_model_path` (can be None)
   - (removed) `grpo.max_advantage_magnitude`
   - `grpo.beta_anneal` (can be None if beta=0)
   - Everything else is REQUIRED

4. **Fail-fast validation**
   - Config loads with strict `_require()` validation
   - Clear error message: `"Missing required config key: 'X' in Y"`
   - Forces explicit configuration

## File Structure

```
configs/dense_rl/
├── dense_base.yaml      # Immutable constants only
├── debug.yaml           # Fast iteration (100 steps)
├── standard.yaml        # Production training
└── README.md           # This file
```

## Required Sections

Every child config MUST provide these sections:

### 1. Paths
```yaml
paths:
  model_path: "path/to/sft/checkpoint"
  ref_model_path: "path/to/ref/model"  # Optional, can be null
  train_data_path: "data/train.jsonl"
  val_data_path: "data/val.jsonl"
  data_root: "data/"
  output_dir: "outputs/rl_run"
  tb_dir: "tb_logs"
```

### 2. Experiment
```yaml
experiment:
  run_name: "my_grpo_run"
  seed: 17
  tags: []  # Optional list
```

### 3. Model
```yaml
model:
  attn_implementation: "flash_attention_2"  # or "eager" or "sdpa"
```

### 4. Sampling
```yaml
sampling:
  prompt_batch_size: 8      # Number of prompts per optimizer step
  sample_k: 8               # GLOBAL completions per prompt across all ranks
  reward_average_window: 5

# NOTE: gradient_accumulation_steps is AUTO-COMPUTED as:
#   local_k × prompt_batch_size
# where local_k = sample_k / world_size (require divisibility)
```

**Important**: The actual gradient accumulation is computed automatically based on the sampling window. Do NOT specify `gradient_accumulation_steps` manually.

Tip: `quad_giou` and `line_giou` require polygon/line geometry computations. For best fidelity, install `shapely`; otherwise they fall back to AABB GIoU (quad) and endpoint L1 (line).

### 5. Generation
```yaml
generation:
  max_new_tokens: 1500
  min_new_tokens: 0
  temperature: 1.1
  top_p: 0.95
  # top_k and repetition_penalty omitted; use HF defaults
  dynamic_length:
    enabled: true
    estimator: "tokenizer"
    alpha: 1.2
    eos_margin: 16
    min_cap: 64
    max_cap: 2048
    hard_cap: true
```

### 6. GRPO
```yaml
grpo:
  epsilon_low: 0.2
  epsilon_high: 0.2
  beta_start: 0.05
  beta_anneal:  # Optional
    type: "cosine"  # or "linear"
    ratio: 0.67     # Fraction of training for annealing (e.g., 0.67 = 2/3)
  loss_type: "dr_grpo"
  scale_rewards: true
  mask_truncated_completions: false
  max_advantage_magnitude: 5.0  # Optional, can be null
```

- Mapping to TRL (runner wiring):
  - `epsilon_low` → `HFGRPOConfig.epsilon`
  - `epsilon_high` → `HFGRPOConfig.epsilon_high`
  - `loss_type` → `HFGRPOConfig.loss_type`
  - Reference policy KL is disabled by default in our TRL path (no `beta` used by trainer; keep `beta_start: 0` to match).

**Beta Annealing (epoch-based):**
- `beta_start`: Initial KL coefficient
- `beta_anneal.type`: "linear" or "cosine" decay
- `beta_anneal.ratio`: Fraction of total training for annealing
  - Example: ratio=0.67 with 3000 steps → anneal over 2010 steps
  - Beta decays from beta_start to 0 over this period
- Optional: Can be omitted or set to null if beta_start=0

### 7. Normalization
```yaml
normalization:
  cross_rank_advantages: false
```

### 8. Training
```yaml
training:
  num_train_epochs: 3
  dataset_size: -1           # -1 = auto-detect from len(train_dataset)
  warmup_ratio: 0.1          # Warmup as fraction of total steps (e.g., 0.1 = 10%)
  lr_scheduler_type: "cosine"
  gradient_checkpointing: false       # Enable to reduce memory (slight speed cost)
  dataloader_num_workers: 8
  pin_memory: true
  prefetch_factor: 4         # Prefetch batches when num_workers > 0
  bf16: true
  fp16: false
```

- Runner defaults `gradient_checkpointing=True` for TRL when constructing `HFGRPOConfig` (YAML value can remain false/omitted; runner overrides).

**Epoch-Based Training:**
- `num_train_epochs`: Number of passes through dataset
- `dataset_size`: Number of samples (-1 for auto-detect)
  - When -1: uses len(train_dataset)
  - When positive: limits to that many samples
- Steps per epoch = `dataset_size // prompt_batch_size`
- Total steps = `num_train_epochs × steps_per_epoch`

**Warmup:**
- `warmup_ratio`: Fraction of total steps for warmup (e.g., 0.1 = 10%)
- Warmup steps = `warmup_ratio × total_steps`

**Step-based configs** (unchanged):
- `eval_every_steps`: Evaluation frequency
- `logging_steps`: Logging frequency  
- `save_steps`: Checkpoint frequency

**Gradient Checkpointing**: 
- Set `gradient_checkpointing: true` to reduce memory usage
- Trades off slight speed for lower VRAM consumption
- Useful for large models or higher resolution images

**Prefetch Factor**:
- Only used when `dataloader_num_workers > 0`
- Prefetches batches to speed up data loading
- Recommended: 2-4 for spinning disks, 4-8 for SSDs

### 9. Optimizer
```yaml
optimizer:
  type: "adamw"
  learning_rates:
    llm: 1.0e-5
    vision: 1.0e-6
    merger: 1.0e-4
  weight_decay: 1.0e-3
  max_grad_norm: 1.0
```

**Note**: AdamW uses default PyTorch values for beta1 (0.9), beta2 (0.999), and epsilon (1e-8). These can be optionally overridden by adding `adam_beta1`, `adam_beta2`, or `adam_epsilon` fields if needed.

### 10. Layer Freezing
```yaml
layer_config:
  vision_tower:
    freeze_patch_embed: true
    freeze_bottom_layers: false
    trainable_top_k_blocks: 2
  llm:
    freeze_bottom_layers: false
    trainable_top_k_blocks: 4
  merger:
    freeze: false
```

### 11. Logging
```yaml
logging:
  logging_steps: 1
  log_level: "INFO"
```

### 12. Checkpointing
```yaml
checkpointing:
  save_steps: 500
  save_total_limit: 3
```

**Note**: Checkpoints always save with strategy="steps" and include tokenizer/processor

### 13. Rewards
```yaml
rewards:
  wrappers: 0.10
  coords: 0.10
  separators: 0.10
  vocab: 0.03
  length_vs_gt: 0.20
  geometry_sanity: 0.10
  coverage: 0.40
  ordering: 0.10
  bbox_giou: 0.35
  quad_l1: 0.10
  line_l1: 0.10
  quad_giou: 0.35
  line_giou: 0.25
  caption_f1: 0.30
  grounding_acc: 0.30

observe_rewards: []

rewards_config:
  clip_sigma: 5.0
  tau_iou: 0.5
  tau_quad: 0.02
  tau_line: 0.02
  line_giou:
    buffer_frac: 0.05
  length_vs_gt:
    estimator: "tokenizer"
    lower: 0.7
    upper: 1.2
    gamma: 3.0
    tail_numeric_weight: 0.4
```

### 14. Evaluation
```yaml
evaluation:
  enabled: true
  eval_every_steps: 50
  rounds: 1
  per_rank_samples: 1
  save_samples: 20
  log_text_snippets: true
  seed: 17
```

## Auto-Computed Values

These values are **automatically computed** by the trainer:

1. **`max_steps`** = `num_train_epochs × (dataset_size // prompt_batch_size)`
   - Where `dataset_size` auto-detects from `len(train_dataset)` if set to -1

2. **`gradient_accumulation_steps`** = `local_k × prompt_batch_size`
   - Where `local_k = sample_k / world_size` (require `sample_k % world_size == 0`)

3. **`per_device_train_batch_size`** = `1` (hardcoded)

Example: With `num_train_epochs=3`, `dataset_size=1000`, `prompt_batch_size=8`, `sample_k=8`, `world_size=4`:
- `steps_per_epoch = 1000 // 8 = 125`
- `max_steps = 3 × 125 = 375`
- `local_k = 8/4 = 2`
- `gradient_accumulation_steps = 2 × 8 = 16`

## Usage

### Basic
```bash
# Load and validate config
python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode load

# Train
python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode train
```

### With Launcher
```bash
bash scripts/run_dense_grpo.sh configs/dense_rl/standard.yaml
```

## Error Handling

### Missing Required Key
```
ConfigValidationError: Missing required config key: 'train_data_path' in paths
This value must be explicitly set in your YAML config.
```

**Solution**: Add the missing key to your YAML file.

### Wrong Type
```
ConfigValidationError: beta_anneal.type must be 'linear' or 'cosine', got: 'exponential'
```

**Solution**: Use a valid value.

## Migration from Old Format

### Old (Deprecated)
```yaml
# OLD - no longer supported
model_path: "..."
train_data_path: "..."
# Scattered keys with defaults
gradient_accumulation_steps: 4  # This is now auto-computed!
```

### New (Required)
```yaml
extends: [./dense_base.yaml]

# All keys grouped logically and explicitly set
paths:
  model_path: "..."
  train_data_path: "..."
  # ...

experiment:
  run_name: "..."
  seed: 17

sampling:
  prompt_batch_size: 8
  sample_k: 8
  # gradient_accumulation_steps is AUTO-COMPUTED, don't specify it!

# ... all other required sections
```

## Best Practices

1. **Start from template**: Copy `debug.yaml` or `standard.yaml`
2. **Change values explicitly**: Never rely on defaults
3. **Group related settings**: Use the section structure
4. **Validate early**: Run `--mode load` before training
5. **Document changes**: Add comments for non-obvious values
6. **Don't specify auto-computed values**: Let the trainer compute them

## Troubleshooting

### Q: Config loading fails with "Missing required config key"
**A**: Add the missing key to your YAML file. There are NO defaults.

### Q: Want to disable a feature?
**A**: Set the corresponding flag explicitly:
- Evaluation: `evaluation.enabled: false`
- Beta annealing: `grpo.beta_anneal: null` or omit (if beta_start=0)

### Q: How to inherit from base?
**A**: Use `extends: [./dense_base.yaml]` at the top of your config.

### Q: How is max_steps determined?
**A**: It's auto-computed from `num_train_epochs × (dataset_size // prompt_batch_size)`. Control training length via `num_train_epochs` and optionally `dataset_size`.

### Q: Can I add new optional parameters?
**A**: Only if they are truly optional (have `Optional[T] = None` in the dataclass).
All hyperparameters must be required.

## Reference

- Configuration module: `src_new/config/rl_config_v2.py`
- Runner: `src_new/rl/runner.py`
- Trainer: `src_new/rl/grpo_trainer.py`
- Plan: `RL-refactoring-plan/config_claude4.5.md`
- Summary: `RL-refactoring-plan/IMPLEMENTATION_SUMMARY.md`
