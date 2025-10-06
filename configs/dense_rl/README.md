# Dense RL Configuration System

## Overview

The Dense RL configuration system has been refactored to eliminate redundancy and enforce explicit configuration management. This prevents configuration errors by removing fallback defaults that could mask missing critical settings.

## Configuration Structure

### Inheritance Pattern

```
dense_base.yaml  (INCOMPLETE - universal constants only)
    ├── debug.yaml     (complete debug environment)
    └── standard.yaml  (complete production environment)
```

### Design Principles

1. **Base is incomplete by design** - Cannot be launched alone
2. **No fallback defaults** - All required values must be explicitly set (legacy batch keys removed)
3. **Fail-fast validation** - Missing configuration detected immediately
4. **Clear error messages** - Developers know exactly what's missing
5. **YAML-only policy** - Config values are loaded strictly from YAML; no env-var expansion and no CLI overrides for config keys are supported. Use YAML fields exclusively.

## File Descriptions

### `dense_base.yaml` - Universal Constants

Contains only constants that **never change** across debug/standard environments:

- **Model constants**: `bf16`, `torch_dtype`, `image_max_pixels`, `trust_remote_code`
- **Loss weights**: LossManager compatibility ratios
- **GRPO algorithm**: Core algorithm parameters (`epsilon_low`, `epsilon_high`, etc.)
- **Runtime constants**: `ddp_backend`, `local_rank`
- **Generation constants**: Universal generation settings

**Missing (must be provided by child configs):**
- All paths (`model_path`, `data_paths`, `output_dir`)
- All training parameters (`learning_rate`, `batch_size`, etc.)
- All environment-specific settings

### `debug.yaml` - Fast Iteration Environment

Inherits from base and provides:
- **Debug paths**: Local checkpoint and data paths
- **Fast training**: `max_steps: 20`, `logging_steps: 1`
- Uses `grpo.sample_k` for per-update sample count (no separate global batch key)
- **Minimal freezing**: Limited layer training for speed
- **Small sampling**: `sample_k: 2` for faster iteration

### `standard.yaml` - Production Environment

Inherits from base and provides:
- **Production paths**: Full checkpoint and data paths  
- **Full training**: `max_steps: 1000`, proper warmup
- Per-update sample count is driven by `grpo.sample_k` (no separate global batch key)
- **Complete freezing**: Full layer training configuration
- **Optimal sampling**: `sample_k: 4` for better GRPO performance

## RL Conversation Shape

- RL runs are single‑turn only; no teacher‑student pairing.
- Expect standard single‑image batches per sample.
- Datasets and builders used by RL MUST NOT introduce additional turns beyond the student prompt.

## Required Configuration Sections

Every complete configuration **must** explicitly define:

### Core Paths (all required)
```yaml
model_path: "/path/to/checkpoint"
ref_model_path: "/path/to/ref_checkpoint"  
train_data_path: "/path/to/train.jsonl"
val_data_path: "/path/to/val.jsonl"
data_root: "/path/to/data"
output_dir: "/path/to/output"
```

### Experiment Tracking (all required)
```yaml
tb_dir: "/path/to/tensorboard"
run_name: "experiment_name"
seed: 42
```

### Model Configuration (required)
```yaml
model:
  attn_implementation: "flash_attention_2"  # required
  # base provides: torch_dtype, use_cache, trust_remote_code, image_max_pixels
```

### Layer Freezing (all sections required)
```yaml
layer_config:
  vision_tower:
    freeze_patch_embed: true
    freeze_bottom_layers: true  
    trainable_top_k_blocks: 4
  llm:
    freeze_bottom_layers: false
    trainable_top_k_blocks: -1
  merger:
    freeze: false
```

### Training Configuration (all required)
```yaml
training:
  max_steps: 1000
  warmup_steps: 25
logging:
  logging_steps: 10
checkpointing:
  save_steps: 200
optimizer:
  weight_decay: 0.0
```

### Generation Configuration (all required)
```yaml
grpo:
  sample_k: 4
  max_new_tokens: 1024
  temperature: 1.0
  top_p: 0.95
  repetition_penalty: 1.1
```

### Reward Configuration (required, at least one non-zero)
```yaml
rewards:
  parse: 0.30
  wrappers: 0.10
  coords: 0.10
  separators: 0.10
  vocab: 0.05
  length: 0.10
  length_window: 0.10
  bbox_giou: 0.15
```

## Validation Behavior

### Explicit Validation
- **No `.get(key, default)`** patterns in runner code
- **Direct key access** with immediate failure if missing
- **Type validation** for all configuration sections
- **Non-empty validation** for critical sections like rewards

### Error Examples

Missing model_path:
```
ValueError: Missing required 'model_path' (absolute path to SFT checkpoint)
```

Missing grpo section:
```
ValueError: Missing required 'grpo' section in config
```

Missing specific grpo parameter:
```
ValueError: Missing required 'grpo.epsilon_low' - must be explicitly set
```

## Usage

### Launch Debug Environment
```bash
python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode train
```

### Launch Production Environment  
```bash
python -m src_new.rl.runner --config configs/dense_rl/standard.yaml --mode train
```

### Base Alone (Will Fail)
```bash
python -m src_new.rl.runner --config configs/dense_rl/dense_base.yaml --mode load
# ValueError: Missing required 'model_path' (absolute path to SFT checkpoint)
```

## Benefits of New Structure

1. **Reduced Redundancy**: Common values defined once in base
2. **Explicit Configuration**: No hidden defaults that can cause issues
3. **Fail-Fast**: Missing config detected before training starts
4. **Clear Environment Separation**: Debug vs production clearly differentiated
5. **Maintenance**: Easy to update universal constants in one place
6. **Safety**: Impossible to accidentally run with wrong/missing config

## Migration Notes

When updating configurations:

1. **Remove redundant values** that are already in base
2. **Add missing required values** explicitly 
3. **Test with validation** to ensure completeness
4. **Use inheritance properly** - only override what differs from base

The new system prevents the common issue of "it worked on my machine" by ensuring all environments have explicitly defined, complete configurations.