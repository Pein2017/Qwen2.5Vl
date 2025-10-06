# Configuration Structure

This directory contains the reorganized, simplified configuration system for Qwen2.5-VL training.

## ✅ System Status
**All configurations are validated and working!** The system has been thoroughly tested and all 8 configurations load successfully and work with the training pipeline.

## Structure Overview

```
configs/
├── base/                       # Base configurations (universal settings only)
│   └── sft_base.yaml          # SFT universal settings
├── phase_1/                   # Phase 1 SFT configs
│   ├── standard.yaml          # Standard phase 1 training
│   └── debug.yaml             # Debug phase 1 training
├── phase_2/                   # Phase 2 SFT configs  
│   ├── standard.yaml          # Standard phase 2 training
│   └── debug.yaml             # Debug phase 2 training
├── phase_3/                   # Phase 3 SFT configs
│   ├── standard.yaml          # Standard phase 3 training
│   └── debug.yaml             # Debug phase 3 training
├── phase_summary/             # Summary generation SFT configs
│   ├── standard.yaml          # Standard summary training
│   └── debug.yaml             # Debug summary training
├── dense_rl/                  # Dense-caption RL (src_new) configs
│   ├── standard.yaml
│   └── debug.yaml
├── rl/                        # Group QC RL (src_post) configs
│   └── group_qc_grpo.yaml     # Group QC GRPO training
└── README.md                  # This documentation
```

## Design Principles

1. **Minimal base configs**: Only universal, non-tunable settings in base configs
2. **Everything else tunable**: Model paths, loss configs, features, etc. in specific configs
3. **No redundancy**: Avoid repeating settings between configs
4. **Clear separation**: SFT and the two RL pipelines have separate bases/locations
5. **Two-layer hierarchy**: Only base configs + specific configs (no intermediate layers)
6. **Phase-specific variants**: Each phase has standard and debug variants

## Base Configurations

### `base/sft_base.yaml`
Contains **only** universal, non-tunable settings:
- **Data root**: `data_root` path
- **Torch configs**: `torch_dtype`, `use_cache`
- **Universal training**: optimizer/scheduler types, `max_grad_norm`, `gradient_checkpointing`
- **Dataloader settings**: `dataloader_num_workers`, `dataloader_pin_memory`, `prefetch_factor`, `remove_unused_columns`
- **Sequence limits**: `max_total_length`
- **System settings**: CUDA/NCCL logging suppression
- **Runtime**: seed

## What's Tunable (in Phase/RL Configs)

All specific configurations contain the tunable parameters:
- **Model paths**: `model_path`, `merge_size`, `max_pixels`
- **Data paths**: `train_data_path`, `val_data_path`, `teacher_pool_file`
- **Training parameters**: learning rates, epochs, batch sizes, variant ratios
- **Loss configuration**: loss weights, geometry settings, grouped losses
- **Features**: phase settings, teacher pairing, augmentation, checkpointing
- **Layer configuration**: freezing settings, trainable blocks
- **Advanced settings**: debug flags, span settings
- **Output configuration**: run names, output directories

## Phase Configurations

Each phase has multiple variants: **standard** and **debug**:

### Phase 1: Vision Encoder Fine-tuning
- **Focus**: Train vision encoder, freeze LLM completely
- **Configs**: 
  - `phase_1/standard.yaml` - Standard phase 1 training
  - `phase_1/debug.yaml` - Quick debug with 100 samples, 1 epoch
- **Key settings**: 
  - LLM frozen (`llm_lr: 0.0`, `llm_top_k_block: 0`)
  - Pure dense captioning or mixed variants
  - No or light teacher examples

### Phase 2: Progressive LLM Unfreezing  
- **Focus**: Train top LLM layers + vision with coordinate understanding
- **Configs**: 
  - `phase_2/standard.yaml` - Standard phase 2 training
  - `phase_2/debug.yaml` - Quick debug with 100 samples, 1 epoch
- **Key settings**:
  - Train top 4 LLM blocks (`llm_top_k_block: 4`)
  - Add coordinate variants (`coords_to_desc: 0.2`, `desc_to_coords: 0.2`)
  - Light to moderate teacher pairing

### Phase 3: Full Model Fine-tuning
- **Focus**: Train all layers with all variants
- **Configs**: 
  - `phase_3/standard.yaml` - Standard phase 3 training
  - `phase_3/debug.yaml` - Quick debug with 100 samples, 1 epoch
- **Key settings**:
  - Train all LLM and vision blocks (`llm_top_k_block: -1`)
  - All conversation variants including summary
  - Full teacher pairing (`num_teacher_samples: 1500`)

### Phase Summary: Summary Generation Fine-tuning
- **Focus**: Specialized training for summary generation
- **Configs**: 
  - `phase_summary/standard.yaml` - Standard summary training
  - `phase_summary/debug.yaml` - Quick debug with 100 samples, 1 epoch
- **Key settings**:
  - Train all LLM and vision blocks (`llm_top_k_block: -1`)
  - Summary variant only (`summary: 1.0`)
  - No teacher pairing

## Configuration Variants

### Standard Configs
- **Purpose**: Production training with full settings
- **Data**: Full dataset, appropriate teacher pairing
- **Training**: Multiple epochs, proper learning rates
- **Features**: Full augmentation, checkpointing, logging

### Debug Configs  
- **Purpose**: Quick testing and debugging
- **Data**: Limited to 100 samples, minimal teacher examples
- **Training**: 1 epoch, frequent logging (every step)
- **Features**: No augmentation, frequent saves/eval, debug alignment enabled

## RL Configuration

### Dense Captioning GRPO (src_new)
- **Focus**: Improve dense captioning quality (object descriptions + geometry) with GRPO on the same HF-first pipeline as SFT.
- **Configs**: `dense_rl/standard.yaml`, `dense_rl/debug.yaml`
- **Runner**: `python -m src_new.rl.runner --config /abs/configs/dense_rl/standard.yaml --mode {load|train}`
- **Eval**: `python -m src_new.rl.eval --config /abs/configs/dense_rl/standard.yaml --input_file /abs/val.jsonl --data_root /abs/data_root`
- **Launcher**: `bash scripts/run_dense_grpo.sh` (auto-distributed for train; single-process eval)
- **Schema**: `src_new.config.rl_config.EnhancedRLConfig`

### Group QC GRPO (src_post)
- **Focus**: Group-level quality control judgment with GRPO
- **Config**: `rl/group_qc_grpo.yaml`
- **Runner**: `python -m src_post.runner --config /abs/configs/rl/group_qc_grpo.yaml --mode {train|eval}`
- **Important**: This is a separate project/module with different objectives and schema. Do not mix `src_post` configs with `src_new` runners, and vice versa.

## Validation Results

All configurations have been thoroughly tested and validated:

### ✅ Configuration Loading Test
All 8 SFT configurations load successfully:
- `phase_1/standard` ✅ - `phase_1/debug` ✅
- `phase_2/standard` ✅ - `phase_2/debug` ✅  
- `phase_3/standard` ✅ - `phase_3/debug` ✅
- `phase_summary/standard` ✅ - `phase_summary/debug` ✅

Dense RL configs (src_new) load via `src_new.rl.runner` and support both `load` and `train` modes. Group QC RL uses `src_post`.

### ✅ Training Pipeline Compatibility
- Training script can load and validate all SFT configurations
- Proper field resolution and inheritance working
- All essential training parameters present
- Launcher script integration working with proper help and examples

### ✅ Key Configuration Parameters
- **Phase 1**: LR 1e-4, LLM blocks: 0 (vision only), no teacher pairing
- **Phase 2**: LR 5e-5, LLM blocks: 4 (top layers), teacher pairing enabled
- **Phase 3**: LR 2e-5, LLM blocks: -1 (all), full augmentation pipeline
- **Phase Summary**: LR 5e-5, LLM blocks: -1 (all), summary variant only

## Usage Examples

### SFT Training
```bash
# Standard training
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_1/standard --log_level INFO
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_2/standard --log_level INFO
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_3/standard --log_level INFO
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_summary/standard --log_level INFO

# Debug training
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_1/debug --log_level INFO
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_2/debug --log_level INFO
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_3/debug --log_level INFO
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_summary/debug --log_level INFO

# Using the launcher script
CONFIG_NAME=phase_3/standard bash scripts/run_new_train.sh
DEBUG_MODE=true CONFIG_NAME=phase_1/debug bash scripts/run_new_train.sh
GPU_DEVICES=0,1,2,3 CONFIG_NAME=phase_2/standard bash scripts/run_new_train.sh

# Configuration validation
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_1/standard --validate-only --log_level INFO
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_1/standard --print-config --log_level INFO
```

### RL Post-training
```bash
# Dense captioning RL (src_new)
bash scripts/run_dense_grpo.sh
RL_CONFIG_PATH=configs/dense_rl/debug.yaml bash scripts/run_dense_grpo.sh
RL_MODE=eval RL_INPUT_FILE=/abs/val.jsonl RL_DATA_ROOT=/abs/data_root bash scripts/run_dense_grpo.sh

# Group QC RL (src_post)
bash scripts/run_group_qc_rl.sh /abs/path/to/configs/rl/group_qc_grpo.yaml
```

## Key Features

### Minimal Base Configs
Base configs contain **only** the settings you specified:
- `data_root` and `max_total_length`
- Torch-relevant configs like `dtype`
- Optimizer type and scheduler
- Training `grad_norm` and `gradient_checkpointing`
- Dataloader settings: `num_workers`, `pin_memory`, `prefetch_factor`, `remove_unused_columns`
- System-level environment variables

### CUDA/NCCL Logging Suppression
All configs include system settings to suppress verbose CUDA/NCCL logs:
```yaml
system:
  environment_vars:
    NCCL_DEBUG: "WARN"
    CUDA_LAUNCH_BLOCKING: "0"
    TRANSFORMERS_VERBOSITY: "warning"
    TOKENIZERS_PARALLELISM: "false"
```

### Consistent Schema Alignment
- SFT configs follow `src_new.config.schema.TrainingConfig` structure
- Dense RL configs follow `src_new.config.rl_config.EnhancedRLConfig` structure
- Group QC RL configs follow `src_post.config` schema
- All configs use proper nested hierarchies

### Auto-resolved Paths
Data paths are auto-resolved from `data_root`:
```yaml
data:
  data_root: data/ds_v2_full
  train_data_path: null    # Auto-resolved to {data_root}/train.jsonl
  val_data_path: null      # Auto-resolved to {data_root}/val.jsonl
```

### Legacy Coordinate Token Removal
All new configs use modern coordinate handling:
```yaml
geometry:
advanced:
  trainable_token_strings: []       # No legacy coordinate tokens
```

## Customization

To create custom configs:

1. **Extend appropriate base**: Use `extends: [../base/sft_base.yaml]` 
2. **Specify all tunable parameters**: Include model_path, loss, features, etc.
3. **Follow naming convention**: Use descriptive names like `phase_X/variant.yaml`
4. **Set unique run names**: Always specify unique `output.run_name`

## Recent Fixes and Improvements

The configuration system has been recently overhauled to fix multiple issues:

### Issues Resolved
1. **Schema Validation Errors**: Fixed field name mismatches (`pin_memory` → `dataloader_pin_memory`) and missing conditional requirements
2. **Configuration Inheritance**: Created robust base configuration and proper inheritance chains
3. **Phase-Specific Problems**: Fixed each phase's unique configuration issues systematically
4. **Training Pipeline Integration**: Ensured all configs work seamlessly with training scripts and launcher

### Key Improvements
- **Robust base configuration** (`configs/base/sft_base.yaml`) with universal settings
- **8 working configurations**: 4 phases × 2 variants (standard/debug) each
- **Proper inheritance chains** for maintainable configs
- **Schema compliance** with all required fields
- **Comprehensive validation** - all configs tested and working

## Migration from Old Configs

The new structure provides:
- **Truly minimal base configs**: Only universal, system-level settings
- **Complete phase configs**: All tunable parameters explicitly specified
- **No hidden defaults**: Everything is visible and configurable
- **Clear separation**: Universal vs. tunable settings are clearly distinguished
- **Phase-specific variants**: Debug and sanity configs organized by phase

This makes it easy to:
- Understand exactly what each config changes
- Maintain consistent universal settings
- Add new training variants
- Debug configuration issues
- Ensure no important settings are hidden in base configs
- Quick access to debug variants for each phase
