# ⚙️ Configuration Reference — Dual Configuration System

*Last updated: 2025-01-18 – Reference for Legacy + New Domain-Specific configuration systems*

---
## 0. Configuration System Overview

### Legacy System (DirectConfig)
**Single Source of Truth**: `configs/base_flat.yaml`  
**Access Pattern**: Direct flat access via `src.config.config.parameter_name`  
**Validation**: All parameters required, no defaults in code  
**Loading**: `init_config("configs/base_flat.yaml")` at application startup

### New Domain-Specific System
**Enhanced Configuration**: `configs/base_flat_v2.yaml`  
**Access Pattern**: Domain-specific access via `config_manager.model.parameter_name`  
**Validation**: Cross-domain validation and dependency checking  
**Loading**: `ConfigManager().load_from_yaml("configs/base_flat_v2.yaml")` with `--use-new-config` flag

### Migration Guide
1. **Use Legacy**: `python scripts/train.py --config base_flat --log_level INFO --log_verbose true`
2. **Use New**: `python scripts/train.py --config base_flat_v2 --use-new-config --log_level INFO --log_verbose true`

---
## 1. Model Configuration (10 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `model_path` | str | Path to Qwen2.5-VL model (local or HF) | `"/data4/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"` |
| `model_size` | str | Model size identifier | `"3B"` |
| `model_max_length` | int | Maximum sequence length | `120000` |
| `attn_implementation` | str | Attention implementation | `"flash_attention_2"` |
| `torch_dtype` | str | Model precision | `"bfloat16"` |
| `use_cache` | bool | Enable KV caching | `false` |
| `model_hidden_size` | int | Hidden dimension size | `3584` |
| `model_num_layers` | int | Number of transformer layers | `28` |
| `model_num_attention_heads` | int | Number of attention heads | `28` |
| `model_vocab_size` | int | Vocabulary size | `152064` |

---
## 2. Training Configuration (18 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `num_train_epochs` | int | Number of training epochs | `30` |
| `per_device_train_batch_size` | int | Training batch size per device | `4` |
| `per_device_eval_batch_size` | int | Evaluation batch size per device | `4` |
| `gradient_accumulation_steps` | int | Steps to accumulate gradients | `2` |
| `learning_rate` | float | Base learning rate (if uniform) | `0` |
| `vision_lr` | float | Vision encoder learning rate | `5e-7` |
| `merger_lr` | float | Vision-language merger LR | `1e-5` |
| `llm_lr` | float | Language model learning rate | `5e-6` |
| `detection_lr` | float | Detection head learning rate | `1e-5` |
| `adapter_lr` | float | Adapter modules learning rate | `5e-3` |
| `warmup_ratio` | float | Warmup ratio of total steps | `0.1` |
| `weight_decay` | float | L2 regularization strength | `0.001` |
| `max_grad_norm` | float | Gradient clipping threshold | `0.5` |
| `lr_scheduler_type` | str | Learning rate scheduler | `"cosine"` |
| `gradient_checkpointing` | bool | Enable gradient checkpointing | `true` |
| `bf16` | bool | Use bfloat16 training | `true` |
| `fp16` | bool | Use float16 training | `false` |
| `mixed_precision` | str | Mixed precision mode | `"bf16"` |

---
## 3. Data Configuration (12 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `train_data_path` | str | Training data JSONL path | `"data/chinese-train.jsonl"` |
| `val_data_path` | str | Validation data JSONL path | `"data/chinese-val.jsonl"` |
| `data_root` | str | Data root directory | `"./"` |
| `max_total_length` | int | Maximum total sequence length | `12000` |
| `use_candidates` | bool | Use candidate phrases | `true` |
| `candidates_file` | str | Candidate phrases file | `"data_conversion/candidate_phrases.json"` |
| `teacher_pool_file` | str | Teacher pool JSONL file | `"data/teacher_pool.jsonl"` |
| `num_teacher_samples` | int | Number of teacher samples | `1` |
| `collator_type` | str | Collator type ("standard"/"packed") | `"packed"` |
| `teacher_ratio` | float | Ratio using teacher samples | `0.7` |
| `max_examples` | int | Maximum examples per sample | `1` |
| `language` | str | Primary language | `"chinese"` |

---
## 4. Detection Head Configuration (13 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `detection_enabled` | bool | Enable detection head | `false` |
| `detection_num_queries` | int | Number of object queries | `100` |
| `detection_max_caption_length` | int | Max caption token length | `32` |
| `detection_decoder_dim_feedforward_factor` | float | Decoder FFN dimension factor | `2.0` |
| `detection_decoder_num_layers` | int | Number of decoder layers | `2` |
| `detection_caption_decoder_dim_feedforward_factor` | float | Caption decoder FFN factor | `2.0` |
| `detection_caption_decoder_num_layers` | int | Caption decoder layers | `4` |
| `detection_head_dropout` | float | Detection head dropout | `0.1` |
| `detection_adapter_bottleneck_ratio` | int | Adapter bottleneck ratio | `8` |
| `detection_adapter_num_layers` | int | Adapter layers | `1` |
| `detection_bbox_weight` | float | Bounding box loss weight | `10.0` |
| `detection_giou_weight` | float | GIoU loss weight | `20.0` |
| `detection_objectness_weight` | float | Objectness loss weight | `10.0` |
| `detection_caption_weight` | float | Caption loss weight | `0.02` |
| `detection_focal_loss_gamma` | float | Focal loss gamma parameter | `2.0` |
| `detection_focal_loss_alpha` | float | Focal loss alpha parameter | `0.25` |
| `detection_freeze_epochs` | int | Epochs to freeze detection head | `0` |

---
## 5. Vision Processing Configuration (3 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `patch_size` | int | Spatial patch size of vision encoder | `14` |
| `merge_size` | int | Merge size from vision to LLM encoder | `2` |
| `temporal_patch_size` | int | Temporal patch size | `2` |

---
## 6. Evaluation Configuration (5 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `eval_strategy` | str | Evaluation strategy | `"steps"` |
| `eval_steps` | int | Steps between evaluations | `10` |
| `save_strategy` | str | Checkpoint save strategy | `"steps"` |
| `save_steps` | int | Steps between saves | `50` |
| `save_total_limit` | int | Maximum checkpoints to keep | `2` |

---
## 7. Logging Configuration (8 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `logging_steps` | int | Steps between log outputs | `1` |
| `logging_dir` | str | Logging directory (null=auto) | `null` |
| `log_level` | str | Log level | `"INFO"` |
| `report_to` | str | Reporting backend | `"tensorboard"` |
| `verbose` | bool | Verbose logging | `true` |
| `disable_tqdm` | bool | Disable progress bars | `true` |
| `tb_dir` | str | TensorBoard directory | `"tb"` |

---
## 8. Performance Configuration (8 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `use_flash_attention` | bool | Use Flash Attention 2 | `true` |
| `dataloader_num_workers` | int | Data loader worker processes | `8` |
| `pin_memory` | bool | Pin memory for data loading | `true` |
| `prefetch_factor` | int | Prefetch factor for data loading | `2` |
| `batching_strategy` | str | Batching strategy | `"standard"` |
| `remove_unused_columns` | bool | Remove unused columns | `false` |

---
## 9. Output Configuration (3 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `output_dir` | str | Base output directory | `"output-626"` |
| `run_name` | str | Specific run name | `"626-random_teacher-packed-04mini"` |

---
## 10. Stability & Recovery Configuration (9 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `max_consecutive_nan` | int | Max consecutive NaN before abort | `5` |
| `max_consecutive_zero` | int | Max consecutive zero losses | `5` |
| `max_nan_ratio` | float | Max ratio of NaN in batch | `0.3` |
| `nan_monitoring_window` | int | Window size for NaN monitoring | `100` |
| `allow_occasional_nan` | bool | Allow occasional NaN values | `true` |
| `nan_recovery_enabled` | bool | Enable NaN recovery | `true` |
| `learning_rate_reduction_factor` | float | LR reduction on instability | `0.5` |
| `gradient_clip_reduction_factor` | float | Grad clip reduction factor | `0.5` |

---
## 11. Debug & Testing Configuration (2 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `test_samples` | int | Number of test samples | `1` |
| `test_forward_pass` | bool | Test forward pass | `false` |

---
## 12. Learning Rate Scaling Configuration (2 parameters)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `lr_reference_batch_size` | int | Reference batch size for LR scaling | `0` |
| `auto_scale_lr` | bool | Auto-scale LR based on collator | `false` |

---
## 13. Derived Configuration (Auto-Generated)

These are computed automatically from other parameters:

| Property | Description |
|----------|-------------|
| `run_output_dir` | `{output_dir}/{run_name}` |
| `tensorboard_dir` | `{run_output_dir}/tb` |
| `log_file_dir` | `{run_output_dir}/logs` |
| `tune_vision` | `vision_lr > 0` |
| `tune_mlp` | `merger_lr > 0` |
| `tune_llm` | `llm_lr > 0` |
| `tune_detection` | `detection_lr > 0` |
| `use_differential_lr` | Multiple unique LR values |

---
## 14. Parameter Groups for Optimization

The trainer automatically assigns parameters to LR groups:

| Group | Parameters | LR Parameter |
|-------|------------|--------------|
| `vision` | `visual.*` | `vision_lr` |
| `merger` | `merger.*` | `merger_lr` |
| `llm` | `model.*`, `lm_head.*` | `llm_lr` |
| `detection` | `detection_head.*` | `detection_lr` |
| `adapter` | `*adapter*` | `adapter_lr` |

**Fail-Fast**: Any parameter not matching these patterns raises `KeyError`.

---
## 15. Validation Rules

1. **Required Parameters**: All 149 parameters must be explicitly defined
2. **Type Checking**: Automatic validation against dataclass type hints
3. **Range Validation**: Learning rates ≥ 0, batch sizes > 0, etc.
4. **Path Validation**: File paths checked for existence where applicable
5. **Consistency**: Detection parameters validated when `detection_enabled: true`

---
## 16. Example Complete Configuration

See `configs/base_flat.yaml` for a complete working configuration with all 149 parameters defined.

---
This reference documents every configurable aspect of the training pipeline. All parameters are required and validated at startup for fail-fast behavior. 