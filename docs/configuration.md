# Configuration Reference (New System)

> **Purpose:** Summarize the domain-specific YAML parameters validated by the new `ConfigManager`.

---

## 1. New Configuration System
The new system uses a `ConfigManager` that loads and validates domain-specific configurations from a YAML file (e.g., `configs/base_flat_v2.yaml`). This provides better organization and validation than the old flat `DirectConfig`.

The system is enabled with the `--use-new-config` flag.

## 2. Domain-Specific Configurations
Configurations are now split into logical domains:

### ModelConfig (`src/config/domain_configs.py`)
| Name | Type | Example |
|------|------|---------|
| `model_path` | str | `/path/to/model` |
| `model_max_length` | int | 12000 |
| `attn_implementation`| str | `flash_attention_2`|

### TrainingConfig (`src/config/domain_configs.py`)
| Name | Type | Example |
|------|------|---------|
| `num_train_epochs` | int | 30 |
| `per_device_train_batch_size` | int | 4 |
| `learning_rate` | float | 5e-6 |
| `vision_lr` | float | 5e-7 |
| `llm_lr` | float | 5e-6 |
| `detection_lr` | float | 1e-5 |

### DataConfig (`src/config/domain_configs.py`)
| Name | Type | Example | Notes |
|------|------|---------|-------|
| `collator_type` | str | `packed` | `packed` for Flash-Attention 2. |
| `teacher_ratio` | float | 0.7 | Fraction of teacher batches. |
| `language` | str | `chinese` | Affects prompt selection. |

## 3. Parameter Group Management
Parameter groups and their learning rates are now managed by the `ParameterGroupManager`, which is configured via the `TrainingConfig`. The manager assigns parameters to groups based on their names (e.g., `visual`, `llm`, `detection_head`).

---

### Related source files
* `src/config/config_manager.py`
* `src/config/domain_configs.py`
* `src/training/parameter_manager.py`
* `configs/base_flat_v2.yaml` 