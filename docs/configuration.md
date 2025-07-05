# Configuration Reference (DirectConfig)

> **Purpose:** Summarise all YAML parameters validated by `src/config/global_config.py`.  No defaults are hard-coded; every field must appear in YAML.

---

## 1. Model
| Name | Type | Example |
|------|------|---------|
| `model_path` | str | `/data4/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct` |
| `model_max_length` | int | 12000 |
| `attn_implementation` | str | `flash_attention_2` |
| `torch_dtype` | str | `bfloat16` |

## 2. Training (excerpt)
| Name | Type | Example |
|------|------|---------|
| `num_train_epochs` | int | 30 |
| `per_device_train_batch_size` | int | 4 |
| `gradient_accumulation_steps` | int | 2 |
| `vision_lr` / `merger_lr` / `llm_lr` / … | float | 5e-7 / 1e-5 / 5e-6 |

*(See YAML for full list of 149 parameters.)*

## 3. Parameter Groups
`BBUTrainer` auto-assigns parameters to groups → learning rates:
```
vision      → visual.*          → vision_lr
merger      → merger.*          → merger_lr
llm         → model.* + lm_head → llm_lr
detection   → detection_head.*  → detection_lr
adapter     → *adapter*         → adapter_lr
```
Unmatched parameters raise at startup (fail-fast).

## 4. Data & Collator settings
| Name | Type | Example | Notes |
|------|------|---------|-------|
| `collator_type` | str | `standard` / `packed` | `packed` enables Flash-Attention 2 variable-length path. |
| `teacher_ratio` | float | 0.7 | Fraction of batches that include teacher demonstrations. |
| `num_teacher_samples` | int | 1-2 | How many teacher examples to prepend. |
| `language` | str | `chinese` / `english` | Affects prompt selection. |

---

### Related source files
* `src/config/global_config.py` 