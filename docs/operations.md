# Operations, Diagnostics & Fail-Fast Philosophy

> **Purpose:** Summarise runtime checks, monitoring hooks, and common failure modes in the new modular system.

---

## 1. Special tokens (IDs are asserted at startup)
| Token | String | Purpose |
|-------|--------|---------|
| IM_START | `<|im_start|>` | Start of chat turn |
| IM_END   | `<|im_end|>`   | End of turn & sequence |
| VISION_START | `<|vision_start|>` | Vision prefix |
| VISION_END   | `<|vision_end|>`   | Vision suffix |
| IMAGE_PAD    | `<|image_pad|>`    | One image patch |

## 2. Fail-fast guards (executed before training)
* **Configuration Validation**: `ConfigManager` validates domain-specific configs and cross-config dependencies at startup.
* **Path Consistency**: All data paths in the configuration are checked for existence.
* **Parameter Grouping**: `ParameterGroupManager` ensures all trainable parameters are assigned to a learning rate group.

## 3. Monitoring hooks (New System)
| Metric | Logged by | Location |
|--------|-----------|----------|
| `lm_loss`, `teacher_lm_loss`, `student_lm_loss` | `LossManager` | `training/loss_manager.py` |
| `bbox_*`, `objectness_loss`, `caption_loss` | `LossManager` | `training/loss_manager.py` |
| Gradient / weight norms | `BBUTrainer` | `training/trainer.py` |

## 4. Common pitfalls (and fixes)
| Symptom | Likely cause |
|---------|--------------|
| "BOX BOX BOX" string, no coords | Using `model.generate()` instead of the detection pipeline via `Inference.predict_detection`. |
| Vision token mismatch assertion | Images were not correctly processed by the `ChatProcessor`. |
| **`ConfigValidation-Error`** | A required parameter is missing from your YAML config, or a value is invalid. Check the error message for details. |
| **`KeyError` during optimizer creation** | A trainable parameter was not assigned to a group by the `ParameterGroupManager`. |

## 5. Collator-specific assertions
`PackedDataCollator` adds two extra safety checks:
1. **Boundary masking** – The label at each sample boundary (`cu_seqlens[1:-1]`) is set to `-100` to prevent cross-sample supervision.
2. **Position-ID reset** – Verifies that `position_ids[0, cu_seqlens[:-1]] == 0` so every packed sample restarts the rotary cache.

---

### Related source files
* `src/config/config_manager.py`
* `src/training/training_coordinator.py`
* `src/training/loss_manager.py`
* `src/training/parameter_manager.py` 