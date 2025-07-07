# Project Overview

> **Purpose:** Explain the overall goal, scope, and high-level design decisions of the Qwen-BBU-VL codebase.

---

## 0. 30-second TL;DR
We fine-tune **Qwen-2.5-VL-3B** end-to-end for *simultaneous* dense object detection **and** captioning in BBU rooms. A modular, coordinator-based training system has replaced the original monolithic script, providing better structure and maintainability while remaining fully backward compatible.

## 1. Canonical Data Schema
```jsonc
{
  "teachers": [
    {"images": ["ds_rescaled/<img>.jpeg"], "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}]}
  ],
  "student": {
    "images": ["ds_rescaled/<img>.jpeg"],
    "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}]
  }
}
```
Key facts: absolute pixel boxes, natural-language descriptions, pre-scaled JPEGs, and a teacher-student training format.

## 2. End-to-End Execution Flow (New System)
```
bash → python scripts/train.py --config base_flat_v2 --use-new-config
         ↳ ConfigManager.load_from_yaml() # Domain-specific configs
         ↳ create_trainer_with_coordinator() # Factory function
              ↳ TrainingCoordinator       # Orchestrates training
              ↳ LossManager               # Computes loss
              ↳ ParameterGroupManager     # Manages param groups
              ↳ BBUTrainer.train()        # Starts training
```
Outputs live in `output-{run_name}/` (checkpoints, logs, TensorBoard). The legacy system can still be used by omitting the `--use-new-config` flag.

## 3. Source-Tree Reference (Modular Architecture)
| Path | Responsibility |
|------|----------------|
| `src/config/` | New domain-specific and legacy configuration systems. |
| `src/core/` | Core factories for models, data, and checkpoints. |
| `src/training/`| Modular training components (Coordinator, Managers, Trainer). |
| `src/detection/`| Detection-specific model heads and loss functions. |
| `src/models/` | Model wrappers and patches. |
| `src/inference.py` | Stand-alone inference pipeline. |
| `data_conversion/` | Raw JSON → clean JSONL converter. |

---

### Related Documentation
* `docs/architecture.md` for a detailed technical breakdown.
* `docs/MIGRATION_GUIDE.md` for switching from legacy to the new system.
* `docs/critical_fixes_log.md` for a history of major bug fixes.