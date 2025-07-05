# Project Overview

> **Purpose:** Explain the overall goal, scope, and high-level design decisions of the Qwen-BBU-VL codebase.

---

## 0. 30-second TL;DR
We fine-tune **Qwen-2.5-VL-3B** end-to-end for *simultaneous* dense object detection **and** captioning in BBU rooms.  A modular training system ("New System") replaces the original monolithic script while remaining backward compatible.

## 1. Canonical Data Schema (Current)
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
Key facts: absolute pixel boxes, natural-language descriptions, pre-scaled JPEGs, 70 % teacher ratio.

## 2. End-to-End Execution Flow
```
bash → python scripts/train.py --config …
         ↳ init_config()            # YAML → DirectConfig
         ↳ create_trainer()         # BBUTrainer
              ↳ setup_model()       # Qwen2.5-VL + patches
              ↳ setup_data_module() # BBUDataset + collator
              ↳ trainer.train()     # Multi-task optimisation
```
Outputs live in `output-{run_name}/` (checkpoints, logs, TensorBoard).

## 3. Source-Tree Reference
| Path | Responsibility |
|------|----------------|
| `src/config/global_config.py` | DirectConfig (149 params) |
| `src/schema.py` | Torchtyping tensor schema |
| `src/data.py` | BBUDataset + collators |
| `src/models/` | Model wrapper & detection head |
| `src/training/trainer.py` | Custom `BBUTrainer` |
| `src/inference.py` | Stand-alone inference pipeline |
| `data_conversion/` | Raw JSON → clean JSONL converter |

---

### Related source files
* `src/config/global_config.py`
* `src/data.py`
* `src/training/trainer.py`