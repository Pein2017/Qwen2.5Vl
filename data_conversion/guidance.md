# Qwen2.5-VL Data Conversion & Preprocessing Guide

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Script Reference](#script-reference)
- [Data Formats](#data-formats)

---

<a name="overview"></a>
## Overview
This guide explains each step of the data conversion pipeline for Qwen2.5-VL, turning raw JSON annotations and images into training-ready JSONL files. Examples of inputs and outputs are provided to illustrate what each script does under the hood.

<a name="prerequisites"></a>
## Prerequisites
- Activate the Conda environment: `conda activate ms`
- Ensure the following are installed: Python ≥3.8, Pillow, torch, torchvision, requests
- Set `PYTHONPATH` to the project root:
  ```bash
  export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
  ```

<a name="quick-start"></a>
## Quick Start
To run the entire pipeline with default settings:
```bash
cd /data4/Qwen2.5-VL-main
data_conversion/convert_dataset.sh
```
The script will orchestrate all steps. Customize parameters at the top of `convert_dataset.sh` as needed.

<a name="pipeline-steps"></a>
## Pipeline Steps

### 0. Class Hierarchy Configuration
**Configuration File:** `data_conversion/label_hierarchy.json`
**Purpose:** Explicitly define each `object_type` and its allowed `property` values.
**Behavior:** Segments beyond declared `property` values are concatenated as `extra_info`. Control which levels appear via `--response_types` flags (e.g., `"object_type property"`).

### 1. Convert Raw JSON to Intermediate JSONL
**Script:** `data_conversion/convert_pure_json.py`
**Description:** Reads raw JSON annotations from `ds/`, validates dimensions, optionally resizes images using `vision_process.smart_resize`, and outputs:
```jsonc
{"images": ["ds_rescaled/<img>.jpeg"], "objects": {"ref": [...], "bbox": [...]}, "height": H, "width": W}
```
**Command:**
```bash
python data_conversion/convert_pure_json.py \
  --input_folder ds \
  --output_image_folder ds_rescaled \
  --output_jsonl data_conversion/qwen_combined.jsonl \
  --language chinese \
  --resize true \
  --response_types "object_type property" \
  --map_file data_conversion/token_map_zh.json \
  --log_level DEBUG
```

### 2. Extract Candidate Phrases
**Script:** `data_conversion/extract_candidates.py`
**Description:** Parses intermediate JSONL, extracts unique phrases per `--response_types`, counts frequencies, and saves to `data_conversion/candidate_phrases.json`.
**Command:**
```bash
python data_conversion/extract_candidates.py \
  --input_jsonl data_conversion/qwen_combined.jsonl \
  --output_phrases data_conversion/candidate_phrases.json \
  --min_frequency 1 \
  --response_types "object_type property"
```

### 3. Create Teacher Pool
**Script:** `data_conversion/create_teacher_pool.py`
**Description:** Builds a diverse set of teacher samples covering all labels and scene types. Outputs a JSON list of image paths to `data_conversion/teacher_pool.json`.
**Command:**
```bash
python data_conversion/create_teacher_pool.py \
  --data_path data_conversion/qwen_combined.jsonl \
  --hierarchy data_conversion/label_hierarchy.json \
  --max_teachers 10 \
  --output data_conversion/teacher_pool.json
```

### 4. Filter Student Pool
**Description:** Removes teacher samples from intermediate JSONL to form student pool.
**Implementation:** Inline Python snippet in `convert_dataset.sh` writes filtered entries to `data_conversion/student_combined.jsonl`.

### 5. Convert to Clean Semantic Data
**Script:** `data_conversion/qwen_converter_unified.py`
**Description:** Transforms student pool JSONL into clean semantic format with `bbox_2d` and slash-separated `desc`. Teacher pool samples are used separately for multi-chat conversations during training.
**Command:**
```bash
python data_conversion/qwen_converter_unified.py \
  --input_jsonl data_conversion/student_combined.jsonl \
  --output_train data/chinese_train.jsonl \
  --output_val data/chinese_val.jsonl \
  --val_ratio 0.1 \
  --seed 42 \
  --response_types "object_type property"
```

**Note:** Teacher pool (`data_conversion/teacher_pool.json`) contains image paths for diverse examples used in multi-chat training conversations. The training pipeline uses mixed training (70% teacher-student, 30% single-shot) to ensure the model learns both multi-chat and single-shot patterns.

### 6. Validate Outputs (Optional)
**Script:** `data_conversion/simple_validate.py`
**Description:** Validates clean semantic JSONL structure and required fields.
**Command:**
```bash
python data_conversion/simple_validate.py data/chinese_train.jsonl data/chinese_val.jsonl
```

---

<a name="script-reference"></a>
## Script Reference
- `convert_dataset.sh`: Full pipeline orchestration
- `convert_pure_json.py`: Raw JSON to intermediate JSONL with image resizing
- `extract_candidates.py`: Candidate phrase extraction and frequency analysis
- `create_teacher_pool.py`: Teacher sample selection for multi-chat diversity
- `qwen_converter_unified.py`: Clean semantic conversion & train/val split
- `simple_validate.py`: Basic validation for clean semantic JSONL format
- `core_modules.py`: Shared utilities for token mapping, validation, and formatting

<a name="data-formats"></a>
## Data Formats

### Raw JSON
Proprietary telecom inspection schema with `info`, `dataList` or `markResult.features`.

### Intermediate JSONL
```json
{
  "images": ["path/to/img.jpeg"],
  "objects": {"ref": [...], "bbox": [[x1,y1,x2,y2], ...]},
  "height": H, "width": W
}
```

### Clean Semantic JSONL (Simple)
```json
{
  "images": ["path/to/img.jpeg"],
  "objects": [
    {"bbox_2d": [x1,y1,x2,y2], "desc": "type/property"},
    ...
  ]
}
```

### Clean Semantic JSONL (Multi-Round)
```json
{
  "examples": [...],
  "target": { ... }
}
```

---