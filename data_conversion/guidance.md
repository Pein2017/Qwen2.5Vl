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

### 1. Fix EXIF Orientation (Optional)
**Script:** `data_conversion/strip_exif_orientation.py`

**What it does:**
Applies EXIF orientation transforms in-place to ensure image pixel data matches annotations, then strips metadata.

**Example:**
```
$ python data_conversion/strip_exif_orientation.py ds --dry-run
2025-05-01 12:00:00 - INFO - Processed 100 images, modified 30
$ python data_conversion/strip_exif_orientation.py ds
2025-05-01 12:00:10 - INFO - Processed 100 images, modified 30
```
- **Input:** Folder `ds/` with JPEG/PNG images
- **Output:** Same folder with images rotated and metadata removed

---

### 2. Raw JSON → Intermediate JSONL
**Script:** `data_conversion/convert_pure_json.py`

**What it does:**
1. Reads each raw JSON annotation under `ds/`.
2. Validates `info.width/height` against actual image dimensions (fail-fast on mismatch).
3. Extracts bounding boxes from `dataList` or `markResult.features`.
4. Reads labels (`object_type`, `property`, `extra_info`), applies token mapping.
5. Sorts objects top-to-bottom, left-to-right.
6. Resizes images to multiples of 28px via `vision_process.smart_resize`, scales bboxes.
7. Saves rescaled images under `ds_rescaled/` preserving directory structure.
8. Writes intermediate JSONL: one sample per line.

**Sample Raw JSON (ds/0001.json):**
```json
{
  "info": {"width": 900, "height": 1200},
  "dataList": [
    {
      "coordinates": [[100,50],[400,300]],
      "properties": {"contentZh": {"标签": ["挡风板/安装"]}}
    }
  ]
}
```

**Command:**
```bash
python data_conversion/convert_pure_json.py \
  --input_folder ds \
  --output_image_folder ds_rescaled \
  --output_jsonl data_conversion/qwen_combined.jsonl \
  --language chinese \
  --resize true \
  --response_types "object_type property"
```

**Sample Output Line:**
```json
{
  "images": ["ds_rescaled/0001.jpeg"],
  "objects": {
    "ref": ["挡风板/安装"],
    "bbox": [[100,50,400,300]]
  },
  "height": 784,
  "width": 448
}
```
*(Note: heights/widths shown after smart_resize.)*

---

### 3. Extract Candidate Phrases
**Script:** `data_conversion/extract_unique_phrases.py`

**What it does:**
Scans the intermediate JSONL, parses each `objects.ref` string, extracts meaningful tokens per specified response types, and counts frequencies.

**Sample Input JSONL Entry:**
```json
{
  "objects": {"ref": [
      "object_type:bbu;property:match;extra_info:none",
      "object_type:install screw correct;property:none;extra_info:none"
  ], "bbox": [[0,0,85,140],[304,353,390,438]]}
}
```

**Command:**
```bash
python data_conversion/extract_unique_phrases.py \
  --input_jsonl data_conversion/qwen_combined.jsonl \
  --output_phrases data_conversion/candidate_phrases.json \
  --min_frequency 1 \
  --response_types "object_type property" 
```

**Sample Output (candidate_phrases.json):**
```json
{
  "metadata": {
    "total_unique_phrases": 2,
    "min_frequency_threshold": 1,
    "most_common_phrase": ["bbu", 100]
  },
  "phrases": {"bbu": 100, "install screw correct": 80},
  "phrase_list": ["bbu", "install screw correct"]
}
```

---

### 4. Extract Few-Shot Examples
**Script:** `data_analysis/extract_examples_from_conversations.py`

**What it does:**
Analyzes intermediate JSONL samples to select representative few-shot examples by complexity (object count, diversity, question types).
Categorizes into `sparse`, `medium`, `dense`, `diverse`, and `rare`, then picks top samples per category.

**Sample Input (qwen_combined.jsonl):**
```json
{
  "images": ["ds_rescaled/0001.jpeg"],
  "objects": {"ref":["object_type:bbu;property:match;extra_info:none"],
               "bbox":[[100,50,400,300]]}
}
```

**Command:**
```bash
python data_analysis/extract_examples_from_conversations.py \
  data_conversion/qwen_combined.jsonl \
  --output data_analysis/training_examples.json \
  --num_examples 5 \
  --seed 42 \
  --response_types object_type property
```

**Sample Output (training_examples.json):**
```json
{
  "sparse": {
    "image": "ds_rescaled/0001.jpeg",
    "objects": [
      {"bbox": [100,50,400,300], "description": "bbu/match"}
    ]
  },
  "medium": { ... },
  "dense": { ... }
}
```

---

### 5. Convert to Clean Semantic Data
**Script:** `data_conversion/qwen_converter_unified.py`

**What it does:**
Transforms intermediate JSONL into training-ready JSONL with simple or multi-round format:
- **Simple**: list of images + objects with `box` and `desc` fields
- **Multi-round**: includes `examples` array followed by `target`

**Sample Input:**
```json
{
  "images": ["ds_rescaled/0001.jpeg"],
  "objects": {"ref":["object_type:bbu;property:match;extra_info:none"],
               "bbox":[[100,50,400,300]]}
}
```

**Command (simple):**
```bash
python data_conversion/qwen_converter_unified.py \
  --input_jsonl data_conversion/qwen_combined.jsonl \
  --output_train data/clean_train.jsonl \
  --output_val data/clean_val.jsonl \
  --val_ratio 0.1 \
  --seed 42 \
  --response_types "object_type property"
```

**Sample Simple Output Line:**
```json
{
  "images": ["ds_rescaled/0001.jpeg"],
  "objects": [
    {"box": [100,50,400,300], "desc": "bbu/match"}
  ]
}
```

**Command (multi-round):** add `--multi_round --include_examples --examples_file data_analysis/training_examples.json --max_examples 1`

**Sample Multi-Round Output:**
```json
{
  "examples": [
    {"images": ["ds_rescaled/0002.jpeg"], "objects": [
      {"box": [200,100,500,350], "desc": "cabinet fully occupied"}
    ]}
  ],
  "target": {
    "images": ["ds_rescaled/0001.jpeg"],
    "objects": [
      {"box": [100,50,400,300], "desc": "bbu/match"}
    ]
  }
}
```

---

### 6. Validate Outputs
**Script:** `data_conversion/validate_jsonl.py`

**What it does:**
Checks JSONL syntax, conversation structure, and special token consistency.

**Command:**
```bash
python data_conversion/validate_jsonl.py \
  --train_file data/clean_train.jsonl \
  --val_file data/clean_val.jsonl \
  --include_examples
```

**Sample Validation Log:**
```
Validating JSON format...
✅ data/clean_train.jsonl: Valid JSONL format
✅ data/clean_val.jsonl: Valid JSONL format
Validating conversation structure...
✅ data/clean_train.jsonl: Valid conversation structure
✅ data/clean_val.jsonl: Valid conversation structure
Validating special tokens...
✅ data/clean_train.jsonl: Valid special token format
```

---

<a name="script-reference"></a>
## Script Reference
- `convert_dataset.sh`: Full pipeline orchestration
- `strip_exif_orientation.py`: EXIF fix and metadata strip
- `convert_pure_json.py`: Raw JSON to intermediate JSONL
- `extract_unique_phrases.py`: Candidate phrase extraction
- `data_analysis/extract_examples_from_conversations.py`: Few-shot example selection
- `qwen_converter_unified.py`: Clean semantic conversion & split
- `validate_jsonl.py`: JSONL and conversation validation

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
    {"box": [x1,y1,x2,y2], "desc": "type/property"},
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