# Data Schema & Conversion Pipeline

> **Purpose:** Define the JSONL formats used by the training pipeline and document how raw vendor annotations are converted into them.

---

## 1. Teacher-Student JSONL (training & validation)
```jsonc
{
  "teachers": [
    {
      "images": ["ds_output/img001.jpeg"],
      "objects": [
        {"bbox_2d": [x1, y1, x2, y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}
      ]
    }
  ],
  "student": {
    "images": ["ds_output/img002.jpeg"],
    "objects": [
      {"bbox_2d": [x1, y1, x2, y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}
    ]
  }
}
```
**Rules**
1. `bbox_2d` is **absolute** pixel coordinates after smart-resize.
2. Descriptions are natural-language Chinese phrases (English fallback allowed).
3. 70 % of samples include teachers (configurable `teacher_ratio`).

## 2. Enhanced Conversion Pipeline (data_conversion/)
```
ds/ (raw JSON + images) → convert_dataset.sh → ① clean_raw_json.py (NEW)
                                             → ② copy images to ds_output
                                             → ③ processor.py (unified)
                                             → ④ vision_process.smart_resize
                                             → data/ (train.jsonl, val.jsonl, teacher.jsonl)
```

**Pipeline Steps:**
1. **JSON Cleaning**: `clean_raw_json.py` strips unnecessary metadata while preserving essential structure (`info`, `markResult`, `features`)
2. **Image Processing**: Smart resize with bbox scaling using accurate coordinate transformation
3. **Unified Processing**: Single `processor.py` handles sample processing, teacher selection, and train/val splitting
4. **Output**: All data references `ds_output/` paths for consistency

**Key invariants checked during conversion:**
* All image files exist after resize and are copied to output directory.
* BBox bounds satisfy `0 ≤ x1 < x2 ≤ width` & `0 ≤ y1 < y2 ≤ height` after scaling.
* Original JSON structure preserved with cleaned content.
* Path consistency: all JSONL files reference correct `ds_output/` image paths.

## 3. Runtime validation (src/schema.py)
The dataclass `GroundTruthObject` and friends use **torchtyping** to validate shapes at runtime.  Any violation raises immediately (fail-fast).

## 4. Enhanced Conversion Code Map (`data_conversion/`)
| Module | Key classes / functions | Responsibility |
|--------|-------------------------|----------------|
| `convert_dataset.sh` | **Main Pipeline Script** | **NEW Enhanced**: Orchestrates complete pipeline with environment setup, JSON cleaning, image copying, and processing. |
| `clean_raw_json.py` | `clean_annotation_file()` | **NEW**: Strips unnecessary metadata while preserving essential JSON structure (`info`, `markResult`, `features`). Language-aware content filtering. |
| `processor.py` | `DataProcessor` | Unified entry-point that orchestrates loading, processing, splitting, and writing JSONL files. |
|  | `DataProcessor.process()` | Returns counts & writes `train.jsonl`, `val.jsonl`, `teacher.jsonl` with correct `ds_output/` paths. |
| `sample_processor.py` | `SampleProcessor` | Converts *one* vendor JSON + image into cleaned sample dict; handles label mapping & image processing with path updates. |
|  | `SampleProcessor._process_image_file()` | **Enhanced**: Copies/resizes images to output directory and returns updated paths for JSONL. |
| `teacher_selector.py` | `TeacherSelector.select_teachers()` | Greedy-diversity algorithm that builds the teacher demonstration pool. |
| `data_splitter.py` | `DataSplitter.split()` | Deterministic train/val split with `seed` + `val_ratio`. |
| `core_modules.py` | `TokenMapper`, `ObjectProcessor.scale_bbox()` | Token mapping (English mode) + **Enhanced**: Accurate bbox scaling with validation. |
| `vision_process.py` | `smart_resize()` | Ensures images satisfy pixel constraints with proper aspect ratio handling. |

---

### Related source files
* `data_conversion/convert_dataset.sh` *(Enhanced main pipeline)*
* `data_conversion/clean_raw_json.py` *(NEW: JSON cleaning)*
* `data_conversion/processor.py` *(Unified processing)*
* `data_conversion/sample_processor.py` *(Enhanced with path updates)*
* `data_conversion/core_modules.py` *(Enhanced bbox scaling)*
* `src/schema.py` 