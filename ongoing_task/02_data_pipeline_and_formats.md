# 🗄️ Data Pipeline & Format Reference — Qwen-BBU-VL (Updated)

*Last updated: 2025-01-18 – updated with current teacher-student schema and conversion pipeline*

---
## 0. Pipeline Overview
```
Raw vendor JSONs → data_conversion/convert_dataset.sh → Clean JSONL + ds_rescaled/
       ↓                    ↓                               ↓
Annotation files    convert_pure_json.py              Teacher-Student JSONL
                    vision_process.smart_resize         + Rescaled images
                    create_teacher_pool.py              + Teacher pool
                    split_train_val.py                  + Train/val split
```

All images are pre-processed through `smart_resize()` and stored in `ds_rescaled/` alongside JSONL files.

---
## 1. Raw Annotation Format (Input)

**Source**: Vendor annotation files (one JSON per image)  
**Used Section**: Only `markResult.features` is processed  
**Feature Structure**:
```jsonc
{
  "geometry": {
    "type": "ExtentPolygon",
    "coordinates": [[x1,y1], [x2,y1], [x2,y2], [x1,y2], [x1,y1]]
  },
  "properties": {
    "content": { "label": "connection_point/install_screw/install_screw_true" },
    "contentZh": { "标签贴纸": "螺丝连结点/BBU安装螺丝/连接正确" }
  }
}
```

**Key Processing Rules**:
1. **Polygon → BBox**: Extract axis-aligned bounding rectangle from 5-point polygon
2. **Language Priority**: `contentZh` (Chinese) preferred over `content.label` (English)
3. **Coordinate Space**: Original image resolution, later scaled during conversion

---
## 2. Conversion Pipeline (data_conversion/)

### Core Scripts:
- **`convert_dataset.sh`**: Main orchestrator script
- **`convert_pure_json.py`**: Raw JSON → clean JSONL conversion
- **`vision_process.py`**: Image resizing with `smart_resize()`
- **`create_teacher_pool.py`**: Curated teacher sample selection
- **`split_train_val.py`**: Train/validation dataset splitting

### Processing Steps:
1. **Extract & Validate**: Read JSONs, extract image dimensions, validate against JPEG files
2. **Smart Resize**: Resize images to meet `min_pixel`/`max_pixel` constraints (typically 896×1344)
3. **Scale Coordinates**: Transform bounding box coordinates to new image resolution
4. **Label Processing**: Map vendor labels through `label_hierarchy.json` and `token_map.json`
5. **Output Generation**: Create teacher-student JSONL format

**Critical Rules**:
- **No EXIF Rotation**: Never apply `ImageOps.exif_transpose()` 
- **Pre-scaled Only**: All images must be processed through `vision_process.smart_resize()`
- **Fail-Fast**: Invalid bounding boxes or missing images cause immediate errors

---
## 3. Current Data Schema (Teacher-Student Format)

### JSONL Structure:
```jsonc
{
  "teachers": [
    {
      "images": ["ds_rescaled/img001.jpeg"],
      "objects": [
        {
          "bbox_2d": [x1, y1, x2, y2],
          "desc": "螺丝连接点/BBU安装螺丝/连接正确"
        }
      ]
    }
  ],
  "student": {
    "images": ["ds_rescaled/img002.jpeg"],
    "objects": [
      {
        "bbox_2d": [x1, y1, x2, y2],
        "desc": "螺丝连接点/BBU安装螺丝/连接正确"
      }
    ]
  }
}
```

### Schema Components:

**ImageSample** (both teachers and student):
- `images`: List of pre-scaled JPEG paths (≥1 image)
- `objects`: List of GroundTruthObject instances

**GroundTruthObject**:
- `bbox_2d`: Tensor `[x1, y1, x2, y2]` in absolute pixels (rescaled image space)
- `desc`: Natural language description (simplified from old hierarchical format)

### Key Changes from Old Format:
- **Teacher-Student Structure**: Explicit separation for differential learning
- **Natural Language Descriptions**: Simplified from `object_type/property/extra_info` format
- **Tensor BBoxes**: `bbox_2d` as torch.Tensor, not `box` as list
- **Multi-image Support**: Each sample can contain multiple images

---
## 4. Data Loading & Validation (src/data.py)

### BBUDataset Class:
```python
class BBUDataset(Dataset):
    def __init__(self, jsonl_path: str, ...)
    def __getitem__(self, idx) -> MultiChatSample
```

**Validation Rules** (Fail-Fast):
1. **Path Existence**: All image paths under `ds_rescaled/` must exist
2. **BBox Bounds**: `0 ≤ x1 < x2 ≤ width` and `0 ≤ y1 < y2 ≤ height`
3. **Description Format**: Non-empty string descriptions
4. **Schema Compliance**: Validated against dataclass definitions in `src/schema.py`

### Collator Types:

**StandardDataCollator**:
- **Padding**: Sequences padded to max length in batch
- **Shape**: `(B, L_max)` where B = batch size
- **Use Case**: Default, clear sample-to-tensor mapping

**PackedDataCollator**:
- **Concatenation**: All sequences concatenated into single row
- **Shape**: `(1, ΣL_i)` where ΣL_i = sum of all sequence lengths
- **Efficiency**: Zero padding waste, ideal for variable-length data
- **Requirement**: Usually `per_device_train_batch_size: 1`

---
## 5. Teacher Pool Management

### Teacher Pool Creation:
- **Script**: `create_teacher_pool.py`  
- **Selection Strategy**: Greedy + random sampling for diversity
- **Criteria**:
  1. **Semantic Coverage**: Every label represented
  2. **Scene Complexity**: Mix of sparse/medium/dense scenes
  3. **Spatial Distribution**: 3×3 grid coverage of bbox centers
  4. **Size Diversity**: Small/medium/large object representation

### Runtime Usage:
- **TeacherPoolManager** (in `src/data.py`) loads pre-selected pool
- **Sampling**: Controlled by `teacher_ratio` config parameter
- **Training Mix**: ~70% teacher-student, ~30% single-shot (validation format)

---
## 6. Data Conversion Commands

### Full Pipeline:
```bash
# Convert entire dataset
bash data_conversion/convert_dataset.sh \
    --input_folder raw_vendor_jsons/ \
    --output_root data/

# Validate processed data
python data_conversion/simple_validate.py \
    --jsonl_path data/chinese-train.jsonl \
    --image_root ds_rescaled/
```

### Individual Steps:
```bash
# Raw JSON conversion
python data_conversion/convert_pure_json.py \
    --input_folder raw_jsons/ \
    --output_jsonl data/raw_converted.jsonl

# Teacher pool creation
python data_conversion/create_teacher_pool.py \
    --input_jsonl data/raw_converted.jsonl \
    --output_jsonl data/teacher_pool.jsonl

# Train/val split
python data_conversion/split_train_val.py \
    --input_jsonl data/raw_converted.jsonl \
    --train_output data/chinese-train.jsonl \
    --val_output data/chinese-val.jsonl
```

---
## 7. Schema Validation System

### Runtime Validation (src/schema.py):
- **TensorType Annotations**: Shape validation via torchtyping
- **Dataclass Assertions**: Automatic `__post_init__` checks
- **Fail-Fast Philosophy**: Invalid data causes immediate errors

### Key Validation Points:
1. **ChatProcessorOutput**: Token sequence consistency, vision token alignment
2. **CollatedBatch**: Batch dimension consistency, pixel values validation
3. **GroundTruthObject**: BBox format and bounds checking
4. **ModelInputs**: Input tensor compatibility validation

---
## 8. Migration Notes (Old → New)

### Breaking Changes:
1. **Schema**: `box` → `bbox_2d`, `examples`/`target` → `teachers`/`student`
2. **Descriptions**: Hierarchical format → natural language
3. **Validation**: Added comprehensive tensor shape checking
4. **Teacher System**: Explicit teacher-student structure vs implicit few-shot

### Compatibility:
- **Backward**: Old data must be re-converted
- **Forward**: New schema supports both teacher-student and single-shot modes

---
## 9. Current File Structure

```
data/
├── chinese-train.jsonl          # Training data (teacher-student format)
├── chinese-val.jsonl            # Validation data (single-shot format)
├── teacher_pool.jsonl           # Curated teacher demonstrations
└── candidate_phrases.json       # Label vocabulary

ds_rescaled/                     # Pre-processed images
├── img001.jpeg                  # Resized, no EXIF rotation
├── img002.jpeg
└── ...

data_conversion/
├── convert_dataset.sh           # Main conversion orchestrator
├── convert_pure_json.py         # Raw JSON → JSONL converter
├── vision_process.py            # Image resizing utilities
├── create_teacher_pool.py       # Teacher selection algorithm
├── split_train_val.py           # Dataset splitting
├── simple_validate.py           # Data validation script
├── label_hierarchy.json         # Label mapping hierarchy
└── token_map*.json              # Token mappings
```

---
**The data pipeline is now fully automated and validated, with comprehensive error checking at every stage.** 🚀 