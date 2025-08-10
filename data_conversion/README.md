# Qwen2.5-VL Data Conversion Pipeline

> **Unified Processing Architecture – August 2025**
>
> Streamlined data conversion pipeline using `unified_processor.py` for BBU equipment annotation processing. 
> Features coordinate transformation management, hierarchical Chinese descriptions, and flexible object type 
> filtering for progressive multi-task learning.

---

## Table of Contents
1. [Overview](#overview)
2. [Object-Oriented Training](#object-oriented-training)
3. [Quick Start Guide](#quick-start-guide)
4. [Hierarchical Description System](#hierarchical-description-system)
5. [Configuration System](#configuration-system)
6. [Advanced Training Combinations](#advanced-training-combinations)
7. [Output Format & Structure](#output-format--structure)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The unified pipeline converts **V2 JSON annotations + images** into training-ready format via `unified_processor.py`:

```
ds_v2/ (V2 JSON/images) → unified_processor.py → data/{dataset_name}/ (train.jsonl, val.jsonl, teacher.jsonl + processed images)
```

**Architecture Components:**
- **`unified_processor.py`** - Main orchestrator with integrated sample processing
- **`coordinate_manager.py`** - EXIF orientation, rescaling, smart resize transformations  
- **`flexible_taxonomy_processor.py`** - V2 annotation processing with hierarchical descriptions
- **`validation_manager.py`** - Comprehensive validation with detailed error reporting
- **`config.py`** - Type-safe configuration management

### 🎯 Key Features

* **Object-Oriented Training** – Flexible combinations of 6 equipment types for progressive learning
* **Hierarchical Descriptions** – Precise comma/slash separated attribute formatting
* **Geometry Constraints** – Line objects (fiber/wire) vs Square/BBox objects (equipment/labels)
* **Multi-Task Learning** – Train individual object types, then combine for joint training
* **Chinese BBU Optimization** – Streamlined for telecommunications equipment detection
* **Exact Key Matching** – Uses precise Chinese question keys for attribute extraction

### 📊 Supported Object Types

| Object Type | Chinese Label | Geometry | Sample Count | Training Focus |
|-------------|---------------|----------|--------------|----------------|
| `bbu` | BBU设备 | square/bbox | ~187 | Equipment detection, brand recognition |
| `bbu_shield` | 挡风板 | square/bbox | ~62 | Windshield detection, installation assessment |
| `connect_point` | 螺丝、光纤插头 | square/bbox | ~769 | Connection hardware, compliance checking |
| `label` | 标签 | square/bbox | ~608 | Text recognition, content extraction |
| `fiber` | 光纤 | line | ~272 | Fiber cable routing, protection assessment |
| `wire` | 电线 | line | ~204 | Wire management, organization checking |

---

## Object-Oriented Training

### Training Strategy

The system supports **progressive multi-task learning** through flexible object type combinations:

#### 1. **Individual Object Training** (Foundation)
```bash
# Train BBU equipment detection only
OBJECT_TYPES="bbu bbu_shield"     # ~249 samples - Equipment focus

# Train connection infrastructure only  
OBJECT_TYPES="connect_point"      # ~769 samples - Hardware focus

# Train text recognition only
OBJECT_TYPES="label"              # ~608 samples - OCR focus

# Train cable systems only
OBJECT_TYPES="fiber wire"         # ~476 samples - Line geometry focus
```

#### 2. **Combined Object Training** (Integration)
```bash
# Train equipment + text recognition
OBJECT_TYPES="bbu bbu_shield label"

# Train all hardware components
OBJECT_TYPES="bbu bbu_shield connect_point"

# Train all cable systems
OBJECT_TYPES="fiber wire"

# Train complete system (no filtering)
OBJECT_TYPES="full"               # All ~2102 objects
```

#### 3. **Progressive Learning Pipeline**
```bash
# Step 1: Train individual components
OBJECT_TYPES="bbu"           → model_bbu.pth
OBJECT_TYPES="connect_point" → model_connections.pth  
OBJECT_TYPES="fiber"         → model_cables.pth
OBJECT_TYPES="label"         → model_text.pth

# Step 2: Train combined systems
OBJECT_TYPES="bbu connect_point" → model_equipment.pth
OBJECT_TYPES="fiber wire"        → model_cables.pth

# Step 3: Joint training
OBJECT_TYPES="full"              → model_complete.pth
```

### Geometry-Aware Training

Objects are automatically grouped by geometry type for specialized training:

- **Line Objects** (`fiber`, `wire`): Complex multi-point coordinate prediction
- **Square/BBox Objects** (`bbu`, `bbu_shield`, `connect_point`, `label`): Rectangular boundary detection

---

## Quick Start Guide

### 1. Configure Object-Oriented Training

Edit `/data3/Qwen2.5-VL-main/data_conversion/convert_dataset.sh`:

```bash
# Essential Configuration - EDIT THESE VALUES
INPUT_DIR="ds_v2"                    # Your V2 data directory
OUTPUT_DIR="data"                    # Base output directory
DATASET_NAME="ds_v2_bbu"            # Dataset identifier
OBJECT_TYPES="bbu bbu_shield"       # Object types to train (or "full" for all)
VAL_RATIO="0.1"                     # 10% validation split
MAX_TEACHERS="10"                   # Teacher samples for few-shot learning
RESIZE="true"                       # Enable smart image resizing
SEED="17"                          # Reproducible random seed
```

### 2. Run Object-Oriented Processing

```bash
cd /data3/Qwen2.5-VL-main
./data_conversion/convert_dataset.sh
```

### 3. Verify Object-Oriented Output

```bash
ls data/ds_v2_bbu/
# Expected output:
# train.jsonl           - Training samples (filtered by object types)
# val.jsonl             - Validation samples
# teacher.jsonl         - Teacher samples
# all_samples.jsonl     - Combined samples
# label_vocabulary.json - Object type statistics
# images/               - Processed images (if RESIZE=true)
```

**Success Indicators:**
```bash
✅ Dataset ds_v2_bbu processed successfully!
📁 Output: data/ds_v2_bbu/
🚀 Ready for training!

Object-Oriented Results:
- Training: 138 samples → train.jsonl
- Validation: 15 samples → val.jsonl  
- Teacher: 5 samples → teacher.jsonl
- Total: 158 BBU samples processed (filtered from 2102 total objects)
- Object Types: 2 (bbu, bbu_shield)
```

---

## Hierarchical Description System

### Description Format Rules

The system generates **exact hierarchical descriptions** using precise separator logic:

- **`,` (comma)** → Separates attributes at the **same hierarchical level**
- **`/` (slash)** → Separates **different attribute levels** or conditional sub-attributes

### Example Hierarchical Descriptions

#### **BBU Equipment:**
```
BBU设备/华为,显示完整,无需安装
BBU设备/中兴,只显示部分,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板
```
- Level 0: `BBU设备` (object type)
- Level 1: `华为,显示完整,无需安装` (basic attributes, comma-separated)
- Level 2: `这个BBU设备按要求配备了挡风板` (conditional sub-attribute)

#### **Connection Hardware:**
```
螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求
螺丝、光纤插头/机柜处接地螺丝,只显示部分,不符合要求/未拧紧,露铜
```
- Level 1: `BBU安装螺丝,显示完整,符合要求` (type, completeness, compliance)
- Level 2: `未拧紧,露铜` (specific issues when non-compliant)

#### **Fiber Cables:**
```
光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管
光纤/无遮挡,无保护措施,弯曲半径不合理（弯曲半径<4cm或者成环）
```
- Level 1: `有遮挡,有保护措施,弯曲半径合理` (obstruction, protection, radius)
- Level 2: `蛇形管` (protection details when protected)

#### **Text Labels:**
```
标签/5G-BBU-接地线
标签/NR900-RRU1-光纤
标签/不能
```

### Attribute Hierarchy Mapping

The system uses **exact Chinese question keys** for precise attribute extraction:

| Object | Level 1 Attributes | Level 2 Conditionals |
|--------|-------------------|---------------------|
| **BBU设备** | 品牌,显示完整性,挡风板需求 | 挡风板配备符合性,特殊情况 |
| **挡风板** | 品牌,显示完整性,遮挡情况,安装方向 | 特殊情况 |
| **螺丝、光纤插头** | 种类,显示完整性,符合要求 | 具体问题,特殊情况 |
| **光纤** | 遮挡情况,保护措施,弯曲半径 | 保护措施详情,特殊情况 |
| **电线** | 遮挡情况,捆扎整齐 | 特殊情况 |
| **标签** | 文字内容 | - |

---

## Configuration System

### 🔧 Object-Oriented Configuration (Required)

| Variable | Example | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `"ds_v2"` | Directory containing V2 JSON/image files |
| `OUTPUT_DIR` | `"data"` | Base output directory |
| `OBJECT_TYPES` | `"bbu label"` or `"full"` | Object types to include (space-separated or "full") |
| `VAL_RATIO` | `"0.1"` | Validation split ratio (10% = 0.1) |
| `MAX_TEACHERS` | `"10"` | Maximum teacher samples for few-shot learning |
| `RESIZE` | `"true"` | Enable smart image resizing |

### ⚙️ Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_NAME` | Auto-detect from INPUT_DIR | Dataset identifier for output folder |
| `HIERARCHY_FILE` | Built-in hierarchical mapping | Custom attribute hierarchy file path |
| `LOG_LEVEL` | `"INFO"` | Logging verbosity: DEBUG/INFO/WARNING/ERROR |
| `SEED` | `"17"` | Random seed for reproducible splits |

### 🎯 Object Type Combinations

```bash
# Individual object types
OBJECT_TYPES="bbu"                    # BBU equipment only
OBJECT_TYPES="connect_point"          # Connection hardware only
OBJECT_TYPES="fiber"                  # Fiber cables only
OBJECT_TYPES="label"                  # Text labels only

# Combined training
OBJECT_TYPES="bbu bbu_shield"         # All equipment
OBJECT_TYPES="fiber wire"             # All cables
OBJECT_TYPES="connect_point label"    # Hardware + text

# Complete training
OBJECT_TYPES="full"                   # All object types (no filtering)
```

---

## Advanced Training Combinations

### Progressive Training Examples

#### **Equipment Detection Pipeline**
```bash
# Phase 1: Individual equipment training
./convert_dataset.sh  # OBJECT_TYPES="bbu"
./convert_dataset.sh  # OBJECT_TYPES="bbu_shield"

# Phase 2: Combined equipment training  
./convert_dataset.sh  # OBJECT_TYPES="bbu bbu_shield"

# Phase 3: Equipment + infrastructure
./convert_dataset.sh  # OBJECT_TYPES="bbu bbu_shield connect_point"
```

#### **Cable System Pipeline**
```bash
# Phase 1: Individual cable training
./convert_dataset.sh  # OBJECT_TYPES="fiber"
./convert_dataset.sh  # OBJECT_TYPES="wire"

# Phase 2: Combined cable training
./convert_dataset.sh  # OBJECT_TYPES="fiber wire"
```

#### **Complete System Pipeline**
```bash
# Phase 1: Specialized training
./convert_dataset.sh  # OBJECT_TYPES="bbu bbu_shield"      → Equipment model
./convert_dataset.sh  # OBJECT_TYPES="connect_point"       → Hardware model  
./convert_dataset.sh  # OBJECT_TYPES="fiber wire"          → Cable model
./convert_dataset.sh  # OBJECT_TYPES="label"               → Text model

# Phase 2: Joint training
./convert_dataset.sh  # OBJECT_TYPES="full"                → Complete model
```

### Direct Python Usage

```python
from data_conversion.unified_processor import UnifiedProcessor
from data_conversion.config import DataConversionConfig

# Configure unified processor
config = DataConversionConfig(
    input_dir="ds_v2",
    output_dir="data", 
    object_types=["bbu", "label"],     # Specific object types
    resize=True,
    val_ratio=0.1,
    max_teachers=10,
    seed=17
)

# Run unified processing pipeline
processor = UnifiedProcessor(config)
results = processor.process()

print(f"✅ Processed {results['total_processed']} samples")
print(f"📊 Object types: {config.object_types}")
```

**Key Features:**
- **Coordinate Transformation Pipeline**: EXIF orientation → dimension rescaling → smart resize
- **Hierarchical Processing**: V2 annotations with comma/slash separator logic
- **Comprehensive Validation**: Strict validation with detailed error reporting and fix suggestions
- **Teacher Selection**: Automated teacher sample selection for few-shot learning

---

## Output Format & Structure

### Object-Oriented Sample Format

Training samples use **native multi-geometry** with **hierarchical descriptions**:

```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"
    },
    {
      "square": [704, 487, 670, 554, 973, 644, 993, 590],
      "desc": "标签/4G-RRU3-光纤"
    },
    {
      "line": [614, 1271, 498, 1179, 419, 1216, 280, 1280, 117, 1456, 3, 1721],
      "desc": "光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管"
    }
  ],
  "width": 532,
  "height": 728
}
```

### Object-Oriented Output Structure

```
data/ds_v2_bbu/                    # Object-type specific dataset
├── train.jsonl                   # Training samples (filtered by object types)
├── val.jsonl                     # Validation samples
├── teacher.jsonl                 # Teacher samples  
├── all_samples.jsonl             # Combined samples
├── label_vocabulary.json         # Object type statistics
└── images/                       # Processed images (if RESIZE=true)
    ├── QC-20230217-0000279_19621.jpeg
    └── ... (filtered image set)
```

### Object Type Statistics

The `label_vocabulary.json` provides **object-oriented statistics**:

```json
{
  "metadata": {
    "total_samples": 158,
    "total_objects": 249,
    "object_types_included": ["bbu", "bbu_shield"],
    "object_types_filtered": ["connect_point", "label", "fiber", "wire"],
    "language": "chinese"
  },
  "statistics": {
    "unique_labels_count": 15,
    "object_types_count": 2,
    "properties_count": 3,
    "full_descriptions_count": 20
  },
  "object_distribution": {
    "bbu": 187,
    "bbu_shield": 62
  },
  "vocabulary": {
    "object_types": ["BBU设备", "挡风板"],
    "hierarchical_descriptions": [
      "BBU设备/华为,显示完整,无需安装",
      "BBU设备/华为,只显示部分,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板",
      "挡风板/华为,显示完整,挡风板无遮挡,安装方向正确"
    ]
  }
}
```

---

## Troubleshooting

### Object-Oriented Training Issues

#### 1. **No Samples After Object Filtering**
```
✅ Sample processing complete: 0 processed, 209 skipped
```
**Solution**: 
- Check if specified object types exist in your dataset
- Use `OBJECT_TYPES="full"` to process all objects
- Verify object type names: `bbu`, `bbu_shield`, `connect_point`, `label`, `fiber`, `wire`

#### 2. **Geometry Constraint Violations**
```
Geometry constraint violation: fiber with square
```
**Solution**: Normal behavior - fiber/wire objects require line geometry, others use square/bbox

#### 3. **Hierarchical Description Issues**
```
Missing required attribute brand for bbu
```
**Solution**: 
- Check if source data contains the expected Chinese question keys
- Verify attribute mapping in `hierarchical_attribute_mapping.json`
- Use `LOG_LEVEL="DEBUG"` for detailed attribute extraction info

#### 4. **Object Type Configuration Errors**
```
Invalid object type: equipment. Valid types: {bbu, bbu_shield, label, fiber, wire, connect_point}
```
**Solution**: Use exact object type names from the supported list

### Performance Optimization

1. **Large Datasets**: Use specific object types instead of `"full"` for faster processing
2. **Memory**: Use line objects (`fiber wire`) for smaller memory footprint
3. **Speed**: Use square/bbox objects (`bbu label`) for faster coordinate processing
4. **Debug**: Use `LOG_LEVEL="DEBUG"` for hierarchical description debugging

### Validation Commands

Test unified processor functionality:

```bash
# Test unified processor initialization
/root/miniconda3/envs/ms/bin/python -c "
from data_conversion.unified_processor import UnifiedProcessor
from data_conversion.config import DataConversionConfig
config = DataConversionConfig(input_dir='ds_v2', output_dir='data', object_types=['bbu'])
processor = UnifiedProcessor(config)
print('✅ UnifiedProcessor initialized successfully')
"

# Test coordinate transformation
/root/miniconda3/envs/ms/bin/python -c "
from data_conversion.coordinate_manager import CoordinateManager
print('✅ CoordinateManager ready for EXIF/rescaling/resize transformations')
"

# Test validation manager
/root/miniconda3/envs/ms/bin/python -c "
from data_conversion.validation_manager import ValidationManager
vm = ValidationManager('strict')
print('✅ ValidationManager ready with strict validation mode')
"
```

---

## Migration from Previous Versions

### Key Changes in Object-Oriented Version

1. **Object Type Filtering**: Replaced `response_types` with `object_types` for training focus
2. **Hierarchical Descriptions**: Comma/slash separator logic for precise attribute formatting
3. **Geometry Constraints**: Automatic validation of line vs square/bbox objects
4. **Progressive Training**: Support for individual → combined → joint training pipelines

### Backward Compatibility

- **Shell script**: Same interface (`convert_dataset.sh`) with new `OBJECT_TYPES` parameter
- **Output format**: Enhanced JSONL structure with hierarchical descriptions
- **Python API**: Extended `UnifiedProcessor` with object type filtering
- **Configuration**: New object-oriented parameters while maintaining existing options

The object-oriented architecture enables **precise multi-task learning** for BBU equipment detection with hierarchical understanding.