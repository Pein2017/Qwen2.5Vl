# Qwen2.5-VL Data Conversion Pipeline - Comprehensive Guide

> **Advanced Multi-Geometry Architecture – July 2025**
>
> This pipeline features a sophisticated multi-geometry processing system with flexible
> taxonomy-based classification. It handles complex V2 annotations with bbox_2d, square,
> and line geometries while maintaining full backward compatibility.

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Complete Pipeline Flow](#complete-pipeline-flow)
4. [Configuration System](#configuration-system)
5. [Architecture & Components](#architecture--components)
6. [Output Format & Structure](#output-format--structure)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The advanced pipeline converts **V2 JSON annotations + images** into training-ready format with comprehensive geometry support:

```
ds_v2/ (V2 JSON/images) → output/ds_v2/ (train.jsonl, val.jsonl, teacher.jsonl + processed images)
```

### 🎯 Key Features

* **Multi-Geometry Support** – bbox_2d, square (四边形), line (LineString) coordinates
* **Flexible Taxonomy System** – Attribute-based classification using `attribute_taxonomy.json`
* **Advanced Coordinate Processing** – EXIF orientation, dimension rescaling, smart resize
* **Comprehensive V2 Coverage** – Handles markResult features and dataList formats
* **Teacher Sample Selection** – Intelligent diversity-based teacher pool creation
* **Smart Image Processing** – Automatic resizing with MAX_PIXELS constraints

### ⚡ Core Components

1. **UnifiedProcessor** – Main orchestrator with integrated sample extraction
2. **CoordinateManager** – Unified geometry transformation pipeline
3. **FlexibleTaxonomyProcessor** – Attribute-based hierarchical classification
4. **TeacherSelector** – Diversity-based teacher sample selection
5. **ImageProcessor** – Smart image processing with EXIF handling

---

## Quick Start Guide

### 1. Configure the Pipeline

Edit `/data3/Qwen2.5-VL-main/data_conversion/convert_dataset.sh`:

```bash
# Essential Configuration - EDIT THESE VALUES
INPUT_DIR="ds_v2"                    # Your V2 data directory
OUTPUT_DIR="output"                  # Base output directory (creates output/ds_v2/)
DATASET_NAME="ds_v2"                 # Dataset identifier
LANGUAGE="chinese"                   # "chinese" or "english"
RESPONSE_TYPES="object_type property extra_info"  # Description components
VAL_RATIO="0.1"                     # 10% validation split
MAX_TEACHERS="10"                   # Teacher samples for few-shot learning
RESIZE="true"                       # Enable smart image resizing (MAX_PIXELS=512*28*28)
```

### 2. Run the Pipeline

```bash
cd /data3/Qwen2.5-VL-main/data_conversion
./convert_dataset.sh
```

### 3. Check Results

```bash
ls output/ds_v2/
# Expected output:
# train.jsonl           - Training samples (187 samples)
# val.jsonl             - Validation samples (~21 samples)
# teacher.jsonl         - Teacher samples for few-shot learning (10 samples)
# all_samples.jsonl     - Combined samples (209 total)
# label_vocabulary.json - Complete label statistics (241 unique labels)
# images/               - Processed images (if RESIZE=true)
```

---

## Complete Pipeline Flow

### Entry Point: `convert_dataset.sh`

The shell script orchestrates the entire pipeline:

```bash
convert_dataset.sh
├── Environment setup (UTF-8 locale, Python paths)
├── Configuration validation (INPUT_DIR existence)
├── Auto-detect DATASET_NAME if not provided
└── Executes: /root/miniconda3/envs/ms/bin/python data_conversion/processor.py
    ├── Creates: DataConversionConfig from arguments
    ├── Initializes: UnifiedProcessor with config
    └── Runs: complete processing pipeline
```

### Detailed Pipeline Execution Flow

```
🚀 COMPLETE PIPELINE EXECUTION FLOW

1. 📋 INITIALIZATION & CONFIGURATION
   ├── Load DataConversionConfig from command line arguments
   ├── Setup logging (INFO/DEBUG/WARNING/ERROR levels)
   ├── Initialize UnifiedProcessor with:
   │   ├── TokenMapper (if token_map_path provided)
   │   ├── Label hierarchy (from hierarchy_path or default)
   │   ├── HierarchicalProcessor (compatibility layer)
   │   ├── ImageProcessor (with smart resize settings)
   │   ├── TeacherSelector (diversity-based selection)
   │   └── DataSplitter (train/val splitting)

2. 📁 SAMPLE PROCESSING (process_all_samples)
   ├── Find all JSON files in INPUT_DIR using FileOperations.find_json_files()
   ├── For each JSON file (process_single_sample):
   │   ├── Load JSON data with structure validation
   │   ├── Find corresponding image file (.jpeg/.jpg)
   │   ├── Extract image dimensions (with EXIF handling)
   │   ├── Process annotation data:
   │   │   ├── dataList format → extract_objects_from_datalist()
   │   │   └── markResult format → extract_objects_from_markresult()
   │   │       └── Uses HierarchicalProcessor → FlexibleTaxonomyProcessor
   │   │           ├── Process V2 features with attribute taxonomy
   │   │           ├── Handle geometry: bbox_2d, square, line
   │   │           ├── Extract hierarchical attributes by groups
   │   │           └── Generate descriptions: "object_type/property/extra_info"
   │   ├── Apply unified coordinate transformation pipeline:
   │   │   ├── EXIF orientation compensation
   │   │   ├── Dimension rescaling (JSON vs actual image)
   │   │   └── Smart resize scaling (MAX_PIXELS=512*28*28)
   │   ├── Process images (ImageProcessor):
   │   │   ├── Copy/resize images to output/ds_v2/images/
   │   │   └── Apply EXIF orientation and smart resize
   │   ├── Sort objects by position (top-left to bottom-right)
   │   └── Create training sample with relative image paths
   └── Returns: List[processed_samples] with validation

3. 🎯 DATASET SPLITTING (split_into_sets)
   ├── TeacherSelector.select_teachers():
   │   ├── Analyze geometry diversity (bbox_2d/square/line)
   │   ├── Ensure label coverage across all object types
   │   ├── Consider spatial distribution and object density
   │   └── Select up to MAX_TEACHERS diverse samples
   ├── Remove teacher samples from student pool
   └── DataSplitter.split():
       ├── Shuffle remaining samples with fixed seed
       └── Split into train/val with VAL_RATIO

4. 💾 OUTPUT GENERATION (write_outputs)
   ├── Write JSONL files:
   │   ├── train.jsonl (training samples)
   │   ├── val.jsonl (validation samples)
   │   ├── teacher.jsonl (teacher samples)
   │   └── all_samples.jsonl (combined samples)
   ├── Generate label_vocabulary.json:
   │   ├── Extract all unique labels from descriptions
   │   ├── Categorize: object_types, properties, full_descriptions
   │   ├── Generate statistics and metadata
   │   └── Include usage notes for training
   └── Final validation and cleanup
```

### Sample Output Format

Training samples support multiple geometry formats:

```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "螺丝、光纤插头/显示完整/BBU安装螺丝/符合要求",
      "geometry": {
        "coordinates": [[263.67, 143.83], [326.26, 143.83], [326.26, 200.71], [263.67, 200.71], [263.67, 143.83]],
        "type": "ExtentPolygon"
      }
    },
    {
      "bbox_2d": [248, 184, 367, 244],
      "desc": "标签/4G-RRU3-光纤",
      "square": [704, 487, 670, 554, 973, 644, 993, 590],
      "geometry": {
        "lineType": ["LLLLL"],
        "coordinates": [[[260.02, 184.49], [247.71, 210.19], [359.36, 244.32], [366.75, 223.67], [260.02, 184.49]]],
        "type": "Square"
      }
    },
    {
      "bbox_2d": [2, 568, 288, 828],
      "desc": "电线/有遮挡，捆扎整齐",
      "line": [614, 1271, 498, 1179, 419, 1216, 280, 1280, 117, 1456, 3, 1721],
      "geometry": {
        "lineMode": 1,
        "lineType": "LLLLLL",
        "coordinates": [[288.22, 611.86], [233.92, 567.51], [196.72, 585.04], [131.35, 615.98], [54.93, 700.54], [1.63, 828.42]],
        "type": "LineString"
      }
    }
  ],
  "width": 532,
  "height": 728
}
```

---

## Configuration System

### 🔧 Essential Configuration (Required)

| Variable | Example | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `"ds_v2"` | Directory containing V2 JSON/image files |
| `OUTPUT_DIR` | `"output"` | Base output directory (creates output/dataset_name/) |
| `LANGUAGE` | `"chinese"` | Language mode: `chinese` or `english` |
| `RESPONSE_TYPES` | `"object_type property extra_info"` | Description components to include |
| `VAL_RATIO` | `"0.1"` | Validation split ratio (10% = 0.1) |
| `MAX_TEACHERS` | `"10"` | Maximum teacher samples for few-shot learning |
| `RESIZE` | `"true"` | Enable smart image resizing (MAX_PIXELS=512*28*28) |

### ⚙️ Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_NAME` | Auto-detect from INPUT_DIR | Dataset identifier for output folder |
| `HIERARCHY_FILE` | Built-in default | Custom label hierarchy file path |
| `TOKEN_MAP_PATH` | None | Token mapping file (required for English mode) |
| `LOG_LEVEL` | `"INFO"` | Logging verbosity: DEBUG/INFO/WARNING/ERROR |
| `SEED` | `"17"` | Random seed for reproducible splits |

### 🎛️ Advanced Processing Options

Advanced options are configured in the Python code:

```python
# In DataConversionConfig class:
geometry_diversity_weight: float = 4.0    # Weight for geometry diversity in teacher selection
fail_fast: bool = True                     # Stop on first error vs continue processing

# In vision_process.py constants:
IMAGE_FACTOR = 28                          # Image dimension factor for smart resize
MIN_PIXELS = 4 * 28 * 28                  # Minimum image pixels (3,136)
MAX_PIXELS = 512 * 28 * 28                # Maximum image pixels (401,408)
MAX_RATIO = 200                           # Maximum aspect ratio allowed
```

---

## Architecture & Components

### Advanced Multi-Geometry Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UnifiedProcessor                                  │
│  (Main orchestrator with integrated sample extraction)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ • DataConversionConfig management                                           │
│ • Content field extraction (Chinese/English with token mapping)            │
│ • Object filtering via flexible label hierarchy                            │
│ • Sample processing orchestration                                          │
│ • Output generation & comprehensive statistics                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌─────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
│ CoordinateManager│    │  FlexibleTaxonomy      │    │   ImageProcessor    │
│ (Multi-Geometry) │    │  Processor             │    │  (Smart Processing) │
├─────────────────┤    ├─────────────────────────┤    ├─────────────────────┤
│ • EXIF transform│    │ • Attribute taxonomy    │    │ • EXIF orientation  │
│ • Dimension scale│    │ • V2 feature processing│    │ • Smart resize      │
│ • Smart resize   │    │ • Multi-geometry support│    │ • Path management   │
│ • bbox/square/line│    │ • Hierarchical desc.   │    │ • RGB conversion    │
│ • Validation     │    │ • Content extraction    │    │ • Output copying    │
└─────────────────┘    └─────────────────────────┘    └─────────────────────┘
        │                              │                              │
┌─────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
│ TeacherSelector │    │ HierarchicalProcessor  │    │   DataSplitter      │
│ (Diversity)     │    │ (Compatibility Layer)  │    │  (Train/Val Split)  │
├─────────────────┤    ├─────────────────────────┤    ├─────────────────────┤
│ • Geometry div. │    │ • Backward compatibility│    │ • Reproducible      │
│ • Label coverage│    │ • V2 format bridge      │    │ • Configurable ratio│
│ • Spatial dist. │    │ • Legacy support        │    │ • Validation        │
│ • Density analysis│   │ • Format conversion     │    │ • Shuffling         │
└─────────────────┘    └─────────────────────────┘    └─────────────────────┘
```

### Key Architecture Features

1. **Multi-Geometry Support**: Native handling of bbox_2d, square (四边形), line (LineString)
2. **Flexible Taxonomy System**: Attribute-based classification using `attribute_taxonomy.json`
3. **Advanced Coordinate Processing**: Complete transformation pipeline with EXIF, scaling, resize
4. **Intelligent Teacher Selection**: Diversity-based selection considering geometry and spatial distribution
5. **Comprehensive Output**: Detailed statistics, label vocabulary, and training-ready formats

### Flexible Taxonomy System

The system uses comprehensive attribute-based classification from `attribute_taxonomy.json`:

```json
{
  "object_types": {
    "bbu": {
      "chinese_label": "BBU设备",
      "geometry_types": ["bbox_2d", "square"],
      "content_key": "bbu"
    },
    "fiber": {
      "chinese_label": "光纤",
      "geometry_types": ["line"],
      "content_key": "fiber"
    }
  },
  "attribute_groups": {
    "physical_properties": {
      "attributes": {
        "visibility_completeness": {
          "chinese_questions": ["这个BBU设备是否显示完整"],
          "content_mapping": {"bbu": "bbu_stituation"},
          "values": {
            "显示完整": ["bbu_stituation_complete"],
            "只显示部分": ["bbu_stituation_part"]
          }
        },
        "brand_identification": {
          "chinese_questions": ["这个BBU设备是什么品牌"],
          "content_mapping": {"bbu": "bbu_brand"},
          "values": {"华为": "huawei", "中兴": "zhongxing"}
        }
      }
    }
  }
}
```

---

## Output Format & Structure

### Complete Output Structure

The pipeline generates a comprehensive output structure:

```
output/ds_v2/
├── train.jsonl              # Training samples (187 samples)
├── val.jsonl                # Validation samples (~21 samples)
├── teacher.jsonl            # Teacher samples (10 samples)
├── all_samples.jsonl        # Combined samples (209 total)
├── label_vocabulary.json    # Comprehensive label statistics
└── images/                  # Processed images (if RESIZE=true)
    ├── QC-20230217-0000279_19621.jpeg
    ├── QC-20230323-0001285_216052.jpeg
    └── ... (all processed images)
```

### Label Vocabulary Structure

The `label_vocabulary.json` provides comprehensive statistics:

```json
{
  "metadata": {
    "total_samples": 209,
    "total_objects": 2102,
    "language": "chinese",
    "extraction_date": "2025-07-21T12:57:12.745934"
  },
  "statistics": {
    "unique_labels_count": 241,
    "object_types_count": 6,
    "properties_count": 188,
    "full_descriptions_count": 249
  },
  "vocabulary": {
    "all_unique_labels": ["4G-BBU-接地线", "BBU设备", "华为", "显示完整", ...],
    "object_types": ["BBU设备", "光纤", "电线", "标签", "螺丝、光纤插头", "挡风板"],
    "properties": ["华为", "显示完整", "只显示部分", "符合要求", ...],
    "full_descriptions": ["BBU设备/华为，显示完整/机柜空间充足，需要安装", ...]
  },
  "usage_notes": {
    "training_prompts": "Use 'all_unique_labels' for comprehensive label-aware training",
    "object_detection": "Use 'object_types' for class-specific detection tasks"
  }
}
```

---

## Advanced Features

### Custom Processing Pipeline

You can use the UnifiedProcessor directly in Python:

```python
from data_conversion.unified_processor import UnifiedProcessor
from data_conversion.config import DataConversionConfig

# Create configuration
config = DataConversionConfig(
    input_dir="ds_v2",
    output_dir="output",
    language="chinese",
    response_types=["object_type", "property", "extra_info"],
    resize=True,
    val_ratio=0.1,
    max_teachers=10,
    seed=42
)

# Run pipeline
processor = UnifiedProcessor(config)
results = processor.process()

print(f"Processed {results['total_processed']} samples")
print(f"Train: {results['train']}, Val: {results['val']}, Teachers: {results['teacher']}")
```

### Custom Attribute Taxonomy

Create your own attribute taxonomy file:

```json
{
  "object_types": {
    "custom_equipment": {
      "chinese_label": "自定义设备",
      "geometry_types": ["bbox_2d"],
      "content_key": "custom"
    }
  },
  "attribute_groups": {
    "custom_properties": {
      "attributes": {
        "custom_attribute": {
          "chinese_questions": ["自定义问题"],
          "content_mapping": {"custom_equipment": "custom_field"},
          "values": {"值1": "value1", "值2": "value2"}
        }
      }
    }
  }
}
```

Then use it:
```bash
HIERARCHY_FILE="data_conversion/custom_taxonomy.json"
```

### Batch Processing Multiple Datasets

```bash
# Process multiple datasets with same configuration
for dataset in ds_v2 ds_v3 ds_experimental; do
    INPUT_DIR="$dataset"
    DATASET_NAME="$dataset"
    OUTPUT_DIR="output"
    ./convert_dataset.sh
done
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Configuration Errors
```
ValueError: Unsupported language: english
```
**Solution**: Ensure `TOKEN_MAP_PATH` is provided for English mode. Chinese mode works without token mapping.

#### 2. Geometry Processing Issues
```
Dimension mismatch: JSON says 1920x1080 but image is 1080x1920
```
**Solution**: This is normal for EXIF-rotated images. The CoordinateManager handles this automatically with proper coordinate transformation.

#### 3. Empty Output
```
No valid samples were processed
```
**Solution**: Check that:
- Input directory contains valid JSON files with `.json` extension
- JSON files have corresponding `.jpeg` or `.jpg` images
- JSON structure contains either `dataList` or `markResult.features`
- Objects pass the label hierarchy filtering

#### 4. Memory Issues
```
Out of memory during processing
```
**Solution**:
- Set `RESIZE="false"` to disable image processing
- Process smaller batches
- Set `LOG_LEVEL="WARNING"` to reduce memory usage

### Performance Optimization

1. **Large Datasets**: Set `LOG_LEVEL="WARNING"` to reduce log output
2. **Memory Usage**: Disable resizing for datasets with pre-sized images
3. **Speed**: Use SSD storage for input/output directories
4. **Parallel Processing**: Process multiple datasets in parallel

### Debug Mode

Enable detailed debugging:
```bash
LOG_LEVEL="DEBUG"
./convert_dataset.sh
```

This shows:
- Detailed coordinate transformations with EXIF handling
- Geometry processing steps for bbox_2d/square/line
- Sample filtering decisions based on taxonomy
- File I/O operations and image processing
- Teacher selection diversity analysis

### Validation Commands

Test the pipeline components:

```bash
# Test complete pipeline
cd /data3/Qwen2.5-VL-main/data_conversion
/root/miniconda3/envs/ms/bin/python -c "
from unified_processor import UnifiedProcessor
from config import DataConversionConfig
print('✅ All imports successful')
"

# Test coordinate processing
/root/miniconda3/envs/ms/bin/python -c "
from coordinate_manager import CoordinateManager
print('✅ Coordinate processing ready')
"

# Test taxonomy system
/root/miniconda3/envs/ms/bin/python -c "
from flexible_taxonomy_processor import FlexibleTaxonomyProcessor
processor = FlexibleTaxonomyProcessor()
print('✅ Taxonomy system loaded')
"
```

The advanced multi-geometry system provides comprehensive V2 data processing with flexible taxonomy-based classification and intelligent coordinate transformations.