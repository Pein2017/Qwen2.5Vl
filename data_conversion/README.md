# Qwen2.5-VL Data Conversion Pipeline - Streamlined Architecture

> **Refactored & Optimized – July 2025**
>
> This pipeline features a streamlined, Chinese-focused architecture with reduced redundancy
> and improved maintainability. The refactored system eliminates duplicate code while
> maintaining full multi-geometry support for BBU equipment detection training.

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Complete Pipeline Flow](#complete-pipeline-flow)
4. [Configuration System](#configuration-system)
5. [Streamlined Architecture](#streamlined-architecture)
6. [Output Format & Structure](#output-format--structure)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The streamlined pipeline converts **V2 JSON annotations + images** into training-ready format optimized for Chinese BBU equipment detection:

```
ds_v2/ (V2 JSON/images) → data/ds_v2/ (train.jsonl, val.jsonl, teacher.jsonl + processed images)
```

### 🎯 Key Features

* **Chinese-Only Optimization** – Streamlined for BBU equipment detection without multilingual overhead
* **Multi-Geometry Support** – bbox_2d, square (四边形), line (LineString) coordinates  
* **Unified Processing** – Single `UnifiedProcessor` eliminates duplicate code paths
* **Fail-Fast Design** – Immediate error exposure per CLAUDE.local.md guidelines
* **Smart Coordinate Pipeline** – EXIF orientation, dimension rescaling, smart resize
* **Comprehensive Output** – Training splits, teacher samples, label vocabulary

### ⚡ Core Components (Refactored)

1. **UnifiedProcessor** – Single entry point with integrated processing logic
2. **CoordinateManager** – Unified geometry transformation (consolidated from multiple sources)
3. **FileOperations** – Centralized file handling (merged from data_loader)
4. **HierarchicalProcessor** – V2-compatible Chinese processing
5. **ImageProcessor** – Smart image processing with EXIF handling

### 📊 Refactoring Benefits

- **~518 lines eliminated** - Removed redundant `sample_processor.py` and `data_loader.py`
- **Unified architecture** - Single processing pipeline instead of fragmented modules
- **Improved maintainability** - Consolidated coordinate transformations and file operations
- **Better error handling** - Consistent fail-fast approach throughout

---

## Quick Start Guide

### 1. Configure the Pipeline

Edit `/data3/Qwen2.5-VL-main/data_conversion/convert_dataset.sh`:

```bash
# Essential Configuration - EDIT THESE VALUES
INPUT_DIR="ds_v2"                    # Your V2 data directory
OUTPUT_DIR="data"                    # Base output directory (creates data/ds_v2/)
DATASET_NAME="ds_v2"                 # Dataset identifier
RESPONSE_TYPES="object_type property extra_info"  # Description components
VAL_RATIO="0.1"                     # 10% validation split
MAX_TEACHERS="10"                   # Teacher samples for few-shot learning
RESIZE="true"                       # Enable smart image resizing
SEED="17"                          # Reproducible random seed
```

### 2. Run the Pipeline

```bash
cd /data3/Qwen2.5-VL-main
./data_conversion/convert_dataset.sh
```

### 3. Verify Success

```bash
ls data/ds_v2/
# Expected output:
# train.jsonl           - Training samples (180 samples)
# val.jsonl             - Validation samples (19 samples)  
# teacher.jsonl         - Teacher samples (10 samples)
# all_samples.jsonl     - Combined samples (209 total)
# label_vocabulary.json - Complete label statistics (244 unique labels)
# images/               - Processed images (if RESIZE=true)
```

**Success Indicators:**
```bash
✅ Dataset ds_v2 processed successfully!
📁 Output: data/ds_v2/
🚀 Ready for training!

Final Results:
- Training: 180 samples → train.jsonl
- Validation: 19 samples → val.jsonl  
- Teacher: 10 samples → teacher.jsonl
- Total: 209 samples processed
- Labels: 244 unique, 6 object types, 184 properties
```

---

## Complete Pipeline Flow

### Entry Point: `convert_dataset.sh`

The shell script orchestrates the streamlined pipeline:

```bash
convert_dataset.sh
├── Environment setup (UTF-8 locale, Python paths)
├── Configuration validation (INPUT_DIR existence)
├── Auto-detect DATASET_NAME if not provided
└── Executes: /root/miniconda3/envs/ms/bin/python data_conversion/processor.py
    ├── Creates: DataConversionConfig from arguments
    ├── Initializes: UnifiedProcessor (single entry point)
    └── Runs: streamlined processing pipeline
```

### Streamlined Pipeline Execution Flow

```
🚀 STREAMLINED PIPELINE EXECUTION FLOW

1. 📋 INITIALIZATION (Unified Configuration)
   ├── Load DataConversionConfig from command line arguments
   ├── Setup logging with fail-fast error handling
   ├── Initialize UnifiedProcessor with:
   │   ├── Built-in label hierarchy (Chinese BBU equipment)
   │   ├── HierarchicalProcessor (V2 compatibility)
   │   ├── ImageProcessor (smart resize + EXIF)
   │   ├── TeacherSelector (diversity-based selection)  
   │   └── DataSplitter (reproducible train/val splitting)

2. 📁 UNIFIED SAMPLE PROCESSING (Single Code Path)
   ├── Find all JSON files using centralized FileOperations
   ├── For each JSON file (process_single_sample):
   │   ├── Load JSON + find image with validation
   │   ├── Extract V2 annotation data:
   │   │   ├── dataList format → direct bbox extraction
   │   │   └── markResult format → HierarchicalProcessor
   │   │       └── Multi-geometry: bbox_2d, square, line
   │   ├── Unified coordinate transformation pipeline:
   │   │   ├── EXIF orientation compensation
   │   │   ├── Dimension rescaling (JSON vs actual)
   │   │   └── Smart resize (MAX_PIXELS=401,408)
   │   ├── Process images with EXIF handling
   │   ├── Sort objects by position (top→bottom, left→right)
   │   └── Generate training sample with relative paths
   └── Returns: Validated samples list (fail-fast on errors)

3. 🎯 INTELLIGENT DATASET SPLITTING
   ├── TeacherSelector: diversity-based selection
   │   ├── Ensure label coverage across object types
   │   ├── Consider geometry diversity and spatial distribution
   │   └── Select up to MAX_TEACHERS diverse samples
   ├── Remove teachers from student pool (no overlap)
   └── DataSplitter: reproducible train/val split with seed

4. 💾 COMPREHENSIVE OUTPUT GENERATION  
   ├── Write JSONL files (train/val/teacher/all_samples)
   ├── Generate detailed label_vocabulary.json:
   │   ├── Extract all unique labels with statistics
   │   ├── Categorize: object_types, properties, descriptions
   │   └── Include training usage recommendations
   └── Final validation (ensure no overlapping samples)
```

### Sample Output Format

Training samples support native multi-geometry:

```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {
      "bbox_2d": [264, 144, 326, 201],
      "desc": "螺丝、光纤插头/显示完整/BBU安装螺丝"
    },
    {
      "square": [704, 487, 670, 554, 973, 644, 993, 590],
      "desc": "标签/4G-RRU3-光纤"
    },
    {
      "line": [614, 1271, 498, 1179, 419, 1216, 280, 1280, 117, 1456, 3, 1721],
      "desc": "电线/有遮挡，捆扎整齐"
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
| `OUTPUT_DIR` | `"data"` | Base output directory (creates data/dataset_name/) |
| `RESPONSE_TYPES` | `"object_type property extra_info"` | Description components to include |
| `VAL_RATIO` | `"0.1"` | Validation split ratio (10% = 0.1) |
| `MAX_TEACHERS` | `"10"` | Maximum teacher samples for few-shot learning |
| `RESIZE` | `"true"` | Enable smart image resizing |

### ⚙️ Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_NAME` | Auto-detect from INPUT_DIR | Dataset identifier for output folder |
| `HIERARCHY_FILE` | Built-in BBU hierarchy | Custom label hierarchy file path |
| `LOG_LEVEL` | `"INFO"` | Logging verbosity: DEBUG/INFO/WARNING/ERROR |
| `SEED` | `"17"` | Random seed for reproducible splits |

### 🎛️ Advanced Processing Constants

Advanced options are configured in the code:

```python
# Smart resize parameters (in vision_process.py)
IMAGE_FACTOR = 28                          # Image dimension factor
MIN_PIXELS = 4 * 28 * 28                  # Minimum pixels (3,136)
MAX_PIXELS = 512 * 28 * 28                # Maximum pixels (401,408)
MAX_RATIO = 200                           # Maximum aspect ratio

# Processing behavior
fail_fast: bool = True                     # Stop on first error (CLAUDE.local.md)
geometry_diversity_weight: float = 4.0    # Teacher selection diversity weight
```

---

## Streamlined Architecture

### Refactored Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UnifiedProcessor (Single Entry Point)                   │
│                    Consolidated Processing Logic                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Streamlined configuration management                                     │
│ • Chinese-only content extraction (no token mapping overhead)             │  
│ • Integrated object filtering with built-in BBU hierarchy                 │
│ • Unified sample processing (eliminated SampleProcessor redundancy)       │
│ • Comprehensive output generation with statistics                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌─────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
│ CoordinateManager│    │ HierarchicalProcessor  │    │  FileOperations     │
│ (Consolidated)   │    │ (V2 Compatibility)     │    │  (Centralized I/O)  │
├─────────────────┤    ├─────────────────────────┤    ├─────────────────────┤
│ • Unified geometry  │    │ • V2 feature processing │    │ • JSON loading/saving │
│ • EXIF + scaling │    │ • Multi-geometry support │    │ • Image discovery    │
│ • Smart resize   │    │ • Chinese description   │    │ • Path management    │
│ • Validation     │    │ • Content extraction    │    │ • Dimension handling │
└─────────────────┘    └─────────────────────────┘    └─────────────────────┘
        │                              │                              │
┌─────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
│ TeacherSelector │    │   ImageProcessor        │    │   DataSplitter      │
│ (Diversity)     │    │  (Smart Processing)     │    │  (Train/Val Split)  │
├─────────────────┤    ├─────────────────────────┤    ├─────────────────────┤
│ • Label coverage│    │ • EXIF orientation      │    │ • Reproducible      │
│ • Geometry div. │    │ • Smart resize          │    │ • Configurable ratio│
│ • Spatial dist. │    │ • Path management       │    │ • Validation        │
│ • BBU-optimized │    │ • RGB conversion        │    │ • Shuffling         │
└─────────────────┘    └─────────────────────────┘    └─────────────────────┘
```

### Refactoring Improvements

**Before**: Fragmented architecture with duplicate logic
- `SampleProcessor` (393 lines) + `UnifiedProcessor` overlap
- `data_loader.py` (125 lines) + `FileOperations` duplication  
- Multiple coordinate transformation implementations
- Inconsistent error handling approaches

**After**: Streamlined architecture with single responsibility
- ✅ **Single `UnifiedProcessor`** handles all sample processing
- ✅ **Centralized `FileOperations`** for all I/O operations
- ✅ **Unified `CoordinateManager`** for all geometry transformations
- ✅ **Consistent fail-fast** error handling throughout
- ✅ **Chinese-only optimization** removes multilingual complexity

---

## Output Format & Structure

### Complete Output Structure

```
data/ds_v2/
├── train.jsonl              # Training samples (180 samples)
├── val.jsonl                # Validation samples (19 samples)  
├── teacher.jsonl            # Teacher samples (10 samples)
├── all_samples.jsonl        # Combined samples (209 total)
├── label_vocabulary.json    # Comprehensive label statistics
└── images/                  # Processed images (if RESIZE=true)
    ├── QC-20230217-0000279_19621.jpeg
    ├── QC-20230323-0001285_216052.jpeg
    └── ... (209 processed images)
```

### Label Vocabulary Structure

The `label_vocabulary.json` provides comprehensive BBU equipment statistics:

```json
{
  "metadata": {
    "total_samples": 209,
    "total_objects": 2102,
    "language": "chinese", 
    "extraction_date": "2025-07-21T14:32:41.152268"
  },
  "statistics": {
    "unique_labels_count": 244,
    "object_types_count": 6,
    "properties_count": 184,
    "full_descriptions_count": 255
  },
  "vocabulary": {
    "object_types": ["螺丝、光纤插头", "标签", "BBU设备", "光纤", "电线", "挡风板"],
    "all_unique_labels": ["4G-BBU-接地线", "4G-BBU-电源线1", "华为", "显示完整", ...],
    "properties": ["华为", "显示完整", "只显示部分", "符合要求", "有遮挡", ...],
    "full_descriptions": ["螺丝、光纤插头/显示完整/BBU安装螺丝", ...]
  },
  "usage_notes": {
    "training_prompts": "Use 'all_unique_labels' for comprehensive label-aware training",
    "object_detection": "Use 'object_types' for class-specific detection tasks"
  }
}
```

---

## Advanced Features

### Direct Python Usage

Use the streamlined processor directly:

```python
from data_conversion.unified_processor import UnifiedProcessor
from data_conversion.config import DataConversionConfig

# Create configuration
config = DataConversionConfig(
    input_dir="ds_v2",
    output_dir="data", 
    response_types=["object_type", "property", "extra_info"],
    resize=True,
    val_ratio=0.1,
    max_teachers=10,
    seed=17
)

# Run streamlined pipeline
processor = UnifiedProcessor(config)
results = processor.process()

print(f"✅ Processed {results['total_processed']} samples")
print(f"📊 Train: {results['train']}, Val: {results['val']}, Teachers: {results['teacher']}")
```

### Custom Label Hierarchy

Override the built-in BBU hierarchy:

```json
{
  "螺丝、光纤插头": ["BBU安装螺丝", "BBU端光纤插头"],
  "标签": [],
  "BBU设备": ["华为", "中兴"],
  "光纤": [],
  "电线": [],
  "挡风板": ["华为"]
}
```

Then use it:
```bash
HIERARCHY_FILE="data_conversion/custom_hierarchy.json"
```

### Batch Processing Multiple Datasets

```bash
# Process multiple BBU datasets
for dataset in ds_v2 ds_v3 ds_production; do
    INPUT_DIR="$dataset"
    DATASET_NAME="$dataset"
    OUTPUT_DIR="data"
    ./convert_dataset.sh
    echo "✅ Completed $dataset"
done
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Pipeline Execution Errors
```
No valid samples were processed
```
**Solution**: 
- Verify JSON files contain `markResult.features` or `dataList`
- Check corresponding `.jpeg/.jpg` images exist
- Ensure objects pass label hierarchy filtering

#### 2. Geometry Processing Issues
```
Dimension mismatch: JSON says 1920x1080 but image is 1080x1920
```
**Solution**: Normal for EXIF-rotated images. The unified `CoordinateManager` handles this automatically.

#### 3. Memory Issues
```
Out of memory during processing
```
**Solution**:
- Set `RESIZE="false"` to disable image processing
- Set `LOG_LEVEL="WARNING"` to reduce output
- Process smaller input directories

#### 4. Configuration Errors
```
Configuration validation failed
```
**Solution**: Ensure all required parameters are set in `convert_dataset.sh`:
- `INPUT_DIR`, `OUTPUT_DIR`, `VAL_RATIO`, `MAX_TEACHERS`, `SEED`

### Performance Optimization

1. **Large Datasets**: Use `LOG_LEVEL="WARNING"` for reduced output
2. **Memory**: Disable resizing if images are pre-sized
3. **Speed**: Use SSD storage for faster I/O
4. **Debug**: Use `LOG_LEVEL="DEBUG"` for detailed processing info

### Validation Commands

Test the streamlined architecture:

```bash
# Test unified processor
cd /data3/Qwen2.5-VL-main/data_conversion
/root/miniconda3/envs/ms/bin/python -c "
from unified_processor import UnifiedProcessor
from config import DataConversionConfig
print('✅ Streamlined architecture ready')
"

# Test coordinate manager
/root/miniconda3/envs/ms/bin/python -c "
from coordinate_manager import CoordinateManager  
print('✅ Unified coordinate processing ready')
"

# Test file operations
/root/miniconda3/envs/ms/bin/python -c "
from utils.file_ops import FileOperations
print('✅ Centralized file operations ready')
"
```

---

## Migration from Previous Versions

### Key Changes in Refactored Version

1. **Removed Files**: `sample_processor.py`, `data_loader.py` (functionality merged)
2. **Unified Processing**: Single `UnifiedProcessor` entry point  
3. **Chinese-Only**: Optimized for BBU equipment detection (no multilingual overhead)
4. **Fail-Fast**: Consistent error handling throughout pipeline
5. **Centralized I/O**: All file operations through `FileOperations`

### Backward Compatibility

- **Shell script**: Same interface (`convert_dataset.sh`)
- **Output format**: Identical JSONL structure
- **Configuration**: Same environment variables
- **Python API**: Same `UnifiedProcessor` class

The streamlined architecture provides the same functionality with improved maintainability and reduced codebase size.