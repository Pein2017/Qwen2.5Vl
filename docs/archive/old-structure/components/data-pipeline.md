# Data Pipeline Components

**Detailed documentation for the data processing pipeline (`data_conversion/`)**

## Overview

The data pipeline transforms raw BBU annotations into training-ready data through a 5-stage processing system. It supports object-oriented training with multi-geometry handling and hierarchical Chinese descriptions.

## Pipeline Architecture

```
data_conversion/
├── convert_dataset.sh           # Main pipeline script
├── pipeline_manager.py          # 5-stage pipeline orchestrator
├── unified_processor.py         # Core processing engine
├── coordinate_manager.py        # Coordinate transformations
├── flexible_taxonomy_processor.py # Chinese annotation processing
├── data_splitter.py            # Train/val splitting
├── clean_raw_json.py           # JSON cleaning
└── config.py                   # Data conversion configuration
```

## PipelineManager (`pipeline_manager.py`)

### Component Contract

**Input**:
- `input_dir`: Directory with raw JSON files and images
- `output_dir`: Directory for processed training data
- `object_types`: List of object types to process (e.g., ["bbu", "label"])
- `resize`: Boolean flag for image resizing
- `val_ratio`: Validation set ratio (default: 0.1)

**Output**:
- `train.jsonl`: Training data in chat format
- `val.jsonl`: Validation data in chat format
- `teacher.jsonl`: High-quality teacher samples
- Processing summary and statistics

**Dependencies**:
- UnifiedProcessor for core processing
- DataSplitter for train/val splitting
- Various utility modules

**Side Effects**:
- Creates output directory structure
- Processes and transforms raw data files
- Generates processing logs and statistics

### Key Features

#### 5-Stage Processing System
1. **Stage 1**: Clean raw JSON files (remove metadata)
2. **Stage 2**: Apply token mapping (optional, for bilingual support)
3. **Stage 3**: Process samples (core transformation)
4. **Stage 4**: Validate output (quality assurance)
5. **Stage 5**: Generate summary report (statistics and metrics)

#### Object-Oriented Training Support
```bash
# Train on specific object types
OBJECT_TYPES="bbu"           # Equipment detection only
OBJECT_TYPES="label"         # Text recognition only
OBJECT_TYPES="fiber wire"    # Cable system model
OBJECT_TYPES="full"          # All object types
```

#### Progress Tracking and Logging
- Stage-by-stage progress reporting
- Error tracking and recovery
- Performance metrics collection
- Detailed processing logs

### Usage Example

```bash
# Command line usage
python data_conversion/pipeline_manager.py \
    --input_dir ds_v2 \
    --output_dir data \
    --object_types "bbu label fiber" \
    --resize true \
    --val_ratio 0.1

# Shell script wrapper
bash data_conversion/convert_dataset.sh
```

```python
# Python API usage
from data_conversion.pipeline_manager import PipelineManager
from data_conversion.config import DataConversionConfig

config = DataConversionConfig(
    input_dir="ds_v2",
    output_dir="data",
    object_types=["bbu", "label", "fiber"],
    resize=True,
    val_ratio=0.1
)

manager = PipelineManager(config)
manager.run_pipeline()
```

## UnifiedProcessor (`unified_processor.py`)

### Component Contract

**Input**:
- Cleaned JSON files with BBU annotations
- Configuration for object filtering and processing
- Image files for coordinate validation

**Output**:
- Processed samples in chat format
- Coordinate tokens embedded in conversations
- Multi-geometry support (bbox, square, line)

**Dependencies**:
- CoordinateManager for coordinate transformations
- FlexibleTaxonomyProcessor for Chinese descriptions
- ChatProcessor for format conversion

**Side Effects**:
- Validates and transforms coordinate data
- Filters objects by type
- Converts to training format

### Key Features

#### Multi-Geometry Processing
```python
# Supported geometry types
geometries = {
    'bbox_2d': [x1, y1, x2, y2],                    # Standard rectangles
    'square': [x1, y1, x2, y2, x3, y3, x4, y4],     # Rotated quadrilaterals
    'line': [x1, y1, x2, y2, ..., xN, yN]           # Multi-point lines
}
```

#### Object Type Filtering
- Configurable object type selection
- Hierarchical filtering (e.g., all BBU-related objects)
- Quality-based filtering (remove low-quality annotations)

#### Coordinate Validation and Transformation
- EXIF orientation compensation
- Dimension mismatch rescaling
- Smart resize scaling for VLM optimization

### Usage Example

```python
from data_conversion.unified_processor import UnifiedProcessor

processor = UnifiedProcessor(
    object_types=["bbu", "label"],
    resize_images=True,
    validate_coordinates=True
)

# Process a batch of samples
processed_samples = processor.process_samples(raw_samples)
```

## CoordinateManager (`coordinate_manager.py`)

### Component Contract

**Input**:
- Raw coordinates from JSON annotations
- Image metadata (dimensions, EXIF data)
- Target image size for VLM

**Output**:
- Transformed coordinates for training
- Coordinate tokens for text embedding
- Validation results

**Dependencies**:
- PIL for image processing
- EXIF data handling utilities
- Coordinate transformation functions

**Side Effects**:
- None (pure transformation component)

### Key Features

#### 3-Stage Coordinate Transformation
```python
# Stage 1: EXIF Orientation Compensation
if exif_orientation in [3, 6, 8]:
    coords = apply_orientation_transform(coords, orientation)

# Stage 2: Dimension Mismatch Rescaling
scale_x = image_width / json_width
scale_y = image_height / json_height
coords = apply_dimension_scaling(coords, scale_x, scale_y)

# Stage 3: Smart Resize Scaling
new_size = smart_resize(original_size, factor=28)
coords = apply_smart_resize_scaling(coords, original_size, new_size)
```

#### Multi-Geometry Support
- **bbox_2d**: Standard rectangular transformations
- **square**: Quadrilateral point transformations
- **line**: Multi-point line transformations with variable length

#### Coordinate Validation
- Range checking (0-2047 for coordinate tokens)
- Geometry consistency validation
- Degenerate case handling

### Usage Example

```python
from data_conversion.coordinate_manager import CoordinateManager

manager = CoordinateManager(
    target_size=(448, 448),
    coordinate_range=(0, 2047)
)

# Transform coordinates
transformed_coords = manager.transform_coordinates(
    coordinates=raw_coords,
    geometry_type="bbox_2d",
    image_size=(1920, 1080),
    exif_orientation=1
)
```

## FlexibleTaxonomyProcessor (`flexible_taxonomy_processor.py`)

### Component Contract

**Input**:
- Raw Chinese BBU annotations
- Object type and attribute information
- Hierarchical description structure

**Output**:
- Formatted hierarchical descriptions
- Standardized Chinese text
- Attribute extraction and organization

**Dependencies**:
- Chinese text processing utilities
- BBU domain knowledge
- Hierarchical formatting rules

**Side Effects**:
- None (pure text processing component)

### Key Features

#### Hierarchical Description Processing
```python
# Chinese hierarchy format: object_type/attributes,level1,level2/conditional_details
examples = [
    "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求",
    "光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管",
    "标签/5G-BBU,清晰可见"
]
```

#### Attribute Extraction
- Automatic extraction of equipment attributes
- Standardization of Chinese terminology
- Quality assessment integration

#### Chinese-Only Processing Mode
- Optimized for Chinese BBU annotations
- No English translation required
- Preserves original Chinese semantics

### Usage Example

```python
from data_conversion.flexible_taxonomy_processor import FlexibleTaxonomyProcessor

processor = FlexibleTaxonomyProcessor(
    language="zh",
    preserve_hierarchy=True
)

# Process Chinese descriptions
formatted_desc = processor.process_description(
    raw_description="BBU设备安装螺丝显示完整符合要求",
    object_type="connect_point"
)
# Output: "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"
```

## Data Splitter (`data_splitter.py`)

### Component Contract

**Input**:
- Processed samples in chat format
- Split ratios (train/val/teacher)
- Splitting strategy configuration

**Output**:
- Training set (train.jsonl)
- Validation set (val.jsonl)
- Teacher set (teacher.jsonl)

**Dependencies**:
- Random number generation for seeded splitting
- Sample quality metrics for teacher selection

**Side Effects**:
- Creates output JSONL files
- Maintains consistent splits across runs

### Key Features

#### Stratified Splitting
- Maintains object type distribution across splits
- Ensures representative validation sets
- Balanced teacher sample selection

#### Teacher Pool Creation
- Selects high-quality samples for teacher-student learning
- Geometry diversity weighting
- Quality score-based selection

#### Seeded Splitting
- Reproducible splits across runs
- Configurable random seeds
- Consistent train/val boundaries

### Usage Example

```python
from data_conversion.data_splitter import DataSplitter

splitter = DataSplitter(
    train_ratio=0.8,
    val_ratio=0.1,
    teacher_ratio=0.1,
    random_seed=42
)

# Split processed samples
train_samples, val_samples, teacher_samples = splitter.split_samples(
    samples=processed_samples,
    stratify_by="object_type"
)
```

## Integration Patterns

### Complete Pipeline Execution
```bash
# Shell script approach (recommended)
bash data_conversion/convert_dataset.sh

# Python approach (for customization)
python data_conversion/pipeline_manager.py \
    --input_dir ds_v2 \
    --output_dir data \
    --object_types "bbu label fiber" \
    --resize true
```

### Custom Processing Pipeline
```python
# Step-by-step processing for customization
from data_conversion import *

# Stage 1: Clean raw JSON
clean_raw_json(input_dir="ds_v2", output_dir="ds_v2_clean")

# Stage 2: Process samples
processor = UnifiedProcessor(object_types=["bbu", "label"])
samples = processor.process_directory("ds_v2_clean")

# Stage 3: Split data
splitter = DataSplitter(train_ratio=0.8, val_ratio=0.2)
train, val, teacher = splitter.split_samples(samples)

# Stage 4: Save outputs
save_jsonl(train, "data/train.jsonl")
save_jsonl(val, "data/val.jsonl")
save_jsonl(teacher, "data/teacher.jsonl")
```

### Object-Oriented Training Configurations
```bash
# Equipment detection model
export OBJECT_TYPES="bbu bbu_shield"
bash data_conversion/convert_dataset.sh

# Text recognition model
export OBJECT_TYPES="label"
bash data_conversion/convert_dataset.sh

# Cable system model
export OBJECT_TYPES="fiber wire"
bash data_conversion/convert_dataset.sh

# Complete system model
export OBJECT_TYPES="full"
bash data_conversion/convert_dataset.sh
```

## Performance and Optimization

### Processing Speed
- **Typical throughput**: 100-200 samples/second
- **Bottlenecks**: Image loading and coordinate transformation
- **Optimization**: Parallel processing and caching

### Memory Usage
- **Peak memory**: ~2-4GB for large datasets
- **Streaming processing**: Handles datasets larger than memory
- **Garbage collection**: Automatic cleanup of processed samples

### Quality Metrics
- **Coordinate accuracy**: Validation against image bounds
- **Description quality**: Hierarchical format compliance
- **Object coverage**: Distribution across object types

## Debugging and Validation

### Pipeline Health Checks
```bash
# Check input data
ls -la ds_v2/ | head -5
find ds_v2/ -name "*.json" | wc -l

# Check output data
ls -la data/
wc -l data/*.jsonl

# Validate output format
head -1 data/train.jsonl | python -m json.tool
```

### Common Issues and Solutions

#### Missing Input Data
```bash
# Issue: ds_v2/ directory not found
# Solution: Verify raw data location
ls -la ds_v2/ || echo "Raw data directory missing"
```

#### Processing Failures
```bash
# Issue: Pipeline stage failures
# Solution: Check logs and restart from failed stage
tail -20 data_conversion/pipeline.log
python data_conversion/pipeline_manager.py --resume-from-stage 3
```

#### Coordinate Validation Errors
```python
# Issue: Invalid coordinates
# Solution: Check coordinate transformation
from data_conversion.coordinate_manager import validate_coordinates
validate_coordinates(coords, image_size=(1920, 1080))
```

---

**Next Steps**:
- **Training System**: [training-system.md](training-system.md)
- **Model System**: [model-system.md](model-system.md)
- **Configuration**: [configuration.md](configuration.md)
