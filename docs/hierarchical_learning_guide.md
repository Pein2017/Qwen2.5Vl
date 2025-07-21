# Hierarchical Learning Framework for Qwen2.5-VL

> **New in v2.0**: Progressive learning stages for AI quality inspection with support for multiple annotation formats

---

## Overview

The Hierarchical Learning Framework enables progressive training of Qwen2.5-VL models through structured learning stages. Instead of learning everything at once, the model progresses through increasingly complex understanding levels:

1. **Object Identification** - Basic object detection and classification
2. **Property Recognition** - Object attributes and characteristics  
3. **Complex Attributes** - Detailed properties, installation status, compliance
4. **OCR & Special Cases** - Text recognition and edge cases

## Key Features

### 🎯 **Progressive Learning Stages**
- **Stage 1**: Object identification only (`螺丝、光纤插头`)
- **Stage 1-2**: Object + basic properties (`螺丝、光纤插头/BBU安装螺丝`)
- **Stage 1-3**: Object + properties + complex attributes (`螺丝、光纤插头/BBU安装螺丝/符合要求`)
- **All Stages**: Complete hierarchical learning including OCR

### 📐 **Multiple Annotation Formats**
- **bbox_2d**: Standard rectangular bounding box `[x1, y1, x2, y2]`
- **square**: Four-point polygon (四边形) `[x1, y1, x2, y2, x3, y3, x4, y4]`
- **line**: Multi-point line annotation `[x1, y1, x2, y2, ..., xn, yn]`

### 🔄 **Flexible Description Strategies**
- **Slash-separated**: `object/property/extra_info` (default)
- **Natural language**: `一个华为BBU设备，显示完整，符合要求`
- **Structured JSON**: `{"object_type": "BBU", "properties": ["华为"], ...}`
- **Progressive stages**: `[object_identification]BBU设备/[property_recognition]华为`

---

## Quick Start

### Basic Usage

```python
from data_conversion.hierarchical_processor import HierarchicalProcessor

# Initialize processor
processor = HierarchicalProcessor(
    language="chinese",
    response_types={"object_type", "property", "extra_info"}
)

# Process v2 format features
features = data["markResult"]["features"]
objects = processor.extract_objects_from_markresult(features)

# Each object now contains:
# - Multiple geometry formats (bbox_2d, square, line)
# - Progressive descriptions for all learning stages
# - Hierarchical content categorization
```

### Create Progressive Datasets

```bash
# Using the example script
python data_conversion/hierarchical_example.py \
  --input_dir ds_v2 \
  --output_dir data_hierarchical \
  --language chinese \
  --demo
```

This creates separate datasets for each learning stage:
```
data_hierarchical/
├── stage_stage_1/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── teacher.jsonl
├── stage_stage_1_2/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── teacher.jsonl
└── ...
```

---

## Architecture

### Core Components

```
HierarchicalLearningFramework
├── LearningStage (Enum)
├── HierarchicalLabel (DataClass)
└── GeometryInfo (DataClass)

DescriptionConcatenator
├── ConcatenationStrategy (Enum)
├── ConcatenationConfig (DataClass)
└── Progressive Description Generation

HierarchicalProcessor
├── Content Field Extraction
├── Stage-Specific Sample Creation
└── Geometry Processing Integration

GeometryProcessor (Enhanced)
├── extract_hierarchical_geometry()
├── scale_hierarchical_geometry()
└── Multiple Format Support
```

### Data Flow

```
Raw v2 JSON → Content Extraction → Stage Categorization → Description Concatenation → Training Samples
     ↓              ↓                    ↓                      ↓                    ↓
Geometry Processing → Coordinate Scaling → Multiple Formats → Progressive Stages → Stage-Specific Datasets
```

---

## Configuration

### Label Hierarchy v2

The new label hierarchy supports progressive learning stages:

```json
{
  "object_hierarchy": {
    "螺丝、光纤插头": {
      "category": "connect_point",
      "stage_1_object_identification": {
        "aliases": ["连接点", "螺丝", "光纤插头"]
      },
      "stage_2_property_recognition": {
        "types": ["BBU安装螺丝", "BBU端光纤插头"],
        "visibility": ["显示完整", "只显示部分"]
      },
      "stage_3_complex_attributes": {
        "compliance": ["符合要求", "不符合要求"]
      }
    }
  }
}
```

### Concatenation Strategies

```python
from data_conversion.description_concatenator import ConcatenationConfig, ConcatenationStrategy

# Slash-separated (default)
config = ConcatenationConfig(
    strategy=ConcatenationStrategy.SLASH_SEPARATED,
    language="chinese"
)

# Natural language
config = ConcatenationConfig(
    strategy=ConcatenationStrategy.NATURAL_LANGUAGE,
    language="chinese"
)

# Custom separators
config = ConcatenationConfig(
    strategy=ConcatenationStrategy.SLASH_SEPARATED,
    stage_separators={
        "between_stages": " | ",
        "within_stage": " & "
    }
)
```

---

## Training Data Format

### Progressive Learning Output

Each training object contains multiple description levels:

```json
{
  "bbox_2d": [100, 100, 200, 200],
  "square": [100, 100, 200, 100, 200, 200, 100, 200],
  "desc": "螺丝、光纤插头/BBU安装螺丝/符合要求",
  "progressive_descriptions": {
    "stage_1": "螺丝、光纤插头",
    "stage_1_2": "螺丝、光纤插头/BBU安装螺丝", 
    "stage_1_2_3": "螺丝、光纤插头/BBU安装螺丝/符合要求",
    "all_stages": "螺丝、光纤插头/BBU安装螺丝/符合要求"
  },
  "geometry": {
    "type": "Square",
    "coordinates": [[[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]],
    "lineType": ["LLLLL"]
  }
}
```

### Stage-Specific Training

For progressive training, use stage-specific datasets:

```python
# Stage 1: Object identification only
{
  "bbox_2d": [100, 100, 200, 200],
  "desc": "螺丝、光纤插头"
}

# Stage 1-2: Object + properties
{
  "bbox_2d": [100, 100, 200, 200], 
  "desc": "螺丝、光纤插头/BBU安装螺丝"
}

# All stages: Complete description
{
  "bbox_2d": [100, 100, 200, 200],
  "desc": "螺丝、光纤插头/BBU安装螺丝/符合要求"
}
```

---

## Advanced Usage

### Custom Learning Stages

```python
from data_conversion.hierarchical_learning_framework import LearningStage

# Define custom target stages
custom_stages = [
    LearningStage.OBJECT_IDENTIFICATION,
    LearningStage.PROPERTY_RECOGNITION
]

# Create stage-specific samples
stage_objects = processor.create_stage_specific_samples(
    objects, 
    target_stage="stage_1_2"
)
```

### Geometry Processing

```python
from data_conversion.geometry_processor import GeometryProcessor

# Extract hierarchical geometry
geometry_dict = GeometryProcessor.extract_hierarchical_geometry(geometry)

# Scale coordinates for different annotation types
scaled_geometry = GeometryProcessor.scale_hierarchical_geometry(
    geometry_dict, 
    scale_x=0.5, 
    scale_y=0.5
)
```

### Validation

```python
# Validate hierarchical objects
validation = processor.validate_hierarchical_object(obj)
print(validation)
# {
#   'has_description': True,
#   'has_geometry': True, 
#   'has_progressive_descriptions': True,
#   'valid_bbox': True,
#   'valid_coordinates': True
# }
```

---

## Migration Guide

### From v1 to v2

The hierarchical learning framework is fully backward compatible:

1. **Existing pipelines** continue to work unchanged
2. **New features** are opt-in through configuration
3. **Output format** maintains compatibility with existing training code

### Enabling Hierarchical Features

```python
# Old way (still works)
config = DataConversionConfig(
    input_dir="ds",
    output_dir="data", 
    language="chinese"
)

# New way (with hierarchical features)
config = DataConversionConfig(
    input_dir="ds_v2",
    output_dir="data_hierarchical",
    language="chinese",
    hierarchy_path="data_conversion/label_hierarchy_v2.json"
)
```

---

## Testing

### Run Tests

```bash
# Test hierarchical learning framework
python temporal/test_hierarchical_learning.py

# Test description strategies
python temporal/test_description_strategies.py

# Test pipeline integration
python temporal/test_hierarchical_pipeline_integration.py
```

### Validation

```bash
# Validate output with real v2 data
python data_conversion/hierarchical_example.py \
  --input_dir ds_v2 \
  --output_dir test_output \
  --demo
```

---

## Best Practices

### 1. Progressive Training Strategy

- **Start with Stage 1**: Train object identification first
- **Gradually increase complexity**: Add properties, then complex attributes
- **Monitor performance**: Validate each stage before proceeding
- **Use teacher samples**: Leverage diverse teacher pool for guidance

### 2. Annotation Format Selection

- **bbox_2d**: Use for simple rectangular objects
- **square**: Use for rotated or irregular rectangular objects  
- **line**: Use for cables, wires, and linear structures

### 3. Description Strategy

- **Slash-separated**: Best for structured training data
- **Natural language**: Better for human-readable descriptions
- **Progressive stages**: Useful for debugging and analysis

### 4. Quality Control

- **Validate geometry**: Ensure coordinates are within image bounds
- **Check descriptions**: Verify progressive descriptions make sense
- **Monitor coverage**: Ensure all object types are represented

---

## Troubleshooting

### Common Issues

**Q: Progressive descriptions are empty**
A: Check that your label hierarchy v2 file is properly configured and accessible.

**Q: Geometry scaling produces invalid coordinates**
A: Verify that scale factors are positive and reasonable (typically 0.1-2.0).

**Q: Stage-specific datasets are identical**
A: Ensure your content has sufficient hierarchical information across different stages.

**Q: Import errors with hierarchical modules**
A: Make sure you're using relative imports and the project root is in PYTHONPATH.

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate hierarchical objects
validation = processor.validate_hierarchical_object(obj)
if not all(validation.values()):
    print(f"Validation issues: {validation}")
```

---

## Performance Notes

- **Memory usage**: Hierarchical objects contain more metadata
- **Processing time**: Slightly increased due to stage categorization
- **Storage**: Progressive descriptions increase file sizes by ~30%
- **Training speed**: Stage-specific datasets enable faster convergence

---

For more examples and detailed API documentation, see the `temporal/` test files and `data_conversion/hierarchical_example.py`.
