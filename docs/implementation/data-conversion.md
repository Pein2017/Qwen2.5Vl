# Data Conversion Module Documentation

## Overview

The `data_conversion/` module provides a comprehensive pipeline for processing, transforming, and validating BBU equipment detection data. It handles conversion from raw annotation formats to training-ready datasets with support for multi-geometry annotations, coordinate transformations, and robust validation.

## Core Architecture

### Primary Pipeline Components

#### 1. UnifiedProcessor (`unified_processor.py`)
The main orchestrator for the data processing pipeline.

**Key Features:**
- End-to-end sample processing from JSON/image pairs
- Multi-format annotation support (dataList, markResult)
- Coordinate transformation pipeline integration
- Teacher-student data splitting
- Comprehensive validation and reporting

**Main Methods:**
```python
def process() -> Dict[str, int]:
    """Execute complete processing pipeline"""
    
def process_single_sample(json_path: Path) -> Optional[Dict]:
    """Process JSON/image pair into clean sample"""
    
def split_into_sets(all_samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split into train/val/teacher sets"""
```

**Configuration:**
- Supports Chinese-only processing mode
- Configurable label hierarchy filtering
- Validation parameters (min object size, coordinate bounds)
- Output directory management

#### 2. CoordinateManager (`coordinate_manager.py`)
Centralized coordinate transformation and geometry processing.

**Transformation Pipeline:**
1. **EXIF Orientation Compensation** - Handles image rotation metadata
2. **Dimension Mismatch Rescaling** - Corrects JSON vs actual image dimensions
3. **Smart Resize Scaling** - Applies vision processing constraints

**Supported Geometries:**
- **Simple bbox**: `[x1, y1, x2, y2]`
- **Square**: 4-point polygon `[x1,y1,x2,y2,x3,y3,x4,y4]`
- **LineString**: Multi-point line `[x1,y1,x2,y2,...,xn,yn]`
- **ExtentPolygon**: GeoJSON-style polygons

**Key Methods:**
```python
@classmethod
def transform_geometry_complete(
    cls, geometry_input: Union[List, Dict], image_path: Path,
    json_width: int, json_height: int, enable_smart_resize: bool = True
) -> Tuple[List[float], Union[List, Dict], int, int]:
    """Apply complete geometry transformation pipeline"""

@staticmethod
def normalize_object_coordinates(
    obj: Dict[str, Any], width: int, height: int
) -> Dict[str, Any]:
    """Normalize coordinates within object preserving native geometry"""
```

**Coordinate Normalization Features:**
- Canonical line ordering to resolve directional ambiguity
- Degenerate geometry handling with padding
- Bounds checking and clamping
- Multi-element coordinate validation

#### 3. ValidationManager (`validation_manager.py`)
Comprehensive validation system with detailed error reporting.

**Validation Modes:**
- `strict` - Fail on critical errors
- `lenient` - Warning for non-critical issues
- `warning_only` - Log all issues but continue processing

**Validation Checks:**
```python
def validate_sample(
    self, sample: Dict[str, Any], sample_id: str,
    image_width: Optional[int] = None, image_height: Optional[int] = None
) -> Tuple[bool, ValidationReport]:
    """Comprehensive sample validation with error reporting"""

def filter_valid_objects(
    self, objects: List[Dict[str, Any]], 
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    sample_id: str = "unknown"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter objects returning valid and invalid with error details"""
```

**Error Categories:**
- Structure validation (required fields, data types)
- Geometry validation (coordinate bounds, format compliance)
- Content validation (description requirements, object size limits)
- Coordinate validation (bounds checking, reasonable values)

### Supporting Components

#### 4. Core Processing Modules (`core_modules.py`)

**TokenMapper**: Handles coordinate-to-token conversion
```python
class TokenMapper:
    def convert_bbox_to_tokens(self, bbox: List[int], width: int, height: int) -> List[str]
    def convert_tokens_to_bbox(self, tokens: List[str]) -> List[int]
```

**FieldStandardizer**: Normalizes annotation field formats
**ResponseFormatter**: Formats descriptions for training
**ObjectProcessor**: Handles object-level transformations
**DataValidator**: Basic data structure validation

#### 5. Hierarchical Processing (`flexible_taxonomy_processor.py`)

**HierarchicalProcessor**: Handles V2 data with complex taxonomies
```python
class HierarchicalProcessor:
    def extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]
    """Extract objects from markResult with native geometry support"""
```

**AnnotationSample**: Data structure for hierarchical annotations

#### 6. Pipeline Management (`pipeline_manager.py`)

**PipelineManager**: Orchestrates processing steps
**PipelineStep**: Individual processing step wrapper

#### 7. Vision Processing (`vision_process.py`)

**Image Processing Constants:**
```python
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
```

**Key Functions:**
```python
def smart_resize(height: int, width: int, factor: int, 
                min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Resize image maintaining aspect ratio within pixel constraints"""

def fetch_image(image_path: str) -> Image.Image:
    """Load and process image with EXIF handling"""
```

**ImageProcessor**: Handles image transformations and copying

### Utility Components

#### 8. File Operations (`utils/file_ops.py`)

**FileOperations**: Centralized file I/O operations
```python
class FileOperations:
    @staticmethod
    def load_json_data(json_path: Path) -> Dict
    @staticmethod
    def find_image_file(json_path: Path) -> Path
    @staticmethod
    def get_image_dimensions(image_path: Path) -> Tuple[int, int]
    @staticmethod
    def write_jsonl(data: List[Dict], output_path: Path) -> None
```

#### 9. Configuration (`config.py`)

**DataConversionConfig**: Configuration management
```python
@dataclass
class DataConversionConfig:
    input_dir: str
    output_dir: str
    val_ratio: float = 0.1
    max_teachers: int = 500
    object_types: List[str] = field(default_factory=list)
    response_types: List[str] = field(default_factory=lambda: ["chinese"])
    resize: bool = True
    fail_fast: bool = False
```

## Usage Examples

### Basic Pipeline Execution

```python
from data_conversion.config import DataConversionConfig
from data_conversion.unified_processor import UnifiedProcessor

# Configure processing
config = DataConversionConfig(
    input_dir="/path/to/raw/data",
    output_dir="/path/to/output",
    val_ratio=0.1,
    max_teachers=500,
    object_types=["BBU设备", "光纤", "标签"],
    resize=True
)

# Execute pipeline
processor = UnifiedProcessor(config)
results = processor.process()

# Results: {'train': 1200, 'val': 150, 'teacher': 500, 'total_processed': 1850}
```

### Custom Coordinate Transformation

```python
from data_conversion.coordinate_manager import CoordinateManager
from pathlib import Path

# Transform complex geometry
geometry_input = {
    "type": "LineString",
    "coordinates": [[100, 200], [150, 250], [200, 300]]
}

bbox, transformed_geom, final_w, final_h = CoordinateManager.transform_geometry_complete(
    geometry_input=geometry_input,
    image_path=Path("image.jpg"),
    json_width=1920,
    json_height=1080,
    enable_smart_resize=True
)
```

### Validation with Custom Rules

```python
from data_conversion.validation_manager import ValidationManager

# Create validator with custom rules
validator = ValidationManager(
    validation_mode="strict",
    min_object_size=20,
    max_coordinate_value=10000,
    require_non_empty_description=True,
    check_coordinate_bounds=True
)

# Validate sample
is_valid, report = validator.validate_sample(
    sample=sample_data,
    sample_id="sample_001",
    image_width=1920,
    image_height=1080
)

if not is_valid:
    print(f"Validation errors: {report.get_summary()}")
```

## Output Formats

### Training Data Format

```json
{
  "images": ["relative/path/to/image.jpg"],
  "objects": [
    {
      "bbox_2d": [100, 150, 200, 250],
      "desc": "BBU设备/华为"
    },
    {
      "line": [50, 60, 70, 80, 90, 100],
      "desc": "光纤"
    },
    {
      "square": [10, 20, 30, 20, 30, 40, 10, 40],
      "desc": "标签"
    }
  ],
  "width": 1792,
  "height": 1008
}
```

### Validation Reports

```json
{
  "sample_id": "sample_001",
  "is_valid": false,
  "errors": [
    {
      "error_type": "coordinates_out_of_bounds",
      "severity": "critical",
      "message": "Object 0 coordinates exceed image bounds",
      "fix_suggestion": "Ensure coordinates are within image (1920x1080)",
      "object_index": 0,
      "field_name": "bbox_2d"
    }
  ]
}
```

## Performance Considerations

### Memory Management
- Processes samples individually to limit memory usage
- Streams JSON data rather than loading all at once
- Configurable batch processing for large datasets

### Error Handling  
- Fail-fast mode for development and debugging
- Graceful degradation in production mode
- Comprehensive error logging and reporting

### Scalability
- Parallel processing support for large datasets
- Efficient coordinate transformation with caching
- Minimal file I/O operations

## Integration Points

### Model Training Integration
- Compatible with BBU trainer coordinate token system
- Outputs normalized coordinates for VLM training
- Supports teacher-student learning splits

### Validation Integration  
- Hooks into training pipeline validation
- Provides detailed error reports for debugging
- Supports multiple validation modes for different use cases

### Configuration Integration
- YAML configuration file support
- Environment variable overrides
- Validation of configuration parameters

This module provides a robust, scalable foundation for converting raw BBU equipment annotation data into high-quality training datasets with comprehensive validation and error handling.