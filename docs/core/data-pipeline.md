# Data Schema & Conversion Pipeline - V2 Multi-Geometry Format

> **Purpose:** Define the V2 JSONL formats with multi-geometry support used by the training pipeline and document how raw vendor annotations are converted using the object-oriented processing system.

> **Migration Status**: **COMPLETE** - Fully migrated from V1 to V2 data format with multi-geometry support (bbox_2d, square, line)

---

## 1. V2 Multi-Geometry JSONL Format (Training Data)

### 1.1 V2 Multi-Geometry Sample Structure
```jsonc
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

### 1.2 V2 Geometry Types Supported
- **`bbox_2d`**: Standard rectangular bounding box `[x1, y1, x2, y2]`
- **`square`**: Rotated/perspective quadrilateral `[x1, y1, x2, y2, x3, y3, x4, y4]`  
- **`line`**: Multi-point line/curve `[x1, y1, x2, y2, ..., xN, yN]` (variable length)

### 1.3 V2 Data Format Rules
1. **Multi-Geometry Coordinates**: Each object has one geometry type (`bbox_2d`, `square`, or `line`) with absolute pixel coordinates after transformation pipeline
2. **Hierarchical Descriptions**: Chinese descriptions use comma/slash hierarchy: `object_type/attributes,level1,level2/conditional_details`
3. **Geometry Constraints**: Line objects (fiber/wire) use `line` geometry; equipment/labels use `bbox_2d`/`square`
4. **Object-Oriented Processing**: Filter training by object types (`bbu`, `label`, `fiber`, etc.) for specialized training
5. **Image Dimensions**: Include `width` and `height` for coordinate validation and scaling

### 1.3 Language Support
- **Chinese Format**: `object_type/property/extra_info` (compact)
- **English Format**: `object_type:value;property:value` (structured)
- **Token Mapping**: Automatic Chinese-to-English mapping when enabled

## 2. V2 Object-Oriented Conversion Pipeline (Current Architecture)

### 2.1 V2 Pipeline Architecture - 5-Stage Processing System
```
V2 Raw Data (ds_v2/) → PipelineManager → ① Clean Raw JSON Files
                                       → ② Apply Token Mapping (optional)
                                       → ③ Process Samples (UnifiedProcessor)
                                       → ④ Validate Output
                                       → ⑤ Generate Summary Report
                                       → V2 Training Data (data/)
```

#### Core Pipeline Components

| Component | File | Purpose | Key Features |
|-----------|------|---------|--------------|
| **PipelineManager** | `pipeline_manager.py` | Pipeline orchestration | 5-stage processing, error handling, progress tracking |
| **UnifiedProcessor** | `unified_processor.py` | Core processing engine | Multi-geometry support, object-oriented training |
| **CoordinateManager** | `coordinate_manager.py` | Coordinate transformations | EXIF handling, smart resize, geometry processing |
| **FlexibleTaxonomyProcessor** | `flexible_taxonomy_processor.py` | V2 annotation processing | Hierarchical descriptions, attribute extraction |

#### Supported Object Types & Geometries (Object-Oriented Training)
| Object Type | Chinese Label | Geometry | Usage | Training Combinations |
|-------------|---------------|----------|-------|----------------------|
| `bbu` | BBU设备 | `bbox_2d`/`square` | Equipment detection | Individual or combined |
| `bbu_shield` | 挡风板 | `bbox_2d`/`square` | Shield detection | With bbu for equipment model |
| `connect_point` | 螺丝、光纤插头 | `bbox_2d`/`square` | Connection hardware | Hardware-focused training |
| `label` | 标签 | `bbox_2d`/`square` | Text recognition | Text model training |
| `fiber` | 光纤 | `line` | Fiber cable routing | Cable system training |
| `wire` | 电线 | `line` | Wire management | With fiber for cable model |

#### Stage 1b: JSON Cleaning Pipeline  
The `clean_raw_json.py` component addresses data quality issues by:

**Purpose**:
- **Removing unnecessary metadata** that bloats file size and processing time
- **Preserving essential structure** required by the training pipeline  
- **Language-aware filtering** to support both Chinese and English workflows
- **Maintaining compatibility** with existing data loader expectations

**What Gets Preserved**:
- **`info`**: Image dimensions (`width`, `height`, `depth`)
- **`tagInfo`**: Task metadata (`mode`, `dataId`, `taskId`, `timestamp`)  
- **`version`**: JSON format version
- **`markResult`**: Complete annotation structure
- **`geometry`**: Bounding box coordinates
- **`properties`**: Filtered content based on language selection

**What Gets Removed**:
- Statistical summaries (`statistcs`)
- Quality control metadata (`qualityResult`, `quality`)
- Workflow tracking (`submitWorkflow`)
- Workload metrics (`workload`)
- Administrative fields not needed for training

**Performance Impact**:
- **Before**: Vendor JSON files ~24KB with extensive metadata
- **After**: Cleaned files ~2-3KB with only essential data
- **Reduction**: ~85% smaller files, faster I/O and JSON parsing

**Language Filtering Options**:
```bash
# Chinese mode - preserves contentZh
python data_conversion/clean_raw_json.py input_dir output_dir --lang zh

# English mode - preserves content  
python data_conversion/clean_raw_json.py input_dir output_dir --lang en

# Both languages
python data_conversion/clean_raw_json.py input_dir output_dir --lang both
```

### 2.2 5-Stage Pipeline Processing Details

#### Stage 1: Clean Raw JSON Files (`clean_raw_json.py`)
**Purpose**: Remove unnecessary metadata and optimize file size for processing

**What Gets Preserved**:
- **`info`**: Image dimensions (`width`, `height`, `depth`)
- **`tagInfo`**: Task metadata (`mode`, `dataId`, `taskId`, `timestamp`)
- **`version`**: JSON format version
- **`markResult`**: Complete annotation structure with geometry and properties

**What Gets Removed**:
- Statistical summaries, quality control metadata, workflow tracking
- Administrative fields not needed for training
- **Performance Impact**: ~85% file size reduction (24KB → 2-3KB)

#### Stage 2: Apply Token Mapping (Optional)
**Purpose**: Convert Chinese tokens to English equivalents when needed
- Skipped for Chinese-only processing
- Uses token mapping files for bilingual support

#### Stage 3: Process Samples (`UnifiedProcessor`)
**Core Processing Engine** with integrated components:

| Sub-Component | Purpose | Key Features |
|---------------|---------|--------------|
| **HierarchicalProcessor** | V2 annotation processing | Chinese-only mode, object type filtering |
| **ImageProcessor** | Image transformations | EXIF handling, smart resize |
| **TeacherSelector** | Teacher pool creation | Geometry diversity weighting |
| **DataSplitter** | Train/validation split | Configurable ratios, seeded splitting |

#### Stage 4: Validate Output
**Purpose**: Ensure data quality and format compliance
- File existence and format validation
- Coordinate range checking
- Sample count verification

#### Stage 5: Generate Summary Report
**Purpose**: Provide processing statistics and quality metrics
- Processing time and throughput
- Object type distribution
- Error summary and recommendations

### 2.3 Coordinate Transformation Pipeline (`CoordinateManager`)
The coordinate transformation system handles multi-geometry processing:

#### 3-Stage Coordinate Transformation
Each geometry (bbox, square, line) undergoes sophisticated coordinate transformation:

1. **EXIF Orientation Compensation**
   ```python
   # Handle image rotation metadata
   if exif_orientation in [3, 6, 8]:
       coords = apply_orientation_transform(coords, orientation)
   ```

2. **Dimension Mismatch Rescaling**
   ```python
   # Compensate for JSON vs image dimension differences
   scale_x = image_width / json_width
   scale_y = image_height / json_height
   coords = apply_dimension_scaling(coords, scale_x, scale_y)
   ```

3. **Smart Resize Scaling**
   ```python
   # Apply VLM-optimized resizing
   new_size = smart_resize(original_size, factor=28)
   coords = apply_smart_resize_scaling(coords, original_size, new_size)
   ```

#### Multi-Geometry Support
- **bbox_2d**: `[x1, y1, x2, y2]` - Standard rectangular bounding boxes
- **square**: `[x1, y1, x2, y2, x3, y3, x4, y4]` - Rotated/perspective quadrilaterals
- **line**: `[x1, y1, x2, y2, ..., xN, yN]` - Multi-point lines (variable length)

### 2.4 Object-Oriented Training System
The pipeline supports flexible object type combinations for progressive learning:

#### Training Combinations
```bash
# Individual object type training
OBJECT_TYPES="bbu"           # Equipment detection only
OBJECT_TYPES="label"         # Text recognition only
OBJECT_TYPES="fiber"         # Fiber cable routing only

# Combined training
OBJECT_TYPES="bbu bbu_shield"    # Equipment model
OBJECT_TYPES="fiber wire"        # Cable system model
OBJECT_TYPES="full"              # All object types
```

#### Hierarchical Description Processing
Chinese descriptions use comma/slash hierarchy format:
```
object_type/attributes,level1,level2/conditional_details
```

Example:
```
"螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"
"光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管"
```

### 2.5 Data Validation and Quality Assurance
The pipeline includes comprehensive validation:

**Coordinate Validation:**
- BBox bounds: `0 ≤ x1 < x2 ≤ width` & `0 ≤ y1 < y2 ≤ height`
- Coordinate precision and boundary checking
- Transformation accuracy validation

**Structure Validation:**
- JSONL format compliance
- Required field presence
- Data type consistency

**Set Validation:**
- No overlap between train/val/teacher sets
- Proper sample distribution
- Label coverage verification

## 3. Runtime validation (src/schema.py)
The dataclass `GroundTruthObject` and friends use **torchtyping** to validate shapes at runtime.  Any violation raises immediately (fail-fast).

## 4. Current Implementation Code Map (`data_conversion/`)

### 4.1 Core Pipeline Components
| Module | Key Classes/Functions | Responsibility |
|--------|----------------------|----------------|
| `convert_dataset.sh` | **Main Pipeline Script** | Bash wrapper providing backward compatibility and environment setup |
| `pipeline_manager.py` | `PipelineManager` | **NEW**: Python-based pipeline orchestration with progress tracking and error handling |
| `unified_processor.py` | `UnifiedProcessor`, `SampleExtractor` | **Core Engine**: Complete processing workflow with centralized coordinate transformations |
| `clean_raw_json.py` | `clean_annotation_file()` | Strips unnecessary metadata while preserving essential JSON structure |

### 4.2 Processing Components
| Module | Key Classes/Functions | Responsibility |
|--------|----------------------|----------------|
| `core_modules.py` | `TokenMapper`, `FieldStandardizer`, `ResponseFormatter`, `ObjectProcessor` | Fundamental processing classes shared across pipeline |
| `sample_processor.py` | `SampleProcessor` | Individual sample processing with bbox scaling and image handling |
| `teacher_selector.py` | `TeacherSelector` | Multi-objective teacher selection using greedy optimization |
| `coordinate_manager.py` | `CoordinateManager` | **NEW**: Centralized 3-stage coordinate transformation system |
| `image_processor.py` | `ImageProcessor` | Unified image processing with EXIF handling and smart resizing |

### 4.3 Utility Components
| Module | Key Classes/Functions | Responsibility |
|--------|----------------------|----------------|
| `data_splitter.py` | `DataSplitter` | Deterministic train/val split with proper randomization |
| `vision_process.py` | `smart_resize()` | VLM-optimized resizing with factor constraints (divisible by 28) |
| `utils/file_ops.py` | `FileOperations` | Centralized file operations with JSON validation |
| `utils/transformations.py` | `CoordinateTransformer`, `FormatConverter` | Mathematical transformations and format conversions |
| `utils/validators.py` | `DataValidator`, `StructureValidator` | Comprehensive validation systems |

### 4.4 Configuration and Management
| Module | Key Classes/Functions | Responsibility |
|--------|----------------------|----------------|
| `config.py` | `DataConversionConfig` | Type-safe configuration with automatic validation |
| `simple_validate.py` | `validate_pipeline_output()` | Quick validation script for pipeline output |
| `test_pipeline.py` | `test_complete_pipeline()` | Comprehensive pipeline testing |

### 4.5 Data Flow Through Components

```python
# Complete data flow
raw_data = load_raw_data("ds/")
cleaned_data = clean_raw_json(raw_data)                    # Stage 1
token_mapped = apply_token_mapping(cleaned_data)           # Stage 2 (optional)
samples = SampleExtractor().extract_samples(token_mapped)  # Stage 3a
processed = UnifiedProcessor().process_samples(samples)    # Stage 3b
train, val, teachers = split_and_select(processed)         # Stage 3c
validate_outputs(train, val, teachers)                     # Stage 4
generate_summary(train, val, teachers)                     # Stage 5
```

### 4.6 Key Implementation Features

**Centralized Coordinate Management:**
- Single source of truth for all coordinate transformations
- Consistent bbox validation across the pipeline
- Proper handling of edge cases and boundary conditions

**Unified Processing Interface:**
- Single entry point (`unified_processor.py`) for all processing
- Consistent error handling and logging
- Configurable fail-fast behavior

**Comprehensive Validation:**
- Multi-level validation (structure, coordinates, sets)
- Early error detection with clear messages
- Automatic boundary and consistency checking

**Teacher Selection Algorithm:**
- Multi-objective optimization for label coverage
- Spatial distribution and size diversity considerations
- Greedy selection with random diversification

---

### Related source files
* `data_conversion/convert_dataset.sh` *(Enhanced main pipeline)*
* `data_conversion/clean_raw_json.py` *(NEW: JSON cleaning)*
* `data_conversion/processor.py` *(Unified processing)*
* `data_conversion/sample_processor.py` *(Enhanced with path updates)*
* `data_conversion/core_modules.py` *(Enhanced bbox scaling)*
* `src/schema.py` 