# Data Schema & Conversion Pipeline

> **Purpose:** Define the JSONL formats used by the training pipeline and document how raw vendor annotations are converted using the current unified processing system.

---

## 1. Current JSONL Format (Training Data)

### 1.1 Teacher-Student JSONL Structure
```jsonc
{
  "teachers": [
    {
      "images": ["ds_output/img001.jpeg"],
      "objects": [
        {
          "bbox_2d": [x1, y1, x2, y2], 
          "description": "螺丝连接点/BBU安装螺丝/连接正确",
          "object_type": "螺丝连接点",
          "property": "BBU安装螺丝", 
          "extra_info": "连接正确"
        }
      ]
    }
  ],
  "student": {
    "images": ["ds_output/img002.jpeg"],
    "objects": [
      {
        "bbox_2d": [x1, y1, x2, y2],
        "description": "螺丝连接点/BBU安装螺丝/连接正确",
        "object_type": "螺丝连接点", 
        "property": "BBU安装螺丝",
        "extra_info": "连接正确"
      }
    ]
  }
}
```

### 1.2 Data Format Rules
1. **Coordinates**: `bbox_2d` contains absolute pixel coordinates `[x1, y1, x2, y2]` after 3-stage coordinate transformation
2. **Descriptions**: Natural language Chinese phrases with structured decomposition
3. **Teacher Ratio**: 70% of samples include teachers (configurable via `teacher_ratio`)
4. **Path References**: All image paths reference `ds_output/` directory for consistency
5. **Field Standardization**: `desc` → `description`, `contentZh` → structured fields

### 1.3 Language Support
- **Chinese Format**: `object_type/property/extra_info` (compact)
- **English Format**: `object_type:value;property:value` (structured)
- **Token Mapping**: Automatic Chinese-to-English mapping when enabled

## 2. Current Unified Conversion Pipeline

### 2.1 Pipeline Architecture (5-Stage Process)
```
Raw Data (ds/) → convert_dataset.sh (Pipeline Manager) → ① JSON Cleaning
                                                      → ② Token Mapping (Optional)
                                                      → ③ Unified Processing
                                                      → ④ Output Validation
                                                      → ⑤ Summary Generation
                                                      → Processed Data (data/)
```

#### Stage 1: JSON Cleaning Pipeline
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

### 2.2 Pipeline Orchestration
The pipeline is managed by `pipeline_manager.py` with the following components:

| Stage | Component | Responsibility |
|-------|-----------|---------------|
| **Stage 1** | `clean_raw_json.py` | Strips unnecessary metadata, preserves essential structure |
| **Stage 2** | `TokenMapper` (optional) | Maps Chinese terms to English for standardization |
| **Stage 3** | `unified_processor.py` | Core processing with `SampleExtractor` and `UnifiedProcessor` |
| **Stage 4** | `DataValidator` | Validates structure, coordinates, and set overlaps |
| **Stage 5** | `SummaryGenerator` | Creates processing summary and label vocabulary |

### 2.3 Unified Processing System
The core processing is handled by `unified_processor.py`:

```python
class UnifiedProcessor:
    """Main orchestrator for the complete data processing workflow"""
    
    def process_dataset(self):
        # Extract samples from cleaned JSON
        samples = self.sample_extractor.extract_samples()
        
        # Apply 3-stage coordinate transformation
        processed_samples = self.process_samples(samples)
        
        # Filter using label hierarchy
        filtered_samples = self.filter_by_hierarchy(processed_samples)
        
        # Split into train/val/teacher sets
        train, val, teachers = self.split_and_select(filtered_samples)
        
        # Generate final JSONL files
        self.write_outputs(train, val, teachers)
```

### 2.4 3-Stage Coordinate Transformation
Each bounding box undergoes sophisticated coordinate transformation:

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