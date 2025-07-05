# JSON Cleaning Pipeline

> **Purpose:** Document the new `clean_raw_json.py` component that strips unnecessary metadata from vendor annotation files while preserving essential structure.

---

## Overview

The `clean_raw_json.py` script addresses data quality issues by:
1. **Removing unnecessary metadata** that bloats file size and processing time
2. **Preserving essential structure** required by the training pipeline  
3. **Language-aware filtering** to support both Chinese and English workflows
4. **Maintaining compatibility** with existing data loader expectations

## Usage

### Basic Usage
```bash
python data_conversion/clean_raw_json.py input_dir output_dir --lang zh
```

### Parameters
- `input_dir`: Directory containing raw vendor JSON files
- `output_dir`: Directory for cleaned JSON files  
- `--lang`: Language filter (`zh`, `en`, or `both`)

### Integration in Pipeline
The script is automatically called by `convert_dataset.sh`:
```bash
# Automatic language mapping
LANGUAGE="chinese" → JSON_LANG="zh"
LANGUAGE="english" → JSON_LANG="en"
```

## What Gets Preserved

### Essential Metadata
- **`info`**: Image dimensions (`width`, `height`, `depth`)
- **`tagInfo`**: Task metadata (`mode`, `dataId`, `taskId`, `timestamp`)  
- **`version`**: JSON format version
- **`markResult`**: Complete annotation structure

### Annotation Features
- **`type`**: GeoJSON feature type
- **`geometry`**: Bounding box coordinates
- **`properties`**: Filtered content based on language selection

## What Gets Removed

- Statistical summaries (`statistcs`)
- Quality control metadata (`qualityResult`, `quality`)
- Workflow tracking (`submitWorkflow`)
- Workload metrics (`workload`)
- Administrative fields not needed for training

## Language Filtering

### Chinese Mode (`--lang zh`)
Preserves:
- `properties.contentZh`: Chinese label hierarchies
- Filters out English content to reduce file size

### English Mode (`--lang en`)  
Preserves:
- `properties.content`: English label mappings
- Filters out Chinese content

### Both Mode (`--lang both`)
Preserves both language variants for bilingual workflows.

## Structure Validation

The script ensures cleaned files maintain compatibility with:
- **`data_loader.py`**: Expects `data.get("info", {})` structure
- **`sample_processor.py`**: Requires `markResult.features` arrays
- **Training pipeline**: Needs consistent image dimension metadata

## Error Handling

Common issues and solutions:
- **Empty features**: File skipped with warning
- **Missing info section**: Error with detailed path information  
- **Invalid JSON**: Descriptive error with file location
- **Write permissions**: Clear filesystem error messages

## Performance Impact

### File Size Reduction
- **Before**: Vendor JSON files ~24KB with extensive metadata
- **After**: Cleaned files ~2-3KB with only essential data
- **Reduction**: ~85% smaller files, faster I/O

### Processing Speed
- Faster JSON parsing during training
- Reduced memory footprint
- Improved batch loading performance

---

### Related source files
* `data_conversion/clean_raw_json.py` *(Main implementation)*
* `data_conversion/convert_dataset.sh` *(Pipeline integration)*
* `data_conversion/data_loader.py` *(Consumes cleaned files)*