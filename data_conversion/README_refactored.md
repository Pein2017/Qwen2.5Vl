# Refactored Data Conversion Pipeline

## Overview

The data conversion pipeline has been **completely refactored** to eliminate redundancy and simplify the process. The new modular architecture converts raw JSON/image files directly to `train.jsonl`, `val.jsonl`, and `teacher.jsonl` in a single step.

## Before vs After

### Before (Multi-step Pipeline)
```
Raw JSON → Intermediate JSONL → Candidate Extraction → Teacher Pool → Student Pool → Train/Val Split
    ↓
7 separate steps, multiple intermediate files, complex bash orchestration
```

### After (Unified Pipeline)
```
Raw JSON → train.jsonl + val.jsonl + teacher.jsonl
    ↓  
Single step, modular components, clean architecture
```

## New Architecture

### Core Components

1. **`processor.py`** - Main orchestrator that coordinates all processing
2. **`data_loader.py`** - Handles JSON/image file loading and validation
3. **`sample_processor.py`** - Processes individual samples (objects, filtering, image processing)
4. **`teacher_selector.py`** - Selects diverse teacher samples with optimal label coverage
5. **`data_splitter.py`** - Splits data into train/validation sets
6. **`convert_dataset.sh`** - Simplified entry point script

### Reused Components
- **`core_modules.py`** - TokenMapper, ObjectProcessor (unchanged)
- **`vision_process.py`** - Image resizing utilities (unchanged)

## Usage

### Simple Usage (Default Settings)
```bash
bash data_conversion/convert_dataset.sh
```

### Advanced Usage (Custom Parameters)
```bash
python data_conversion/processor.py \
    --input_dir ds \
    --output_dir data \
    --language chinese \
    --resize \
    --val_ratio 0.1 \
    --max_teachers 10 \
    --token_map_path data_conversion/token_map_zh.json \
    --hierarchy_path data_conversion/label_hierarchy.json
```

## Key Improvements

### 1. **Eliminated Redundancy**
- No more intermediate JSONL files
- Single-pass object processing
- Unified field extraction and token mapping
- Direct output to final format

### 2. **Modular Design**
- Each component has single responsibility
- Easy to test individual parts
- Reusable across different datasets
- Clear separation of concerns

### 3. **Simplified Configuration**
- All parameters in one place
- Fail-fast validation
- Clear error messages
- Type-safe operations

### 4. **Better Performance**
- Single-pass processing
- No redundant file I/O
- Memory-efficient streaming
- Parallel-ready architecture

## Output Files

The pipeline produces exactly 4 files in the output directory:

- **`train.jsonl`** - Training samples (clean format)
- **`val.jsonl`** - Validation samples (clean format)  
- **`teacher.jsonl`** - Teacher demonstration samples (clean format)
- **`all_samples.jsonl`** - Combined file with all samples (for visualization/analysis)

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dir` | Raw JSON/image directory | `ds` |
| `output_dir` | Output JSONL directory | `data` |
| `language` | chinese/english | `chinese` |
| `resize` | Enable image resizing | `true` |
| `val_ratio` | Validation split ratio | `0.1` |
| `max_teachers` | Max teacher samples | `10` |
| `response_types` | Fields to include | `object_type property` |

## Testing

```bash
# Run component tests
python data_conversion/test_pipeline.py

# Validate syntax
python -m py_compile data_conversion/processor.py
bash -n data_conversion/convert_dataset.sh
```

## Clean Architecture

The `data_conversion/` directory has been cleaned up and organized:

### Active Files (15 total)
```
data_conversion/
├── processor.py              # Main orchestrator
├── data_loader.py           # JSON/image loading
├── sample_processor.py      # Sample processing
├── teacher_selector.py      # Teacher selection
├── data_splitter.py         # Train/val splitting
├── core_modules.py          # Shared utilities
├── vision_process.py        # Image processing
├── convert_dataset.sh       # Entry point
├── simple_validate.py       # Validation
├── test_pipeline.py         # Testing utility
└── legacy_backup/           # Old files (backup)
```

### Migration from Old Pipeline
Old files have been moved to `legacy_backup/` folder:
- `convert_pure_json.py` → `sample_processor.py`
- `create_teacher_pool.py` → `teacher_selector.py`  
- `split_train_val.py` → `data_splitter.py`
- `extract_candidates.py` → No longer needed
- `qwen_converter_unified.py` → `processor.py`

Simply use the new `convert_dataset.sh` script for all data conversion needs.

## Error Handling

The new pipeline follows **fail-fast** principles:
- Validates all inputs upfront
- Surfaces errors immediately with clear messages
- No silent failures or data corruption
- Preserves data integrity throughout

## Visualization Tools

New visualization tools are available in `vis_tools/` for debugging and validation:

### Scaling Comparison Visualization
```bash
# Visualize specific image for debugging scaling
python vis_tools/vis_scaling_comparison.py \
    --input_file data/all_samples.jsonl \
    --image_id QC-20230308-0000619_66516 \
    --output_dir vis_output

# Visualize first 5 samples
python vis_tools/vis_scaling_comparison.py \
    --input_file data/all_samples.jsonl \
    --max_samples 5 \
    --output_dir vis_output
```

This tool creates side-by-side comparisons of original (`ds/`) vs scaled (`ds_rescaled/`) images with bounding boxes, perfect for validating the scaling process.

## Benefits

✅ **Simplified**: 1 command instead of 7 steps  
✅ **Faster**: Single-pass processing  
✅ **Reliable**: Fail-fast error handling  
✅ **Maintainable**: Modular, testable components  
✅ **Flexible**: Easy to customize and extend  
✅ **Consistent**: Follows your coding standards  
✅ **Debuggable**: Visualization tools for validation