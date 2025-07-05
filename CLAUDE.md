# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen2.5-VL fine-tuning project for BBU (Base-Band Unit) equipment detection and captioning. The project implements end-to-end training of a vision-language model for dense object detection with natural language descriptions in both English and Chinese.

This implementation has evolved significantly beyond the standard Qwen2.5-VL fine-tuning approach, featuring a sophisticated multi-task training system with teacher-student learning and DETR-style object detection.

## Environment Setup

The project requires:
- Conda environment: `ms`
- CUDA_VISIBLE_DEVICES for GPU selection
- HF_HOME for model cache (typically `/data4/swift/model_cache`)

**Important Reminders:**
- We need to activate `ms` virtual environment, remember this.

## Enhanced Data Processing Pipeline

### Quick Start
```bash
conda activate ms
bash data_conversion/convert_dataset.sh
```

### Pipeline Overview
The data conversion pipeline has been **enhanced** with improved JSON cleaning and path management:

```
ds/ (raw JSON + images) → convert_dataset.sh → ① clean_raw_json.py (NEW)
                                             → ② copy images to ds_output
                                             → ③ unified processor.py 
                                             → ④ smart resize + bbox scaling
                                             → data/ (train.jsonl, val.jsonl, teacher.jsonl)
```

### Key Improvements
1. **JSON Cleaning**: `clean_raw_json.py` strips unnecessary metadata while preserving essential structure
2. **Unified Output**: All processed data (JSON + images) goes to `ds_output/` directory  
3. **Path Consistency**: JSONL files reference correct `ds_output/` paths
4. **Enhanced Validation**: Improved bbox scaling accuracy and error checking
5. **Environment Integration**: Automatic conda environment activation

### Configuration Options
```bash
# Custom directories
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" bash data_conversion/convert_dataset.sh

# Language selection (chinese/english)
LANGUAGE="english" bash data_conversion/convert_dataset.sh

# Disable resizing for testing
RESIZE="false" bash data_conversion/convert_dataset.sh
```

### Troubleshooting
- **'list' object has no attribute 'get'**: Run JSON cleaning first
- **FileNotFoundError for images**: Check image copying step
- **Bbox out of bounds**: Verify smart_resize parameters

For detailed documentation, see `docs/data_schema.md` and `docs/runbook.md`.

## Code Maintenance Notes
- Don't need the `legacy` or `backward compatibility`, just override the scripts and keep codebase clean.

## Development Guidelines
- Don't need any CLI as well, define everything we need in the bash script or python script at top