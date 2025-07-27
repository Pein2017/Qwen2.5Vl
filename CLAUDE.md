# CLAUDE.md

## Project Overview

Qwen2.5-VL fine-tuning project for BBU (Base-Band Unit) equipment detection and captioning with the following key features:

- **Multi-modal Vision-Language Integration**: End-to-end training for dense object detection with natural language descriptions in English/Chinese
- **Coordinate Token System**: Soft expectation regression with automatic bbox→token conversion
- **Multi-task Training**: Teacher-student learning with span-based loss splitting
- **Modular Architecture**: Component-based design with separation of concerns
- **Enhanced Loss Management**: Mode-aware coordinate vs standard LLM loss computation

## Environment Requirements

- **Conda Environment**: `ms` (MUST be activated before running any scripts)
- **Python Path**: Always use `/root/miniconda3/envs/ms/bin/python` directly
- **Environment Variables**:
  - `CUDA_VISIBLE_DEVICES` for GPU selection
  - `HF_HOME` for model cache (typically `/data3/Qwen2.5-VL-main/model_cache`)
- **Network Constraints**: Located in China, cannot access foreign websites (GitHub, Google, HuggingFace)

## Key Documentation

- **Architecture**: `docs/ARCHITECTURE.md` for system overview
- **Mental Model**: `docs/MENTAL_MODEL.md` for high-level understanding
- **Project Map**: `docs/PROJECT_MAP.md` for codebase navigation
- **Data Migration**: `docs/raw_data_v2.md`, `docs/raw_data_template_数据堂.md`

## Development Guidelines

- **Error Handling**: Don't use `try-except` unless necessary; expose errors immediately
- **Python Execution**: Always use full path to Python in `ms` environment
- **Code Organization**: Follow modular architecture with clear component responsibilities
- **Configuration**: Use typed, explicit configuration with validation

## Entry Points

- **Training**: `python scripts/train.py --config configs/base_flat_v2.yaml`
- **Data Processing**: `bash data_conversion/convert_dataset.sh`
- **Inference**: `python src/inference.py --model_path /path/to/model --image_path /path/to/image`