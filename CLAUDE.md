# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen2.5-VL fine-tuning project for BBU (Base-Band Unit) equipment detection and captioning. The project implements end-to-end training of a vision-language model for dense object detection with natural language descriptions in both English and Chinese.

This implementation has evolved significantly beyond standard Qwen2.5-VL fine-tuning, featuring:
- **Coordinate Token System**: Soft expectation regression with automatic bbox→token conversion
- **Multi-task Training**: Teacher-student learning with span-based loss splitting  
- **Modular Architecture**: Refactored from monolithic to component-based design
- **Enhanced Loss Management**: Mode-aware coordinate vs standard LLM loss computation

## Environment Setup

The project requires:
- Conda environment: `ms`
- CUDA_VISIBLE_DEVICES for GPU selection
- HF_HOME for model cache (typically `/data3/Qwen2.5-VL-main/model_cache`)

**Important Reminders:**
- We need to activate `ms` virtual environment, remember this.
- **NEW**: Directly use `/root/miniconda3/envs/ms/bin/python` to avoid conda activation inconsistencies
- Always use the full path to Python in the `ms` environment to ensure package consistency
- Located in China, cannot access foreign websites like `github` `google` `huggingface`

## Memory Log

### Data Migration and Annotation
- Migrating to v2 data annotation structure
- Reference documentation for migration:
  * `@docs/raw_data_v2.md`
  * `@docs/raw_data_template_数据堂.md`

[Rest of the file remains unchanged...]