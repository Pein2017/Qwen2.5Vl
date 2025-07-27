# Project Overview

## Purpose
Qwen2.5-VL fine-tuning project for BBU (Base-Band Unit) equipment detection and captioning. Implements end-to-end training of a vision-language model for dense object detection with natural language descriptions in both English and Chinese.

## Key Features
- **Coordinate Token System**: Soft expectation regression with automatic bbox↔token conversion
- **Multi-task Training**: Teacher-student learning with span-based loss splitting  
- **Modular Architecture**: Refactored from monolithic to component-based design
- **Enhanced Loss Management**: Mode-aware coordinate vs standard LLM loss computation
- **DETR-Style Detection**: Hungarian matching with dynamic loss weighting
- **Multi-Language Support**: Both English and Chinese annotations

## Advanced Capabilities
- **5-Stage Data Pipeline**: JSON cleaning → Token mapping → Sample processing → Validation → Summary
- **Multi-Format Support**: Handles both `dataList` and `markResult` JSON formats
- **Coordinate Transformation**: 3-stage system with EXIF orientation, dimension rescaling, and smart resize
- **Model Patches**: mRoPE integration, Flash Attention 2, and visual processing fixes
- **Production-Ready**: Fail-fast validation, unified configuration, robust inference

## Tech Evolution
This implementation has evolved significantly beyond standard Qwen2.5-VL fine-tuning, featuring sophisticated multi-geometry support and hierarchical training systems.