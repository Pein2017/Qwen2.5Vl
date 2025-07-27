# Product Overview

## Qwen2.5-VL BBU Fine-tuning System

This is a sophisticated computer vision and natural language processing system for BBU (Base-Band Unit) equipment detection and captioning. The system combines:

- **Multi-modal AI**: Fine-tuned Qwen2.5-VL model for visual understanding and text generation
- **Object Detection**: DETR-style detection with Hungarian matching for equipment localization
- **Multi-language Support**: English and Chinese annotation capabilities
- **Teacher-Student Learning**: Advanced training methodology for improved model performance

## Key Capabilities

### Core AI Features
- **Equipment Detection**: Automated detection and classification of BBU equipment in images
- **Multi-task Learning**: Simultaneous visual language modeling and object detection
- **Coordinate Regression**: Precise bounding box prediction with soft expectation methods
- **Multi-language Support**: Native English and Chinese processing with token mapping

### Production Pipeline
- **End-to-end Processing**: Complete pipeline from raw JSON annotations to trained models
- **5-Stage Data Pipeline**: JSON cleaning → Token mapping → Sample processing → Validation → Summary
- **Teacher-Student Learning**: Advanced training methodology with intelligent teacher pool creation
- **Batch Inference**: Scalable inference engine for production workloads

### Advanced Training Features
- **DETR-Style Detection**: Hungarian matching with dynamic loss weighting
- **Model Patches**: mRoPE integration, Flash Attention 2, and visual processing optimizations
- **Multi-GPU Support**: Distributed training with gradient accumulation
- **Stability Enhancements**: Gradient clipping and training stability utilities

### Evaluation & Analysis
- **Comprehensive Metrics**: COCO-style detection metrics and ROUGE text evaluation
- **Visualization Tools**: Training progress, scaling comparisons, and sample analysis
- **Experiment Comparison**: Tools for comparing different training runs and configurations

## Target Use Cases

- Industrial equipment monitoring and documentation
- Automated quality control and inspection
- Technical documentation generation
- Multi-language equipment cataloging