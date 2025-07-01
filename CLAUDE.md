# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen2.5-VL fine-tuning project for BBU (Base-Band Unit) equipment detection and captioning. The project implements end-to-end training of a vision-language model for dense object detection with natural language descriptions in both English and Chinese.

This implementation has evolved significantly beyond the standard Qwen2.5-VL fine-tuning approach, featuring a sophisticated multi-task training system with teacher-student learning and DETR-style object detection.

## Core Architecture

**Data Flow**: Raw vendor JSON → Preprocessed JSONL → Tensor batches → Multi-task training
- Raw data in `ds/` and `ds_rescaled/` (images + JSON annotations)
- Data conversion pipeline in `data_conversion/`
- Training pipeline in `src/` (custom implementation)
- Reference architecture in `qwen-vl-finetune/` (original baseline)
- Evaluation pipeline in `eval/`
- Documentation in `ongoing_task/`

**Key Components**:
- **Model**: Qwen2.5-VL-3B with custom detection head (DETR-style decoder)
- **Training**: Teacher-student learning with differential learning rates
- **Data**: Multi-image conversations with bounding box annotations
- **Loss**: Multi-task loss (language modeling + detection + captioning)
- **Configuration**: DirectConfig system with 149+ explicit parameters

## Common Development Commands

### Training
```bash
# Main training script
python scripts/train.py --config base_flat --log_level INFO --log_verbose true

# Multi-GPU training (via environment)
export BBU_DEEPSPEED_ENABLED=true
export BBU_DEEPSPEED_CONFIG=scripts/zero2.json
python scripts/train.py --config base_flat --log_level INFO --log_verbose true

# Full training launcher (handles environment setup)
bash scripts/run_train.sh
```

### Inference
```bash
# Single dataset inference
python src/inference.py --model_path output-630/630-filtered_sample-updated_prompt/checkpoint-150 --input_jsonl data/chinese-val.jsonl --output_file results.json

# Batch inference on multiple datasets
bash eval/infer_dataset.sh

# Teacher-guided inference
NUM_TEACHERS=2 bash eval/infer_dataset.sh
```

### Data Processing
```bash
# Convert raw data to training format
bash data_conversion/convert_dataset.sh

# Create teacher pool for demonstrations
python data_conversion/create_teacher_pool.py

# Split train/validation data
python data_conversion/split_train_val.py
```

### Configuration Validation
```bash
# Validate config without training
python scripts/train.py --config base_flat --validate-only

# Print current config
python scripts/train.py --config base_flat --print-config
```

## Project Structure

### Core Training (`src/`) - Custom Implementation
- `config/global_config.py` - DirectConfig system (149+ explicit parameters)
- `data.py` - BBUDataset with teacher-student sampling, packed/standard collators
- `chat_processor.py` - Unified conversation building and tokenization
- `models/wrapper.py` - Model wrapper preserving HF compatibility + detection integration
- `models/detection_head.py` - DETR-style decoder with dual-stream processing
- `detection_loss.py` - Hungarian matching + multi-task loss computation
- `training/trainer.py` - Custom trainer with component-wise loss tracking
- `inference.py` - Production-ready inference with Flash Attention 2
- `schema.py` - Comprehensive type system with runtime validation
- `response_parser.py` - Structured output parsing for detection results
- `teacher_pool.py` - Teacher demonstration management
- `logger_utils.py` - Advanced logging and monitoring utilities

### Reference Architecture (`qwen-vl-finetune/`) - Original Baseline
- `qwenvl/train/train_qwen.py` - Standard fine-tuning entry point
- `qwenvl/train/trainer.py` - Basic trainer with parameter freezing control
- `qwenvl/train/argument.py` - Nested dataclass configuration system
- `qwenvl/data/data_qwen.py` - Standard conversation data processing
- `qwenvl/data/data_qwen_packed.py` - Packed data processing for efficiency
- `tools/process_bbox.ipynb` - Bounding box format conversion utilities
- `scripts/` - Training launch scripts with hyperparameter documentation

### Data Pipeline (`data_conversion/`)
- `qwen_converter_unified.py` - Main data converter
- `convert_dataset.sh` - Full conversion pipeline
- `vision_process.py` - Image preprocessing and resizing
- `create_teacher_pool.py` - Teacher demonstration curation
- `core_modules.py` - Shared conversion utilities

### Evaluation (`eval/`)
- `infer_dataset.sh` - Batch inference runner
- `validate_results.py` - Result validation and statistics
- `run_evaluation.sh` - Full evaluation pipeline

### Configuration
- `configs/base_flat.yaml` - Main training configuration (149 parameters)
- All training parameters must be explicitly defined (no defaults)

## Architecture Comparison: Custom vs Reference

| Aspect | Current Implementation (src/) | Reference (qwen-vl-finetune/) |
|--------|-------------------------------|-------------------------------|
| **Purpose** | BBU equipment detection + captioning | General vision-language fine-tuning |
| **Configuration** | DirectConfig (149+ parameters, flat) | HfArgumentParser (nested dataclasses) |
| **Model Extension** | Detection head + teacher-student | Parameter freezing control |
| **Data Format** | Teacher-student + detection objects | Standard conversation format |
| **Loss Function** | Multi-task (LM + detection + captions) | Single language modeling loss |
| **Training Approach** | Unified multi-task learning | Component-wise training control |

## Key Configuration Parameters

The project uses a **DirectConfig system** where all parameters are explicitly defined in `configs/base_flat.yaml` (no hardcoded defaults):

### Model Settings
- `model_path`: Path to base Qwen2.5-VL model
- `model_max_length`: Maximum sequence length (default: 120000)
- `attn_implementation`: "flash_attention_2" (required for performance)

### Learning Rates (Differential Component Training)
- `vision_lr`: Vision encoder learning rate (5e-7)
- `merger_lr`: Vision-text merger learning rate (1e-5)  
- `llm_lr`: Language model learning rate (5e-6)
- `detection_lr`: Detection head learning rate (1e-5)
- `adapter_lr`: Adapter learning rate (5e-3)
- Automatic LR scaling based on effective batch size and collator type

### Data Configuration
- `train_data_path` / `val_data_path`: JSONL file paths
- `teacher_ratio`: Fraction of samples using teacher guidance (0.7)
- `collator_type`: "standard" (padding) or "packed" (no padding)
- `max_examples`: Maximum examples per batch
- `teacher_pool_path`: Path to teacher demonstration pool

### Detection Settings
- `detection_enabled`: Enable/disable detection head training
- `detection_num_queries`: Number of DETR object queries (100)
- Detection loss weights: `bbox_l1_weight`, `bbox_giou_weight`, `objectness_weight`, `caption_weight`
- `detection_hidden_dim`: Detection transformer hidden dimension
- `detection_num_layers`: Number of decoder layers in detection head

## Data Format

Training data uses teacher-student format in JSONL:
```json
{
  "teachers": [
    {
      "images": ["ds_rescaled/image.jpeg"],
      "objects": [
        {"bbox_2d": [x1, y1, x2, y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}
      ]
    }
  ],
  "student": {
    "images": ["ds_rescaled/image.jpeg"], 
    "objects": [
      {"bbox_2d": [x1, y1, x2, y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}
    ]
  }
}
```

## Environment Setup

The project requires:
- Conda environment: `ms`
- CUDA_VISIBLE_DEVICES for GPU selection
- HF_HOME for model cache (typically `/data4/swift/model_cache`)

## Monitoring and Debugging

### TensorBoard Metrics
- Loss components: `lm_loss`, `teacher_lm_loss`, `student_lm_loss`
- Detection: `bbox_l1_loss`, `bbox_giou_loss`, `objectness_loss`, `caption_loss`
- Parameter norms: `wn/*` (weights), `gn/*` (gradients)

### Key Features
- Fail-fast validation with torchtyping
- Multi-level logging system
- Teacher-student loss splitting for analysis
- Dynamic detection head freezing
- Packed collator for efficiency

## Documentation

Comprehensive documentation is available in `ongoing_task/`:
- Start with `complete_pipeline_guide.md` for end-to-end workflow
- `00_project_overview.md` for high-level architecture
- `02_data_pipeline_and_formats.md` for data flow details
- Individual numbered guides for specific aspects

## Testing and Validation

### Quick Tests
```bash
# Test forward pass only
python scripts/train.py --config base_flat --test_forward_pass true

# Validate config parameters
python scripts/train.py --config base_flat --validate-only

# Quick inference test
MAX_SAMPLES=5 bash eval/infer_dataset.sh
```

### Result Validation
```bash
# Validate inference results
python eval/validate_results.py results.json

# Check data format
python data_conversion/simple_validate.py data/chinese-train.jsonl
```

## Key Implementation Features

### 1. DirectConfig System
- **Flat parameter access**: No nested structures, all 149+ parameters in single namespace
- **No silent defaults**: Every parameter must be explicitly defined in YAML
- **Automatic scaling**: Learning rates scale with effective batch size and collator type
- **Type safety**: Comprehensive type conversion and validation from YAML

### 2. Teacher-Student Learning
- **Dynamic sampling**: Configurable ratio of samples using teacher guidance (default 70%)
- **Teacher pool**: Curated demonstrations loaded from separate file
- **Span tracking**: Token-level attribution for teacher vs student responses
- **Multi-image conversations**: Full conversation history with multiple images

### 3. DETR-Style Detection Head
- **Object queries**: Learnable detection tokens with cross-attention
- **Dual-stream processing**: Separate vision and language feature adaptation
- **Autoregressive captions**: Detection captions use shared vocabulary
- **Hungarian matching**: Optimal assignment between predictions and ground truth

### 4. Multi-Task Training
- **Combined loss**: Language modeling + bbox regression + caption generation
- **Component tracking**: Individual loss accumulation with proper averaging
- **Gradient flow**: Single backward pass through entire model
- **Numerical stability**: Float32 upcasting for loss computation

### 5. Advanced Data Processing
- **Packed collator**: Memory-efficient concatenation without padding
- **Multi-format support**: Teacher/student, examples/target, simple formats
- **Flash Attention ready**: Proper attention mask handling
- **Runtime validation**: torchtyping for tensor shape checking

## Important Notes

1. **Always use `predict_detection()`** for inference with bounding boxes
2. **No hardcoded defaults** - all configuration must be in YAML
3. **Teacher-student training** is enabled by default (70% of samples)
4. **Packed collator** requires careful batch size tuning for memory efficiency
5. **Detection head** can be disabled for language-only training
6. **Image preprocessing** is handled automatically via vision_process.py
7. **Flash Attention 2** is mandatory for inference (no fallbacks)
8. **Component-wise learning rates** enable fine-grained training control
9. **Runtime shape validation** with torchtyping provides fail-fast error detection
10. **HuggingFace compatibility** preserved through model wrapper pattern