# Qwen2.5-VL Multi-modal AI for Engineering Quality Inspection

This project implements an end-to-end AI quality inspection system using the Qwen2.5-VL multi-modal vision-language model. The system specializes in BBU (Base-Band Unit) equipment detection and captioning with sophisticated object detection and natural language descriptions.

## 🚀 Quick Start

### Prerequisites
- Conda environment: `ms`
- CUDA-compatible GPU
- Python 3.8+

### Environment Setup
```bash
# Activate the conda environment
conda activate ms

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your GPU setup
export HF_HOME=/data4/swift/model_cache  # Model cache directory
```

### Data Processing Pipeline
```bash
# Process your dataset (from raw JSON + images to training-ready JSONL)
bash data_conversion/convert_dataset.sh

# With custom settings
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" bash data_conversion/convert_dataset.sh
```

### Training
```bash
# Start training with your processed data
python -m src.training.trainer --config configs/base_flat_v2.yaml
```

### Inference
```bash
# Run inference on new images
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg
```

## 📋 Project Architecture

The project uses a modular architecture with clear separation of concerns:

```
src/
├── core/                      # Central factory classes and managers
│   ├── model_factory.py       # Centralized model creation and configuration
│   ├── data_processor.py      # Unified data processing and dataset creation  
│   └── checkpoint_manager.py  # Model saving/loading and checkpoint management
├── config/                    # Configuration system
│   ├── global_config.py       # Legacy DirectConfig system  
│   └── domain_configs.py      # Domain-specific configuration classes
├── training/                  # Modular training components
│   ├── trainer.py             # Main BBU trainer implementation
│   ├── training_coordinator.py # Training orchestration and state management
│   ├── loss_manager.py        # Multi-task loss computation (LM + detection)
│   ├── parameter_manager.py   # Parameter grouping for differential learning rates
│   └── callbacks.py           # Training callbacks and monitoring
├── models/                    # Model architecture and integration
│   ├── model_loader.py        # Unified model loader
│   ├── wrapper.py             # Qwen2.5-VL wrapper with coordinate tokens
│   └── patches.py             # Model patches and optimizations
├── utils/                     # Support utilities
│   ├── simple_token_manager.py # Simple token handling system (ms-swift approach)
│   ├── coordinate_token_manager.py # Coordinate token system
│   ├── prompt.py              # Prompt templates and conversation formatting
│   └── response_parser.py     # Output parsing and validation
├── data.py                    # BBUDataset with multi-geometry support
├── chat_processor.py          # Conversation building with token integration
├── teacher_pool.py            # Teacher demonstration management
└── inference.py               # Production inference with Flash Attention 2
```

## 🎯 Key Features

### Multi-modal Vision-Language Integration

- **Dynamic Resolution Processing**: Handles images of different sizes with absolute time encoding
- **Multi-Geometry Support**: Processes various geometric annotations including:
  - `bbox_2d`: Standard bounding boxes for BBU equipment and components
  - `square`: Four-point polygons for arbitrary shape annotations (e.g., labels)
  - `line`: Multi-point paths for cable/fiber annotations

### Advanced Training System

- **Teacher-Student Learning**: Uses demonstration examples to guide model learning
- **Multi-Task Training**: Combines language modeling and object detection objectives
- **Simple Token System**: Lightweight token addition using standard HuggingFace infrastructure
- **Coordinate Tokens**: Special tokens for handling geometric object information

### Data Processing Pipeline

- **Hierarchical Annotations**: Processes tree-like structure of object annotations
- **Conversation Format**: Converts annotations to natural conversation format
- **Fail-Fast Validation**: Comprehensive error detection with detailed messages
- **Multi-Format Support**: Processes both raw JSON and structured JSONL formats

### Optimized Inference Engine

- **Flash Attention 2**: Always-on optimization for efficient inference
- **KV Cache**: Optimized key-value caching for faster generation
- **Batch Processing**: Native support for multiple images in a batch
- **Teacher-Guided Inference**: Optional demonstration examples to guide generation

## 📚 Implementation Details

### Token System

The project implements two complementary token systems:

```python
# Simple Token Approach (ms-swift inspired)
"<|box_start|>100, 200, 300, 400<|box_end|> <|object_ref_start|>BBU设备<|object_ref_end|>"
"<|square_start|>150, 10, 211, 35, 218, 16, 166, 0<|square_end|> <|object_ref_start|>标签<|object_ref_end|>"
"<|line_start|>579, 1385, 679, 1451, 764, 1444<|line_end|> <|object_ref_start|>光纤<|object_ref_end|>"
```

### Data Format

The system uses a rich multi-geometry format for representing objects:

```json
{
  "images": ["images/QC-20230217-0000279_19621.jpeg"],
  "objects": [
    {"bbox_2d": [264, 144, 326, 201], "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求"},
    {"square": [704, 487, 670, 554, 973, 644, 993, 590], "desc": "标签/4G-RRU3-光纤"},
    {"line": [614, 1271, 498, 1179, 419, 1216, 280, 1280, 117, 1456, 3, 1721], "desc": "光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管"}
  ],
  "width": 532,
  "height": 728
}
```

### Object Types Supported

The system recognizes the following equipment types:

- **Equipment**: `bbu` (BBU设备), `bbu_shield` (挡风板)
- **Hardware**: `connect_point` (螺丝、光纤插头), `label` (标签)
- **Cables**: `fiber` (光纤), `wire` (电线)

### Model Loading

Unified model loading ensures training-inference consistency:

```python
from src.models.model_loader import load_model_and_processor_unified

model, tokenizer, image_processor = load_model_and_processor_unified(
    config,
    for_inference=False,
    deepspeed_enabled=True
)
```

## 🛠️ Development

### Working with the Codebase

The project follows these development principles:

1. **Fail-Fast Validation**: Explicit error reporting rather than silent fallbacks
2. **Modular Components**: Clear separation of concerns with focused responsibilities
3. **Unified Configuration**: Type-safe configuration with explicit parameters
4. **Training-Inference Consistency**: Same model loading and processing pipeline

### Testing

```bash
# Run the test suite
python -m pytest eval/test_all_evaluations.py

# Validate pipeline output
python data_conversion/simple_validate.py
```

### Memory Optimization

The system implements several memory optimization techniques:

- **Packed Data Collator**: Removes padding for efficient memory usage
- **Flash Attention**: High-speed attention kernel for improved throughput
- **Gradient Checkpointing**: Trades computation for memory efficiency

## 🚨 Important Notes

### Environment Requirements
- **Always use the `ms` conda environment** before running any scripts
- **Note that the project is located in China** and cannot access foreign websites like GitHub, Google, or HuggingFace
- **Configure CUDA devices** appropriately for your GPU setup

### Model Capabilities
- **Multi-Geometry Detection**: Specialized for equipment, components, and cables
- **Quality Assessment**: Automatically identifies installation issues
- **Hierarchical Descriptions**: Structured output format for integration with quality systems

## 📊 Performance

### Inspection Capabilities
- **Component Detection**: High accuracy identification of BBU components
- **Quality Assessment**: Automated verification of installation standards
- **Documentation Match**: Validation of labels against installation documentation

### Resource Requirements
- **GPU Memory**: 24GB+ recommended for training (depends on batch size)
- **CPU**: Multi-core recommended for data processing
- **Storage**: Fast SSD recommended for training data

## 🤝 Contributing

1. Review the existing code structure before making changes
2. Follow the fail-fast development philosophy
3. Prefer refactoring over file duplication
4. Keep code DRY, transparent, and consistent
5. Update documentation for any changes

## 📄 License

This project is for internal use and research purposes.

## 🙏 Acknowledgments

Built on top of the Qwen2.5-VL foundation model by Alibaba Cloud, with significant enhancements for BBU equipment detection and multi-task learning.

---

For detailed documentation, see the [docs/](docs/) directory. For quick operational procedures, check the [runbook](docs/runbook.md).