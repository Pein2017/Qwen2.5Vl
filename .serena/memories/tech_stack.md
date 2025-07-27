# Technology Stack

## Core Framework
- **Base Model**: Qwen2.5-VL-3B-Instruct (Vision-Language Model)
- **Framework**: PyTorch 2.5.1 with transformers 4.51.3
- **Training**: HuggingFace Transformers + custom BBUTrainer
- **Optimization**: DeepSpeed for multi-GPU training
- **Attention**: Flash Attention 2 for efficiency

## Key Dependencies
- **Vision**: OpenCV (opencv-python 4.11.0.86), Pillow 11.1.0
- **Data Processing**: pandas 2.2.3, datasets 3.2.0, jsonlines 4.0.0
- **ML/DL**: torch 2.5.1, transformers 4.51.3, accelerate 1.6.0
- **Chinese Processing**: jieba 0.42.1, pypinyin 0.54.0
- **Evaluation**: rouge-chinese 1.0.3, sentence-transformers 4.1.0
- **Configuration**: pyyaml 6.0.2, omegaconf 2.0.0

## Development Tools
- **Code Quality**: ruff 0.11.2 (linting + formatting)
- **Environment**: Conda environment `ms`
- **Logging**: Custom rank-aware logging system
- **Validation**: Custom validation scripts

## Hardware Requirements
- **GPU**: CUDA-compatible with 24GB+ memory recommended
- **Multi-GPU**: Supported via DeepSpeed and torchrun
- **Storage**: Fast SSD recommended for training data

## Chinese Development Environment
- Located in China, cannot access foreign websites (github, google, huggingface)
- Uses local mirrors and cached resources
- Model cache: `/data3/Qwen2.5-VL-main/model_cache`