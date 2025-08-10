# Qwen2.5-VL BBU Detection Training Pipeline

**Production-ready training system for vision-language model fine-tuning with coordinate token support**

## 🎯 Quick Start

### Training
```bash
cd /data3/Qwen2.5-VL-main
bash scripts/run_new_train.sh
```

### Inference
```bash
python -m src_new.inference --config configs/bbu_v2.yaml --checkpoint path/to/checkpoint --image path/to/image.jpg
```

## 📚 Complete Documentation

**📖 See docs hub: `../docs/SRC_NEW_REFERENCE.md`**

This docs hub links to the single deep-dive source of truth and key guides:

- Input Data Format, Token Conversion Pipeline, Conversation Structure
- Span Detection & Loss Masking, Training Architecture
- Configuration & Setup, Troubleshooting

## 🏗️ System Overview

### Architecture Components
```
Raw Data (JSONL) → Coordinate Conversion → Conversation Templates →
Tokenization → Span Detection → Training → Model Checkpoints
```

### Key Features
- **Vision-Language Integration**: End-to-end dense detection → Chinese captions
- **Coordinate Token System**: Automatic bbox ↔ token conversion via soft regression
- **Multi-Task Training**: Teacher–student span-based loss splits
- **Modular Architecture**: Clean component boundaries, plug-and-play design
- **Loss Management**: Mode-aware switching between coordinate and LLM loss

## 🔧 Core Components

1. **Data Processing** (`src_new/data/`) - Dataset loading, teacher-student conversations, coordinate conversion
2. **Model Components** (`src_new/models/`) - Detection wrapper, dual-loss management, Qwen2.5-VL patches
3. **Training System** (`src_new/training/`) - BBUTrainer, checkpoint saving, distributed training
4. **Processing Pipeline** (`src_new/processing/`) - Templates, coordinate conversion, HuggingFace integration

## 📊 Training Performance

- **Model**: 7B parameters (Qwen2.5-VL base + coordinate tokens)
- **Speed**: ~2-3 samples/second on A100
- **Memory**: ~24GB VRAM for batch_size=1
- **Convergence**: 1000-2000 steps for fine-tuning

## 🧪 Testing

```bash
cd src_new/tests
python run_comprehensive_tests.py
```

## 🚀 Recent Improvements

- ✅ **Fixed Student Response Generation**: Complete teacher-student conversations
- ✅ **Resolved Checkpoint Saving**: Proper `processing_class` handling
- ✅ **Accurate Span Detection**: Offset mapping for token-level alignment
- ✅ **Unified Documentation**: Single authoritative reference
- ✅ **EOS Training Added**: `<|im_end|>` is now included in assistant span labels to teach proper termination
- ✅ **Vision token expansion validation fixed**: We now validate the number of `<|image_pad|>` tokens against the expected count computed from image grids and merge size, i.e. `expected_image_tokens = sum_i (t_i*h_i*w_i) // (merge_size**2)`. This matches the official Qwen2.5‑VL processor behavior and prevents spurious "Image token/grid mismatch" errors.
- ✅ **HF config exposure in wrapper**: `DetectionModel.config` now proxies the underlying HuggingFace model config (and keeps the training dataclass on `training_config`). This preserves integrations that call `model.config.to_json_string()` and similar APIs.

---

*For full deep dive, follow `../docs/SRC_NEW_REFERENCE.md` → `../src_new/UNIFIED_DOCUMENTATION.md`*
