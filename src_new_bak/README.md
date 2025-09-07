# Qwen2.5-VL BBU Detection Training Pipeline

**Production-ready training system for vision-language model fine-tuning with coordinate token support**

## 🎯 Quick Start

### Training
```bash
bash scripts/run_new_train.sh
```

### Inference
```bash
source ~/.bashrc && conda activate ms
python -m src_new.inference --config /abs/path/to/config.yaml --checkpoint /abs/path/to/checkpoint --image /abs/path/to/image.jpg
```

## 📚 Complete Documentation

**📖 See docs hub: `../docs/SRC_NEW_REFERENCE.md` and `src_new/UNIFIED_DOCUMENTATION.md`**

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

## ✅ Coordinate Loss Status

Coordinate tokens map to values 0..max_coord and are trained using auxiliary coordinate losses. See `src_new/models/coordinate_loss.py` and `UNIFIED_DOCUMENTATION.md` for details.

### Auxiliary Coordinate Losses (always enabled)
- Kernelized‑KL (sparse window) around the correct coordinate bin
- Unlikelihood on non‑coordinate tokens at coordinate positions (top‑K)

Behavior:
- CE path is unchanged and continues to train all assistant tokens.
- LossManager reports and uses separate weighted components:
  - `teacher_kce_loss`, `teacher_unlike_loss`, `student_kce_loss`, `student_unlike_loss`
- These replace the legacy soft-expectation L1 loss which has been removed.

YAML keys (see `configs/bbu_v2/coord_aux.yaml`):
```
coord_aux_enabled: true  # Must be true (auxiliary losses are always enabled)
coord_aux_tau: 1.2
coord_aux_sigma_bins: 8
coord_aux_window_bins: 32
coord_aux_topk: 100
coord_aux_lambda_kce: 1
coord_aux_lambda_unlike: 1
```

Run training with auxiliary coordinate losses:
```bash
bash scripts/run_debug.sh
```

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
- ✅ **Vision token expansion validation**: We validate the number of `<|image_pad|>` tokens against the expected count computed from image grids and merge size, i.e. `expected_image_tokens = sum_i (t_i*h_i*w_i) // (merge_size**2)` to match the official Qwen2.5‑VL processor behavior.
- ✅ **HF config exposure in wrapper**: `DetectionModel.config` now proxies the underlying HuggingFace model config (and keeps the training dataclass on `training_config`). This preserves integrations that call `model.config.to_json_string()` and similar APIs.
- ✅ **Joint Training Loss**: Cross-Entropy loss now covers all assistant tokens (text + coordinate tokens) while L1 loss remains exclusively for coordinate tokens, enabling joint learning of language and coordinate prediction.
- ✅ **Aux Coordinate Losses**: Kernelized‑KL + Unlikelihood available via YAML; Laplacian regularizer removed.

---

*For full deep dive, follow `../docs/SRC_NEW_REFERENCE.md` → `../src_new/UNIFIED_DOCUMENTATION.md`*
