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

## ♻️ Refactor Notes (src_new vs `src_new_bak`)
- **HuggingFace-first pipeline**: `ConversationProcessor` replaces custom chat builders; image/token tensors come from the official processor.
- **Strict config**: Single frozen dataclass `src_new/config/config.py` with fail-fast validation and path normalization (supports relative paths and `@src_new` alias).
- **Dynamic coord tokens**: No hard-coded ID ranges; derive via `get_coord_token_range(tokenizer)`; tokenizer/model expanded before DDP.
- **Losses**: `LossManager` uses single-pass CE; coordinate auxiliary losses (Kernelized-KL + Unlikelihood) are optional via YAML.
- **Trainer**: `BBUTrainer` uses local aggregation, no custom distributed ops; `TrainingStateManager` handles metrics; unified checkpoint manager.
- **Utilities**: Centralized tensor validation, rank-aware logging, path/data resolvers, and debug logging.

## 🧭 Roadmap & Training Procedure (high level)
1) Config: author a YAML under `configs/` (relative paths and `@src_new/` alias are accepted and normalized); load via `src_new.config.load_config`.
2) Load base model/tokenizer; extend vocabulary and embeddings using `TokenProcessor` (pre-distributed phase).
3) Dataset: `src_new/data/dataset.py` builds conversations via `ConversationBuilder` with the official processor; labels/spans via offset mapping.
4) Collation: choose `collator_packed` or `collator_standard` via `collator_type`.
5) Model: wrap with `DetectionModel`; enable coord aux via YAML (`coord_aux_enabled: true`) if desired.
6) Trainer: use `training/bbu_trainer.py` with local metrics; checkpoints saved via `training/checkpoint_saver.py` (`CheckpointSaver` + `BestCheckpointManager`).
7) Inference: see `src_new/inference.py` and use `ConversationBuilder.*for_generation` builders; strict parsing of outputs.

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

### Format modes & variants (standardized)
- Format mode (exclusive):
  - `special_tokens` (default when `coordinate_tokens_enabled=false`)
  - `coord_tokens` (when `coordinate_tokens_enabled=true`)
- Conversation variants:
  - `dense_caption`, `coords_to_desc`, `desc_to_coords`, `summary`
- Variant is sampled in the dataset, attached to each sample as `conversation_variant`, propagated by collators, and consumed by the loss path for strict grouping policy.

## ✅ Coordinate Loss Status

Coordinate tokens map to values 0..max_coord and can be trained with optional auxiliary coordinate losses. See `src_new/losses/coord_aux.py` and `UNIFIED_DOCUMENTATION.md` for details.

### Auxiliary Coordinate Losses (optional, enable via YAML)
- Kernelized‑KL (sparse window) around the correct coordinate bin
- Unlikelihood on non‑coordinate tokens at coordinate positions (top‑K)

Behavior:
- CE path is unchanged and continues to train all assistant tokens.
- LossManager reports and uses separate weighted components when enabled:
  - `teacher_kce_loss`, `teacher_unlike_loss`, `student_kce_loss`, `student_unlike_loss`
- Legacy L1 has been retired in favor of auxiliary losses.

YAML keys (see `configs/bbu_v2/coord_aux.yaml`):
```
coord_aux_enabled: true
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
- ✅ **SOLUTION‑1 CE Path**: Single‑pass CE computed once and reused for teacher/student losses; covers all assistant tokens (text + coordinate tokens).
- ✅ **Aux Coordinate Losses**: Kernelized‑KL + Unlikelihood available via YAML; Laplacian regularizer removed; coordinate components reported separately (`*_kce`, `*_unlike`).
- ✅ **Standardized format modes & variants**: Enforced single active format mode (special/plain/coord) and propagated `conversation_variant` to loss path for strict grouping.
- ✅ **Grouping core extraction**: Centralized grouping utilities (`losses/grouping_core.py`); grouping behavior driven by variant.

---

*For full deep dive, follow `../src_new/UNIFIED_DOCUMENTATION.md`
