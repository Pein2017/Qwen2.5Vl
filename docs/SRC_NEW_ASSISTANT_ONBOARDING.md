# src_new Essentials — Assistant Onboarding (10–15 min)

Purpose: Give you only what’s necessary to understand, run, and modify the production `src_new/` pipeline fast. For deep dives, see `SRC_NEW_REFERENCE.md`.

## What this system does
- Vision-language fine-tuning for BBU equipment detection
- Generates Chinese descriptions + coordinates (bbox/quad/line)
- Optionally uses coordinate tokens for sequence-based regression

## Architecture at a glance
- **Entry points**: training via `scripts/train_new.py`; inference via `src_new/inference.py`
- **Core flow**: JSONL → Conversation + Images → Tokenization → Model → Loss → Checkpoints
- **Key modules (read in this order)**:
  - `processing/conversation_processor.py` — builds training/inference conversations
  - `processing/token_processor.py` — coordinate token extension and embeddings
  - `models/wrapper.py` — `DetectionModel` around Qwen2.5-VL
  - `models/loss_manager.py` — teacher/student LLM loss + coordinate L1 loss
  - `training/bbu_trainer.py` — Trainer with local loss aggregation (fixes NCCL)
  - `utils/path_manager.py` — robust path resolution for images and pools
  - `inference.py` — training-matched data prep + generation and validations

## Minimal commands
- Training (recommended defaults):
```bash
python scripts/train_new.py --config bbu_v2
```
- Inference (training-matched):
```bash
python -m src_new.inference \
  --config_path configs/bbu_v2.yaml \
  --model_path checkpoints/best \
  --input_file data/val.jsonl \
  --output_file results/val.json \
  --data_root /abs/path/to/data_root
```
- Tests (if available in your env):
```bash
python -m pytest src_new/tests -q
```

## Data and conversations
- Input JSONL sample:
```json
{"images": ["path/to/img.jpg"], "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": "标签贴纸"}], "width": 532, "height": 728}
```
- Teacher–student format created by `ConversationProcessor` with HF chat templates. In teacher mode, the final assistant turn is truncated for generation during inference.

## Coordinate token system (when enabled)
- Token ID ranges (Qwen2.5-VL base vocab = 151,665):
  - Line tokens: 151,665–151,666 (`<|line_start|>`, `<|line_end|>`)
  - Coordinate tokens: 151,667–153,715 (`<|coord_0|>` … `<|coord_2048|>`, default max)
- Clamping: all coords clamped to `[0, max_coord_value]` before tokenization
- Embeddings: extended with positional-encoding initialization
- Geometry tags mapping is handled in `conversation_processor.py` and `coordinate_converter.py`

### Token name variants (important)
- Both `<|box_start|>/<|box_end|>/<|box_start|>/<|box_end|>` and `<|box_start|>/<|box_end|>` appear in the codebase/tests.
  - `processing/token_processor.py` primarily emits `<|box_start|>/<|box_end|>`/`<|box_start|>/<|box_end|>`
  - `processing/coordinate_converter.py` uses `<|box_start|>`/`<|box_end|>`
- The pipeline accepts either; prefer the module’s native tokens when extending/consuming outputs in that path.

## Losses and spans
- Teacher/student spans detected via tokenizer offset mapping
- Masking: system/user/prefix/image-pad tokens masked; assistant content unmasked
- Loss components (see `models/loss_manager.py`):
  - `teacher_llm_loss`, `student_llm_loss` — cross-entropy on spans
  - `teacher_l1_loss`, `student_l1_loss` — soft-expectation L1 for coordinate tokens

## Configuration keys that matter (see `config/config.py`)
- Model: `model_path`, `model_max_length`, `attn_implementation`, `torch_dtype`
- Training: `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`
- Coordinate: `coordinate_tokens_enabled`, `max_coord_value`, `coordinate_loss_weight`
- Teacher–student: `teacher_ratio`, `num_teacher_samples`, `teacher_loss_weight`, `student_loss_weight`
- Data: `train_data_path`, `val_data_path`, `teacher_pool_file`, `max_total_length`

## Inference rules of thumb (`src_new/inference.py`)
- Always load tokenizer/processor from the checkpoint; confirm extended vocab if using coord tokens
- Teacher guidance:
  - Provide `--teacher_pool_file`; default `num_teachers` becomes 1
  - Forces `batch_size=1`
- Use `PathManager` for `data_root` to resolve all image paths consistently
- Validation before generate(): image token count must align with image tensors

## Fast validation checklist
- Vocab size > 151,665 when coordinate tokens are enabled
- Conversation includes expected `<|image_pad|>` occurrences; tensors report same image count
- Both teacher and student spans exist when `teacher_ratio > 0`
- All loss components are finite during training
- Best checkpoint saved in SafeTensors; tokenizer saved with extended vocab

## Common pitfalls → quick fixes
- “Image features and image tokens do not match”
  - Ensure truncation keeps all teacher images and the student user’s images; only trim student assistant content
  - Confirm `<|image_pad|>` count matches `image_grid_thw` images
- Checkpoint saving error: `'Qwen2VLProcessor' has no get_vocab`
  - Set `trainer.processing_class = tokenizer`; save processor separately
- NCCL timeouts in distributed
  - Use `BBUTrainer` (local aggregation) — no custom dist ops

## Code landmarks (open these first)
- `src_new/inference.py`: `InferenceEngine` end-to-end, training-matched inputs
- `src_new/processing/conversation_processor.py`: build conversations and tensors
- `src_new/processing/token_processor.py`: vocab/model extension, ranges, init
- `src_new/models/wrapper.py`: `DetectionModel.from_pretrained(_fast)` loading
- `src_new/models/loss_manager.py`: span masks, optimized loss computation
- `src_new/training/bbu_trainer.py`: local aggregation, callbacks, best checkpoints
- `src_new/utils/path_manager.py`: unified path resolution

## Deep dive (single source of truth)
- See `SRC_NEW_REFERENCE.md` → `../src_new/UNIFIED_DOCUMENTATION.md` 