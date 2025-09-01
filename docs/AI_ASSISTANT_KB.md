# AI Assistant Knowledge Base

Purpose: Minimal, canonical facts and pointers to operate, debug, and extend `src_new/` and `data_conversion/` without duplication. Content reflects exact src_new implementation.

## Canonical Deep Dives
- src_new deep dive: `../src_new/UNIFIED_DOCUMENTATION.md`
- Data conversion deep dive: `../data_conversion/README.md`

## Entrypoints & Commands
- Training: `python scripts/train_new.py --config bbu_v2`
- Inference (CLI):
  ```bash
python -m src_new.inference \
  --config_path configs/bbu_v2/base.yaml \
  --model_path checkpoints/best \
  --input_file data/val.jsonl \
  --output_file results/val.json \
  --data_root /abs/path
  ```
- Tests: `python -m pytest src_new/tests -q`

## Key Modules Map (read in this order)
- Conversations: `src_new/processing/conversation_processor.py`
- Token/vocab extension: `src_new/processing/token_processor.py`
- Model wrapper: `src_new/models/wrapper.py`
- Losses: `src_new/models/loss_manager.py` (+ `models/coordinate_loss.py`)
- Trainer: `src_new/training/bbu_trainer.py`
- Dataset: `src_new/data/dataset.py`, collators: `src_new/data/collator.py`
- Inference: `src_new/inference.py`
- Paths: `src_new/utils/path_manager.py`

## Data Schema (training JSONL)
- Required: `images: [str]`, `objects: [{bbox_2d|quad|square|line, desc}]`, `width`, `height`
- Geometry: `bbox_2d` [x1,y1,x2,y2]; `quad|square` 8 ints; `line` 2N ints (N≥2)
- Object categories: `bbu`, `bbu_shield`, `connect_point`, `label`, `fiber`(line), `wire`(line)

## Coordinate Token System
- Enabled by config `coordinate_tokens_enabled: true`. Default `max_coord_value` is 2048 (dataset and processors default to 2048 if unset).
- Tokens added:
  - Geometry: `<|line_start|>`, `<|line_end|>` (box/quad tokens already exist in base Qwen2.5‑VL)
  - Coordinate: `<|coord_0|>`…`<|coord_MAX|>` where MAX = `max_coord_value` (count = MAX+1)
- Runtime ID detection (no hardcoded range):
  - The wrapper’s `CoordinateProcessor.set_tokenizer(...)` scans `tokenizer.get_vocab()` for `"<|coord_"` tokens and sets `coordinate_token_range = (min_id, max_id+1)`; training/inference use this dynamic range.
- Embedding/vocab extension (ms‑swift optimizations):
  - Vocab extension via `tokenizer.add_special_tokens({additional_special_tokens: [...]})`
  - Embeddings are resized with `pad_to_multiple_of=128` and lazily initialized
  - Geometry embeddings: line tokens initialized from quad tokens
  - Coordinate embeddings: random normal with std ≈ `min(model.initializer_range, 0.02)`, only for zero rows; output layer mirrored when present

## Conversations & Image Tokens
- Template: HuggingFace `Qwen2VLProcessor.apply_chat_template(...)` with images list; placeholders `<|image_pad|>` are inserted by template (one per image user turn) before processor expansion.
- Pre-tokenization validation (ConversationProcessor): placeholder count in template text must equal the number of PIL images passed. If mismatch → `ImageTokenMismatchError`.
- Post-tokenization expectations (Model wrapper): expected image token count equals `sum_i (t_i*h_i*w_i) // (merge_size**2)` where `merge_size` is taken from `model.config.vision_config.spatial_merge_size` (fallback to training config if present). If mismatch → ValueError with counts.

## Spans, Labels, and EOS
- Dataset span creation (`Dataset._create_masked_labels_with_spans`):
  - Decode full text (no skipping specials) → re-tokenize with `return_offsets_mapping=True`
  - Regex `r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"` finds assistant content char spans; convert to token spans via offset mapping
  - Initialize all labels = -100, then unmask assistant spans
  - CRITICAL: Extend each assistant span to include the immediate `<|im_end|>` token when present (scans up to 5 tokens ahead) so the model learns proper termination
  - Mask all `<|image_pad|>` tokens back to -100
- Trainer validation: recomputes spans from labels by grouping unmasked regions; if the conversation has multiple assistant turns, all but the last are teacher spans, the last is the student span.

## Losses (Dual and Weighted)
- Single-pass CE optimization: per-token CE computed once; masked for teacher/student text tokens separately (coordinate positions excluded from CE when `coord_mask` provided)
- Coordinate L1: soft-expectation over coordinate logits on coordinate positions only, with `coordinate_loss_temperature` (default 1.0)
- Final weighted sum (all values below are the actual optimized scalars):
  - `teacher_llm = teacher_loss_weight * regular_loss_weight * CE_teacher_text`
  - `student_llm = student_loss_weight * regular_loss_weight * CE_student_text`
  - `teacher_l1 = teacher_loss_weight * coordinate_loss_weight * L1_teacher_coords`
  - `student_l1 = student_loss_weight * coordinate_loss_weight * L1_student_coords`
  - `total_loss = teacher_llm + student_llm + teacher_l1 + student_l1`
- Config defaults often used in tests: `coordinate_loss_weight=0.05`, `regular_loss_weight=1.0`, `teacher_loss_weight=0.3`, `student_loss_weight=1.0`, `coordinate_loss_temperature=1.0`.

## Collators & Tensor Shapes
- Collators: `StandardDataCollator` and `PackedDataCollator` return batches with:
  - `input_ids: [batch, seq_len]`, `attention_mask: [batch, seq_len]` (bool, contiguous), `labels: [batch, seq_len]`
  - If images present: expect Qwen2.5-VL flattened patches `pixel_values: [total_patches, patch_features]` and `image_grid_thw: [num_images, 3]` (t,h,w per image)
  - Teacher-student: collators flatten per-sample `[2,3]` image grids into stacked `[total_images,3]`
  - Pre-batch validations fail fast on shape/count mismatches

## Model Wrapper Behaviors
- `DetectionModel.config` proxies the underlying HF model config; training dataclass is kept on `training_config`
- `from_pretrained(...)` detects extended checkpoints via `AutoConfig.vocab_size`; if already extended and coord mode enabled, sets `skip_vocab_extension=True`
- SafeTensors support: prefers single or sharded `.safetensors` files for faster loads
- Coordinate masking at forward: logits in the coordinate ID slice are set to very negative values at non-coordinate positions; coordinate positions are detected via the dynamic `coordinate_token_range` and `get_coordinate_mask(input_ids)`
- Forward validates multimodal inputs (shapes and expected counts) and raises on mismatches before base model call

## Inference Specifics
- Environment tuning: sets `HF_MODULES_CACHE`, `HF_HOME`, `TOKENIZERS_PARALLELISM=false`, and, when using FlashAttention 2, `TRITON_CACHE_DIR`, etc.
- Checkpoint loading mirrors training: tokenizer/processor loaded from checkpoint; wrapper load with eager attention and `bfloat16` on GPU by default
- Teacher guidance:
  - If a teacher pool is found and `num_teachers` is 0, it defaults to 1 for parity with training and forces `batch_size=1`
  - Inputs are built via `ConversationProcessor.create_teacher_student_conversation_for_generation(...)` or the simple generation variant; DO NOT manually re-tokenize or truncate the text
- Generation EOS handling: uses a set of EOS token IDs `[<|im_end|>, tokenizer.eos_token_id, <|endoftext|>]` (present IDs only) to stop; after decoding, `<|im_end|>`, `<|endoftext|>`, `<|im_start|>` are stripped from the returned text (coordinate/geometry tokens are preserved)
- Post-processing: when coordinate mode is enabled, responses are parsed back to objects by inverting the training token format; otherwise attempts JSON parse first, then fallback heuristics

## Configuration Keys That Matter
- Model: `model_path`, `model_max_length`, `attn_implementation`, `torch_dtype`, `use_cache`
- Training: `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `gradient_checkpointing`, `bf16`, `max_total_length`, `collator_type` ("standard"|"packed")
- Coordinate: `coordinate_tokens_enabled`, `max_coord_value`, `coordinate_loss_weight`, `coordinate_loss_temperature`
- Teacher–student: `teacher_ratio` (default 0.5), `num_teacher_samples` (per sample), `teacher_loss_weight`, `student_loss_weight`
- Data: `train_data_path`, `val_data_path`, `teacher_pool_file`, `data_root`, `language`, `max_pixels`

## Data Conversion (object-oriented)
- Shell: `/data3/Qwen2.5-VL-main/data_conversion/convert_dataset.sh` with `OBJECT_TYPES` in {`bbu`,`bbu_shield`,`connect_point`,`label`,`fiber`,`wire`} or `full`
- Python: `data_conversion.unified_processor.UnifiedProcessor` with `DataConversionConfig`
- Outputs: `data/<dataset>/{train.jsonl,val.jsonl,teacher.jsonl,all_samples.jsonl,label_vocabulary.json,images/}`
- Normalization rules in inference utils: `'square' → 'quad'`, `'desc'` fallback from `'label'`; coordinates coerced to ints when possible

## Health Checks (quick)
- Coord mode: tokenizer vocab size > 151,665 and coordinate tokens present; dynamic `coordinate_token_range` detected
- Unmasked span coverage: teacher and student spans present when teacher mode active; `<|im_end|>` included in unmasked spans
- Losses: all components finite; `total_loss` equals the sum of weighted components in logs
- Multimodal: `pixel_values` rows == sum over `image_grid_thw` of (t*h*w); template placeholder count equals number of images; model-side expected tokens equals sum_i (t_i*h_i*w_i)//(merge_size**2)
- Checkpoints: SafeTensors present; tokenizer saved with expanded vocab and chat template; wrapper exposes HF `model.config`

## Where to Edit Common Tasks
- Change coord range: `src_new/processing/token_processor.py` (`TokenConfig.max_coord_value`)
- Adjust span regex/offset logic: `src_new/data/dataset.py` (`_create_masked_labels_with_spans`)
- Tune loss weights/temperature: config YAML → loaded by `src_new/config/config.py`
- Add geometry token behavior: `token_processor._initialize_geometry_tokens`
- Path resolution issues: `src_new/utils/path_manager.py` (used in dataset and inference)

## Error Messages You’ll See (and meanings)
- "Image placeholder count (...) != actual image count (...)" → template vs images mismatch (pre-processor)
- "Image token count mismatch: tokens=..., expected=..." → model-side check against THW and merge size
- "pixel_values rows (...) != sum(t*h*w) from image_grid_thw (...)" → collator/model validation failure
- "Coordinate tokens are enabled ... but none found" → extension failed; load/extend vocab first

## Minimal Pointers
- NCCL timeouts → use `BBUTrainer` (local aggregation; no custom dist ops)
- Checkpoint save error (`get_vocab`) → set `trainer.processing_class = tokenizer`; save processor separately
- Empty student spans → ensure student assistant content exists in training; in inference, use generation builders that omit student assistant content
