## Architecture & Training Snapshot (2025-10)

### Core Principles
- **HF‑first reasoning**: Typed messages → `Qwen2_5_VLProcessor.apply_chat_template()` → official processor tensors. Never hand‑craft `<|image_pad|>`.
- **Fail‑fast validation**: Shapes, token counts, multimodal alignment, special tokens, and paths are all validated before forward/generate.
- **Single‑pass loss**: Cross‑entropy computed once; teacher/student and grouped masks (caption/grounding/formatting) applied afterward.
- **Configuration‑driven**: Strict YAML schemas with inheritance; no in‑code defaults. Missing keys surface as errors.
- **Separation of concerns**: Processing, conversation building, model wrapper, losses, training, inference, RL, and conversion are independent with shared contracts.


### End‑to‑End Dataflow
- **SFT**: JSONL → ConversationBuilder (typed) → HF processor tensors → DetectionModel (validations) → single‑pass CE + grouped masks → SafeTensors + tokenizer/processor.
- **RL (dense, `src_new`)**: Single‑turn dataset → ConversationBuilder (typed) → K generations → rewards (format + geometry) → GRPO update → periodic saves.
- **Data conversion**: Raw V2 JSON + images → canonical geometry + hierarchical CN descriptions → smart‑resized images + strict JSONL (train/val/teacher_pool/all_samples) + validation reports.


### Modules (`src_new`)
- `config/`: strict dataclasses; layered YAML with validation; no in‑code defaults.
- `processing/`: conversation builder and variants; span extraction/mapping; geometry text and parsing; chat templates; special tokens.
- `data/`: dataset loader with augmentation and dynamic teacher pairing; collators; assistant‑span masking and `<|image_pad|>` masking.
- `models/`: `DetectionModel` wrapper with multimodal validations; Qwen2.5‑VL patches; loss manager integration.
- `losses/`: grouped‑token logic in `token_grouping.py` (caption/grounding/formatting).
- `training/`: `BBUTrainer`, `PhaseFreezeManager`, `TrainingStateManager`, `CheckpointSaver`, callbacks and metrics.
- `inference.py`: training‑matched engine with strict image token/tensor validation and geometry‑first parsing.
- `rl/`: GRPO runner/trainer, dataset regeneration via `ConversationBuilder`, reward registry with formatting + detection rewards.
- `utils/`: HF component loader, tensor validators, path/data resolvers, rank‑aware logging, error formatting, seeding.


### Processing Stack (`src_new/processing`)
- **ConversationBuilder**: Builds simple, teacher‑student, summary, and text‑only conversations. Ensures image placeholder count equals provided images. Attaches `conversation_text` and `offset_mapping`.
- **ConversationProcessor**: Applies HF chat template; validates `<|image_pad|>` vs `image_grid_thw`; maps typed images via processor.
- **Variants**:
  - DenseCaption: image‑only user; assistant outputs object+geometry lines.
  - CoordToDesc / DescToCoord: bidirectional geometry↔description transformations.
  - TextOnly: rewrites per‑object JSON into canonical wrapper lines; injects a cached 28×28 black image so vision runs identically; optional bounded jitter/shuffle to rehearse tolerant parsing.
  - Summary: one‑line Chinese per image; no geometry/special tokens (for RL Stage‑A stability).
- **Spans**: `span_extraction` + `span_mapping` + `span_builder` compute assistant spans at char level and align to tokens via `offset_mapping`. Spans include the immediate `<|im_end|>`. Spans are mapped to expanded `input_ids` if HF expansion differs.
- **Special tokens**: Core + geometry wrappers validated at startup. Coordinate tokens are deprecated (numeric geometry only).
- **Geometry**: `geometry_text.py` formatting; `parse_generated.py` geometry‑first parsing, then tolerant, then JSON fallback; deduplication.


### Data Layer (`src_new/data`)
- **Dataset & Labels**: Loads strict JSONL; constructs conversations and masked labels from precomputed spans. Labels outside spans = -100; `<|image_pad|>` always masked.
- **Dynamic contrastive pairing**: Bucket by object‑type majority; optional cross‑bucket exploration. Episode map injects ≤1 teacher before student; only final assistant (student) contributes to gradients. Eval/inference are single‑turn (pairing disabled).
- **Collator**: Standard only; pads text; emits `pixel_values` + `image_grid_thw`; validates THW vs packed patch rows. Packed mode disabled in `src_new`.
- **Augmentation (`src_new/augmentation`)**: Smart resize (factor=28) → geometry‑aware transforms (with coordinate tracking) → photometric ops → line ops → validators for bounds/sanity. Preserves canonical ordering (quads TL→TR→BR→BL; lines leftmost‑first). Teacher augmentation configurable.


### Model & Loss (`src_new/models`, `src_new/losses`)
- **DetectionModel**: Wraps Qwen2.5‑VL; validates:
  - Decoded `<|image_pad|>` count equals `(∑ t*h*w) // merge_size²` from `image_grid_thw`.
  - `pixel_values` rows equal `∑(t*h*w)`; shapes consistent.
  - Assistant spans provided → bypass base loss; single‑pass CE path.
  - Mismatched placeholders raise `ImageTokenMismatchError` (`processing/conversation_processor.py`).
- **Patches** (`models/patches.py`) unify fixes for rotary position embedding, attention implementation, and `prepare_inputs_for_generation` across train/infer/RL.
- **Loss & Grouped tokens** (`losses/token_grouping.py`, `models/loss_manager.py`):
  - CE is computed once on shifted tensors.
  - Masks per role (teacher/student) and per group with strict disjointness and full coverage of the shifted assistant mask:
    - caption: inside `<|object_ref_start|>...<|object_ref_end|>`, excluding punctuation/geometry.
    - grounding: numeric tokens strictly inside geometry spans.
    - formatting: geometry/object‑ref wrappers, punctuation, separators, residuals (incl. `<|im_end|>`).
  - Config requires nonnegative weights: `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight`, plus `teacher_loss_weight`/`student_loss_weight`.
  - Metrics: per‑role/group CE means and token counts; optional aggregates.


### Training Workflow (`src_new/training`)
- **Initialization**: Load YAML via `load_config` (layered inheritance; strict schema). Build HF components once (tokenizer, image processor, model, dtype/attn, device map). Apply Qwen2.5‑VL patches. Validate special tokens and geometry wrappers; enforce dtype/attention contracts.
- **Phases** (`PhaseFreezeManager`):
  - Phase‑1: train `visual.merger` (optionally patch embed); LLM frozen.
  - Phase‑2: unfreeze top‑K LLM blocks; vision mostly frozen; merger trainable.
  - Phase‑3: unfreeze all by default (keep `visual.patch_embed` frozen unless configured).
- **Loop**: `dataset.set_epoch(epoch)` → batch → multimodal validation → `DetectionModel.forward()` → single‑pass CE → teacher/student + grouped weights → metrics via `TrainingStateManager` (no custom distributed ops).
- **Checkpointing** (`CheckpointSaver`): Saves SafeTensors + tokenizer/processor + generation config; rotates by `save_total_limit`. Best checkpoints copied atomically to `best-*`. Ensures auxiliary files; persists minimal `preprocessor_config.json` when needed.
- **Parameter groups**: Vision, merger, LLM, and coord_slice groups are detected by parameter names; learning rates are surfaced with clear labels for logging.


### Inference (training‑matched)
- Same builder and HF processor; strict pre‑generation checks: `input_ids`/`attention_mask` required. If `pixel_values`/`image_grid_thw` present, verify image tokens vs THW; forbid image tensors with zero `<|image_pad|>`.
- EOS is `<|im_end|>` only; decode only newly generated tokens; trim at the first `<|im_end|>`.
- Parsing priority: geometry‑first (strict → tolerant) → JSON list → fallback converter; deduplicate; keep raw integers only.


### RL Post‑Training (dense GRPO, `src_new/rl`)
- **Dataset**: Single‑turn dense JSONL reconstructed via `ConversationBuilder`; vision tensors preserved end‑to‑end.
- **Trainer**: Per‑sample generation with K completions; reward registry combines formatting (`parse`, `wrappers`, `coords`, `separators`, `vocab`, `length`, `length_window`) and detection/geometry (`bbox_giou`, `quad_l1`, `line_l1`, `ordering`, `coverage`, `geometry_sanity`). Optional KL via `beta`.
- **Vision**: Packed 2D `pixel_values` and `image_grid_thw` with strict slice offsets; eager attention preferred; bf16 recommended.
- **Contracts**: Completion masks stop at first `<|im_end|>`; ensure divisibility of `sample_k` across micro‑batches/world size/update steps. `src_post` is a separate RL project (group‑level QC); do not mix with `src_new` dense RL.


### Configuration & Schema (`src_new/config`)
- Strict dataclasses with YAML layering and validation. Required keys for model/runtime, data paths, features (merge size, max pixels), loss weights, collator (standard only), logging, and checkpointing.
- Auto‑resolution: derive `train/val/teacher_pool` paths from `data_root` when null.
- No in‑code defaults; explicit `torch_dtype`/attention implementation enforced. bf16‑only policy: `model.torch_dtype=bfloat16`, `training.bf16=true`, `training.fp16=false`.


### Data Conversion Pipeline (`data_conversion`)
- **Purpose**: Convert V2 annotations + images into training‑ready JSONL with canonical geometry and hierarchical CN descriptions; produce `train.jsonl`, `val.jsonl`, `teacher_pool.jsonl`, `all_samples.jsonl`, processed images, label vocabulary, and validation reports.
- **Strict pipeline (order)**: 1) EXIF orientation correction → 2) Rescale if JSON dims differ from actual image dims → 3) Smart resize (grid factor=28, pixel budget bounds).
- **Geometry**: Formats: `bbox_2d` (4 ints), `quad` (8 ints TL→TR→BR→BL), `line` (even length ≥4, leftmost‑first). Clamp to bounds; canonicalize quads/lines; filter degenerates.
- **Descriptions (Chinese)**: One object per line; commas within levels, slashes across levels; conditional segments appear only when parents demand; final optional free‑text “remarks” at last level only. Remove occlusion tokens; normalize `标签/*` to `标签/无法识别` when empty‑like.
- **Teacher Pool**: Greedy coverage over fixed vocabulary (fallback to free vocab) with determinism; brand/geometry balancing; stats exported.
- **Outputs**: Strict flat JSONL per sample: `images: [str]`, `objects: [dict]`, `width`, `height`; optional `summary` string for SFT summary variant.
- **Implementation points**: `coordinate_manager.py`, `vision_process.py`, `summary_builder.py`, `teacher_selector.py`.


### Critical Contracts & Health Checks (runtime‑enforced)
- **Chat template & images**: Typed images in user content; placeholder count equals image count; mismatches raise `ImageTokenMismatchError`.
- **Multimodal alignment**: `<|image_pad|>` count equals expected tokens from `image_grid_thw`. `pixel_values` rows equal packed patch sums. `image_grid_thw` must be `[num_images, 3]`. Centralized checks: `src_new/utils/tensor_validation.py`, `src_new/models/wrapper.py`.
- **Spans & labels**: Assistant spans token‑aligned and include `<|im_end|>`; labels outside spans set to -100; `<|image_pad|>` always masked.
- **Grouping invariants**: Caption/grounding/formatting masks are disjoint; union equals shifted assistant mask per role.
- **Geometry tokens**: Wrappers + raw integers only (no coordinate tokens); numeric tokens inside geometry count as grounding.
- **Checkpoints**: Always save tokenizer/processor; wrapper exposes `model.config` for HF compatibility; best‑checkpoint rotation is atomic.


### Troubleshooting
- **Common failure modes & diagnosis**
  - Image token mismatch: decode prompt and count `<|image_pad|>`; compare against `image_grid_thw`; verify `processor(images=[...])` count equals placeholders.
  - Span/label drift: confirm `offset_mapping`; ensure assistant spans include `<|im_end|>`; labels mask outside spans.
  - Formatting parse failures: check wrapper/separator balance; try tolerant geometry parsing; fallback converter for JSON‑like outputs.
  - Augmentation issues: ensure factor‑aligned smart resize; validate coordinate tracking; enable visualization debug flags; check bounds/canonicalization.
  - RL vision packing: `pixel_values` packed 2D and slice offsets consistent with `image_grid_thw` per chunk.
- **Checklist**
  - Do image tokens exist when image tensors are present? If not, rebuild conversation via builder and re‑process with HF processor; check `image_grid_thw` shape.
  - Do assistant spans include `<|im_end|>` and are labels masked outside spans? If not, inspect span builder logs and offset mapping.
  - Are grouped masks disjoint and covering assistant masks after shift? If not, inspect `losses/token_grouping.py` and logits/labels alignment.
  - Are geometry wrappers present with raw integers only? If coordinate tokens leak, disable legacy paths and validate tokenizer special tokens.
  - In RL, are `pixel_values` packed 2D and slice offsets consistent with `image_grid_thw`? If not, rebuild packed vision tensors per chunk.


### Debugging & Reading Order
- **Quick reading order**: `processing/conversation_processor.py` → span builder/mapping → `data/dataset.py` → `models/wrapper.py` → `models/loss_manager.py` → `losses/token_grouping.py`.
- **Augmentation**: `augmentation/compose.py` → `wrappers.py` → `image_ops.py` → `validators.py`.
- **Inference & RL**: `inference.py` → `rl/runner.py` → `rl/trainer.py` → rewards registry.
- **Conversion**: `data_conversion/unified_processor.py` → `coordinate_manager.py` → `validation_manager.py` → `summary_builder.py` → `teacher_selector.py`.
- **Debug configs**: `phase_*/debug.yaml` (100 samples, 1 epoch); `debug_alignment: true` logs token‑span mapping; `debug_visualization: true` saves augmentation before/after.
- **Performance monitor**: Initialization, training, inference timing; memory profiling via accumulation and CUDA cache management.
- **Checkpoint validation**: `src_new/tools/checkpoint_validator.py`.


### Utilities
- `PathManager`: Resolves absolute/relative image paths consistently against `data_root` and avoids double‑prefix issues; safe resolution returns `None` per path on failure.
- `DataResolver`: Validates expected dataset layout and auto‑resolves `train/val/teacher_pool/images` under a given `data_root`.
- `rank_aware_logging`: INFO/DEBUG on rank 0 only; WARNING/ERROR on all ranks; helper to configure global log level.


### References
- SFT docs hub: `src_new/UNIFIED_DOCUMENTATION.md`
- RL post‑training: `src_new/rl/README.md`
- Data conversion deep dive: `data_conversion/README.md`
- Configuration system: `configs/README.md`


### Glossary
- **Assistant span**: token‑aligned region for the assistant turn (includes `<|im_end|>`).
- **Grounding tokens**: numeric tokens inside geometry spans.
- **Formatting tokens**: geometry wrappers + punctuation + residuals not categorized as caption/grounding.


### Assistant behavior (how to help effectively)
- Default to HF‑first reasoning; apply templates via processor; never craft `<|image_pad|>`.
- Preserve strict ordering and invariants; validate shapes/counts explicitly.
- Do not introduce in‑code defaults; surface missing config keys clearly.
- Keep geometry tokens and coord ranges dynamic; avoid hard‑coded IDs when editing.
- Prefer improving processing/validation flows over adding ad‑hoc run scripts.

