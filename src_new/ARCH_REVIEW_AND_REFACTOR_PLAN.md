## Qwen2.5-VL src_new: Critical Code Review and Refactor Plan

This document provides a deep, critical review of `src_new/` and a concrete plan to refactor toward higher standards of accuracy, abstraction, type safety, and PyTorch/AI best practices. It targets correctness, maintainability, performance, and reliability in distributed environments.

### TL;DR (Top Priority Fixes)
- Unify geometry token naming to canonical "box" everywhere; add invariant tests.
- Derive coordinate token range dynamically from tokenizer; remove hard-coded ranges.
- Standardize multimodal batch shapes end-to-end; fix collator/model contracts.
- Clean up imports/types: use `Qwen2_5_VLForConditionalGeneration` or a generic `PreTrainedModel` consistently.
- Align distributed launch: don’t override `local_rank` under torchrun/DeepSpeed; let env drive rank detection.
- Split monolith modules into focused subpackages; add public typed APIs.

---

## Architectural Review

### What’s strong
- Clear modular separation (`data/`, `processing/`, `models/`, `training/`, `utils/`, `config/`).
- Fail-fast config validation with explicit dataclass fields and path defaults.
- Rank-aware logging preventing cross-rank log spam.
- Dual-loss system: span-based LLM CE + soft expectation L1 coordinate loss.
- HuggingFace-first conversation building (dramatic simplification vs custom logic).
- Unified checkpoint management with best-copy and SafeTensors.

### Critical correctness issues
- Geometry token naming drift (“bbox” vs “box”) can break parsing and evaluation.
- Hard-coded coordinate ranges in several places; config allows different `max_coord_value`.
- Collator correctly flattens patches and emits `image_grid_thw: [num_images, 3]`; wrapper still contains a fallback squeeze `[B,1,3]→[B,3]` that should be removed to avoid mis-shaping. Standardize on `[num_images, 3]` only.
- Mixed model class names and types (Qwen2VL vs Qwen2_5_VL) reduce clarity and can cause subtle mismatches.
- TrainingArguments forcibly set `local_rank=-1` while using torchrun/DeepSpeed.

### Maintainability issues
- Large monolithic modules (1k–1.6k lines) hinder review and evolution:
  - `processing/conversation_processor.py`
  - `training/bbu_trainer.py`
  - `models/wrapper.py`
  - `data/collator.py`
  - `models/loss_manager.py`
- Coordinate logic duplicated across `CoordinateProcessor` and `TokenProcessor`.
- Incomplete type coverage for non-trivial data passed across modules.

---

## Type System and Contracts (Proposed)

Introduce a dedicated `src_new/types/` package to define shared, explicit types with mypy/pyright coverage, plus optional runtime validation where valuable.

### New modules
- `src_new/types/conversation.py`
  - `ConversationType (Enum)`
  - `TurnRole (Enum)`
  - `ConversationMessage (TypedDict)`: { role: TurnRole, content: str | list[dict] }
  - `TeacherStudentConversation (TypedDict)` with explicit message/image invariants
  - `ConversationValidationResult (dataclass)`
- `src_new/types/geometry.py`
  - `GeometryType (Enum)`: BOX, QUAD, LINE
  - `Box = tuple[int, int, int, int]`
  - `Quad = tuple[int, int, int, int, int, int, int, int]`
  - `Line = tuple[int, ...]` (even length ≥ 4)
  - `ObjectAnnotation (TypedDict)`: { desc: str, bbox_2d?: Box, quad?: Quad, line?: Line }
- `src_new/types/coords.py`
  - `CoordTokenRange (NamedTuple)`: start_id: int, end_id_exclusive: int
- `src_new/types/batch.py`
  - `Span = tuple[int, int]`  (end-exclusive)
  - `Spans = list[Span]`
  - `TeacherStudentSpans (TypedDict)`: { teacher: Spans, student: Spans }
  - `TextBatch (Protocol)`: input_ids: Tensor[B, T], attention_mask: Tensor[B, T], labels: Tensor[B, T]
  - `ImageGrid = Tensor[NI, 3]`  (t, h, w per image)
  - `PatchTensor = Tensor[total_patches, patch_features]`
  - `MultimodalBatch (TypedDict)` with optional `pixel_values: PatchTensor` and `image_grid_thw: ImageGrid` and `spans: TeacherStudentSpans`
- `src_new/types/model.py`
  - `ModelForwardInputs (TypedDict)` aligning with model.forward kwargs
  - `ModelOutputs (Protocol)` exposing `.loss`, `.logits`, `.hidden_states | None`
- `src_new/types/loss.py`
  - Promote `LossComponents` to shared types; ensure all members typed

### Typing standards
- Use `from typing import Protocol, TypedDict, NamedTuple, Literal` to express contracts at boundaries.
- For device/dtype, annotate helper functions, but avoid over-typing tensors internally where the shapes vary; use shape comments or torchtyping if desired.
- Prefer Enums for canonical token names and geometry types; centralize special tokens.

---

## Shape and API Contracts (Multimodal)

Define and enforce one contract across dataset → collator → model:

- Text tensors: `input_ids: [batch, seq_len]`, `attention_mask: [batch, seq_len]`, `labels: [batch, seq_len]`.
- Images (Qwen2.5-VL flattened patches):
  - `pixel_values: [total_patches_across_batch, patch_features]` (from HF processor; never reshape into image tensors)
  - `image_grid_thw: [num_images_across_batch, 3]` where each row is (t, h, w) for the corresponding image
- Counts must satisfy: sum over images of (t*h*w) == pixel_values.shape[0]
- Additionally, expected image token count in `input_ids` equals sum_i (t_i*h_i*w_i) // (merge_size**2) using `vision_config.spatial_merge_size`.
- Span lists are per sample. Collator must map per-sample spans to batch-level lists (list[Spans]) aligned to batch index.

Actions:
- In `data/collator.py`:
  - Keep batch dimension; produce `image_grid_thw: [num_images_total, 3]` and not nest `[batch, 1, 3]`.
  - Maintain lists of spans per batch item: `teacher_assistant_spans: list[Spans]`, `student_assistant_spans: list[Spans]`.
  - Assert shape invariants before returning batch.
- In `models/wrapper.py`:
  - Remove the fallback that squeezes `[B,1,3]→[B,3]`. Accept only `image_grid_thw: [num_images, 3]` and validate using counts.
  - Provide a utility to validate `pixel_values` vs `image_grid_thw` early, with actionable error messages.

---

## Canonical Tokens and Ranges

- Centralize geometry special tokens in one module (e.g., `src_new/types/geometry.py` or `src_new/processing/special_tokens.py`).
  - BOX: `<|box_start|>`, `<|box_end|>` — canonical
  - QUAD: `<|quad_start|>`, `<|quad_end|>`
  - LINE: `<|line_start|>`, `<|line_end|>`
- Ensure all code paths (converter, processor, docs) use BOX naming, not “bbox”.
- Derive coordinate token range from tokenizer at runtime via `TokenProcessor.get_coordinate_token_range(tokenizer) -> CoordTokenRange` and thread this into:
  - Mask creation (coordinate mask)
  - Loss slicing (`logits[..., start:end]`) including `SoftExpectationCoordinateLoss(coord_start_id, coord_end_id)`
  - Saved `coordinate_config.json` (no hard-coding)

---

## Logging and Error-Handling Standards

- Prefer a single rank-aware logging system; remove module-local fallbacks where possible.
- Make all validation errors explicit and fail-fast, especially on shape mismatches and missing coordinate tokens.
- Promote a small subset of critical errors to include contextual tips (e.g., "check chat template propagation ") to speed diagnosis.

---

## Distributed and Launch Alignment

- Under torchrun/DeepSpeed:
  - Do not hard-set `local_rank=-1` in `TrainingArguments`; leave it to HF Trainer env detection. In `scripts/train_new.py#create_training_arguments_with_deepspeed`, remove `local_rank=-1` and let torchrun set ranks.
  - Preserve current NCCL timeouts (1800s default) unless evidence suggests otherwise.
  - Keep unified checkpoint saving rank0-only; barriers only where necessary.

- In `scripts/run_new_train.sh`:
  - Keep adaptive DeepSpeed enablement.
  - Let Trainer infer rank; remove conflicting overrides.
  - Retain flash-attn env only when `attn_implementation == "flash_attention_2"`.

---

## Depth-first pipeline walkthrough (train_new.py → src_new/)

This captures the effective end-to-end initialization and training flow based on `scripts/train_new.py` and the `src_new/` modules.

1) Entry and logging
- `scripts/train_new.py` parses CLI, initializes rank-aware logging, and applies Qwen2.5-VL compatibility patches early.

2) TrainingArguments creation
- `create_training_arguments_with_deepspeed` builds `transformers.TrainingArguments` and (currently) sets `local_rank=-1` (should be removed under torchrun).
- DeepSpeed enablement controlled via `BBU_DEEPSPEED_ENABLED` and `BBU_DEEPSPEED_CONFIG` envs (from `run_new_train.sh`).

3) Tokenizer and image processor
- `AutoTokenizer.from_pretrained(config.model_path, use_fast=True)` for offset mapping support.
- `Qwen2VLImageProcessor.from_pretrained(config.model_path)` for image pre-processing; `max_pixels` overridden from config.

4) Dataset and teacher pool
- `Dataset(...).set_processor(hf_processor)` performed via trainer to inject the processor later; dataset uses HuggingFace-first `ConversationProcessor` to build conversations, with robust teacher-student or simple flows.
- Teacher pool loaded via `TeacherPoolManager(teacher_pool_file)`.

5) Data collator
- `create_data_collator(collator_type)` returns either Standard or Packed collator wrapped by `TrainerCompatibleDataCollator`.
- Collators produce text tensors `[B, T]`, `pixel_values` flattened patches, and `image_grid_thw: [num_images, 3]` across batch. They also propagate per-sample teacher/student spans.

6) Model loading and expansion
- Base `Qwen2_5_VLForConditionalGeneration.from_pretrained` with optimized kwargs.
- `perform_pre_distributed_expansion`: builds coordinate tokens and extends embeddings BEFORE distributed training, preventing DDP sync race conditions.
- Wrap into `DetectionModel(base_model, config, tokenizer, skip_expansion=True)`.

7) Trainer and processor bindings
- `BBUTrainer(...)` created; unified checkpoint manager embedded.
- `trainer.processing_class = tokenizer` ensures checkpoint saving doesn’t hit `.get_vocab()` issues.
- A `Qwen2VLProcessor` instance is created to preserve chat_template; then re-instantiated with the extended tokenizer and image processor; injected via `trainer.set_processor` for saving.

8) Training loop and metrics
- `BBUTrainer` overrides `_maybe_log_save_evaluate` to avoid NCCL conflicts, logs metrics via `TrainingStateManager`, and uses unified checkpointing with best-copy.
- `BBUTrainer.create_optimizer` builds param groups with distinct LRs (vision/merger/llm), labels LR logs correctly.

9) Model forward and loss
- `DetectionModel.forward` validates multimodal tensors and image token alignment, optionally bypasses the base loss when spans are provided, and computes detailed loss components via `LossManager` (single-pass CE optimization) plus coordinate L1 (soft expectation).

10) Checkpoints
- Inference-ready SaveTensors checkpoints saved by default; tokenizers/processors and coordinate metadata saved.

---

## Qwen2.5‑VL model mechanics clarified

- Placeholders and tokens
  - Chat templates contain one placeholder token (config.image_token_id) per image. After processing, processor expands `<|image_pad|>` into a large run of vision tokens during tokenization; you observe the placeholder count in `input_ids` via `sum(input_ids == image_token_id)`.

- Image grid and patches
  - `image_grid_thw: [num_images, 3]` lists (t, h, w) tiles for each image. For static images, t=1.
  - The processor flattens vision patches; `pixel_values: [total_patches, patch_features]` where `total_patches = Σ_i (t_i*h_i*w_i)`.
  - The number of generated text “image pad” tokens in the tokenizer output relates to the merged visual grid by the model’s spatial merge size `merge_size`: expected `<|image_pad|>` expansion is `Σ_i (t_i*h_i*w_i) // (merge_size**2)`. We already validate the patch counts; we should add optional validation for decoded `<|image_pad|>` counts when practical.

- Required invariants (codified already or to finalize):
  - `sum(input_ids == image_token_id) == image_grid_thw.shape[0]` (placeholders ↔ images)
  - `Σ_i (t_i*h_i*w_i) == pixel_values.shape[0]` (patch rows ↔ total patches)
  - Optional informational check: decoded `<|image_pad|>` count density vs merge_size as a warning, not a hard error, to avoid false positives when HF internals change.

- mRoPE and FlashAttention v2 compatibility
  - `patches.py` fixes `apply_multimodal_rotary_pos_emb` to avoid doubled mrope_section; `torch.library.wrap_triton` compatibility added for PyTorch 2.5.1. Attention implementation fallback forces eager when FA2 is unavailable.

- Coordinate tokens
  - Tokenizer extended with `<|coord_N|>` tokens and optional geometry tokens (`<|line_start|>`, `<|line_end|>`). Coordinate token range MUST be derived from tokenizer at runtime and reused in masking, loss slicing, and checkpoint metadata.

---

## Additional actionable checks (to implement)

- Wrapper
  - Remove `[B,1,3]→[B,3]` squeezing fallback and only accept `image_grid_thw: [num_images, 3]`. Validate via invariant counts; provide clear remediation tips on failure.

- Collator
  - Add explicit asserts with helpful messages showing: total_images, sum(t*h*w), pixel_values rows, and placeholder counts where available.

- TrainingArguments
  - Under torchrun/DeepSpeed, stop forcing `local_rank=-1` to let HF Trainer pick up env rank; provide a CLI flag `--single_gpu` that sets `CUDA_VISIBLE_DEVICES=0` when desired for single-card dev.

- Coordinate metadata
  - Save `coordinate_config.json` fields strictly from tokenizer-derived range and `max_coord_value`. Add a small validator test to assert this consistency post-save.

- Tests
  - Add a shape contract test for: (1) single-image, (2) teacher+student 2-image, and (3) batch of 2 samples, verifying all invariants and that `DetectionModel.forward` does not reshape tensors.

---

## Refactor Plan (Phased)

### Phase 1: Correctness and invariants (1–2 days)
1) Geometry token unification to BOX everywhere; add a sanity test.
2) Remove hard-coded coordinate ranges; derive from tokenizer at save and at loss/mask time.
3) Enforce multimodal shape contract end-to-end:
   - Collator returns consistent `pixel_values` and `image_grid_thw`.
   - Model wrapper validates counts, not just shapes.
4) Clean imports/types to consistent `Qwen2_5_VLForConditionalGeneration` or `PreTrainedModel` abstraction.
5) Replace `local_rank=-1` when using torchrun/DeepSpeed; rely on env.

Deliverables:
- Invariant tests and minimal integration tests for shape/token checks.
- Updated saving to include tokenizer-derived `coordinate_token_range`.

### Phase 2: Type system rollout (2–3 days)
1) Introduce `src_new/types/` with Enums, TypedDicts, Protocols outlined above.
2) Add type hints to public APIs:
   - Dataset, Collators, ConversationProcessor, LossManager, DetectionModel wrapper, Trainer helpers.
3) Configure mypy/pyright; add a basic CI step locally.
4) Optional runtime validation for critical inputs (beartype/dataclasses).

### Phase 3: Module decomposition (3–5 days)
- `processing/` → `conversation/` (builder/validator/truncation/stats)
- `training/` → split checkpointing/log formatting out of `bbu_trainer.py`
- `models/` → `detection_model.py` and `coordinate_processor.py`
- `data/` → `collator_standard.py`, `collator_packed.py`, `collator_utils.py`
- Deduplicate coordinate helpers into one module consumed by both wrapper and loss.

### Phase 4: Performance polish (ongoing)
- Add small LRU image cache in dataset to reduce disk IO.
- Consider pre-resolving paths with `PathManager` at dataset init.
- Verify tokenizer offset mapping path once per sample to reduce decode+retokenize cost, or reuse processor metadata if available.
- Keep ms-swift-inspired optimizations (pad-to-128, lazy init) but gate on feature flags.

---

## Concrete Changes (How-To)

### 1) Coordinate range derivation
- Add helper:
```python
# src_new/processing/special_tokens.py
from typing import NamedTuple

class CoordTokenRange(NamedTuple):
    start: int
    end_exclusive: int

def get_coord_token_range(tokenizer) -> CoordTokenRange:
    vocab = tokenizer.get_vocab()
    coord_ids = [v for k, v in vocab.items() if k.startswith("<|coord_")]
    if not coord_ids:
        return CoordTokenRange(0, 0)
    return CoordTokenRange(min(coord_ids), max(coord_ids) + 1)
```
- Use this in:
  - `TokenProcessor.get_coordinate_token_range`
  - `CoordinateProcessor.mask_coordinate_logits` & `get_coordinate_mask`
  - `LossManager` coordinate slicing
  - Checkpoint saving metadata

### 2) Canonical geometry tokens
- Centralize token constants:
```python
# src_new/processing/special_tokens.py
BOX_START = "<|box_start|>"; BOX_END = "<|box_end|>"
QUAD_START = "<|quad_start|>"; QUAD_END = "<|quad_end|>"
LINE_START = "<|line_start|>"; LINE_END = "<|line_end|>"
```
- Replace any use of legacy "bbox" tokens and ensure docs match.

### 3) Collator batch contracts
- Enforce batch-first text; emit `image_grid_thw: [num_images_total, 3]`.
- Validate `sum(t*h*w) == pixel_values.shape[0]`; include the counts in error messages.
- Keep `teacher_assistant_spans` and `student_assistant_spans` as `list[list[tuple[int,int]]]` sized to batch.

### 4) Loss manager
- Accept `CoordTokenRange` instead of relying on internal state.
- Ensure `coordinate_loss_temperature` comes from config; remove silent defaults.
- Keep single-pass CE path for teacher/student with clear masks.

### 5) Wrapper
- Validate multimodal inputs using counts vs shapes; do not mutate shapes unless strictly required.
- Accept `TeacherStudentSpans` explicitly to make typing consistent.

---

## Testing Additions

- Token invariants:
  - Geometry tokens (BOX/QUAD/LINE) exist and are used consistently.
  - Coordinate token range derived at runtime matches tokenizer content.
- Shape contracts:
  - Single-image sample → expected `pixel_values` rows and `image_grid_thw` count.
  - Teacher-student sample with 2 images → sums and alignment verified.
  - Mini-batch of 2 samples → aggregated rows equal sum of each sample’s t*h*w.
- Checkpoint metadata:
  - `coordinate_config.json` always reflects derived range and `max_coord_value`.
- Distributed smoke test:
  - torchrun (2 ranks) launch without `local_rank` override; rank0-only checkpointing.

---

## Documentation and Standards

- Update `docs/SRC_NEW_REFERENCE.md` and `src_new/UNIFIED_DOCUMENTATION.md` to reference the canonical token module and the new shape contracts.
- Add `CONTRIBUTING_TYPES.md` with conventions:
  - Enums for canonical strings
  - TypedDict for cross-module data bags
  - Protocols for interface-style dependencies (`ProcessorLike`, `ModelOutputsLike`)
  - Avoid bare `dict[str, Any]` in public APIs

---

## Adoption Checklist

- [ ] Add `src_new/types/` package and initial signatures
- [ ] Centralize special tokens and coordinate range helpers
- [ ] Collator shape contract enforced with tests
- [ ] Wrapper validation updated; no shape mutation assumptions
- [ ] Loss manager uses derived ranges and config temperature
- [ ] Remove `local_rank=-1` override when using torchrun/DeepSpeed
- [ ] Monoliths split into submodules (phase 3)
- [ ] Mypy/pyright baseline clean; CI target: “no new type errors”

---

## Notes on ms-swift alignment

- Keep vocabulary padding to multiple-of-128; document why and where applied.
- Keep smart embedding initialization but guard HF API differences (e.g., `mean_resizing` availability) with capability checks.
- Prefer eager attention for inference stability unless FA2 is explicitly available.

---

## Expected Impact

- Fewer silent mismatches in multimodal alignment; faster root-cause on shape/token issues.
- Stronger type-driven design reduces regressions and improves code navigation.
- Cleaner module boundaries enable faster iteration and onboarding.
- Distributed training reliability maintained; checkpointing faster and safer.
