# Qwen2.5-VL BBU Detection Training Pipeline - Unified Documentation

**Complete Reference for Vision-Language Model Fine-tuning (coordinate tokens deprecated)**

---

## Executive Snapshot (2025-09)

- **Entrypoints**:
  - Training: `/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config <name> --log_level {INFO|DEBUG}`
    - Optional DeepSpeed via env: `BBU_DEEPSPEED_ENABLED=true`, `BBU_DEEPSPEED_CONFIG=scripts/zero2.json`
  - Inference: `/root/miniconda3/envs/ms/bin/python -m src_new.inference --config /abs/config.yaml --model_path /abs/checkpoint --data_root /abs/data --output_file /abs/out.jsonl`

- **Critical invariants**:
  - Fast tokenizer required with `offset_mapping` available; chat template must exist (tokenizer or processor).
  - Coordinate tokens are deprecated. All geometry is emitted as raw integer coordinates with canonical wrappers; no `<|coord_*|>` usage.
- Conversation variants cover both multimodal and text-only flows. `text_only` now injects a cached 28×28 all-black image so the vision tower executes every step; the user message still delivers raw JSON guidance, and the assistant target remains identical to dense captioning.
  - Image alignment: decoded `<|image_pad|>` count equals `sum_i (t*h*w) // (merge_size**2)`; `pixel_values` rows equal `sum_i (t*h*w)`.
  - Labels/spans: assistant spans are precomputed once in the conversation builder (regex+offsets) and mapped to expanded input_ids; include immediate `<|im_end|>`; `<|image_pad|>` always masked.

- **Data → Model → Loss → Checkpoint** (HF‑first): JSONL → `ConversationProcessor.apply_chat_template` → official processor tensors → `DetectionModel` wrapper (validations) → single‑pass CE (teacher/student masks) + grouped LLM losses → optional coord‑aux (when enabled) → local aggregation (no custom distributed ops) → SafeTensors + tokenizer/processor saved; best checkpoint via atomic copy, rotation enabled.

- **Freezing phases**: phased unfreezing for LLM/vision remains. PhaseFreezeManager applies per-phase policies: phase_1 trains `visual.merger` (optionally coord-slice on embeddings/LM head), phase_2 unfreezes last‑K LLM blocks, phase_3 unfreezes all by default (keeping `visual.patch_embed` frozen unless overridden).

---

## Training Pipeline (step‑by‑step)

1) Parse & validate
- Load `configs/phase_1/base.yaml` merged with experiment override; auto‑resolve dataset paths via `data_root` when missing (`config/config.py::load_config`).
- Initialize rank‑aware logging; apply Qwen2.5‑VL patches; seed RNGs.

2) Tokenizer/processor & model
- Fast tokenizer and `Qwen2VLImageProcessor` from `config.model_path`; `image_processor.max_pixels` overridden from YAML.
- Coordinate-token mode is deprecated in src_new; only geometry/object‑ref wrappers are required and validated. Chat template is strictly required and is propagated to the processor at runtime.

3) Dataset & conversations
- `Dataset` reads JSONL; validates per-sample structure; optional augmentation via `ObjectAwareAugmentationPipeline` with epoch-based presets; dynamic contrastive pairing per-epoch when `dynamic_pairing_enabled` (one context teacher max; eval forced single-turn). Text-only variants (currently `text_only`) bypass augmentation/teacher pairing and receive a sentinel image path that resolves to the cached 28×28 black patch, ensuring the conversation retains a single `<image>` placeholder and the vision tower gradients stay in sync across ranks.
- `ConversationProcessor.create_conversation(...)` builds teacher-student or simple conversations, applies HF chat template, threads `conversation_text` + `offset_mapping`, enforces image token consistency. The processor accepts user-spec dictionaries (`{"text": ..., "include_image": bool}`) letting variants decide whether placeholders are emitted. Summary variant uses precomputed `sample['summary']`.
- Spans are computed once in the builder (`processing/span_builder.py`) and attached as token-aligned `teacher_assistant_spans`/`student_assistant_spans`. The dataset consumes these directly to build labels; `<|image_pad|>` is masked; span ends include immediate `<|im_end|>` when present.

4) Collation & packing
- Standard collator pads to the longest in batch; emits `pixel_values` and `image_grid_thw` when present; validates THW vs `pixel_values` rows and enforces `image_grid_thw` shape [num_images, 3].
- Packed mode is disabled in src_new; only the standard collator is supported.

5) Forward & validations (`models/wrapper.py`)
- Validate text/image tensor shapes and token/image counts; compute expected image token count from THW grid and merge size; raise on mismatch.
- When spans provided, bypass base loss and use SOLUTION‑1 CE path; optionally build block‑diagonal attention for packed rows.

6) Loss computation (`models/loss_manager.py`)
- Single‑pass CE once; apply teacher/student masks; grouped LLM losses (caption/grounding/formatting) via `losses/token_grouping.py` with weights from config (teacher/student weighted separately).
- Coordinate auxiliary losses are optional and disabled by default. When enabled (`coord_aux_enabled: true`), LossManager computes per-group kernelized-KL and unlikelihood components only at positions where the label is a coordinate token, and logs diagnostics (window_mass, gt_prob, top1/top5, etc.).

7) Phases & optimizer
- `PhaseFreezeManager` applies `phase_name`; coordinate‑slice specific groups removed; use standard groups: `vision`, `merger`, `llm`.

8) Logging, evaluation, checkpointing
- `TrainingStateManager` aggregates loss components locally; `_maybe_log_save_evaluate` overridden to avoid NCCL conflicts; evaluation accumulates components too.
- `CheckpointSaver` writes SafeTensors and tokenizer/processor. Coordinate config files are no longer produced.

---

## Data Schema & Business Rules (authoritative)

- JSONL fields per sample: `images: [str]`, `objects: [dict]`, `width: int`, `height: int`.
- Supported geometry keys: `bbox_2d` (4 ints), `quad` (8 ints), `line` (even ≥ 4 ints); coordinates clamped to `[0, max_coord_value]`.
- Geometry is emitted with canonical wrappers and raw integers only. Example per object line:
  - `<|object_ref_start|>描述<|object_ref_end|><|box_start|>[x1, y1, x2, y2]<|box_end|>`
  - `<|object_ref_start|>描述<|object_ref_end|><|quad_start|>[x1, y1, x2, y2, x3, y3, x4, y4]<|quad_end|>`
  - `<|object_ref_start|>描述<|object_ref_end|><|line_start|>[x1, y1, x2, y2, …]<|line_end|>`
- Object categories (6): `bbu`, `bbu_shield`, `connect_point`, `label`, `fiber`, `wire`.
- Description policy (Chinese): one object per line; concise but complete; sort outputs top‑to‑bottom then left‑to‑right; line start is leftmost endpoint (break ties by y then x); quad ordering TL → TR → BR → BL.

---

## Configuration (strict)

- Model/runtime: `model_path`, `attn_implementation {flash_attention_2|eager|sdpa}`, `torch_dtype {float16|bfloat16|float32|bf16|fp16}`, `use_cache`.
- Training: `num_train_epochs`, `per_device_{train,eval}_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `vision_lr`, `merger_lr`, `llm_lr`, `warmup_ratio`, `weight_decay`, `max_grad_norm`, `lr_scheduler_type`, `gradient_checkpointing`, `bf16|fp16`.
- Data: `train_data_path`, `val_data_path`, `data_root`, `teacher_pool_file`, `max_total_length`, `num_teacher_samples`, `max_dataset_size`.
- Features: `max_coord_value`, `merge_size`, `max_pixels`, `coordinate_tokens_enabled` (set to false). Coordinate-token mode is deprecated; keep the key in configs for compatibility but disable it.
- Loss: `teacher_loss_weight`, `student_loss_weight`, `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight`.
- Collator: `collator_type: standard` only. Packed mode is disabled in src_new and `packed_segment_isolation` is ignored.
- HF Trainer & Checkpointing: `eval_strategy`, `eval_steps`, `save_steps`, `save_total_limit`, `metric_for_best_model`, `greater_is_better`. HF `save_strategy` is "no"; `CheckpointSaver` writes SafeTensors and tokenizer/processor, promotes best checkpoint by atomic copy at interval ticks, and rotates old `checkpoint-*` folders per `save_total_limit`. `load_best_model_at_end` remains false.
- Logging: `logging_steps`, `report_to`, `disable_tqdm`.
- Dataloader: `dataloader_num_workers`, `pin_memory`, `prefetch_factor`, `remove_unused_columns`.
- LRs (optional): `lr_merger`, `lr_full_model`.
- Variants: `conversation_variant_ratios`.
- Format mode: a single numeric mode (special wrappers + integers). Coordinate‑token mode is deprecated.

---

## Loss Computation & Diagnostics (single source)

- LLM loss: single CE pass (shifted), masks per role; grouped LLM masks from `TokenGroupingPlugin` (caption/grounding/formatting) with teacher/student role weights.
- Coordinate auxiliary losses (kernelized‑KL + unlikelihood) are optional and off by default. Enable with `coord_aux_enabled: true` to compute per‑group aux at coordinate‑labeled targets and log diagnostics (window_mass, coord_slice_mass, gt_prob, expected_mae_bins, top1/top5, etc.).
- Grouping policy keyed by variant:
  - `dense_caption`, `coords_to_desc`, `desc_to_coords`: require geometry/object-ref wrappers; fail fast if absent.
  - Plain JSON (when enabled): require strict JSON Lines layout for variants; fail fast if invalid.
  - `summary`: caption-only grouping over assistant spans; wrappers/JSON not required.

---

## Freezing Phases (separate runs)

- `phase_1`: Train `visual.merger`; freeze most of LLM & vision as needed.
- `phase_2`: Phase 1 + unfreeze last‑K LLM decoder blocks (use `llm_top_k_block` in configs); vision remains frozen; merger trainable.
- `phase_3`: Unfreeze all by default (keep `visual.patch_embed` frozen). For memory‑constrained runs, selectively unfreeze last‑K LLM blocks via `llm_top_k_block` and last‑K vision blocks via `vision_top_k_block`; merger always trainable.

---

## Collation

- Standard: pad sequences to max length; emit `pixel_values` and `image_grid_thw` when present; validate THW vs `pixel_values` rows and ensure `image_grid_thw` has shape [num_images, 3].
- Packed: disabled in src_new. Only the standard collator is supported; `collator_type` must be "standard" and `packed_segment_isolation` is not used.

---

## Inference (training‑matched)

- Loads tokenizer/processor/model with the same validations as training; enforces image token/tensor alignment.
- Teacher‑guided mode (num_teachers > 0) replicates dynamic pairing; batch size forced to 1.
- Parsing priority: geometry tokens with raw numbers → JSON best‑effort; no coordinate‑token parsing.
- Stability recommendations: `attn_implementation="eager"`, `bf16`, batch size 1.

---

## Module Map (refactored `src_new`)

- `config/`: strict dataclass; path normalization; schedule & variant validation.
- `data/`: dataset + augmentation + teacher pool; consumes precomputed assistant spans; `<|image_pad|>` masking.
- `processing/`: conversation builder; span precomputation (`processing/span_builder.py`); numeric geometry conversion; canonical tokens; templates; variant registry.
- `models/`: wrapper + validations; SOLUTION‑1 CE path; grouped LLM losses.
- `losses/`: `token_grouping.py`.
- `training/`: BBUTrainer; PhaseFreezeManager; TrainingStateManager; CheckpointSaver.
- `inference.py`: engine with strict parsing and training-matched prep.

---

## New: Per-image Summary Variant (SFT) & Prompt Alignment

- Summary variant (SFT, optional): image → one-line Chinese summary per image
  - Purpose: reduce distribution shift before RL by teaching a clean, single-line, per-image summary with no coordinates/special tokens.
  - Source: precomputed during data conversion as `summary` in each sample; src_new reads this field directly (no dynamic extraction). The precomputation respects the same slash-level semantics:
    - Level-0: object type literal (e.g., “BBU设备”, “挡风板”, …)
    - Level-1: comma-joined canonical attributes (e.g., visibility, compliance, bend radius, organization)
    - Level-2 (conditional): present only when parent condition is met (e.g., `windshield_conformity`, `specific_issues`, `protection_details`)
    - Level-Last (remarks): optional free-text “special circumstances” (e.g., cannot determine/rectified/space-limited). Detected strictly by position (last slash segment), not by keywords.
  - Hard constraints: per-image only; no group-level or pass/fail decisions; no `<|...|>`, `<`, `>`, `[`, `]`, or coordinates.

- YAML toggle (no code changes needed):
  - Enable summary in training by setting non-zero ratio:
    ```yaml
    conversation_variant_ratios:
      dense_caption: 0.45
      coords_to_desc: 0.20
      desc_to_coords: 0.20
      summary: 0.15
    ```
  - Disable by omitting `summary` or setting `summary: 0.0`.

- Prompt alignment (mitigate drift between SFT and RL):
  - SFT summary system prompt (`SUMMARY_SYSTEM_PROMPT`, `src_new/processing/templates.py`) and RL Stage‑A system prompt (`STAGE_A_SYSTEM_PROMPT`, `src_post/conversation.py`) share the same core constraints:
    - One-line Chinese; per-image only (no group-level/pass/fail)
    - No coordinates or special tokens; no quotes or list-like repetition
    - BBU-domain only; labels judged as clear/unclear; optional short extra_info/remarks
  - Minor divergence by design:
    - SFT summary is mission-agnostic to avoid overfitting; RL Stage‑A can add mission hints/checklists dynamically.
  - User prompts are aligned in spirit:
    - SFT: text instruction + typed image input
    - RL: inline `<image>` placeholder in text; both are equivalent under the official chat template

- Evaluation note:
  - The default eval flow uses `dense_caption`; to evaluate summary behavior, run a small validation pass sampling the `summary` variant and check formatting/cleanliness metrics (symbol leakage=0, length 10–40 chars, coverage of severe slots).

---

## New: Text-Only Variant

- Purpose: give the base SFT curriculum an explicit formatting rehearsal where the model rewrites ground-truth objects into canonical wrapper lines without seeing pixels. This injects repeated practice for `<|object_ref_start|>…<|line_end|>` balance and discourages truncation during long multimodal turns.
- Prompt contract:
  - User turn lists the raw JSON records (`Object k: {"desc": ..., "bbox_2d": [...]}`) and an English instruction mirroring dense wrapper expectations, while still attaching a single dummy image placeholder so the encoded sequence matches the dense path.
  - Assistant target is the same multimodal wrapper string produced in dense captioning (geometry + description wrappers with raw integers).
- Implementation details:
  - Registered as `ConversationVariant.TEXT_ONLY` (`processing/variants.py`); handler returns a spec with `include_image=True` so the chat template emits the placeholder.
  - `Dataset._process_sample_unified` detects the variant, bypasses teacher pairing/augmentation, and swaps the real image path for a sentinel that resolves to the cached 28×28 black patch. `_load_images` feeds that dummy through the Qwen processor, yielding a single image pad token.
- Configuration:
  - Add to YAML via `conversation_variant_ratios.text_only`. Example (phase 3 standard):
    ```yaml
    conversation_variant_ratios:
      dense_caption: 0.8
      text_only: 0.2
    ```
  - Ratios are validated centrally (`config/config.py`) so any typo or negative weight fails fast.
- Testing: `test_conversation_processor_precomputed_spans.py` covers the zero-image path; `test_conversation_variants.py` logs the prompt/target pair to `conversation_variants.log` for manual inspection.
