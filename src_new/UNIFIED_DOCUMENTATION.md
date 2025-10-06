# Qwen2.5-VL BBU Detection Training Pipeline - Unified Documentation

**Complete Reference for Vision-Language Model Fine-tuning (coordinate tokens deprecated)**

---

## Executive Snapshot (2025-09)

- **Entrypoints**:
  - Training: `/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config <name> --log_level {INFO|DEBUG}`
    - Optional DeepSpeed via env: `BBU_DEEPSPEED_ENABLED=true`, `BBU_DEEPSPEED_CONFIG=scripts/zero2.json`
  - Inference: `/root/miniconda3/envs/ms/bin/python -m src_new.inference --config /abs/config.yaml --model_path /abs/checkpoint --data_root /abs/data --output_file /abs/out.jsonl`
  - RL/GRPO (dense captioning, src_new): `/root/miniconda3/envs/ms/bin/python -m src_new.rl.runner --config /abs/configs/dense_rl/standard.yaml --mode {load|train}`
  - Launcher (dense captioning RL): `bash scripts/run_dense_grpo.sh` (routes to `src_new.rl.runner` and `src_new.rl.eval`)

- **Pipeline separation (IMPORTANT)**:
  - `src_new` dense RL: shares the same HF-first stack and objectives as SFT; its goal is to improve dense captioning quality (object descriptions + geometry) via GRPO on the same conversation/template/tensor pipeline.
  - `src_post` group QC RL: a separate project for group-level quality control judgment with different objectives and config schema. Do not mix configs or launchers between these projects.

- **Critical invariants**:
  - Fast tokenizer required with `offset_mapping` available; chat template must exist (tokenizer or processor).
  - Coordinate tokens are deprecated. All geometry is emitted as raw integer coordinates with canonical wrappers; no `<|coord_*|>` usage. Parsing logic is centralized in `src_new/processing/parse_generated.py` and reused by inference and RL rewards.
  - Conversation variants cover both multimodal and text-only flows. `text_only` now injects a cached 28×28 all-black image so the vision tower executes every step; the user message still delivers raw JSON guidance, and the assistant target remains identical to dense captioning.
  - Text-only user prompts can optionally apply bounded geometry jitter/shuffle via `TEXT_ONLY_USER_NOISE_ENABLED` so the model rehearses tolerant parsing.
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
- `build_hf_components` (`src_new/utils/hf_components.py`) is the single entrypoint for loading tokenizer/processor/model across training, inference, and RL. It applies the Qwen2.5 patches once, inherits the chat template, probes bf16 support, defaults to `flash_attention_2` unless overridden, and surfaces the effective dtype/device map.
- Coordinate-token mode is deprecated in src_new; only geometry/object‑ref wrappers are required and validated. Chat template is strictly required and is propagated to the processor at runtime.

3) Dataset & conversations
- `Dataset` reads JSONL; validates per-sample structure; optional augmentation via `ObjectAwareAugmentationPipeline` with epoch-based presets; dynamic contrastive pairing per-epoch when `dynamic_pairing_enabled` (one context teacher max; eval forced single-turn). Text-only variants (currently `text_only`) bypass augmentation/teacher pairing and receive a sentinel image path that resolves to the cached 28×28 black patch, ensuring the conversation retains a single `<image>` placeholder and the vision tower gradients stay in sync across ranks.
- Augmentation order (when enabled): **smart resize → geometry → photometric → line ops → criteria**. The smart-resize stage snaps every canvas to a Qwen2.5-VL-friendly grid (`factor=28`) and constrains total pixels to `[512×28², 768×28²]`, scaling every bbox/quad/line before rotation so that downstream processors see native aspect ratios and patch layouts without extra padding.
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
- Grouping categories (strict): caption = inside `<|object_ref_start|>...<|object_ref_end|>` excluding punctuation and geometry; grounding = numeric tokens only inside geometry spans (plain-number subwords like 0–9, 12, 3456); formatting = geometry wrappers + punctuation/separators + object‑ref wrappers + residual non‑numeric inside geometry. Coordinate tokens `<|coord_*|>` are deprecated by default.

7) Phases & optimizer
- `PhaseFreezeManager` applies `phase_name`; coordinate‑slice specific groups removed; use standard groups: `vision`, `merger`, `llm`.

8) Logging, evaluation, checkpointing
- `TrainingStateManager` aggregates loss components locally; `_maybe_log_save_evaluate` overridden to avoid NCCL conflicts; evaluation accumulates components too.
- `CheckpointSaver` writes SafeTensors and tokenizer/processor. Coordinate config files are no longer produced.

---

## Data Schema & Business Rules (authoritative)

- JSONL fields per sample: `images: [str]`, `objects: [dict]`, `width: int`, `height: int`.
- Supported geometry keys: `bbox_2d` (4 ints), `quad` (8 ints), `line` (even ≥ 4 ints); coordinates are clamped to image bounds during preprocessing.
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
- Features: `merge_size`, `max_pixels`. Coordinate-token mode has been removed; only numeric geometry with wrapper tokens is supported.
- Loss: `teacher_loss_weight`, `student_loss_weight`, `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight`.
- Collator: `collator_type: standard` only. Packed mode is disabled in src_new.
- HF Trainer & Checkpointing: `eval_strategy`, `eval_steps`, `save_steps`, `save_total_limit`, `metric_for_best_model`, `greater_is_better`. HF `save_strategy` is "no"; `CheckpointSaver` writes SafeTensors and tokenizer/processor, promotes best checkpoint by atomic copy at interval ticks, and rotates old `checkpoint-*` folders per `save_total_limit`. `load_best_model_at_end` remains false.
- Logging: `logging_steps`, `report_to`, `disable_tqdm`.
- Dataloader: `dataloader_num_workers`, `pin_memory`, `prefetch_factor`, `remove_unused_columns`.
- LRs (optional): `lr_merger`, `lr_full_model`.
- Variants: `conversation_variant_ratios`.
- Format mode: a single numeric mode (special wrappers + integers). Coordinate‑token mode is deprecated.

---

## Loss Computation & Diagnostics (single source)

- LLM loss: single CE pass (shifted), masks per role; grouped LLM masks from `TokenGroupingPlugin` (caption/grounding/formatting) with teacher/student role weights.
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
- Packed: disabled in src_new. Only the standard collator is supported; `collator_type` must be "standard".

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
  - `Dataset._process_sample_unified` detects the variant, bypasses teacher pairing/augmentation, swaps the real image path for a sentinel that resolves to the cached 28×28 black patch, and, when noise is enabled, applies bounded jitter/shuffle to JSON geometries before rendering the user prompt.
- Configuration:
  - Add to YAML via `conversation_variant_ratios.text_only`. Example (phase 3 standard):
    ```yaml
    conversation_variant_ratios:
      dense_caption: 0.8
      text_only: 0.2
    ```
  - Ratios are validated centrally (`config/config.py`) so any typo or negative weight fails fast.
- Testing: `test_conversation_processor_precomputed_spans.py` covers the zero-image path; `test_conversation_variants.py` logs the prompt/target pair to `conversation_variants.log` for manual inspection.

---

## RL Post-Training (GRPO) on Shared Stack

- **Location & shims**: RL code for dense captioning lives under `src_new/rl/` (`runner.py`, `grpo_trainer.py`, `data/dataset.py`, `prompting/conversation.py`, `rewards/…`, `eval.py`). The legacy `src_rl/` package (if present) is a thin wrapper and deprecated.
- **Shared loader**: RL uses the same `build_hf_components` helper as SFT/inference so tokenizer/processor/model settings are consistent (pad=eos, left padding, eager attention override). Default attention is `flash_attention_2` unless the YAML explicitly requests otherwise; the runner force-overrides to eager only when stability is required.
- **Config surface**: Dense RL YAMLs live in `configs/dense_rl/` and hydrate `src_new/config/rl_config.py` (`EnhancedRLConfig`). Required keys—`model_path`, `train_data_path`, `val_data_path`, `data_root`, geometry settings—fail fast. Do not use `src_post` configs here.
- **Datasets & prompts**: `src_new/rl/data/RLDenseJSONLDataset` reconstructs tensors through `ConversationBuilder`, mirroring the SFT pipeline (single-turn, teacher disabled). Vision tensors are preserved end-to-end so `BBUGRPOTrainer` forwards `pixel_values`/`image_grid_thw` back into `model.generate` and validates the packed counts before every loss call.
- **Rewards & parsing**: Formatting rewards (`format_rewards.py`) and detection-aware rewards (`detection_rewards.py`) import the shared parser from `src_new/processing/parse_generated.py`, eliminating drift between inference metrics and RL reward calculations.
- **Entrypoints**:
  - Direct: `python -m src_new.rl.runner --config /abs/configs/dense_rl/standard.yaml --mode {load|train}`
  - Eval: `python -m src_new.rl.eval --config /abs/configs/dense_rl/standard.yaml --input_file /abs/val.jsonl --data_root /abs/data_root`
  - Launcher: `bash scripts/run_dense_grpo.sh` (uses `src_new.rl.runner` / `src_new.rl.eval`)

- **IMPORTANT separation**: `src_post` is a separate project focused on group-level quality control judgment (different objectives, prompts, and rewards). Use its own configs (e.g., `configs/rl/group_qc_grpo.yaml`) and launcher. Do not mix with `src_new` dense RL.

---

### RL configuration (YAML) quick reference

- **Required**: `model_path`, `train_data_path`, `val_data_path`, `data_root`, `output_dir`.
- **Recommended**: set `attn_implementation: eager` for stability during RL; batch size 1; enable `bf16` when hardware supports it (auto-probed, falls back to float32 otherwise).
- **Optional (common)**: `per_device_train_batch_size` (default 1), `update_steps` (grad accumulation), `learning_rate`, `weight_decay`, `max_steps`, `warmup_steps`, `logging_steps`, `save_steps`, `save_total_limit`, `sample_k`, `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`, `bf16`.
- **DeepSpeed**: set `BBU_DEEPSPEED_ENABLED=true` and optionally `BBU_DEEPSPEED_CONFIG` env to enable ZeRO; device-map is disabled when DeepSpeed is on.
- **Resume**: `resume_from_checkpoint: /abs/path/to/checkpoint-xxxx` is honored by the runner.

Minimal example:
```yaml
model_path: /abs/checkpoints/sft
data_root: /abs/data_root
train_data_path: /abs/data/train.jsonl
val_data_path: /abs/data/val.jsonl
output_dir: /abs/outputs/rl_run
attn_implementation: eager
bf16: true

# Reward weights (set heavy geometry rewards to 0.0 for a minimal start)
rewards:
  parse: 1.0
  wrappers: 0.5
  coords: 0.5
  separators: 0.25
  vocab: 0.25
  bbox_giou: 0.0
  quad_l1: 0.0
  line_l1: 0.0
  ordering: 0.0

per_device_train_batch_size: 1
update_steps: 1
max_steps: 2000
sample_k: 4
max_new_tokens: 256
temperature: 0.9
Top_p: 1.0
```

### Rewards registry (available keys)

- `parse`, `wrappers`, `coords`, `separators`, `vocab`
- `coverage`, `geometry_sanity`, `bbox_giou`, `quad_l1`, `line_l1`, `ordering`
- Tip: Start with formatting/sanitization (`parse`, `wrappers`, `coords`, `separators`, `vocab`) and keep geometry-heavy rewards at `0.0` until stability is confirmed.

### Attention implementation (RL)

- The RL runner honors the YAML `attn_implementation` you set; it does not currently force `eager` automatically.
- For most setups, prefer `attn_implementation: eager` with `bf16` (when available). If you experience instability, switch from `flash_attention_2` to `eager` in the YAML and re-run.

### Dataset expectations (RL)

- `images` must be a non-empty list of paths (resolved via `data_root`).
- `objects`, `width`, `height` are optional for prompt construction, but are surfaced in `meta` for reward functions that need them (e.g., coverage/geometry checks).
- Single-turn only; no teacher pairing or augmentation in RL; pass-through collator (`lambda items: items`).

### Smoke-load output (`--mode load`)

- The loader prints a one-line JSON summary; example:
```json
{"ok": true, "device": "cuda:0", "dtype": "torch.bfloat16", "vocab_model": 152064, "vocab_tokenizer": 152064}
```

### Checkpointing & logs (RL)

- Periodic saves based on `save_steps`; rotation via `save_total_limit`.
- Final model and tokenizer/processor are saved to `output_dir` (HF-compatible layout).
- TensorBoard logging is enabled by default (`report_to: [tensorboard]`).

## RL Dense GRPO — Current Implementation (2025‑09)

- Entrypoints (src_new RL):
  - Launcher: `bash scripts/run_dense_grpo.sh /abs/configs/dense_rl/{standard|debug}.yaml`
  - Direct: `/root/miniconda3/envs/ms/bin/python -m src_new.rl.runner --config /abs/config.yaml --mode {load|train}`
  - Eval: `/root/miniconda3/envs/ms/bin/python -m src_new.rl.eval --config /abs/config.yaml --input_file /abs/val.jsonl --data_root /abs/data`

- TensorBoard logging (required):
  - YAML requires both keys at root: `tb_dir` and `run_name`.
  - Effective TB directory = `tb_dir + '/' + run_name` (created automatically).
  - Example: `tb_dir: tb_post/9-26`, `run_name: standard-prod` → logs under `tb_post/9-26/standard-prod`.
  - Runner prints: `TensorBoard logging: dir=... | run_name=... | report_to=['tensorboard']`.

- Config schema (strict, enforced by runner):
  - Required top-level keys:
    - `model_path`, `train_data_path`, `val_data_path`, `data_root`, `output_dir`
    - `tb_dir`, `run_name`, `bf16: true`
    - `per_device_train_batch_size`, `update_steps`, `learning_rate`, `weight_decay`, `max_steps`, `warmup_steps`, `logging_steps`, `save_steps`, `seed`
    - Generation: `sample_k`, `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`
    - GRPO block `grpo`: `epsilon_low`, `epsilon_high`, `beta`, `loss_type`, `scale_rewards`, `mask_truncated_completions`
    - Rewards map `rewards`: at least one non-zero key (registered names)
    - `layer_config` (applied by PhaseFreezeManager)
  - Required model block `model`:
    - `attn_implementation: {eager|flash_attention_2|sdpa}`
    - `image_max_pixels: <int>` (must be explicitly set)
    - `trust_remote_code: true` (recommended)
  - Optional keys:
    - `micro_batch_size` (defaults to 1) for per-sample logp micro-batching
    - `generation.min_new_tokens`, `generation.repetition_penalty` (override)

- Filesystem layout and validation:
  - Runner now fails fast if any required section/key is missing.
  - DeepSpeed (optional) enabled via env `BBU_DEEPSPEED_ENABLED=true` and `BBU_DEEPSPEED_CONFIG`.

### BBUGRPOTrainer (core behavior)

- Generation & buffering (HF-first):
  - Calls `model.generate` directly with `eos_token_id = <|im_end|>` and respects YAML overrides for `temperature`, `top_p`, `min/max_new_tokens`.
  - Uses `steps_per_generation` to batch prompts, then splits the cached buffer into gradient micro-steps via `src_new/rl/buffer.split_buffer`.
  - Applies a dynamic temperature scale when NaNs/Infs are detected and captures per-step temperature history for debugging.

- Vision tensors (packed format):
  - Dense RL keeps packed 2D `pixel_values` (`[total_patches, hidden]`) and concatenated `image_grid_thw`/`images_per_sample` metadata identical to SFT.
  - Every chunk is revalidated through `validators.assert_patches_match_thw`, preventing drift between packed rows and THW counts.

- Per-token log-probs:
  - Reuses `src_new/rl/logprobs.get_per_token_logps` for sequential per-sample evaluation; divides logits by the effective temperature.
  - Optional reference policy KL is enabled when `grpo.beta_start > 0`, cloning the initial model once and keeping it frozen.

- Advantage handling:
  - Local advantages are normalized per prompt (`scale_rewards=true`), with optional cross-rank normalization via `rl/distributed.compute_global_advantages` when `normalization.cross_rank_advantages: true`.
  - Advantage statistics (std/max) are logged and stored for feature checks.

- Loss (GRPO/DR-GRPO):
  - `losses.compute_grpo_loss` handles symmetric clipping (`epsilon_low`/`epsilon_high`) and optional `beta * KL` penalties.
  - Max grad norm, bf16 autocast, and gradient accumulation mirror the SFT trainer; DeepSpeed ZeRO can wrap the trainer via environment flags.

### Metrics & logging (text + TensorBoard)

- Scalars:
  - Core: `loss`, `learning_rate`, `num_tokens`, `reward`, `reward_std`, `beta`
  - Completions: `completions/mean_length`, `completions/min_length`, `completions/max_length`, `completions/clipped_ratio`, `completions/terminated_ratio`
  - Advantages: `advantages/std`, `advantages/max_abs`
  - Per reward: `rewards/<name>/mean`, `rewards/<name>/std`
- Histories retained in trainer for feature checks: temperature, reward, clip ratio, advantage std/max, per-reward means.
- Text debug prints the first prompt/completion pair each logging interval (rank 0 only) plus high-level metric summaries.

### Rewards registry (notable keys)

- Format/content: `parse`, `wrappers`, `coords`, `separators`, `vocab`
- Geometry/detection: `bbox_giou`, `quad_l1`, `line_l1`, `ordering`, `coverage`, `geometry_sanity`
- Length shaping: `length`, `length_window` (new; soft window on token‑like length)

### Best‑practice settings (dense captioning)

- Sampling/diversity:
  - `sample_k ≥ 2` (GRPO requirement). Ensure `per_device_train_batch_size * world_size * update_steps` is divisible by `sample_k`.
  - `temperature ∈ [0.9, 1.1]`, `top_p ∈ [0.9, 1.0]`, `repetition_penalty ≈ 1.0–1.1`.
- Length/truncation:
  - Prefer `mask_truncated_completions: false` for very long dense outputs; set `generation.min_new_tokens: 0`.
  - Tune `max_new_tokens` high enough to allow `<|im_end|>` to appear naturally; keep `max_total_length` consistent.
- Attention/dtype:
  - `bf16: true` is required; use `flash_attention_2` or `eager` per stability; set explicitly under `model.attn_implementation`.
- Vision format:
  - Keep `pixel_values` in packed 2D format and ensure `pixel_values` rows equal `sum_i (t*h*w)` computed from `image_grid_thw`.

### YAML roles

- `configs/dense_rl/dense_base.yaml` — universal constants only (not runnable). Child configs must provide all required paths and run‑specific values.
- `configs/dense_rl/standard.yaml` — production settings (full dataset, larger `sample_k`, higher `max_new_tokens`).
- `configs/dense_rl/debug.yaml` — fast coverage of all modules (short run, high logging), still exercises full RL stack.

### Example (excerpt)

```yaml
# Required TB keys
tb_dir: tb_post/9-26
run_name: standard-prod

# Model
model:
  attn_implementation: flash_attention_2
  image_max_pixels: 401408
  trust_remote_code: true

# Data & outputs
model_path: /abs/checkpoints/sft
train_data_path: /abs/data/train.jsonl
val_data_path: /abs/data/val.jsonl
data_root: /abs/data
output_dir: /abs/outputs/rl_run
bf16: true

# Training
per_device_train_batch_size: 2
update_steps: 2
learning_rate: 5.0e-6
weight_decay: 0.0
max_steps: 1000
warmup_steps: 25
logging_steps: 10
save_steps: 200
seed: 17

# Generation
sample_k: 4
max_new_tokens: 1024
temperature: 1.0
top_p: 0.95
repetition_penalty: 1.1

grpo:
  epsilon_low: 0.2
  epsilon_high: 0.2
  beta: 0          # set >0.0 to enable KL
  loss_type: bnpo
  scale_rewards: true
  mask_truncated_completions: false

rewards:
  parse: 0.30
  wrappers: 0.10
  coords: 0.10
  separators: 0.10
  vocab: 0.05
  length: 0.05
  length_window: 0.10
  bbox_giou: 0.20

layer_config:
  llm_trainable_top_k_blocks: 4
  vision_trainable_top_k_blocks: 0
  vision_freeze_patch_embed: true
```

### Troubleshooting checklist

- If losses are zero: verify completion masks (EOS present or `mask_truncated_completions=false`) and ensure reward variance across K > 0.
- If shape mismatches: confirm packed 2D `pixel_values` and correct patch slicing by `image_grid_thw`; decoded `<|image_pad|>` count should align with `sum(t*h*w)`.
- If TB logs don’t appear: ensure `tb_dir` and `run_name` are set; runner prints resolved TB path on startup.
