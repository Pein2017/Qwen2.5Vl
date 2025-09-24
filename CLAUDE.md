## Global Project Rules (AI Assistant Prompt)

### What this is
- Single global prompt for any new AI conversation about this repo; focus on pipeline/flow/processing, not commands.
- Prefer absolute paths. Read relevant .md first when present.
- Environment: use conda env `ms` (or `/root/miniconda3/envs/ms/bin/python`).

### Repository components (3)
- `data_conversion/`: V2 annotations → strict JSONL with native geometry (bbox/quad/line), canonical ordering, and hierarchical Chinese descriptions; teacher pool selection; images EXIF-corrected and smart-resized.
- `src_new/`: HF-first SFT pipeline (Qwen2.5‑VL) with optional coordinate auxiliaries (legacy), strict config, span alignment, grouped losses, and robust multimodal validations. Coordinate tokens are deprecated; use raw integers for geometry.
- `src_post/`: RL post‑training (GRPO) for group-level QC decisions using an SFT checkpoint; Stage‑A summaries + Stage‑B pass/fail.

## End‑to‑End Flow (conceptual)
1) Data conversion (strict): raw V2 JSON + images → flat JSONL samples + processed images
2) SFT training (HF-first): build typed conversations → tokenization → span masking → model wrapper with validations → single‑pass CE (+ grouped losses) → checkpoints (+ processor)
3) RL post‑training (optional): Stage‑A summaries → Stage‑B pass/fail GRPO on short text

---

## Core contracts and invariants (must hold)
- Conversations (HF-first):
  - Built via `Qwen2VLProcessor.apply_chat_template(...)` with typed content; user turns inject images as typed `{"type": "image"}` list.
  - The number of image placeholders in rendered text must equal the number of images; else error.
- Multimodal token alignment (model‑side checks):
  - Expected image token count = sum over images of `(t*h*w) // (merge_size**2)` from `image_grid_thw` and model merge size; must equal `<|image_pad|>` count found in `input_ids`.
  - `pixel_values` rows must equal `sum_i (t_i*h_i*w_i)`; else error.
- Assistant spans and labels:
  - Assistant spans are precomputed once in the conversation builder (regex over `<|im_start|>assistant ... <|im_end|>`), stored with token‑aligned indices, then consumed by the dataset to build labels.
  - Labels outside assistant spans are `-100`. The immediate `<|im_end|>` token is included inside each assistant span to teach termination.
  - Teacher–student: teacher assistant spans (earlier turns) and the last assistant span (student) are separated; only student is the generation target by default.
- Coordinate tokens (deprecated in `src_new`):
  - Do not use `<|coord_*|>` in `src_new`. Geometry is emitted as raw integers with canonical wrappers. Dynamic range detection is retained only for legacy compatibility/inference.
- Losses (grouped and optional coord aux):
  - Single‑pass CE computed once and reused for teacher/student masks; grouped CE components (caption/grounding/formatting) built from label IDs and spans.
  - Optional coordinate auxiliary losses exist for legacy coord‑token mode only and are disabled by default.
- Checkpoints & tokenizer:
  - Save tokenizer/processor with checkpoints; wrapper exposes `model.config` to preserve HF integration expectations.
- Configuration (strict):
  - All keys are explicit in YAML; no in‑code hyperparameter defaults. Paths are normalized and validated; `data_root` enables deriving dataset file paths when missing.

---

## Data conversion (strict V2 → training JSONL)
- Inputs: raw V2 JSON (`dataList` or `markResult.features`) + paired image; require `info.width`, `info.height`.
- Object type whitelist: `{bbu, bbu_shield, connect_point, label, fiber, wire}`; geometry constraints: `fiber|wire → line`, others → bbox/quad.
- Geometry & coordinate pipeline (order is mandatory):
  1) Apply EXIF orientation to geometry
  2) Rescale if JSON dims ≠ actual image dims
  3) Smart resize (multiple‑of‑28 within pixel budget)
- Canonicalization:
  - BBox: x_min < x_max, y_min < y_max; clamp to bounds
  - Quad: 8 ints, vertices ordered clockwise starting at top‑left
  - Line: even number of coords, preserve path; direction canonicalized so the first point is the leftmost endpoint (tie‑break by y); reverse full sequence if needed
- Text normalization:
  - Strict hierarchical Chinese descriptions; remove occlusion tokens containing “遮挡”.
- Ordering & splitting:
  - Sort objects top‑to‑bottom, then left‑to‑right by first coordinate pair; images processed to match final transform.
  - Teacher pool via greedy coverage (fixed vocabulary if available, else free‑vocab fallback), then deterministic train/val split.
- Outputs (flat format everywhere):
  - `train.jsonl`, `val.jsonl`, `teacher_pool.jsonl`, `all_samples.jsonl`, `label_vocabulary.json`, validation reports, and processed `images/`.

---

## SFT pipeline (src_new, HF‑first)
- Conversation building:
  - Use `ConversationProcessor` (HF‑first) which wraps the official processor.
  - Variants: dense caption (default), coords→desc, desc→coords; optional `summary` (image→one‑line CN) and `text_only` (dummy image + JSON guidance). Teacher–student conversations interleave teacher examples before the student turn; inference is single‑turn.
- Span detection and labeling:
  - Spans are precomputed once in the builder (`processing/span_builder.py`) and attached as token‑aligned `teacher_assistant_spans`/`student_assistant_spans`. The dataset consumes these directly to build labels; include immediate `<|im_end|>`; mask `<|image_pad|>` back to `-100`.
- Collation & shapes:
  - Standard collator pads to the longest in batch; emits `pixel_values` and `image_grid_thw` when present; validates THW vs `pixel_values` rows and enforces `image_grid_thw` shape [num_images, 3].
  - Packed mode is disabled in `src_new`.
- Model wrapper (`DetectionModel`):
  - Fail‑fast multimodal validations (shapes, counts, expected `<|image_pad|>`); obtains merge size from model vision config (fallback to training config).
  - Intelligent checkpoint handling; tokenizer/vocab extension with ms‑swift embedding padding; coordinate token range detection retained only for legacy.
  - Loss path: bypass base loss when spans provided; compute single‑pass CE once; apply teacher/student masks and grouped masks; optional coord aux is legacy only.
- Grouped LLM losses (token‑ID based):
  - caption (inside object‑ref), grounding (plain digits inside geometry spans), formatting (punctuation + object‑ref/geometry wrappers + residual). Masks are aligned to shifted CE.

---

## Coordinate token system (legacy only)
- `src_new` disables coordinate‑token mode by default; keep `coordinate_tokens_enabled: false` in configs. Raw integers are authoritative. Range detection utilities remain for backward compatibility (e.g., legacy inference checkpoints).

---

## RL post‑training for group QC (src_post, GRPO)
- Stage‑A: per‑image one‑line Chinese summary (no coordinates or special tokens); optionally sanitized; greedy lines used as context for Stage‑B.
- Stage‑B: text‑only prompt aggregates summaries and checklist hints (mission‑specific); sample K_B responses; parse to `{label, reason}`.
- GRPO update:
  - Compute rewards via modular registry; z‑score to advantages; optimize `-A * logp(y)` over response tokens (length‑norm optional); optional KL to reference policy.
  - Conditional Stage‑A GRPO (optional): vary one image’s summary at a time, recompute Stage‑B reward, and update on summary tokens.
- Decode‑time masking (optional): mask geometry/coord tokens during Stage‑A generation; does not affect gradients.

---

## Health checks (fast)
- Placeholder count in text equals number of images; otherwise error.
- `pixel_values` rows == sum over `image_grid_thw` of (t*h*w).
- `<|image_pad|>` count in `input_ids` equals expected `(t*h*w)//(merge_size**2)` totals.
- Assistant spans found; `<|im_end|>` included; labels outside spans are `-100`.
- Coord mode is legacy only; if coord aux is enabled but no coord‑labeled targets inside spans, fail fast.
- Loss: total equals sum of weighted components; diagnostics finite.
- Checkpoints: Save processor; wrapper exposes HF `model.config`.

---

## Open‑first pointers (code reading order)
- `src_new/processing/conversation_processor.py` (HF‑first builders, validation)
- `src_new/processing/coordinate_converter.py` (object→token conversion)
- `src_new/models/wrapper.py` (DetectionModel, validations, loss path)
- `src_new/models/loss_manager.py` and `src_new/losses/token_grouping.py` (single‑pass CE, grouped masks, coord aux legacy)
- `src_new/data/dataset.py` (precomputed spans, masking, variant dispatch)
- `data_conversion/unified_processor.py` and `coordinate_manager.py` (EXIF→rescale→smart‑resize; canonical geometry)
- `src_post/runner.py` and `src_post/conversation.py` (Stage‑A/B flows, GRPO core)

---

## Assistant behavior (how to help effectively)
- Default to HF‑first reasoning: typed messages → template render → processor tensors; never hand‑craft `<|image_pad|>`.
- Preserve strict ordering and invariants above; validate shapes/counts explicitly.
- Do not introduce in‑code defaults; surface missing config keys clearly.
- When editing, keep geometry tokens and coord ranges dynamic; avoid hard‑coded IDs.
- Prefer improving processing/validation flows over adding ad‑hoc run scripts.

## References
- Data conversion deep dive: `data_conversion/README.md`
- SFT docs hub: `src_new/UNIFIED_DOCUMENTATION.md`
- RL post‑training: `src_post/README.md`