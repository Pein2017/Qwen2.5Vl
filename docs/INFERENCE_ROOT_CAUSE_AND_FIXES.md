# Qwen2.5-VL Inference: Root Cause Analysis and Fixes (Canonical)

Note: A concise operational checklist lives in `AI_ASSISTANT_KB.md` (Inference Specifics). This file preserves the full root-cause analysis and complete fix set.

This document is the single source of truth for the inference issue that caused empty predictions and alignment warnings in the `src_new/` inference path. It consolidates the prior analyses into one practical guide with causes, fixes, and operational notes.

---

## 1) Symptoms

- Intermittent empty predictions in `predictions.json` despite generation logs
- Errors or warnings about image/token alignment (e.g., "Image features and image tokens do not match")
- Template vs. processed token count confusion for `<|image_pad|>`
- Missing `<|coord_xxx|>` tokens when using extended-vocab checkpoints

---

## 2) Root Causes

- **Fragile truncation + tokenizer-only re-tokenization**
  - Full conversations (teachers + student) were correctly built via the HF processor, but a later step truncated by string manipulation and re-tokenized text with the tokenizer only, while reusing image tensors.
  - This desynchronized `<|image_pad|>` placeholders in text from `pixel_values`/`image_grid_thw` tensors.

- **Coordinate tokens removed at decode time**
  - Decoding with `skip_special_tokens=True` can drop extended-vocab geometry tokens (e.g., `<|coord_xxx|>`), making outputs appear empty/missing geometry.

- **Over-strict generation-time validation**
  - Teacher–student assistant-turn rules were enforced during generation (where the final assistant turn is intentionally produced by the model), leading to misleading warnings.
  - Post-processor counts of `<|image_pad|>` were treated as mismatches, even though large counts after processing are expected and not equal to the number of images.

---

## 3) Fixes Implemented

- **Generation-ready conversation builders (no tokenizer-only re-tokenization)**
  - Conversation construction for inference now stays inside `ConversationProcessor` and runs through the HF `processor` together with images.
  - Call sites use generation-specific builders (teacher–student and simple flows) that append the generation prompt and return fully aligned inputs.

- **Preserve coordinate tokens during decoding**
  - Decode with `skip_special_tokens=False` so extended geometry tokens are not dropped.

- **Generation-aware validation + clearer logging**
  - Validate with INFERENCE rules: ensure template placeholder count equals image count before processor tokenization.
  - Treat the large number of post-processor `<|image_pad|>` tokens as informational, not an error.

---

## 4) Results after Fixes

- Non-empty predictions are produced consistently
- `<|coord_xxx|>` tokens appear when using extended-vocab checkpoints
- No image/token alignment errors; template placeholder count equals number of images
- Spurious warnings removed; generation does not require a pre-filled final assistant turn

---

## 5) Operational Notes

- Keep batch size = 1 for teacher-guided inference (matches training)
- Use the exact training tokenizer/processor from the checkpoint (loads the extended vocabulary)
- Prefer `attn_implementation="eager"` for stability in this environment

---

## 6) Checklist for Future Changes

- Do not re-tokenize conversations without reprocessing images via the HF `processor`
- Decode with `skip_special_tokens=False` to preserve geometry tokens
- In generation flows, enforce only template placeholder count vs. image count prior to processor tokenization
- Keep conversation-building responsibilities in `ConversationProcessor`; avoid duplicating logic in runners

---

## 7) Implementation Pointers

- Build generation-ready inputs via `ConversationProcessor` (teacher–student or simple), ensuring the HF `processor(text=[...], images=..., return_tensors="pt")` is used, not `tokenizer(...)` alone after truncation.
- Before generation, validate that the number of `<|image_pad|>` placeholders in the template text equals the number of images.
- Decode generated tokens with `skip_special_tokens=False` to keep `<|coord_xxx|>` tokens.

---

## 8) Retirement Notice

This canonical doc supersedes the following and allows their removal:
- `src_new/INFERENCE_ISSUE.md`
- `src_new/INFERENCE_ROOT_CAUSE_AND_FIXES.md`
- `src_new/UNIFIED_INFERENCE_ISSUE_MERGED.md`

If you encounter any regression (alignment errors or missing coordinate tokens), verify that generation uses the ConversationProcessor generation builders, reprocessing via the HF processor with images, and that decoding does not skip special tokens. 