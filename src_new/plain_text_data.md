## Global Plain Text (JSON Lines) Mode — Implementation Plan (no code changes yet)

### Update (implemented, unified)
- We removed the precomputed group-mask path and any collator/wrapper plumbing for `*_group_*_mask`.
- The existing `TokenGroupingPlugin` now auto-detects plain-text JSON mode (no wrappers present) and dynamically builds caption/grounding/formatting via decode + offset_mapping:
  - caption: the inside of the JSON `"desc"` value (per assistant span, per line)
  - grounding: the geometry key token (`"bbox_2d"|"quad"|"line"`) plus the contents of its `[...]` array
  - formatting: residual inside assistant spans (punctuation/separators)
  - Then intersect with teacher/student assistant spans and shift for CE. Wrapper-mode behavior is unchanged.
- Conversation and collators remain simple: no extra fields; assistant spans are already produced and reused; training semantics and grouped metrics match the wrapper mode.

- **Goal**: Add a global “plain text” mode that removes all special wrapper tokens (`<|object_ref_*>`, `<|box_*>`, `<|quad_*>`, `<|line_*>`) in assistant/user texts and uses JSON Lines instead, while keeping the same training variants (dense_caption, coords_to_desc, desc_to_coords, summary) and preserving assistant-span based grouped losses (caption/grounding/formatting).
- **Output shape (assistant)**: One object per line, canonical JSON with exactly two keys: geometry and desc.
  - Examples:
    - `{"line":[x1,y1,x2,y2],"desc":"电线/分布散乱"}`
    - `{"bbox_2d":[x1,y1,x2,y2],"desc":"BBU设备/华为, 显示完整, 无需安装"}`
    - `{"quad":[x1,y1,x2,y2,x3,y3,x4,y4],"desc":"标签/清晰"}`
  - If `coordinate_tokens_enabled=true`, the geometry array may contain `<|coord_N|>` tokens instead of raw numbers. Otherwise raw integers are used. No other special tokens are used in plain mode.
- **Input shape (user)**:
  - dense_caption: image-only (unchanged)
  - coords_to_desc: lines with geometry only, e.g. `{"line":[...]}`
  - desc_to_coords: lines with desc only, e.g. `{"desc":"..."}`
  - summary: unchanged (one-line natural language; no JSON)


### 1) Configuration (strict)
- Add a required boolean key: `plain_text_mode_enabled`.
  - Validation: explicit True/False in YAML; no silent default.
  - Interactions:
    - If `plain_text_mode_enabled=true` and `coordinate_tokens_enabled=true`, arrays may contain `<|coord_*>`; the rest remains JSON (no geometry wrappers).
    - Summary variant unaffected by this toggle.


### 2) Rendering layer (assistant + user)
- Extend `src_new/processing/coordinate_converter.py` with a JSON renderer path.
  - Add a flag `plain_text_mode_enabled` to the converter constructor.
  - For assistant text: emit newline-joined JSON objects in canonical order: `{GEOM_KEY:[...],"desc":"..."}` with no spaces (stable formatting).
  - Preserve current path for wrapper mode when `plain_text_mode_enabled=false`.
- Update `src_new/processing/geometry_text.py` helpers to respect plain mode for user prompts:
  - `format_geometry_for_user`: return `{"line":[...]}`, `{"bbox_2d":[...]}`, or `{"quad":[...]}` when plain mode; otherwise keep wrappers.
  - `format_object_ref`: return `{"desc":"..."}` when plain mode; otherwise keep `<|object_ref_start|>...<|object_ref_end|>`.


### 3) System and user prompts (global)
- Update `src_new/processing/templates.py::get_system_prompt` to branch on `plain_text_mode_enabled`:
  - Replace wrapper-centric guidance with strict JSON Lines guidance:
    - One object per line; exactly two keys `{geometry, desc}`; geometry key ∈ {bbox_2d, quad, line}.
    - Arrays contain either integers or `<|coord_*>` depending on `coordinate_tokens_enabled`.
    - No extra keys, no trailing commas, no whitespace beyond newlines between objects.
  - Keep the existing numeric vs token wording for coordinates, adapted to JSON arrays.
- Leave `SUMMARY_SYSTEM_PROMPT` unchanged (still forbids brackets/special tokens; summary variant continues to be plain text, not JSON).


### 4) Conversation building (global, not a new variant)
- Pass `plain_text_mode_enabled` into `ConversationProcessor`/`ConversationBuilder` and the converter.
- For each assistant turn we already build `assistant_text`. In plain mode:
  - Build JSON Lines with a deterministic builder that also returns intra-content character spans for:
    - caption: the desc value content (inside quotes, excluding quotes)
    - grounding: the geometry array content (all values inside `[...]`) plus the geometry key token span (e.g., `"line"` / `"bbox_2d"` / `"quad"`) to capture geometry classification
    - formatting: will be derived later as “assistant content minus (caption ∪ grounding)”
  - After `apply_chat_template(...)` returns the final `text`, find each assistant content block using the same regex used by span extraction (`ASSISTANT_SPAN_RE`) to obtain `(content_start_char, content_end_char)` per turn. Offset the per-turn intra-content spans by `content_start_char` and attach them to the outputs as absolute character ranges.
  - Emit per-turn role tags so we know which assistant spans correspond to teacher vs student when teacher-student is used.


### 5) Dataset: char spans → token masks
- (Kept for optional debugging) We can compute plain-text char spans → token masks locally using offset mapping and assistant spans. Not required for training since grouping is now fully dynamic inside `TokenGroupingPlugin`.


### 6) Loss integration (unchanged API)
- `LossManager` continues to rely on `TokenGroupingPlugin` for caption/grounding/formatting masks. The plugin now supports both wrapper and plain JSON modes dynamically based on the presence of wrapper tokens.


### 7) Inference (training-matched)
- Generation uses the same builders; in plain mode assistant outputs are JSON Lines. Parsing can be added if needed.


### 8) Health checks & invariants
- Assistant spans found via `<|im_start|>assistant ... <|im_end|>` remain the anchor for masking.
- Group masks are built dynamically to satisfy:
  - Disjointness: caption ∩ grounding ∩ formatting = ∅
  - Coverage: (caption ∪ grounding ∪ formatting) == assistant masks (after shift)


### 9) Testing plan (adjusted)
- Plugin plain-mode path: decode→offset_mapping→JSON-line scan correctness; coverage/disjointness and counts match wrapper mode on synthetic cases.
- End-to-end: processor→dataset→collator→wrapper unchanged public interface; grouped metrics produced as before.


### 10) Rollout steps (implementation order)
1) Config: add `plain_text_mode_enabled` with validation.
2) Converter + geometry_text: implement JSON rendering and user prompt helpers.
3) Templates: add JSON-mode system prompt builder branch.
4) Conversation builder: produce assistant JSON, compute intra-content char spans, attach absolute group char spans with roles.
5) Dataset: build token-level group masks from char spans; attach to sample (shifted).
6) Collators: pass/merge masks.
7) Wrapper/LossManager: plumb optional masks and skip plugin when present.
8) Docs: update `UNIFIED_DOCUMENTATION.md` (new section) + add YAML example.


### 11) YAML snippet (example)
```yaml
# Global toggle (required)
plain_text_mode_enabled: true
# Optional: still allow coord tokens inside JSON arrays
coordinate_tokens_enabled: false  # set true to use <|coord_*> in arrays

# Variants (sampling unchanged)
conversation_variant_ratios:
  dense_caption: 0.45
  coords_to_desc: 0.20
  desc_to_coords: 0.20
  summary: 0.15
```
