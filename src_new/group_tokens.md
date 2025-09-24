## Grouped Token Losses (Caption / Grounding / Formatting)

### Purpose
- Separate assistant targets into three semantic groups and apply per‑group CE weights:
  - caption: natural‑language description of the object
  - grounding: geometry numeric coordinates (plain digits inside geometry spans)
  - formatting: structural glue (wrappers and punctuation)
- Provide clearer gradient signals while preserving teacher/student weighting and coordinate auxiliary losses.

### Exact token categorization
- **caption**
  - Tokens strictly inside the `<|object_ref_start|> ... <|object_ref_end|>` content
  - Excludes punctuation/brackets and any geometry tokens
- **grounding**
  - Only numeric tokens inside geometry spans (plain-number subwords like `0`–`9`, `12`, `3456`)
  - Coordinate tokens `<|coord_N|>` are deprecated and ignored by default
- **formatting**
  - Geometry wrappers: `<|box_start|>`, `<|box_end|>`, `<|quad_start|>`, `<|quad_end|>`, `<|line_start|>`, `<|line_end|>`
  - Punctuation/brackets/separators: `[ ] { } ( ) , : " /`
  - Object‑ref wrappers: `<|object_ref_start|>`, `<|object_ref_end|>`
  - Residual assistant tokens not covered by caption/grounding (e.g., the span terminator `<|im_end|>` included via residual assignment)

Notes:
- Geometry type prediction (bbox/quad/line) is considered part of formatting (via geometry wrapper tokens).
- “Taxonomy” words are not split as a separate group; they live inside caption.

### Span alignment and masks (offset‑mapping based)
- Assistant spans are discovered at character level and precisely mapped to token indices via `offset_mapping` in `src_new/data/dataset.py::_create_masked_labels_with_spans`.
- Spans are extended to include the immediate `<|im_end|>` token when present.
- Labels are set to original `input_ids` only within assistant spans; outside labels are `-100`.

Grouping masks are built in `src_new/losses/token_grouping.py`:
- Construct unshifted masks by scanning label IDs:
  - caption scope: inside `<|object_ref_start|>...<|object_ref_end|>`, minus punctuation and geometry
  - grounding scope: numeric tokens strictly inside geometry spans
  - formatting scope: geometry wrappers + punctuation/separators + object‑ref wrappers + residual non‑numeric inside geometry
- Intersect each scope with teacher/student assistant spans.
- Shift to match next‑token CE: masks → `mask[:, 1:]` (aligns with `logits[:, :-1]` vs `labels[:, 1:]`).
- Enforce disjointness and coverage:
  - Pairwise intersections are empty
  - `(caption | grounding | formatting) == assistant_mask` for both teacher and student
  - Any uncovered assistant token is assigned to formatting

### Loss computation
- Compute per‑token cross‑entropy once on shifted tensors (`[batch, seq_len‑1]`).
- For teacher and student independently:
  - Compute per‑group sums and counts over the CE matrix
  - Form a per‑token weighted average: `L_grouped = (w_c * sum_cap + w_g * sum_grd + w_f * sum_fmt) / (cap_cnt + grd_cnt + fmt_cnt)`
  - This becomes `teacher_llm_loss` or `student_llm_loss`
- Apply existing outer weights as before in `LossManager`:
  - `teacher_loss_weight * regular_loss_weight * teacher_llm_loss`
  - `student_loss_weight * regular_loss_weight * student_llm_loss`
- Coordinate auxiliary losses (Kernelized‑KL + Unlikelihood) are unchanged and added on top when enabled (legacy coord‑token mode only).

### Required config (strict)
- `caption_loss_weight: float`
- `grounding_loss_weight: float`
- `formatting_loss_weight: float`
- All must be ≥ 0 and not all zeros. Grouping is always enabled; weights alone control contribution.

Example:
```yaml
caption_loss_weight: 0.5
grounding_loss_weight: 1.0
formatting_loss_weight: 0.2
```

### Metrics and logging
- Per‑group CE (means) and token counts are produced for each side:
  - `teacher_caption_ce`, `teacher_grounding_ce`, `teacher_formatting_ce`
  - `student_caption_ce`, `student_grounding_ce`, `student_formatting_ce`
  - `teacher_caption_tokens`, `teacher_grounding_tokens`, `teacher_formatting_tokens`
  - `student_caption_tokens`, `student_grounding_tokens`, `student_formatting_tokens`
- Aggregate (teacher+student) metrics are also computed:
  - `group_caption_ce`, `group_grounding_ce`, `group_formatting_ce`
  - `group_caption_tokens`, `group_grounding_tokens`, `group_formatting_tokens`
- By default, the trainer logs keys beginning with `teacher_` and `student_`. If you also want `group_*` curves, extend the include‑filter in `TrainingStateManager` to accept `group_` keys.

### Invariants tied to spans and shifting
- Masks are aligned to the same shifted frame as CE (`[:, 1:]`).
- Disjointness: no overlap between caption/grounding/formatting per side.
- Coverage: union equals assistant masks (after shift) per side.
- Numeric tokens appearing inside geometry fall under grounding; coordinate tokens are deprecated and treated as formatting unless legacy coord‑token mode is explicitly enabled.
- Object‑ref wrappers and geometry wrappers fall under formatting.

### Code reference
- Grouping: `src_new/losses/token_grouping.py`
- Loss integration: `src_new/models/loss_manager.py`
- Spans/labels (offset mapping): `src_new/data/dataset.py::_create_masked_labels_with_spans`

### Failure modes (fail‑fast)
- Missing required config weights → `ValueError` during config validation.
- Tokenizer missing required special tokens or punctuation encoding → `ValueError` in grouping plugin construction.
- Invalid span bounds or shape mismatches → `ValueError` during mask building.
