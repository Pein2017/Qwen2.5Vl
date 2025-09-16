## Grouped Token Losses (JSON mode: Caption / Grounding / Formatting)

### Purpose
- Separate assistant targets into three semantic groups and apply per‑group CE weights:
  - caption: natural‑language content of object labels
  - grounding: numeric coordinates inside geometry arrays
  - formatting: JSON syntax and keys (structural glue)
- Provide clearer gradient signals while preserving teacher/student weighting. Coordinate auxiliary losses are not used in JSON mode.

### Exact token categorization (JSON outputs only)
- **caption**
  - The character spans corresponding to string values of the `label` field in the assistant JSON.
  - Quotes around the string are excluded; only the inner text is included.
- **grounding**
  - Numeric literals that appear inside geometry arrays of the following keys:
    - `box_points`: [[x1,y1],[x2,y2]]
    - `quadrilateral_points`: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    - `line_points`: [[x1,y1], ...]
  - Only digits (with an optional leading sign) and the contiguous number text are included in this group.
- **formatting**
  - JSON syntax and structural tokens: `{`, `}`, `[`, `]`, `:`, `,`, quotes around strings, and whitespace.
  - JSON keys themselves: `label`, `box_points`, `quadrilateral_points`, `line_points`.
  - Any residual assistant tokens not covered by caption/grounding (e.g., the span terminator `<|im_end|>` that is included inside assistant spans by training) are assigned to formatting.

Notes:
- Geometry type prediction is implicit in JSON via the presence of the corresponding key. The text of the key is considered formatting, while the numbers are grounding.
- “Taxonomy” words live inside `label` values and therefore belong to caption.

### Span alignment and masks (offset‑mapping based)
- Assistant spans are discovered at character level (regex over `<|im_start|>assistant ... <|im_end|>`) and precisely mapped to token indices via `offset_mapping` in `src_new_json/data/dataset.py::_create_masked_labels_with_spans`.
- The immediate `<|im_end|>` token is included at the end of each assistant span to teach termination (falls under formatting by residual assignment).
- Labels are set to original `input_ids` only within assistant spans; outside labels are `-100`.

Grouping masks are built in `src_new_json/losses/token_grouping.py`:
- Parse the assistant JSON text for each assistant turn (teacher and student) using a strict JSON parser.
- Compute character‑level spans for:
  - caption: string values of `label` fields (excluding surrounding quotes)
  - grounding: number tokens inside arrays under `box_points|quadrilateral_points|line_points`
  - formatting: remaining JSON syntax and keys not covered by caption/grounding
- Convert these character spans to token index spans using the tokenizer `offset_mapping` (decode → re‑tokenize flow).
- Intersect each group’s token spans with teacher/student assistant spans.
- Shift to match next‑token CE: masks → `mask[:, 1:]` (aligns with `logits[:, :-1]` vs `labels[:, 1:]`).
- Enforce disjointness and coverage:
  - Pairwise intersections are empty.
  - `(caption | grounding | formatting) == assistant_mask` for both teacher and student.
  - Any uncovered assistant token is assigned to formatting.

### Loss computation
- Compute per‑token cross‑entropy once on shifted tensors (`[batch, seq_len‑1]`).
- For teacher and student independently:
  - Compute per‑group sums and counts over the CE matrix.
  - Form a per‑token weighted average:
    - `L_grouped = (w_c * sum_cap + w_g * sum_grd + w_f * sum_fmt) / (cap_cnt + grd_cnt + fmt_cnt)`
  - This becomes `teacher_llm_loss` or `student_llm_loss`.
- Apply outer weights in `LossManager` as before:
  - `teacher_loss_weight * regular_loss_weight * teacher_llm_loss`
  - `student_loss_weight * regular_loss_weight * student_llm_loss`

### Required config (strict)
- `caption_loss_weight: float`
- `grounding_loss_weight: float`
- `formatting_loss_weight: float`
- All must be ≥ 0 and not all zeros. Grouping is always enabled; weights alone control contribution.

Example:
```yaml
caption_loss_weight: 1.2
grounding_loss_weight: 1.5
formatting_loss_weight: 0.3
```

### Metrics and logging
- Per‑group CE (means) and token counts are produced for each side:
  - `teacher_caption_ce`, `teacher_grounding_ce`, `teacher_formatting_ce`
  - `student_caption_ce`, `student_grounding_ce`, `student_formatting_ce`
  - `teacher_caption_tokens`, `teacher_grounding_tokens`, `teacher_formatting_tokens`
  - `student_caption_tokens`, `student_grounding_tokens`, `student_formatting_tokens`
- Aggregate (teacher+student) metrics may also be computed if enabled:
  - `group_caption_ce`, `group_grounding_ce`, `group_formatting_ce`
  - `group_caption_tokens`, `group_grounding_tokens`, `group_formatting_tokens`

### Invariants tied to spans and shifting
- Masks are aligned to the same shifted frame as CE (`[:, 1:]`).
- Disjointness: no overlap between caption/grounding/formatting per side.
- Coverage: union equals assistant masks (after shift) per side.
- Numeric coordinates inside geometry arrays always fall under grounding masks.
- JSON keys and punctuation fall under formatting.

### Failure modes (fail‑fast)
- Invalid or non‑JSON assistant output → raise with: offending substring and caret position (when available).
- Schema violation → raise with: expected keys, found keys, object index.
- Token span mismatch or invalid bounds → raise with actionable error.
- Empty group coverage (e.g., no caption tokens when `label` strings exist) → raise with counts for diagnostics.

### Implementation references
- Grouping: `src_new_json/losses/token_grouping.py`
- Loss integration: `src_new_json/models/loss_manager.py`
- Spans/labels (offset mapping): `src_new_json/data/dataset.py::_create_masked_labels_with_spans`

### Wrapper‑token implementation (legacy note)
- The `src_new/` pipeline uses wrapper/coord tokens for geometry. The grouping policy there is equivalent in spirit but implemented via token‑ID predicates over special tokens and coordinate ranges. The `src_new_json/` pipeline replaces that with JSON parsing and character‑span mapping to remain tokenizer‑agnostic.
