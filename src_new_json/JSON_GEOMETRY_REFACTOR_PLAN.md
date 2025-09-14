### JSON-based Geometry Refactor Plan (src_new_json)

Goal: remove coordinate tokens and special wrapper tokens; unify geometry I/O as strict JSON. Maintain teacher–student flows, HF-first pipeline, span alignment, and grouped CE losses using JSON-aware grouping.

---

#### 1) New target JSON schema for geometry outputs
- Assistant response (dense caption): a single JSON array, one object per instance.
```json
[
  {"box": [[x1, y1], [x2, y2]], "label": "..."},
  {"quad": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], "label": "..."},
  {"line_points": [[x1, y1], [x2, y2], [x3, y3]], "label": "..."}
]
```
- Variants mapping:
  - dense_caption: assistant returns full array (label + geometry)
  - coords_to_desc: user provides array items with only geometry; assistant fills "label"
  - desc_to_coords: user provides array items with only "label"; assistant fills geometry
  - summary: unchanged (single-line Chinese summary, no JSON)
- Strict validation:
  - Keys allowed: "label", "box", "quad", "line_points"
  - Exactly one geometry field per object; integer absolute coordinates
  - No extra keys; fail-fast with actionable messages

---

#### 2) Remove coordinate-token and wrapper-token plumbing
- Delete usage of `<|coord_*|>`, `<|object_ref_start|>`, `<|box_start|>`, `<|quad_end|>`, `<|line_*|>` across processing, losses, and inference. Replace with JSON.
- Affected (evidence and intent):
  - `src_new_json/processing/coordinate_converter.py`: remove token emission and wrappers
  - `src_new_json/processing/geometry_text.py`: replace wrapper formatting with JSON formatting
  - `src_new_json/processing/token_processor.py`: remove coordinate/line token vocabulary additions and validations
  - `src_new_json/processing/special_tokens.py`: drop GEOMETRY_TOKENS, coord token range helpers; keep IM_START/IM_END/IMAGE_PAD only
  - `src_new_json/losses/token_grouping.py` and `src_new_json/losses/grouping_core.py`: replace grouping predicates over token IDs (wrappers/coord range) with JSON-structure grouping (see §4)
  - `src_new_json/inference.py`: remove wrapper/coord-token parsing paths; add strict JSON parser
  - `src_new_json/models/loss_manager.py`: remove coord-aux paths and any expectations of coord-label targets
  - `src_new_json/models/wrapper.py`: remove checkpoint extension/coord-token detection logic

---

#### 3) Conversation building (JSON-first)
- Introduce `JsonGeometryFormatter` (new module `src_new_json/processing/json_formatter.py`):
  - `build_dense_caption(objects) -> str`: create strict JSON string per schema
  - `build_coords_to_desc_user(objects) -> str`: geometry-only JSON for user turn
  - `build_desc_to_coords_user(objects) -> str`: label-only JSON for user turn
  - Utilities: validate schema, reorder keys, ensure ints, clamp to image bounds if needed (optional)
- Update registry/variants:
  - `src_new_json/processing/variants.py`: replace `CoordinateTokenConverter` calls with `JsonGeometryFormatter`
  - `src_new_json/processing/templates.py`: update prompts to instruct “return valid JSON only; no extra text”; provide short examples per variant
- Keep teacher–student interleaving and span extraction unchanged (assistant spans still derived from `<|im_start|>assistant ... <|im_end|>`)

---

#### 4) Grouped LLM losses over JSON (caption/grounding/formatting)
- Redefine groups:
  - caption: JSON string values of the "label" fields only (exclude quotes)
  - grounding: numeric literals within geometry arrays (numbers in "box", "quad", "line_points")
  - formatting: JSON syntax (brackets, braces, commas, colons, quotes), whitespace, and JSON keys ("label", "box", "quad", "line_points")
- Implementation approach:
  - Decode assistant text (already available via span pipeline); parse with a strict JSON parser (fail-fast) inside group builder
  - Compute character span masks for caption/grounding/formatting by traversing the parsed JSON and mapping back to the original string indices
  - Use tokenizer offset mapping (existing decode→re-tokenize flow) to convert char spans to token index spans; intersect with teacher/student spans; shift to CE frame
- Code changes:
  - Replace internals of `src_new_json/losses/token_grouping.py` to call a new helper (e.g., `build_json_based_group_masks(...)`)
  - Replace `src_new_json/losses/grouping_core.py` with JSON-centric primitives (remove ID-set logic for wrappers/coord ranges)
  - Update `src_new_json/group_tokens.md` to document new definitions and invariants

---

#### 5) Inference pipeline
- `src_new_json/inference.py`:
  - Remove `_parse_coordinate_token_response`, `_parse_geometry_token_response`
  - Add `parse_json_prediction(response: str) -> List[Dict[str, Any]]` (strict schema)
  - Continue using `_convert_objects_to_vis_format(...)` / `_convert_objects_to_training_format(...)` by mapping schema back to existing internal shapes `{bbox_2d|quad|line} + desc`
  - Enforce JSON-only outputs in prompts; add fail-fast diagnostics (sample id, offending substring) on parse errors

---

#### 6) Config cleanup
- Remove keys:
  - `coordinate_tokens_enabled`, `coordinate_init_mode`, `max_coord_value`, `new_geometry_tokens`
  - All coord-aux fields: `coord_aux_enabled`, `coord_aux_*`
- Remove validations and logs referencing the above (e.g., pre-expanded checkpoint checks)
- Add optional JSON schema aliases (future-proof):
```yaml
json_geometry:
  label_key: label
  box_key: box
  quad_key: quad
  line_key: line_points
```
- Keep `conversation_variant_ratios` as-is (keys: dense_caption, coords_to_desc, desc_to_coords, summary)

---

#### 7) Dataset and teacher pool usage
- Input remains your existing `data/ds_v2_full/*.jsonl` (objects with `bbox_2d|quad|line` and `desc`)
- Conversation builder formats those into JSON strings per variant
- Add lightweight validator during dataset preprocessing to ensure assistant targets are valid JSON (optional dev gate)

---

#### 8) Tests/docs to update
- Update tests to assert JSON snippets rather than wrapper/coord tokens:
  - `src_new_json/tests/test_processing_integration.py`
  - `src_new_json/tests/test_processing_integration_real.py`
  - `src_new_json/tests/test_conversation_variants.py`
  - `src_new_json/tests/test_group_token_losses.py` (construct assistant JSON and verify mask counts)
- Remove coord-aux tests; remove wrapper-token assertions
- Update docs:
  - `src_new_json/group_tokens.md` (JSON grouping)
  - `src_new_json/data_details.md` (note: assistant outputs are JSON; desc still hierarchical within label)

---

#### 9) Training and wrappers
- `src_new_json/models/wrapper.py`: drop coordinate token detection/extension; keep HF-first integrations
- `src_new_json/models/loss_manager.py`: remove coord-aux computation and related diagnostics/metrics; keep grouped CE path
- `src_new_json/data/collator_*` and `src_new_json/data/dataset.py`: no geometry-token coupling; ensure no imports rely on special tokens beyond IM_START/IM_END

---

#### 10) Stepwise implementation (recommended PR sequence)
1) Config & tokenizer cleanup
   - Remove coord-token settings, new-geometry-tokens, and pre-expanded checkpoint validations
   - Ensure build/lint/tests run
2) JSON formatter + variants
   - Add `processing/json_formatter.py` and wire into `processing/variants.py`
   - Update `templates.py` prompts + simple examples
3) Dataset & minimal E2E
   - Use JSON assistant/user texts in conversation builder
   - Run a small training sanity test (without grouped losses) to ensure CE path works
4) Grouped losses (JSON-based)
   - Implement JSON span → token masks; integrate into `token_grouping.py`
   - Update metrics & logs
5) Inference JSON parser
   - Replace old parsers; enforce JSON-only response; map back to vis/training formats
6) Tests & docs
   - Update unit/integration tests and docs per above

---

#### 11) Failure modes & diagnostics (fail-fast)
- JSON parse error → raise with: sample id, offending substring, caret position
- Schema violation → raise with: expected keys, found keys, object index
- Grouped mask coverage mismatch → raise with: which group is empty and counts
- Inference non-JSON output → raise with first 200 chars for debugging

---

#### 12) Acceptance criteria
- Training runs with JSON outputs; grouped CE emits non-zero caption/grounding/formatting token counts
- Inference returns valid JSON per schema; visualization succeeds
- No references to `<|coord_*|>` or geometry wrapper tokens remain in `src_new_json/`

---

#### 13) File-by-file checklist (initial)
- Remove/replace wrappers & coord tokens:
  - processing: `coordinate_converter.py`, `geometry_text.py`, `special_tokens.py`, `token_processor.py`, `variants.py`, `conversation_processor.py` (builder hooks)
  - losses: `token_grouping.py`, `grouping_core.py`, remove `coord_aux.py`
  - models: `loss_manager.py` (drop coord aux), `wrapper.py` (drop coord detection)
  - inference: `inference.py` (JSON parser)
  - config: `config.py` (remove coord fields; add optional json_geometry aliases; remove packed collator and `packed_segment_isolation`)
  - tests: `test_*` updated to JSON expectations
  - docs: `group_tokens.md`, `data_details.md`

Notes:
- Your raw input dataset (`teacher_pool.jsonl`) maps directly to the new JSON outputs (geometry+desc → label); no conversion needed on disk—only how we render assistant/user texts changes.
- Keep the teacher–student mechanism and image/token validations intact.

---

#### 14) File-by-file review and modularization plan

This pass ensures we don’t miss any downstream references and guides decoupled design with clean interfaces.

- processing/
  - conversation_processor.py
    - Keeps HF `apply_chat_template` and assistant span invariants. No hard dependency on wrapper/coord tokens; safe. Ensure it delegates variant text building to a pluggable formatter (see JsonGeometryFormatter) via a narrow interface (e.g., VariantRegistry with callables returning strings).
  - variants.py
    - Replace `CoordinateTokenConverter` with `JsonGeometryFormatter`. Ensure handler methods return JSON strings. Keep the registry pattern and expose a minimal interface: `build_user_text(objects) -> Optional[str]`, `build_assistant_text(objects) -> str`. This keeps the module reusable and testable.
  - geometry_text.py
    - Will be deprecated or kept as a thin shim that raises if called. Prefer removing wrapper-formatting and let formatter produce JSON only.
  - coordinate_converter.py
    - Remove; superseded by `json_formatter.py`.
  - templates.py
    - Update prompts to JSON-only instructions. Keep a single place for user/system prompt texts. Avoid embedding schema in too many places; instead import schema key names from a single `json_schema.py`.
  - span_extraction.py
    - Independent of wrapper tokens; remains as-is. Reuse for JSON grouping by mapping decoded char spans to tokens.
  - token_processor.py, special_tokens.py
    - Remove geometry/wrapper/coord-token logic. Keep only IM_START/IM_END/IMAGE_PAD and general helpers. Leave hooks for future special-token additions as a list (pluggable) to preserve extensibility.

- losses/
  - token_grouping.py
    - Refactor to call a new `JsonGroupingEngine` with a stable interface:
      - `build_masks_from_json(decoded_text, offset_mapping, assistant_spans, tokenizer, schema) -> GroupMasks`
    - Internally uses a JSON span finder. No tokenizer ID lists necessary; robust to tokenizer changes.
  - grouping_core.py
    - Replace with JSON-centric helpers: JSON traversal → char spans; char spans → token spans (via provided offset mapping). Keep pure functions; no global state.
  - coord_aux.py
    - Remove; not used.

- models/
  - loss_manager.py
    - Remove coord-aux usage and diagnostics. Depend on `TokenGroupingPlugin` only. Keep the grouped CE logic unchanged.
  - wrapper.py
    - Remove checkpoint coord-token analysis and pre-expansion logic. Keep tokenizer/model config alignment and HF integration.
  - coord_metrics.py
    - Remove or gate behind a feature flag; not used without coord tokens.

- data/
  - dataset.py
    - Ensure it only depends on `ConversationBuilder` public API; not on wrappers. Add optional validation for assistant JSON outputs during preprocessing (development mode only) to fail-fast.
  - collator_standard.py
    - Continue as the sole collator. Ensure assistant spans are attached consistently.
  - collator_packed.py
    - Removed.
  - teacher_pool.py
    - No changes; still serves raw objects with `bbox_2d|quad|line` + `desc`.

- config/
  - config.py
    - Remove coord-related fields; add optional `json_geometry` alias block. Keep `conversation_variant_ratios`. Ensure train entry (`scripts/train_new.py`) continues to work with updated config by ignoring removed fields (handle missing fields gracefully with clear errors).

- inference.py
  - Replace geometry-token parsing with `parse_json_prediction(...)`. Add `JsonSchema` object injected via config or default. Enforce JSON-only outputs with clear remediation hints.

- tests/
  - Update expectations to JSON outputs. Build unit tests for:
    - Formatter (objects → JSON text)
    - Grouping engine (JSON → masks with correct counts)
    - Inference JSON parser (robustness to minor formatting variations when allowed)

- utils/
  - validation.py
    - Reuse for JSON schema validation: introduce a small `json_utils.py`:
      - `validate_geometry_json(text: str, schema: JsonSchema) -> List[Issue]`
      - `extract_char_spans(text: str, schema: JsonSchema) -> Dict[group, List[(start,end)]]`
    - Keep failure messages actionable; include sample id when available.

- types/
  - format.py
    - Remove `FormatMode.COORD_TOKENS`; retain enum for variants. Add a dataclass `JsonSchema` with keys for label/box/quad/line_points to centralize schema customizations.
  - geometry.py, coords.py
    - Retire coord-range types; add simple typed aliases for geometry lists to improve clarity.

---

#### 15) Decoupled, pluggable components (interfaces)

- JsonGeometryFormatter (processing/json_formatter.py)
  - Inputs: `objects: List[Dict[str, Any]]`, `schema: JsonSchema`
  - Outputs: `str` (strict JSON)
  - Methods: `build_dense_caption`, `build_coords_to_desc_user`, `build_desc_to_coords_user`

- JsonGroupingEngine (losses/json_grouping.py)
  - Inputs: `decoded_text: str`, `offset_mapping: Tensor`, `assistant_spans: List[List[(s,e)]]`, `schema: JsonSchema`
  - Outputs: `GroupMasks`
  - Pure; no tokenizer-vocab assumptions beyond offset mapping

- JsonSchema (types/json_schema.py)
  - Dataclass with `label_key`, `box_key`, `quad_key`, `line_key`
  - Default values match plan; configurable via config

- JsonParser/Validator (utils/json_utils.py)
  - `parse_and_validate(text: str, schema: JsonSchema) -> List[Dict[str, Any]]`
  - `to_internal_objects(parsed) -> List[Dict[str, Any]]` mapping JSON → `{bbox_2d|quad|line}+desc`

This separation enables reusability and easier future changes (e.g., adding polygons/beziers) without touching losses or conversation building logic.

---

#### 16) Migration notes & compatibility switches
- Provide a temporary adapter that can still consume legacy wrapper-token datasets during a transition window (optional). A compile-time or config flag `use_legacy_wrappers` could select old converter vs JSON formatter. Default off.
- Keep all fail-fast checks but add a `debug_json_tolerant_parse` optional flag to auto-fix common JSON mistakes in development.

---

#### 17) Concrete rename/remove list (search-and-destroy checklist)
- Remove files or sections:
  - `processing/coordinate_converter.py` (entire)
  - `processing/geometry_text.py` (entire or leave shim raising NotImplementedError)
  - `processing/special_tokens.py` geometry token sets and coord token range helpers
  - `processing/token_processor.py` coord/line token vocabulary code paths
  - `losses/coord_aux.py`, `models/coord_metrics.py` (entire)
  - Wrapper/coord parsing in `inference.py`
  - Coord extension analysis in `models/wrapper.py`
- Add files:
  - `