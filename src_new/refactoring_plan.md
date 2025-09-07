## Objectives

- Build a scalable, low-coupling pipeline with well-defined APIs between modules.
- Eliminate redundancy across conversation building, span extraction, and loss grouping.
- Make teacher/student symmetry first-class to avoid duplicate code and metrics.
- Keep behavior identical while refactoring (phase-in with compatibility aliases).

## Guiding Principles

- **Single responsibility**: each module does one job; rely on clear, typed inputs/outputs.
- **Symmetry by design**: role-agnostic logic, with role as a parameter (teacher|student).
- **Centralize invariants**: system prompt construction, chat-template application, span detection.
- **Composable variants**: add new conversation variants via registry, not new methods.
- **Zero hidden defaults**: use explicit config for all behaviors (already enforced in `config.py`).

## Current Pain Points (Observed)

- Repeated chat-template try/except and system prompt building across builders.
- Near-identical simple vs teacher-student builder flows per variant.
- Variant branching duplicated in dataset and builder.
- Span detection re-tokenizes full text; regex compiled per call; EOS inclusion logic spread out.
- Group losses recompute masked sums/cnts helpers multiple times; teacher/student metrics emitted as separate flat keys (hard to extend).
- Constants accessed via `CONSTANTS` and direct imports; legacy aliases linger.

## High-Level Refactoring Themes

- **Centralize conversation plumbing**: one wrapper to apply chat template safely and validate images; cache system prompt.
- **Unify variant dispatch**: registry + single entrypoint (simple/teacher-student paramized) to remove method explosion.
- **Extract common geometry and user-text formatting**: single utility used by both user builders and converters.
- **Span detection service**: one module that detects assistant spans with optional offset mapping (no re-tokenization if available).
- **Role-parameterized grouping and metrics**: compute once per role; provide combined and role-specific views via a small aggregator, not duplicate keys.
- **Compatibility layer**: maintain current behavior/keys while emitting the new structured metrics in parallel.

## Detailed Plan by Module

### processing/templates.py
- **Now**: Unified prompts (`BASE_USER_PROMPT`, `COORD_TO_DESC_USER_PROMPT`, `DESC_TO_COORD_USER_PROMPT`) + `CONSTANTS` (with legacy aliases).
- **Change**:
  - Prefer direct imports for internal usage. Keep `CONSTANTS` only for backwards-compat and external integrations.
  - Add `get_user_prompt(variant)` convenience if needed by inference tools.

### processing/coordinate_converter.py
- **Now**: Converts objects to tokens/desc-only/geometry-only; internal geometry extraction.
- **Change**:
  - Extract pure text-format helpers to `processing/geometry_text.py` (new):
    - `format_geometry_for_user(obj) -> str`
    - `format_object_ref(desc) -> str`
    - Shared punctuation/separator rules.
  - Keep `CoordinateTokenConverter` focused on assistant-side outputs.

### processing/conversation_processor.py
- **Now**: 6 builders (simple/teacher-student × 3 variants); repeated template calls; repeated system prompt building.
- **Change**:
  - Cache `self._system_prompt = get_system_prompt(coordinate_tokens_enabled)` once in `__init__`.
  - Add `_apply_chat_template_safe(messages, images, add_generation_prompt: bool) -> str` to unify try/except and image interleaving + validation.
  - Introduce `VariantHandler` protocol:
    - `build_user_text(objects) -> str`
    - `build_assistant_text(objects) -> str`
  - Add `VARIANT_REGISTRY: Dict[str, VariantHandler]` for `dense_caption`, `coords_to_desc`, `desc_to_coords`.
  - Collapse builders into:
    - `_build_simple(sample, images, variant_key)`
    - `_build_teacher_student(student_sample, teacher_samples, student_images, teacher_images_list, variant_key)`
  - Public methods route through a single `create_conversation(..., variant_key, teacher_samples=None, teacher_images_list=None)`; keep old names as thin wrappers calling the unified API.

### processing/conversation/builder.py
- **Now**: Thin façade proxying each concrete builder.
- **Change**:
  - Replace per-variant methods with a unified
    - `create_conversation(sample, images, variant, teacher_samples=None, teacher_images_list=None)`.
  - Keep existing methods as aliases to the unified one during migration.

### data/dataset.py
- **Now**: Duplicated variant branching in both teacher and student paths; re-tokenization for offsets later.
- **Change**:
  - Extract `_sample_variant() -> str` and `_prepare_images_and_samples(structured_sample) -> (has_teachers, student_images, teacher_images_list, teacher_samples)`.
  - Single branch:
    - `inputs = conversation_processor.create_conversation(structured_sample, student_images, variant, teacher_samples, teacher_images_list)`.
  - Offset mapping:
    - Modify `_process_text_and_images` to request `return_offsets_mapping=True` if the HF processor forwards it (guarded).
    - Store `offset_mapping` into `inputs` when available; update `_create_masked_labels_with_spans` to use it first, else fall back to re-tokenization.

### processing/special_tokens.py
- **Now**: `ASSISTANT_SPAN_PATTERN` as string.
- **Change**:
  - Add compiled regex: `ASSISTANT_SPAN_RE = re.compile(ASSISTANT_SPAN_PATTERN, re.DOTALL)`; export it.
  - Update span detection to accept a compiled regex.

### span extraction (new: processing/span_extraction.py)
- **Purpose**: centralize assistant span detection and eos-inclusion policy.
- **API**:
  - `find_assistant_spans(full_text: str, offset_mapping: Optional[torch.Tensor], tokenizer, include_eos: bool) -> List[(start_token, end_token, is_teacher)]`
  - Accept `has_teachers: bool` to tag first assistant span as teacher, others as student (current behavior preserved).
  - Use `ASSISTANT_SPAN_RE`. If `offset_mapping` missing, re-tokenize with tokenizer (fast tokenizer enforced by config).

### models/loss_manager.py
- **Now**: `_masked_sum_and_count` duplicated blocks; emits many flat keys (`student_caption_loss`, etc.).
- **Change**:
  - Private helpers: `_masked_sum_and_count`, `_masked_mean` once per class.
  - Introduce a small structure internally: `losses[role]['caption'|'grounding'|'formatting']` and totals.
  - Emission layer:
    - Preserve current flat keys (compatibility).
    - Also emit structured dict in `diagnostics` (e.g., `diagnostics['group_losses'] = losses`).
  - Prepare for future single-role aggregations by plugging a role iterator rather than duplicating code.

### losses/token_grouping.py
- **Now**: Single authority for caption/grounding/formatting masks; good separation.
- **Change**:
  - None functionally. Document invariants: caption excludes punctuation & geometry; grounding includes wrappers & coords; formatting is residual glue.

### models/wrapper.py
- **Now**: Wires `LossManager`, grouping plugin enablement, span usage, fallback path.
- **Change**:
  - No functional change. Consider passing a `roles=('teacher','student')` tuple to LossManager in the future to generalize beyond two roles.

### training/bbu_trainer.py
- **Now**: Accumulates loss components locally; surfaces student grouped metrics.
- **Change**:
  - Optionally log combined group losses (teacher+student) with keys like `eval/group_caption_loss_total` for at-a-glance trend, without removing current student-only keys.
  - Keep compute_loss pipeline intact.

### config/config.py
- **Now**: Strict, normalized, with legacy key mapping for ratios.
- **Change**:
  - Add `span.include_im_end_in_labels: bool` (default True) to centralize EOS inclusion policy used by span extraction.
  - After migration, deprecate legacy ratio keys and `CONSTANTS` indirection once docs and tests are updated.

### training/utils & callbacks
- **Change**:
  - Add a tiny `metrics_adapter` that can map structured `diagnostics['group_losses']` to flat keys for logs, easing the future removal of duplicate per-role flat keys.

## APIs After Refactor (Stable Contracts)

- **Conversation**
  - `ConversationBuilder.create_conversation(sample, images, variant, teacher_samples=None, teacher_images_list=None) -> Dict[str, torch.Tensor]`
  - Variant extensibility via `VariantHandler` registry (add new handler, register key).

- **Span Extraction**
  - `find_assistant_spans(full_text, offset_mapping, tokenizer, include_eos=True, has_teachers: bool) -> List[(start, end, is_teacher)]`

- **Loss Grouping**
  - `LossManager.compute_loss_from_logits(...)` returns current flat keys + `diagnostics['group_losses']` structured view.

## Migration Plan & Milestones

- **Milestone 1 (S)**: Centralize chat-template wrapper + cache system prompt.
  - Risk: None; behavior identical.
- **Milestone 2 (M)**: Variant registry + unified builder entrypoint; keep existing methods as delegating aliases.
  - Risk: Low; extensive unit tests for all three variants (simple & teacher-student).
- **Milestone 3 (M)**: Dataset unification: `_sample_variant`, `_prepare_images_and_samples`, single call to unified builder.
  - Risk: Low; relies on existing helpers; add tests for teacher and student flows.
- **Milestone 4 (M)**: Span extraction module + offset_mapping threading.
  - Risk: Medium; depends on processor support. Implement safe fallback to re-tokenization.
- **Milestone 5 (M)**: LossManager helpers + structured group-loss diagnostics. Keep current flat keys.
  - Risk: Low; only adds diagnostics. Trainer keeps current metrics surfacing.
- **Milestone 6 (S)**: Config addition `span.include_im_end_in_labels` and usage.
  - Risk: Low; default mirrors current behavior.
- **Milestone 7 (S)**: Documentation & tests update; deprecation notices for legacy keys and `CONSTANTS` reliance.

## Testing Strategy

- Conversation snapshots: golden raw chat templates for all variants (simple/teacher-student), numeric & token coord modes.
- Span alignment: property tests asserting spans cover exactly assistant content and include `<|im_end|>` when enabled.
- Group losses: synthetic samples targeting caption-only, geometry-only, and mixed; assert masking/magnitude behaves as expected.
- Dataset sampling: ratio-driven variant frequency; deterministic with fixed seed.

## Performance Considerations

- Avoid extra `apply_chat_template` calls; reuse cached prompt and centralized wrapper.
- Use offset_mapping path when available to skip re-tokenization.
- Minimize Python-level string handling by consolidating formatting helpers.

## Deprecations (Post-Migration)

- Remove `CONSTANTS` for internal calls; keep only for external-facing compatibility if needed.
- Drop legacy ratio keys and synonyms (`dense_captioning`, `coord_to_desc`, `desc_to_coord`) after configs are migrated; enforce canonical: `dense_caption`, `coords_to_desc`, `desc_to_coords`.
- Remove per-variant builder methods from `ConversationBuilder` once all callsites use the unified entry.

## Expected Impact

- Reduced method count and code duplication in conversation building.
- Cleaner role-parameterized metrics; easier to add new roles or group categories.
- Faster data path with cached system prompt and offset-aware span detection.
- Easier onboarding for new variants via registry-based design.

---

## Progress Log (executed)

- [x] Milestone 1: Cached system prompt in `ConversationProcessor.__init__` (`self._system_prompt`) and added `_apply_chat_template_safe(...)` to unify validation + interleaving + template application.
- [x] Milestone 2 (part): Added unified variant dispatcher in `ConversationProcessor`:
  - `_get_variant_handlers(variant)` to return user/assistant builder lambdas.
  - `_build_simple_conversation_unified(...)` and `_build_teacher_student_conversation_unified(...)` using handlers.
  - Public `create_conversation(...)` unified entrypoint.
  - Kept existing per-variant methods as compatibility wrappers calling the same internals.
- [x] Builder façade: Exposed `ConversationBuilder.create_conversation(...)` that forwards to processor.
- [x] Milestone 3: Dataset unification to call unified entrypoint; centralized `_sample_variant()` and image loading; preserved augmentation & labels.
- [x] Milestone 4 (part): Added `processing/span_extraction.py` and compiled `ASSISTANT_SPAN_RE`; dataset now uses `find_assistant_spans(...)` with config flag `span_include_im_end_in_labels`.
- [x] Milestone 4 (finish): Threaded `offset_mapping` from processor outputs to dataset; dataset now consumes `conversation_text` and `offset_mapping` when present and falls back to re-tokenization otherwise.
- [x] Milestone 5 (done): LossManager helper de-dup (masked sum/mean) + structured diagnostics (`diagnostics['group_losses']`) while preserving flat keys.
- [x] Milestone 6 (done): VariantHandler registry and de-duplication
  - Implemented `processing/variants.py` with handlers and `create_default_variant_registry()`; `ConversationProcessor` now resolves via registry.
  - Moved user-text formatting out of the processor.
- [x] Milestone 7 (done): Extract geometry text helpers
  - Added `processing/geometry_text.py` with `format_geometry_for_user` and `format_object_ref`.
  - `ConversationProcessor` uses shared helpers through the registry.
- [x] Milestone 9 (done): Variant curriculum (optional)
  - `Dataset.set_epoch()` supports `conversation_variant_schedule`, updating active ratios per epoch.
- [x] Milestone 10 (done): Metrics adapter and combined group-loss logging
  - Added `training/metrics_adapter.py`; `BBUTrainer` augments logs with per-role and combined totals.
- [x] Fail-fast adoption: Removed best-effort and fallback paths in conversation processing and inference
  - Strict `apply_chat_template` (no TypeError fallback), strict tokenizer `offset_mapping` requirement in `_process_text_and_images`.
  - Inference path raises on sys.path adjustments, teacher pool resolution, and builder failures.

Config in use for development: `configs/phase_1/debug.yaml`.

Next:

- [x] Milestone 11 (done): Remove legacy per-variant builder APIs
  - Deprecated per-variant methods in `ConversationBuilder` and `ConversationProcessor`; they now raise with guidance to use `create_conversation(..., variant=...)`.
  - Unified canonical keys: `dense_caption`, `coords_to_desc`, `desc_to_coords`; removed alias handling across the stack.


