## Redundancy and Usage Inventory (src_new)

Scope: quick, grep-based audit of modules under `/data3/Qwen2.5-VL-main/src_new` as used by the new training/inference entry points (`/data3/Qwen2.5-VL-main/scripts/train_new.py`, `/data3/Qwen2.5-VL-main/src_new/inference.py`).

### Confirmed active (keep)
- `/data3/Qwen2.5-VL-main/src_new/models/wrapper.py` — `DetectionModel`; core forward, integrates loss manager and coord flow; used by train/inference.
- `/data3/Qwen2.5-VL-main/src_new/models/loss_manager.py` — loss orchestration (regular + coordinate aux); wired from wrapper and trainer.
- `/data3/Qwen2.5-VL-main/src_new/models/coord_metrics.py` — diagnostics; imported by loss manager, trainer, and training state manager.
- `/data3/Qwen2.5-VL-main/src_new/training/bbu_trainer.py` — primary trainer; used by `scripts/train_new.py`.
- `/data3/Qwen2.5-VL-main/src_new/training/unified_checkpoint_manager.py` — best-checkpoint copy/cleanup; used by trainer.
- `/data3/Qwen2.5-VL-main/src_new/training/callbacks.py` — progressive unfreeze, loss summaries; active at runtime (see training logs).
- `/data3/Qwen2.5-VL-main/src_new/data/dataset.py` — dataset and image loading; used by training.
- `/data3/Qwen2.5-VL-main/src_new/data/collator.py`, `/data3/Qwen2.5-VL-main/src_new/data/collator_standard.py`, `/data3/Qwen2.5-VL-main/src_new/data/collator_packed.py`, `/data3/Qwen2.5-VL-main/src_new/data/collator_utils.py` — data collation stack; used by trainer.
- `/data3/Qwen2.5-VL-main/src_new/processing/conversation_processor.py`, `/data3/Qwen2.5-VL-main/src_new/processing/conversation/{__init__.py,builder.py,validator.py}`, `/data3/Qwen2.5-VL-main/src_new/processing/templates.py` — conversation building/validation; used by train/inference.
- `/data3/Qwen2.5-VL-main/src_new/processing/token_processor.py` — token config/processing; used by inference.
- `/data3/Qwen2.5-VL-main/src_new/utils/rank_aware_logging.py` — logging backbone; used broadly (train, inference, utils).
- `/data3/Qwen2.5-VL-main/src_new/utils/path_manager.py` — unified path resolution; used by dataset and inference.
- `/data3/Qwen2.5-VL-main/src_new/utils/debug_logging.py` — optional runtime debug facility; referenced by trainer, appears in run logs.
- `/data3/Qwen2.5-VL-main/src_new/utils/validation.py`, `/data3/Qwen2.5-VL-main/src_new/utils/tensor_validation.py`, `/data3/Qwen2.5-VL-main/src_new/utils/error_formatting.py`, `/data3/Qwen2.5-VL-main/src_new/utils/data_resolver.py` — supporting utilities; referenced by active modules.
- Types: `/data3/Qwen2.5-VL-main/src_new/types/{__init__.py,coords.py,arrays.py,shapes.py,batch.py}` — used across collators/processing.

### Authoritative single sources of truth (once, forever)
- Coordinate token range (end‑exclusive): derive once via `src_new/processing/special_tokens.get_coord_token_range(tokenizer)` → tuple `(start_id, end_exclusive)`.
  - Store on `DetectionModel.coordinate_processor.coordinate_token_range` and pass through; do not rescan vocab in downstream code.
  - `LossManager` reads and caches `(start_id, end_exclusive)` at init.
  - `training/bbu_trainer.py` persists `[start_id, end_exclusive]` into `coordinate_config.json`.
  - `src_new/inference.py` uses the persisted range or derives once on startup; no repeated scans.
- Geometry tokens: `src_new/processing/special_tokens.GEOMETRY_TOKENS` is the only canonical set. Legacy `<|bbox_*|>` is not accepted. `OBJECT_REF_SYNONYMS` only for start/end wrappers.
- Conversation formatting: use `processor.apply_chat_template(...)` everywhere; source `chat_template` from the tokenizer and set on the processor once.
- Logging: use `src_new/utils/rank_aware_logging` exclusively as the acquisition surface.
- Configuration: a frozen dataclass with `schema_version`, no in‑code defaults for hyperparameters/runtime settings; explicit `None` for truly optional parameters only.
- Image processor settings: explicitly apply `max_pixels` and `size` overrides from config at construction time in both training and inference (no implicit defaults, no dummy assignments).
- Paths: enforce absolute paths for all critical inputs/outputs in config and CLI. `remove_unused_columns: false` must be asserted by trainer creation.

### Likely unused or low‑value (candidates to deprecate/move)
- `/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py`
  - No direct imports from active runtime found. Coordinate aux logic lives in `loss_manager.py` and `losses/coord_aux.py`.
  - Action: deprecate and update docs that point to this file, or fold any unique helpers into `losses/coord_aux.py`.
- `/data3/Qwen2.5-VL-main/src_new/utils/performance_monitor.py`
  - Only exported via `utils/__init__.py`; no runtime usage detected.
  - Action: either wire into init paths or move to dev‑only tooling and drop from public exports.
- `/data3/Qwen2.5-VL-main/src_new/utils/checkpoint_validator.py`
  - Not used by the new trainer/inference path; retain as a tooling script but remove from public `__all__`.
- `/data3/Qwen2.5-VL-main/src_new/utils/logger_factory.py`
  - Mixed adoption; prefer `rank_aware_logging`.
  - Action: keep as a thin shim (temporarily) or deprecate.
- `/data3/Qwen2.5-VL-main/src_new/processing/coordinate_converter.py`
  - Verify single point of use; if duplicated with conversation processing, merge or drop.

### Public API cleanups
- Shrink `src_new/utils/__init__.py::__all__` to the active runtime surface. Remove `PerformanceMonitor`, `validate_checkpoint`, and other legacy exports not referenced by `scripts/train_new.py` or `src_new/inference.py`.
- Consolidate unlikelihood helpers:
  - Keep `unlikelihood_topk_text` in `src_new/losses/coord_aux.py`.
  - Provide `build_noncoord_vocab_mask(vocab_size, start, end_exclusive)` there and import it in `loss_manager.py` (avoid re‑creating masks ad‑hoc).
  - If `losses/unlikelihood_text.py` is not needed beyond digit utilities, move `get_digit_token_ids`/`unlikelihood_topk_digits` under `coord_aux.py` or clearly mark `unlikelihood_text.py` as legacy.

### Concrete refactors (by file)
- `src_new/models/wrapper.py`
  - Coordinate range sourcing: in `CoordinateProcessor.set_tokenizer(...)` and `update_after_extension(...)`, replace vocab scans with `get_coord_token_range(tokenizer)`. Set `self.coordinate_token_range = (rng.start_id, rng.end_exclusive)` directly.
  - `mask_coordinate_logits(...)`: remove the `coordinate_token_range` parameter and use `self.coordinate_token_range` exclusively. Keep slices end‑exclusive.
- `src_new/models/loss_manager.py`
  - At init, continue to retrieve `(start_id, end_exclusive)` once and cache as `_coord_start_id`, `_coord_end_id`.
  - Replace in‑line non‑coord mask construction with `build_noncoord_vocab_mask(V, start, end_exclusive)` from `coord_aux.py`.
  - Keep all slices and `K = (end_exclusive - start) - 1` semantics.
- `src_new/inference.py`
  - Derive coordinate range once (or load from `coordinate_config.json`) and reuse; remove repeated scans.
  - Call `validate_geometry_tokens(tokenizer)` once after tokenizer load.
  - CLI: assert absolute paths for `--config_path`, `--model_path`, `--data_root`, `--output_file` before running.
  - Unify `trust_remote_code` policy across tokenizer/processor/image processor and document it.
- `scripts/train_new.py`
  - After creating `Qwen2VLImageProcessor`, apply explicit overrides from config:
    - Set `image_processor.do_resize` explicitly.
    - Set `image_processor.size` (e.g., `{"shortest_edge": X}`) and/or `image_processor.max_pixels` when supported by HF version.
  - Before `TrainingArguments`, assert `config.remove_unused_columns is False`.
- `src_new/config/config.py`
  - Make `Config` frozen and add `schema_version: str`.
  - Remove defaults for hyperparameters/runtime settings; require explicit YAML values (only truly optional fields remain Optional with `None`).
  - Validate that critical paths are absolute: `model_path`, `output_dir`, `tb_dir`, `data_root`, `train_data_path`, `val_data_path`, `teacher_pool_file`.
- `src_new/utils/__init__.py`
  - Remove unused exports from `__all__` and stop re‑exporting deprecated utilities.

### Deprecations and moves
- Mark the following as deprecated and update docs accordingly:
  - `src_new/models/coordinate_loss.py`
  - `src_new/utils/performance_monitor.py` (move to tools or wire into init if kept)
  - `src_new/utils/logger_factory.py` (shim only; remove when all modules import `rank_aware_logging` directly)
  - `src_new/processing/coordinate_converter.py` (merge or remove after verification)

### Implementation checklist (actionable)
- Image processor overrides:
  - [x] Training: set `do_resize`, `size`, and/or `max_pixels` in `scripts/train_new.py` explicitly (mirror inference behavior).
- Config hardening:
  - [x] Add `@dataclass(frozen=True)` and `schema_version`.
  - [ ] Remove defaults for required hyperparameters/runtime settings; require explicit YAML values.
  - [x] Assert `remove_unused_columns: false` before trainer creation.
- Logging consolidation:
  - [x] Migrate imports to `rank_aware_logging` in `models/wrapper.py`, `processing/token_processor.py`, `processing/conversation_processor.py`, `data/teacher_pool.py`, `config/config.py`.
  - [x] Shrink `utils/__all__`.
- Loss utilities consolidation:
  - [x] Deprecate or merge `losses/unlikelihood_text.py` (keep digit utilities if needed).
- Public API cleanups:
  - [x] Remove unused exports in `utils/__init__.py`.
  - [x] Move dev‑only utilities to a tools area and stop exporting them.

- Fail-fast enforcement (global):
  - [x] Remove silent fallbacks and broad try/except blocks across modules so misconfigurations raise immediately (wrapper, inference init, dataset/collators, processing, utils).
    - ✅ Fixed wrapper.py: Removed getattr defaults, broad try/except blocks, added explicit validation
    - ✅ Fixed inference.py: Removed getattr defaults, improved error handling for critical paths
    - 🔄 Additional modules (dataset/collators, processing, utils) may need review for remaining patterns

### Notes
- All slices and masks must treat coordinate ranges as end‑exclusive. Keep `(start_id, end_exclusive)`