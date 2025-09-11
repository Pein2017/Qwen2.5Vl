# Qwen2.5‑VL Global Development Guide (Authoritative)

This file defines non‑negotiable rules for all code and docs in this repository. Keep it short, keep it strict. Deep dives live elsewhere.

## Scope and Sources of Truth
- Use this document as the global baseline for every change.
- For specifics and deeper explanations, consult:
  - module-specific docs: `module_dir/*.md`
  - shared documentation: `./docs/`

## Project Outline
- **src_new**: Training/models/processing for the detection‑focused VL pipeline. HF‑first integration (official processors), strict config (`src_new/config/config.py`), conversation and span alignment, coordinate token system, deterministic training, and inference utilities. See `src_new/README.md` and `src_new/UNIFIED_DOCUMENTATION.md`.
- **data_conversion**: Unified processor to convert V2 annotations + images into training JSONL with strict object‑type filtering, geometry constraints, and hierarchical descriptions. See `data_conversion/README.md`.

## Workflow (E2E, concise)
1) **Data conversion** → `data_conversion/convert_dataset.sh` (or `UnifiedProcessor`) produces `data/{dataset_name}/train.jsonl`, `val.jsonl`, `teacher.jsonl` (+ processed images).
2) **Training** → `scripts/run_new_train.sh` launches the `src_new` pipeline using a YAML config. No in‑code hyperparameter defaults; pass everything explicitly.
3) **Inference** → `python -m src_new.inference --config ... --checkpoint ... --image ...` for single‑image runs.

Use the Entry Points section below for exact commands and environment activation.

## Coding Rules (At a Glance)
- No in-code defaults for hyperparameters; pass via YAML/entry scripts and validate early with actionable errors.
- Type hints on all arguments, return types, class attributes, module-level constants, and local variables; clean, full-word naming; small single-responsibility functions with early returns.
- No wildcard imports, hidden globals, or top-level I/O on import; explicit control flow and interfaces only.
- Immutable configs: single frozen dataclass schema; explicit Optional handling; no implicit defaults inside libraries.
- Prefer deterministic seeds at entry points (best-effort); do not swallow errors; prefer linear-time passes in hot paths.

## Execution Environment
- **Required Environment**: Use the `ms` conda virtual environment for all development and execution.
  - Activate with: `conda activate ms`
  - Or use direct Python path: `/root/miniconda3/envs/ms/bin/python`
- **Conda Activation**: For all shell sessions (manual, AI-assisted, or new):
  - **Standard activation**: `conda activate ms`
  - **If conda not initialized**: `source ~/.bashrc && conda activate ms`
  - **Direct Python**: `/root/miniconda3/envs/ms/bin/python`
  - **Note**: Manual terminals auto-source `~/.bashrc`; AI-assisted terminals may need explicit sourcing

  
## Development Mode: Fail-Fast (AI-friendly)

* **Fail-Fast (严控失败，快速暴露/定位)**
  * Never swallow errors; **stop on first failure** and fix at the source.
  * Validate inputs at boundaries; raise with **actionable** messages (`what`, `where`, `how to fix`).
  * Avoid “silent defaults” for hyperparameters/config; require explicit values or validated config objects.
  * Prefer explicit constructor args + dataclass/pydantic validation over permissive `dict.get`/implicit defaults.



## Hyperparameters and Configuration
- No silent defaults for core hyperparameters. Require explicit values via YAML/entry scripts and validate early with actionable errors.
- Derived values allowed for non-core path fields when `data_root` is provided (e.g., auto-resolving `train_data_path`, `val_data_path`, `teacher_pool_file` via `DataResolver`); log the derivation.
- Optional feature toggles may have explicit safe defaults that do not change core training semantics (e.g., `prog_unfreeze_coord_slice_only=True`, or deriving `new_geometry_tokens` when `coordinate_tokens_enabled=True`); validate consistency and document behavior.
- Use a single schema (`@dataclass(frozen=True)`) with required fields first; optional fields typed as `Optional[...]` and handled explicitly. Provide a dedicated validation function that fails fast.

## Permitted Mechanics (Strict)
- `try/finally` and non-suppressing context managers are allowed strictly for deterministic cleanup and must re-raise exceptions. Do not mask or downgrade errors.
- Use explicit conditionals with `raise` for validation. Reserve `assert` for internal invariants and tests only (not user input or runtime contracts).
- Avoid wildcard imports in library code. Permitted only in vendor reference modules under `src_new/reference/offical_huggingface_qwen2_5_vl` (TYPE_CHECKING only) and in ad‑hoc scripts/notebooks via `src_new.utils.common_imports` (not in production modules). Also prohibited: implicit re-exports, magic numbers, hidden global state, `print` in libraries (CLI may print; libraries must use structured logging), silent `pass`, mutation of shared state across module boundaries, dynamic monkey patching in production paths, top-level I/O/network/GPU initialization at import time.

## Module Alignment and Contracts

### src_new (Training, Models, Processing)
- **Processing/vision tokens**
  - Validate vision token expansion exactly: `expected_image_tokens = sum_i (t_i*h_i*w_i) // (merge_size**2)`. Raise on any mismatch; do not coerce or pad silently.
  - Image grid, merge size, and processor assumptions must be explicit and validated upfront.
- **Models/config integration**
  - `DetectionModel.config` must proxy the underlying HuggingFace model config. Keep training dataclass on `training_config` only.
  - All integrations that call `model.config.to_json_string()` or similar must continue to work.
- **Conversation and labels**
  - Teacher–student conversations must be complete and consistent. Include `<|im_end|>` in assistant span labels during training.
  - Span detection must rely on explicit offset mapping; raise if alignment is missing or ambiguous.
- **Training (BBUTrainer and shells)**
  - Fail immediately on non‑finite losses or invalid gradients.
  - Checkpoint saving must preserve `processing_class` and any metadata required for clean reloads.
  - **Determinism**: set seeds at entry points where feasible; CUDA/CuDNN deterministic flags recommended; document trade-offs; exact determinism is best-effort, not mandatory.

### data_conversion (Unified Processor)
- **UnifiedProcessor pipeline**
  - Enforce object type filtering using the exact supported set: `{bbu, bbu_shield, connect_point, label, fiber, wire}`. Raise on unknown types.
  - Enforce geometry constraints: `fiber, wire` are lines; others are square/bbox. Raise on any mismatch.
  - Hierarchical description formatting is strict: comma separates attributes at the same level, slash separates levels.
  - Validation must fail fast with actionable messages; do not auto-correct or infer missing fields.
- **Configuration (DataConversionConfig)**
  - Define a frozen dataclass schema with explicit types and no defaults.
  - Treat `input_dir`, `output_dir`, and `object_types` as required. Other parameters (e.g., `val_ratio`, `max_teachers`, `resize`, `seed`) must be passed explicitly by callers; do not rely on library defaults.
- **CLI scripts**
  - `convert_dataset.sh` must require explicit values for all variables used in processing; exit non‑zero if any are missing or invalid.

## Coding Standards (Be Professional and Critical)
- Type hints everywhere: annotate function signatures, return types, module-level constants, and non-trivial locals.
- Clean naming: descriptive, full-word identifiers. Avoid abbreviations.
- Small, composable functions with single responsibilities; prefer guard clauses and early returns over deep nesting.
- DRY: extract shared logic; centralize constants; avoid duplication.
- Abstraction boundaries:
  - Data layer (I/O, serialization)
  - Processing/logic layer (pure or side‑effect constrained)
  - Presentation/CLI layer
- Error messages: include variable names and key values (sanitized), expected vs. actual, and a remediation hint.
- Immutability by default: prefer `@dataclass(frozen=True)` and do not mutate inputs; return new objects.
- Import hygiene: prefer absolute imports; allow relative imports within `src_new` and `data_conversion` to improve locality and avoid long paths; never cross package boundaries via relative imports; define `__all__` for public APIs only.
- Explicit interfaces: no hidden magic, no implicit conversions; explicit returns and control flow.
- Performance: prefer linear-time passes and minimal allocations in hot paths; measure before optimizing.



## Strict Typing Conventions
- Centralize shared domain types in `src_new/types/` and import from there. Public surface is exported via `src_new/types/__init__.py`.
- Prefer `@dataclass(frozen=True)` for value objects (e.g., `Box`, `Quad`, `Line` in `src_new/types/geometry.py`).
- Use `TypedDict`/`NamedTuple` where appropriate for structured mappings and read-only tuples (e.g., `MultimodalBatch`, `CoordTokenRange`).
- For tensors, use shape-annotated aliases with `jaxtyping` + `beartype` at boundaries (see `src_new/types/arrays.py`: `InputIds`, `AttentionMask`, `Labels`, `PixelValuesPacked`, `ImageGridTHW` and `jaxtyped_beartype`).
- Annotate locals in non-trivial logic blocks to improve debuggability and IDE navigation; avoid implicit `Any` and disable it in type checkers.
- Explicit `Optional[...]` and `None` checks; do not rely on truthiness for control flow decisions.

## Documentation Rules
- Keep this file concise; defer details to `module_dir/*.md` and `./docs/`.
- Every public function/class has a docstring describing purpose, inputs, outputs, and failure modes.
- Update adjacent docs when changing behavior, shapes, or interfaces.

## Entry Points (Minimal, for orientation only)
- Training:
  ```bash
  cd /data3/Qwen2.5-VL-main
  source ~/.bashrc && conda activate ms
  bash scripts/run_new_train.sh
  ```
- Inference:
  ```bash
  source ~/.bashrc && conda activate ms
  python -m src_new.inference --config /abs/path/to/config.yaml \
    --checkpoint /abs/path/to/checkpoint \
    --image /abs/path/to/image.jpg
  ```

## CI and Local Gates (Recommended)
- Run linters and type checks locally before committing.
  - Style/quality: ruff or flake8; formatting: black
  - Types: pyright (preferred) or mypy; prefer strict; error on implicit/unused `Any`
- Treat warnings as errors during tests for early surfacing.
- Pre-commit hooks for format, lint, and type checks are encouraged.
- Commit messages should describe intent and impact; reference modules and user-visible effects.

## When in Doubt
- Stop, write a failing test, and clarify the contract.
- Prefer explicit validation and early raising over defensive defaults.
- If a rule here conflicts with a module doc, this file wins. Move necessary detail to the module doc and link it from here.# Qwen2.5‑VL Global Development Guide (Authoritative)

This file defines non‑negotiable rules for all code and docs in this repository. Keep it short, keep it strict. Deep dives live elsewhere.

## Scope and Sources of Truth
- Use this document as the global baseline for every change.
- For specifics and deeper explanations, consult:
  - module-specific docs: `module_dir/*.md`
  - shared documentation: `./docs/`

## Project Outline
- **src_new**: Training/models/processing for the detection‑focused VL pipeline. HF‑first integration (official processors), strict config (`src_new/config/config.py`), conversation and span alignment, coordinate token system, deterministic training, and inference utilities. See `src_new/README.md` and `src_new/UNIFIED_DOCUMENTATION.md`.
- **data_conversion**: Unified processor to convert V2 annotations + images into training JSONL with strict object‑type filtering, geometry constraints, and hierarchical descriptions. See `data_conversion/README.md`.

## Workflow (E2E, concise)
1) **Data conversion** → `data_conversion/convert_dataset.sh` (or `UnifiedProcessor`) produces `data/{dataset_name}/train.jsonl`, `val.jsonl`, `teacher.jsonl` (+ processed images).
2) **Training** → `scripts/run_new_train.sh` launches the `src_new` pipeline using a YAML config. No in‑code hyperparameter defaults; pass everything explicitly.
3) **Inference** → `python -m src_new.inference --config ... --checkpoint ... --image ...` for single‑image runs.

Use the Entry Points section below for exact commands and environment activation.

## Coding Rules (At a Glance)
- No in-code defaults for hyperparameters; pass via YAML/entry scripts and validate early with actionable errors.
- Type hints on all arguments, return types, class attributes, module-level constants, and local variables; clean, full-word naming; small single-responsibility functions with early returns.
- No wildcard imports, hidden globals, or top-level I/O on import; explicit control flow and interfaces only.
- Immutable configs: single frozen dataclass schema; explicit Optional handling; no implicit defaults inside libraries.
- Prefer deterministic seeds at entry points (best-effort); do not swallow errors; prefer linear-time passes in hot paths.

## Execution Environment
- **Required Environment**: Use the `ms` conda virtual environment for all development and execution.
  - Activate with: `conda activate ms`
  - Or use direct Python path: `/root/miniconda3/envs/ms/bin/python`
- **Conda Activation**: For all shell sessions (manual, AI-assisted, or new):
  - **Standard activation**: `conda activate ms`
  - **If conda not initialized**: `source ~/.bashrc && conda activate ms`
  - **Direct Python**: `/root/miniconda3/envs/ms/bin/python`
  - **Note**: Manual terminals auto-source `~/.bashrc`; AI-assisted terminals may need explicit sourcing

  
## Development Mode: Fail-Fast (AI-friendly)

* **Fail-Fast (严控失败，快速暴露/定位)**
  * Never swallow errors; **stop on first failure** and fix at the source.
  * Validate inputs at boundaries; raise with **actionable** messages (`what`, `where`, `how to fix`).
  * Avoid “silent defaults” for hyperparameters/config; require explicit values or validated config objects.
  * Prefer explicit constructor args + dataclass/pydantic validation over permissive `dict.get`/implicit defaults.



## Hyperparameters and Configuration
- No silent defaults for core hyperparameters. Require explicit values via YAML/entry scripts and validate early with actionable errors.
- Derived values allowed for non-core path fields when `data_root` is provided (e.g., auto-resolving `train_data_path`, `val_data_path`, `teacher_pool_file` via `DataResolver`); log the derivation.
- Optional feature toggles may have explicit safe defaults that do not change core training semantics (e.g., `prog_unfreeze_coord_slice_only=True`, or deriving `new_geometry_tokens` when `coordinate_tokens_enabled=True`); validate consistency and document behavior.
- Use a single schema (`@dataclass(frozen=True)`) with required fields first; optional fields typed as `Optional[...]` and handled explicitly. Provide a dedicated validation function that fails fast.

## Permitted Mechanics (Strict)
- `try/finally` and non-suppressing context managers are allowed strictly for deterministic cleanup and must re-raise exceptions. Do not mask or downgrade errors.
- Use explicit conditionals with `raise` for validation. Reserve `assert` for internal invariants and tests only (not user input or runtime contracts).
- Avoid wildcard imports in library code. Permitted only in vendor reference modules under `src_new/reference/offical_huggingface_qwen2_5_vl` (TYPE_CHECKING only) and in ad‑hoc scripts/notebooks via `src_new.utils.common_imports` (not in production modules). Also prohibited: implicit re-exports, magic numbers, hidden global state, `print` in libraries (CLI may print; libraries must use structured logging), silent `pass`, mutation of shared state across module boundaries, dynamic monkey patching in production paths, top-level I/O/network/GPU initialization at import time.

## Module Alignment and Contracts

### src_new (Training, Models, Processing)
- **Processing/vision tokens**
  - Validate vision token expansion exactly: `expected_image_tokens = sum_i (t_i*h_i*w_i) // (merge_size**2)`. Raise on any mismatch; do not coerce or pad silently.
  - Image grid, merge size, and processor assumptions must be explicit and validated upfront.
- **Models/config integration**
  - `DetectionModel.config` must proxy the underlying HuggingFace model config. Keep training dataclass on `training_config` only.
  - All integrations that call `model.config.to_json_string()` or similar must continue to work.
- **Conversation and labels**
  - Teacher–student conversations must be complete and consistent. Include `<|im_end|>` in assistant span labels during training.
  - Span detection must rely on explicit offset mapping; raise if alignment is missing or ambiguous.
- **Training (BBUTrainer and shells)**
  - Fail immediately on non‑finite losses or invalid gradients.
  - Checkpoint saving must preserve `processing_class` and any metadata required for clean reloads.
  - **Determinism**: set seeds at entry points where feasible; CUDA/CuDNN deterministic flags recommended; document trade-offs; exact determinism is best-effort, not mandatory.

### data_conversion (Unified Processor)
- **UnifiedProcessor pipeline**
  - Enforce object type filtering using the exact supported set: `{bbu, bbu_shield, connect_point, label, fiber, wire}`. Raise on unknown types.
  - Enforce geometry constraints: `fiber, wire` are lines; others are square/bbox. Raise on any mismatch.
  - Hierarchical description formatting is strict: comma separates attributes at the same level, slash separates levels.
  - Validation must fail fast with actionable messages; do not auto-correct or infer missing fields.
- **Configuration (DataConversionConfig)**
  - Define a frozen dataclass schema with explicit types and no defaults.
  - Treat `input_dir`, `output_dir`, and `object_types` as required. Other parameters (e.g., `val_ratio`, `max_teachers`, `resize`, `seed`) must be passed explicitly by callers; do not rely on library defaults.
- **CLI scripts**
  - `convert_dataset.sh` must require explicit values for all variables used in processing; exit non‑zero if any are missing or invalid.

## Coding Standards (Be Professional and Critical)
- Type hints everywhere: annotate function signatures, return types, module-level constants, and non-trivial locals.
- Clean naming: descriptive, full-word identifiers. Avoid abbreviations.
- Small, composable functions with single responsibilities; prefer guard clauses and early returns over deep nesting.
- DRY: extract shared logic; centralize constants; avoid duplication.
- Abstraction boundaries:
  - Data layer (I/O, serialization)
  - Processing/logic layer (pure or side‑effect constrained)
  - Presentation/CLI layer
- Error messages: include variable names and key values (sanitized), expected vs. actual, and a remediation hint.
- Immutability by default: prefer `@dataclass(frozen=True)` and do not mutate inputs; return new objects.
- Import hygiene: prefer absolute imports; allow relative imports within `src_new` and `data_conversion` to improve locality and avoid long paths; never cross package boundaries via relative imports; define `__all__` for public APIs only.
- Explicit interfaces: no hidden magic, no implicit conversions; explicit returns and control flow.
- Performance: prefer linear-time passes and minimal allocations in hot paths; measure before optimizing.



## Strict Typing Conventions
- Centralize shared domain types in `src_new/types/` and import from there. Public surface is exported via `src_new/types/__init__.py`.
- Prefer `@dataclass(frozen=True)` for value objects (e.g., `Box`, `Quad`, `Line` in `src_new/types/geometry.py`).
- Use `TypedDict`/`NamedTuple` where appropriate for structured mappings and read-only tuples (e.g., `MultimodalBatch`, `CoordTokenRange`).
- For tensors, use shape-annotated aliases with `jaxtyping` + `beartype` at boundaries (see `src_new/types/arrays.py`: `InputIds`, `AttentionMask`, `Labels`, `PixelValuesPacked`, `ImageGridTHW` and `jaxtyped_beartype`).
- Annotate locals in non-trivial logic blocks to improve debuggability and IDE navigation; avoid implicit `Any` and disable it in type checkers.
- Explicit `Optional[...]` and `None` checks; do not rely on truthiness for control flow decisions.

## Documentation Rules
- Keep this file concise; defer details to `module_dir/*.md` and `./docs/`.
- Every public function/class has a docstring describing purpose, inputs, outputs, and failure modes.
- Update adjacent docs when changing behavior, shapes, or interfaces.

## Entry Points (Minimal, for orientation only)
- Training:
  ```bash
  cd /data3/Qwen2.5-VL-main
  source ~/.bashrc && conda activate ms
  bash scripts/run_new_train.sh
  ```
- Inference:
  ```bash
  source ~/.bashrc && conda activate ms
  python -m src_new.inference --config /abs/path/to/config.yaml \
    --checkpoint /abs/path/to/checkpoint \
    --image /abs/path/to/image.jpg
  ```

## CI and Local Gates (Recommended)
- Run linters and type checks locally before committing.
  - Style/quality: ruff or flake8; formatting: black
  - Types: pyright (preferred) or mypy; prefer strict; error on implicit/unused `Any`
- Treat warnings as errors during tests for early surfacing.
- Pre-commit hooks for format, lint, and type checks are encouraged.
- Commit messages should describe intent and impact; reference modules and user-visible effects.

## When in Doubt
- Stop, write a failing test, and clarify the contract.
- Prefer explicit validation and early raising over defensive defaults.
- If a rule here conflicts with a module doc, this file wins. Move necessary detail to the module doc and link it from here.