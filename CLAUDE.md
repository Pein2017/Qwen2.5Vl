# Qwen2.5‑VL Global Development Guide (Authoritative)

This file defines non‑negotiable rules for all code and docs in this repository. Keep it short, keep it strict. Deep dives live elsewhere.

## Scope and Sources of Truth
- Use this document as the global baseline for every change.
- For specifics and deeper explanations, consult:
  - module-specific docs: `module_dir/*.md`
  - shared documentation: `./docs/`
- **Paths**: Always use absolute paths in scripts, commands, tooling arguments, and CLI inputs.

## Execution Environment
- **Required Environment**: Use the `ms` conda virtual environment for all development and execution.
  - Activate with: `conda activate ms`
  - Or use direct Python path: `/root/miniconda3/envs/ms/bin/python`
- **Conda Activation**: For all shell sessions (manual, AI-assisted, or new):
  - **Standard activation**: `conda activate ms`
  - **If conda not initialized**: `source ~/.bashrc && conda activate ms`
  - **Direct Python**: `/root/miniconda3/envs/ms/bin/python`
  - **Note**: Manual terminals auto-source `~/.bashrc`; AI-assisted terminals may need explicit sourcing

## Development Mode: Fail‑Fast + Test‑Driven Development
- **Fail‑Fast**: detect issues immediately, stop execution, fix at source.
  - Never catch and hide errors. Do not continue on error.
  - If input is invalid, raise immediately with an actionable message.
  - Validate all inputs upfront; raise on first violation with specific details.
  - No safe/defaulting accessors: `dict.get`, `getattr`, `setdefault`, `defaultdict`, implicit defaults, or silent fallbacks.
  - No implicit default values for function/method parameters that control hyperparameters or external configuration.
- **TDD First**: write or update tests before implementing or changing logic.
  - Place tests under `module_dir/tests/` with `test_*.py` naming.
  - Keep tests fast and deterministic. Use minimal synthetic fixtures.
  - Include negative-path tests (invalid inputs, boundary cases).
  - Suggested run flags: `pytest -q -x` (stop at first failure) and maintain meaningful coverage.

## Hyperparameters and Configuration
- Do not set in-code defaults for hyperparameters or training/runtime settings.
- All hyperparameters must be provided by the entry script and/or a config YAML.
- In code, define parameters as required or optional, but never assign a default value for hyperparameters.
  - Required: must be present; validate and raise immediately if missing/invalid.
  - Optional: use explicit `Optional[...]` types and explicit `None` handling in the caller; absence means the feature is disabled or the caller must decide. Do not inject defaults inside libraries.
- Provide a single schema per config (prefer `@dataclass(frozen=True)` with explicit types, no defaults). Implement a dedicated validation function that raises on the first violation.

## Permitted Mechanics (Strict)
- `try/finally` and non-suppressing context managers are allowed strictly for deterministic cleanup and must re-raise exceptions. Do not mask or downgrade errors.
- Use explicit conditionals with `raise` for validation. Reserve `assert` for internal invariants and tests only (not user input or runtime contracts).
- **Prohibited**: wildcard imports, implicit re-exports, magic numbers, hidden global state, `print` in libraries (CLI may print; libraries must use structured logging), silent `pass`, mutation of shared state across module boundaries, dynamic monkey patching in production paths, top-level I/O/network calls/GPU initialization at import time.

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
  - **Determinism**: set seeds at entry points; configure CUDA/CuDNN deterministic flags where relevant; document trade-offs.
- **Inference**
  - CLI takes absolute paths for `--config`, `--checkpoint`, and `--image`. No downloads at runtime; respect offline caches.

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
- Import hygiene: use absolute imports; avoid circular dependencies; define `__all__` for public APIs only.
- Explicit interfaces: no hidden magic, no implicit conversions; explicit returns and control flow.
- Performance: prefer linear-time passes and minimal allocations in hot paths; measure before optimizing.

## Testing Conventions
- Layout: `module_dir/tests/` (co-located with the code under test).
- Naming: `test_*.py` files; test functions `test_*`.
- Fixtures: keep small; use synthetic inputs; avoid network/filesystem unless explicitly required.
- GPU/accelerator tests: mark explicitly (e.g., `@pytest.mark.gpu`) and provide CPU fallbacks where reasonable.
- Golden tests: snapshot minimal, stable artifacts; update only with clear justification.
- Negative paths: assert failures with `pytest.raises(...)`.
- Assertions: check shapes, dtypes, ranges, and invariants, not just equality.
- Speed: individual tests should complete quickly to encourage frequent runs.

Suggested tests aligned to modules:
- `src_new/tests/test_vision_token_validation.py`: mismatch raises with the exact message and counts.
- `src_new/tests/test_conversation_eos_labels.py`: `<|im_end|>` is included and aligned in spans.
- `src_new/tests/test_detectionmodel_config_proxy.py`: HF config is proxied; `training_config` is separate.
- `data_conversion/tests/test_geometry_and_types.py`: geometry constraints and object type set enforced.
- `data_conversion/tests/test_hierarchy_format.py`: comma/slash rules; invalid inputs raise with specifics.

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
  - Types: mypy (prefer strict); error on implicit/unused `Any`
- Treat warnings as errors during tests for early surfacing.
- Pre-commit hooks for format, lint, and type checks are encouraged.
- Commit messages should describe intent and impact; reference modules and user-visible effects.

## When in Doubt
- Stop, write a failing test, and clarify the contract.
- Prefer explicit validation and early raising over defensive defaults.
- If a rule here conflicts with a module doc, this file wins. Move necessary detail to the module doc and link it from here.
