# Coding Rules

- Development mode (Fail‑Fast):
  - Never swallow errors; stop on first failure and fix at the source.
  - Validate inputs at boundaries; raise with actionable messages (what, where, how to fix).
  - Avoid silent defaults for hyperparameters/config; require explicit values or validated config objects.
  - Prefer explicit constructor args + dataclass/pydantic validation over permissive `dict.get`/implicit defaults.

- Configuration:
  - No silent defaults for core hyperparameters. Require explicit values via YAML/entry scripts and validate early with actionable errors.
  - Derived values allowed for non‑core path fields when `data_root` is provided (auto‑resolve `train_data_path`, `val_data_path`, `teacher_pool_file` via a resolver); log the derivation.
  - Optional feature toggles may have explicit safe defaults that do not change core training semantics.
  - Use a single schema (`@dataclass(frozen=True)`) with required fields first; optional fields typed as `Optional[...]` and handled explicitly. Provide a dedicated validation function that fails fast.

- Permitted mechanics:
  - `try/finally` and non‑suppressing context managers for deterministic cleanup; re‑raise exceptions.
  - Use explicit conditionals with `raise` for validation. Reserve `assert` for internal invariants and tests only.
  - Avoid wildcard imports in production modules. No hidden global state, implicit re‑exports, magic numbers, `print` in libraries, silent `pass`, cross‑module mutable state, dynamic monkey‑patching, or top‑level I/O/GPU init.

- Module contracts — `src_new` (Training, Models, Processing):
  - Processing/vision tokens
    - Validate vision token expansion exactly: `expected_image_tokens = sum_i (t_i*h_i*w_i) // (merge_size**2)`; raise on any mismatch.
    - Image grid, merge size, and processor assumptions must be explicit and validated upfront.
  - Models/config integration
    - `DetectionModel.config` must proxy the underlying HuggingFace model config. Keep training dataclass on `training_config` only.
    - Integrations that call `model.config.to_json_string()` and similar must continue to work.
  - Conversation and labels
    - Teacher–student conversations complete and consistent; include `<|im_end|>` in assistant span labels during training.
    - Span detection must rely on explicit offset mapping; raise if alignment is missing or ambiguous.
  - Training (BBUTrainer and shells)
    - Fail immediately on non‑finite losses/invalid gradients.
    - Checkpoint saving must preserve `processing_class` and metadata required for clean reloads.
    - Determinism: set seeds where feasible; document trade‑offs; exact determinism is best‑effort.

- Module contracts — `data_conversion` (Unified Processor):
  - Enforce object type filtering using `{bbu, bbu_shield, connect_point, label, fiber, wire}`; drop unknown types.
  - Enforce geometry constraints: `fiber, wire` are lines; others are quad/bbox. Raise on any mismatch.
  - Hierarchical description formatting is strict: comma separates same‑level attributes, slash separates levels.
  - Validation fails fast with actionable messages; do not auto‑correct or infer missing fields.
  - Config dataclass is frozen and explicit; treat `input_dir`, `output_dir`, `object_types` as required.
  - CLI scripts require explicit values; exit non‑zero on missing/invalid.

- Coding standards (Be professional and critical):
  - Strong typing on public APIs and non‑trivial locals. Clean naming, small composable functions, guard clauses.
  - DRY, clear abstraction boundaries (data I/O, processing/logic, CLI).
  - Error messages include expected vs actual, key values (sanitized), and remediation hints.
  - Immutability by default; avoid mutating inputs.
  - Import hygiene: prefer absolute imports; define `__all__` for public APIs.
  - Performance: linear‑time passes and minimal allocations in hot paths; measure before optimizing.

- Typing (Strict):
  - Centralize domain types in `src_new/types/`; export via `src_new/types/__init__.py`.
  - Prefer `@dataclass(frozen=True)` for value objects.
  - Use `TypedDict`/`NamedTuple` where appropriate. For tensors, prefer `jaxtyping` + `beartype` at boundaries.
  - Explicit `Optional[...]` and `None` checks; avoid implicit `Any`.

- Documentation:
  - Keep this file concise; defer deep dives to module docs and `./docs/`.
  - Public functions/classes must document purpose, inputs, outputs, and failure modes.
  - Update adjacent docs when behavior, shapes, or interfaces change.

- CI and Local Gates:
  - Run linters/types locally: ruff/flake8, black; pyright/mypy (strict); warnings as errors in tests.
  - Pre‑commit hooks for format, lint, type checks are encouraged.
  - Commit messages state intent and impact; reference modules and user‑visible effects.

- Decision rules (When in doubt):
  - Stop, write a failing test, clarify the contract.
  - Prefer explicit validation and early raising over defensive defaults.
  - If a rule here conflicts with a module doc, this file wins; move the detail to the module doc and link it.