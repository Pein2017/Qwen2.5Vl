# Project (Outline) Rules

- This document defines non‑negotiable rules for all code and docs in this repository. Keep it short; deep dives live elsewhere.
- Use this document as the global baseline for every change.
- For specifics and deeper explanations, consult:
  - module-specific docs: `module_dir/*.md`
  - shared documentation: `./docs/`
- Project outline:
  - **src_new**: Training/models/processing for the detection‑focused VL pipeline. HF‑first integration (official processors), strict config (`src_new/config/config.py`), conversation and span alignment, coordinate token system, deterministic training, and inference utilities. See `src_new/README.md` and `src_new/UNIFIED_DOCUMENTATION.md`.
  - **data_conversion**: Unified processor to convert V2 annotations + images into training JSONL with strict object‑type filtering, geometry constraints, and hierarchical descriptions. See `data_conversion/README.md`.
- Workflow (E2E, concise):
  1) **Data conversion** → `data_conversion/convert_dataset.sh` (or `UnifiedProcessor`) produces `data/{dataset_name}/train.jsonl`, `val.jsonl`, `teacher.jsonl` (+ processed images).
  2) **Training** → `scripts/run_new_train.sh` launches the `src_new` pipeline using a YAML config. No in‑code hyperparameter defaults; pass everything explicitly.
  3) **Inference** → `python -m src_new.inference --config ... --checkpoint ... --image ...` for single‑image runs.
- Entry Points:
  - Training:
    ```bash
    bash scripts/run_new_train.sh
    ```
  - Inference:
    ```bash
    source ~/.bashrc && conda activate ms
    python -m src_new.inference --config /abs/path/to/config.yaml \
      --checkpoint /abs/path/to/checkpoint \
      --image /abs/path/to/image.jpg
    ```


# Environment Rules

- Required environment: use the `ms` conda virtual environment for all development and execution.
  - Activate with: `conda activate ms`
  - Or use direct Python path: `/root/miniconda3/envs/ms/bin/python`
- Conda activation for all shell sessions (manual, AI‑assisted, or new):
  - Standard activation: `conda activate ms`
  - If conda not initialized: `source ~/.bashrc && conda activate ms`
  - Direct Python: `/root/miniconda3/envs/ms/bin/python`
  - Note: Manual terminals auto‑source `~/.bashrc`; AI‑assisted terminals may need explicit sourcing


# Coding Rules

- Development mode (Fail‑Fast):
  - Never swallow errors; stop on first failure and fix at the source.
  - Validate inputs at boundaries; raise with actionable messages (what, where, how to fix).
  - Avoid silent defaults for hyperparameters/config; require explicit values or validated config objects.
  - Prefer explicit constructor args + dataclass/pydantic validation over permissive `dict.get`/implicit defaults.

- Configuration:
  - No silent defaults for core hyperparameters. Require explicit values via YAML/entry scripts and validate early with actionable errors.
  - Derived values allowed for non‑core path fields when `data_root` is provided (e.g., auto‑resolving `train_data_path`, `val_data_path`, `teacher_pool_file` via `DataResolver`); log the derivation.
  - Optional feature toggles may have explicit safe defaults that do not change core training semantics (e.g., `prog_unfreeze_coord_slice_only=True`, or deriving `new_geometry_tokens` when `coordinate_tokens_enabled=True`); validate consistency and document behavior.
  - Use a single schema (`@dataclass(frozen=True)`) with required fields first; optional fields typed as `Optional[...]` and handled explicitly. Provide a dedicated validation function that fails fast.

- Permitted mechanics:
  - `try/finally` and non‑suppressing context managers are allowed strictly for deterministic cleanup and must re‑raise exceptions. Do not mask or downgrade errors.
  - Use explicit conditionals with `raise` for validation. Reserve `assert` for internal invariants and tests only (not user input or runtime contracts).
  - Avoid wildcard imports in library code. Permitted only in vendor reference modules under `src_new/reference/offical_huggingface_qwen2_5_vl` (TYPE_CHECKING only) and in ad‑hoc scripts/notebooks via `src_new.utils.common_imports` (not in production modules).
  - Also prohibited: implicit re‑exports, magic numbers, hidden global state, `print` in libraries (CLI may print; libraries must use structured logging), silent `pass`, mutation of shared state across module boundaries, dynamic monkey patching in production paths, top‑level I/O/network/GPU initialization at import time.

- Module contracts — `src_new` (Training, Models, Processing):
  - Processing/vision tokens
    - Validate vision token expansion exactly: `expected_image_tokens = sum_i (t_i*h_i*w_i) // (merge_size**2)`. Raise on any mismatch; do not coerce or pad silently.
    - Image grid, merge size, and processor assumptions must be explicit and validated upfront.
  - Models/config integration
    - `DetectionModel.config` must proxy the underlying HuggingFace model config. Keep training dataclass on `training_config` only.
    - All integrations that call `model.config.to_json_string()` or similar must continue to work.
  - Conversation and labels
    - Teacher–student conversations must be complete and consistent. Include `<|im_end|>` in assistant span labels during training.
    - Span detection must rely on explicit offset mapping; raise if alignment is missing or ambiguous.
  - Training (BBUTrainer and shells)
    - Fail immediately on non‑finite losses or invalid gradients.
    - Checkpoint saving must preserve `