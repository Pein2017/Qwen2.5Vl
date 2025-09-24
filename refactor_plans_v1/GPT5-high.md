### Qwen2.5‑VL Unified Fine‑Tuning & Post‑Training (SFT + GRPO)

A high‑level refactor plan to merge `src_rl/` into `src_new/` and treat SFT and GRPO uniformly with one shared, HF‑first pipeline. Optimized for minimal files, maximal reuse, explicit configs, and strict invariants.

---

### Objectives (one‑liners)
- Unify SFT and GRPO under one pipeline: same data, same chat template, same processor tensors, same model wrapper, same validations.
- Keep a single source of truth for conversation building and parsing; avoid duplication.
- Share the unfreeze manager, checkpoint saver, and TensorBoard logging across SFT and GRPO.
- Preserve HF‑first invariants: pad=eos, left padding, eager attention, strict image token ↔ tensor alignment.
- Minimize new files and code while improving modularity, reusability, and maintainability.

---

### Non‑negotiable invariants (authoritative)
- HF‑first conversation → tokenizer/processor tensors → model wrapper → loss/reward → checkpoint.
- Strict shape/count checks:
  - Placeholder count in rendered text equals number of images.
  - `<|image_pad|>` count equals expected tokens computed from `image_grid_thw` and merge size.
  - `pixel_values` rows equal sum over `(t*h*w)`; `image_grid_thw` shape is `[num_images, 3]`.
- Coordinate tokens deprecated in `src_new`; keep key present but default false; geometry emitted as wrappers + raw integers.
- No in‑code defaults for critical knobs; fail fast on missing config keys.
- RL evaluation/inference is single‑turn; teacher pairing disabled.

---

### Target end‑state: minimal, unified module layout
- Keep SFT where it is; move RL under `src_new/rl/` and reuse shared modules.
- Keep `src_rl/` only as thin shims (re‑exports + deprecation notice) for a short migration window.

Proposed layout (only what’s needed):
- `src_new/`
  - `processing/` (authoritative, unchanged)
    - `conversation/` (keep `ConversationBuilder` as single source)
    - `parse_generated.py` (NEW, shared tolerant/strict parser for generated geometry; factored from `src_new/inference.py`)
  - `models/`
    - `wrapper.py` (keep `DetectionModel`; RL will use it directly)
  - `training/`
    - `phase_freeze_manager.py` (shared unfreeze manager for SFT + RL)
    - `callbacks.py` (add small adapter to reuse CheckpointSaver + TB logging in RL)
    - `state_manager.py` and `checkpoint_saver.py` (existing; reuse as‑is)
  - `rl/` (migrated from `src_rl/` with minimal edits)
    - `runner.py` (moved; builds tokenizer/processor/model via shared loader; uses `VisionGRPOTrainer`)
    - `trainer.py` (moved; forwards vision tensors throughout GRPO; plugs into shared callbacks/logging)
    - `data/dataset.py` (moved; already emits multimodal tensors via `ConversationBuilder`)
    - `prompting/conversation.py` (moved; single‑turn helpers; unchanged)
    - `rewards/` (moved; keep deterministic registry; import shared parser)
    - `eval.py` (moved; reuses the same loader stack and parser)
    - `config.py` (moved `EnhancedRLConfig`; maps YAML→HF args; reuses shared validation rules)

Legacy shims (temporary):
- `src_rl/__init__.py`, `src_rl/__main__.py`, `src_rl/runner.py` → import from `src_new.rl` and print a deprecation warning.

---

### Single source of truth: data → text → tensors → parsing
- Conversation building: keep using `src_new.processing.conversation.ConversationBuilder` for both SFT and RL (already true in current RL).
- Parsing (NEW shared): factor tolerant/strict regex and normalization into `src_new/processing/parse_generated.py` and import it from:
  - `src_new/inference.py` (replace local parsing logic)
  - `src_new/rl/rewards/*` (replace ad‑hoc parsing; remove duplication)

---

### Unified components loader (minimal change)
- Introduce a tiny helper `src_new/utils/hf_components.py` (or reuse an existing util) that builds:
  - tokenizer (fast, `offset_mapping`), image/video processors, and `DetectionModel`
  - enforces chat template inheritance, pad=eos, left padding, eager attention, and bf16 probe
- Use it from:
  - `scripts/train_new.py` (optional, if the current path is already robust, skip)
  - `src_new/inference.py`
  - `src_new/rl/runner.py`
- This removes the duplicated loader logic today split across RL loader and inference.

---

### Shared unfreeze manager (no new files)
- Reuse `src_new/training/phase_freeze_manager.py` inside RL runner to apply the same phase or last‑K policies:
  - Apply before creating the trainer.
  - Keep defaults conservative for RL: freeze `visual.patch_embed` unless explicitly overridden; allow `merger`/top‑K LLM blocks trainable.

---

### Uniform logging and checkpointing (minimal glue)
- Reuse existing `CheckpointSaver` and rank‑aware TensorBoard logging for RL by adding a tiny adapter in `src_new/training/callbacks.py`:
  - Export a `register_unified_callbacks(trainer, saver, logger)` function that attaches the same periodic save/eval/log hooks to both HF `Trainer` (SFT) and TRL `GRPOTrainer` (RL).
  - Keys/metrics:
    - SFT: existing CE and grouped losses.
    - RL: attach `reward`, `reward_std`, per‑component means via the registry names; keep naming stable: `rewards/<name>/mean`.
- Result: identical checkpoint rotation policy and TB logging surface for SFT and RL.

---

### Configuration surface (explicit, strict; minimal duplication)
- Keep SFT config unchanged under `configs/phase_*/*` and existing `src_new/config` loader.
- Move RL config dataclasses into `src_new/rl/config.py` (from `src_rl/config.py`):
  - `EnhancedRLConfig` remains as the outer dataclass; sub‑configs (`LayerConfig`, `OptimizerConfig`, `TrainingConfig`, `GRPOConfig`, `LoggingConfig`, `CheckpointConfig`).
  - Map RL YAML fields to HF/TR L args explicitly; perform bf16 capability probe and forced eager attention.
  - Reuse shared keys and validations: `model_path`, `data_root`, `train_data_path`, `val_data_path`, `attn_implementation`, `image.max_pixels`, `max_coord_value`, `coordinate_tokens_enabled` (kept false by default).
- No hidden defaults; fail fast on missing required keys.

---

### Trainer roles: same inputs, different objectives
- Batch formation is identical across SFT and RL: `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` (and optional `conversation_text` for logs).
- Forward path identical: use `DetectionModel` with the same validations.
- Divergence point:
  - SFT: `LossManager` path (single‑pass CE + grouped masks), teacher/student weighting.
  - RL: `VisionGRPOTrainer` path (generate K samples → compute reward components → GRPO update). Ensure `_get_per_token_logps` forwards vision tensors and honors alignment checks.

---

### Reward set (deterministic, pure, shared parser)
- Keep existing formatting/sanitization rewards: `parse`, `wrappers`, `coords`, `separators`, `vocab`.
- Keep geometry/content rewards (optional, weighted): `coverage`, `geometry_sanity`, `bbox_giou`, `quad_l1 (AABB proxy)`, `line_l1`, `ordering`, and simple taxonomy checks when available.
- All rewards consume objects parsed by `parse_generated.py` to avoid drift.

---

### CLI and quick commands (unchanged spirit)
- SFT (existing):
```bash
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_3/standard --log_level INFO
```
- RL (new location, same flags):
```bash
python -m src_new.rl.runner --config /abs/path/configs/rl/dense_grpo.yaml --mode train
```
- RL eval:
```bash
python -m src_new.rl.eval --config /abs/path/configs/rl/dense_grpo.yaml \
  --input_file /abs/val.jsonl --data_root /abs/data_root --output_file eval_report.json
```
- Legacy: `python -m src_rl.runner …` prints a deprecation warning and defers to `src_new.rl.runner`.

---

### Migration plan (minimal, safe)
1) Create `src_new/rl/` skeleton; move files from `src_rl/` verbatim, fix imports to `src_new.*`.
2) Add `src_new/processing/parse_generated.py`; refactor `src_new/inference.py` and RL rewards to import it.
3) Replace RL loader logic with shared components loader (or keep local helper but import the same functions), ensuring pad=eos, left padding, eager attention, bf16 probe.
4) Plug `phase_freeze_manager` into RL runner before trainer creation.
5) Add a tiny callback adapter in `src_new/training/callbacks.py` and register it in both trainers to unify checkpoint/TB logging.
6) Keep `src_rl/` shims with deprecation notices.

---

### Acceptance criteria
- Generation uses image tensors in both SFT and RL; no “Image features and image tokens do not match” errors.
- Prompts decode with `<|image_pad|>` and align with `image_grid_thw` counts for the same sample across SFT and RL.
- RL reward report outputs formatting + geometry metrics with stable names; valid parse rate ≥ 0.9 on a small val subset.
- RL checkpoints load in `src_new/inference.py` without changes.
- Checkpoint rotation and TB logs show the same structure for SFT and RL runs.

---

### Risks and mitigations
- TRL/HF version drift → pin tested versions in env; keep TRL‑specific code within `src_new/rl/`.
- Parser divergence → single shared parser in `processing/parse_generated.py` with unit tests.
- Config drift or silent defaults → explicit dataclasses and validations; fail fast with path‑qualified errors.
- Performance instability → force eager attention for RL; probe bf16 and downgrade to float32 with a warning.

---

### Testing strategy (fast and focused)
- Unit:
  - Parser: bbox/quad/line happy paths and edge cases; punctuation normalization.
  - Rewards: formatting + geometry; deterministic outputs.
  - Loader: tokenizer fast path with `offset_mapping`, chat template propagation to processor.
- Integration:
  - `python -m src_new.rl.runner --config … --mode load` → prints device/dtype/vocab parity JSON.
  - Short RL run (`max_steps=2`, `sample_k=2`) verifies vision tensors flow and logging/checkpoints.
  - `src_new/rl/eval.py` generates metrics JSON with non‑zero valid parse rate.
- Regression:
  - SFT pipeline remains unchanged; checkpoints load; inference parity holds.

---

### Minimal change summary (what we actually edit/move)
- MOVE: `src_rl/*` → `src_new/rl/*` (fix imports only).
- ADD: `src_new/processing/parse_generated.py` (shared parser; referenced by inference and RL rewards).
- ADD: small callback adapter in `src_new/training/callbacks.py` to reuse saver/TB logging in RL.
- OPTIONAL ADD: `src_new/utils/hf_components.py` to centralize tokenizer/processor/model loading (or refactor existing helpers).
- SHIM: keep `src_rl/` entrypoints as thin wrappers (deprecate in logs).

This delivers a concise, modular, and reusable codebase where SFT and GRPO share the same core pipeline and runtime behavior, with minimal file churn and clear boundaries for future evolution.
