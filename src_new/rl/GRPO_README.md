## GRPO Post-Training (Dense Captioning) for Qwen2.5‑VL

### What this is
- **Goal**: Reinforcement post-training for dense captioning using GRPO, tightly aligned with the SFT processing stack (HF-first) and the Qwen2.5‑VL DetectionModel validations.
- **Design**: Two interchangeable trainers built on a shared HF-first data path:
- **TRL-based trainer** (`GRPOVLMTrainer`) that wraps Hugging Face’s `trl.GRPOTrainer` while reusing the same source of prompts/rewards.
- **Scope**: Single-turn, image-conditioned dense-caption generation with reward shaping that balances format correctness and detection quality.

### RL refactor v2 (merged, concise)
- **Status**: COMPLETE (2025-10-08)
- **Key changes**: Strict YAML (no defaults); typed `RLConfig` (`src_new/config/rl_config_v2.py`); clean separation of `sampling`/`generation`/`grpo` (algorithm-only); runner/trainer consume typed config only; no `raw_config`/`.get(...)` in critical paths.
- **Files updated**: `configs/dense_rl/{dense_base.yaml,debug.yaml,standard.yaml,README.md}`, `configs/dense_rl/trl_smoke.yaml`, `src_new/config/rl_config_v2.py`, `src_new/rl/{runner.py,grpo_trainer.py,trl_trainer.py,data/collator.py,data/collator_shared.py}`.
- **Required YAML**: 14 sections (~80+ required fields) including `paths`, `experiment`, `model`, `sampling`, `generation`, `grpo`, `normalization`, `training`, `optimizer`, `layer_config`, `logging`, `checkpointing`, `rewards` + `rewards_config`, `evaluation`. Optional only: `paths.ref_model_path`, `grpo.beta_anneal` (when `beta_start=0`).
- **Trainer highlights**:
  - Manual trainer: new `__init__(..., rl_config: RLConfig, output_dir: str)`; rewritten `_build_manual_cfg(rl_config)`; evaluation via `config.evaluation.*`.
  - TRL trainer: `GRPOVLMTrainer` subclasses `trl.GRPOTrainer`, reuses the shared prompt collator + reward buffer, and keeps trust-region log-probs identical to the manual path.
- **Usage**:
- Manual path, validate: `python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode load`
- Manual path, train: `python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode train --trainer manual`
- TRL path, smoke test (GPU required): `python -m src_new.rl.runner --config configs/dense_rl/trl_smoke.yaml --mode train --trainer trl`
- Convenience launcher: `bash scripts/run_dense_grpo.sh` (honours `config`, `mode`, `gpus`, `python` env vars). Conversation dump is configured in YAML under `evaluation.dump_conversations` and `evaluation.dump_max_per_step`.
- **Benefits**: Fail-fast validation with clear errors; no hidden defaults; IDE-friendly typed access; reproducible configs; HF-first dataflow unchanged; optional TRL integration for ecosystems that standardize on `trl.GRPOTrainer`.

### Quality consolidation (2025-10-12)
- **Status**: COMPLETE
- **Key additions (no tensor changes)**:
  - Dataclasses for payloads in `src_new/rl/types.py` (`GenerationResult`, `CompletionSlice`, `ClipDiagnostics`, `PromptBatchMetrics`, `BufferTelemetry`, `TrainingLogs`).
  - Central metric keys in `src_new/rl/metrics_keys.py` to avoid string drift.
  - Diagnostics toggles in `src_new/rl/diagnostics/settings.py` (e.g., `DEBUG_TIMING`, `GENERATION_WARN_THRESHOLD`, `RL_CLEAR_CACHE`).
  - Unified generation args helper `generation.build_generation_args()` used in train/eval paths to align optional kwargs.
  - Trainer uses `MetricsAggregator` for cross-rank stats and clarifies buffer reuse logging (`buffer/reuse_active`).


## Modular Architecture

The trainer has been refactored from a monolithic 2052-line file into specialized components (~1200 lines core + 5 modules), improving maintainability and testability.

### Core Modules

1. **`metrics_aggregator.py`** - Cross-rank tensor gathering
   - `gather_rewards_stats()`, `gather_advantage_stats()`, `gather_per_reward_components()`
   - `gather_termination_ratio()`, `gather_flag()`, `synchronize_index()`

2. **`tensorboard_logger.py`** - Structured logging
   - `log_training_scalars()`, `log_completion_metrics()`, `log_dynamic_length_metrics()`
   - `log_advantage_metrics()`, `log_per_reward_components()`, `log_eval_metrics()`

3. **`console_formatter.py`** - Console output formatting
   - `format_training_summary()`, `format_prompt_batch_summary()`

### Shared types & utilities (consolidation) & utilities (consolidation) & utilities (consolidation)
- **`types.py`** - Runtime dataclasses (`GenerationResult`, `CompletionSlice`, etc.).
- (removed) metrics keys consolidation; TRL logs core metrics directly.
- **`diagnostics/settings.py`** - Env-driven debug toggles (read once).
- **`data/collator_shared.py`** and **`data/collator.py`** - Shared padding / multimodal validation and the prompt-only collator reused by both manual and TRL trainers.
- **`trl_trainer.py`** - TRL wrapper that delegates prompt preparation and rewards to the shared buffer, keeping metrics aligned with TRL.

### Integration

```python
# Initialization in BBUGRPOTrainer.__init__
self._metrics_aggregator = MetricsAggregator(self.accelerator, self._device)
