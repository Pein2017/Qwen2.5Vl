# Refactor Progress Log

## 2025-02-14
- Introduced `src_new/data/collator_shared.py` with shared padding and multimodal validation helpers.
- Rewired `StandardDataCollator` to consume the shared helpers, reducing duplicate tensor logic.
- Added RL-specific `PromptOnlyCollator` (`src_new/rl/data/collator.py`) that reuses the shared utilities and carries metadata for rewards.
- Updated `src_new/rl/data/dataset.py` to rely on the HF-first `read_jsonl` loader for consistency with the SFT pipeline.
- Exposed the shared prompt collator through the RL runner (`src_new/rl/runner.py`) so both manual and TRL paths can reuse it.
- Stubbed `GRPOVLMTrainer` (`src_new/rl/trl_trainer.py`) as the integration point for TRL-based training.
- Runner now accepts `--trainer {manual,trl}` and logs a placeholder warning while TRL integration is in progress (`src_new/rl/runner.py`).
- Implemented the first functional `GRPOVLMTrainer` override by delegating generation/reward flow to the shared buffer stack and carrying forward metrics/logging (`src_new/rl/trl_trainer.py`).
- Added a preliminary TRL training path in the runner that maps key config values onto `GRPOConfig` and instantiates `GRPOVLMTrainer`, wiring in the shared collator/dataset stack (`src_new/rl/runner.py`).
- Added `configs/dense_rl/trl_smoke.yaml` for lightweight TRL smoke tests.
- Simplified launcher script to wrap the new runner interface (`scripts/run_dense_grpo.sh`).
