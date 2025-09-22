# Repository Guidelines

## Project Structure & Module Organization
Work mainly in three roots: `src_new/` for supervised fine-tuning (SFT), `src_post/` for group-level RL post-training, and `data_conversion/` for corpus preparation. Inside `src_new/`, the flow spans `training/`, `models/`, `processing/`, and `utils/`; `src_post/` reuses the wrapper to drive credit assignment, rewards, prompting, and GRPO execution. Configurations for both stages live in `configs/` (`configs/bbu_v2/` for SFT and `configs/rl/group_qc_grpo.yaml` for RL). Treat other top-level folders as experimental forks unless explicitly referenced.

## Build, Test, and Development Commands
Set up the environment via `conda activate ms`. Produce train/dev splits with `bash data_conversion/convert_dataset.sh`. Launch SFT using `python scripts/train_new.py --config configs/bbu_v2/base.yaml`; swap configs as needed. Run group RL with `python -m src_post.runner --config configs/rl/group_qc_grpo.yaml`, which should reference the latest SFT checkpoint. Validate before pushing: `python -m pytest src_new/tests/ -v`, `ruff check .`, `ruff format`, and `python -m pyright src_new src_post` when editing interfaces.

## Coding Style & Naming Conventions
Use four-space indentation and double-quoted strings per Ruff formatting. Keep modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Maintain explicit type hints for dataclasses, configs, and public APIs, mirroring existing `src_new/training` signatures. Keep configuration filenames lowercase with hyphens or underscores that match experiment IDs.

## Testing Guidelines
Add or extend `pytest` suites beside the code they cover (`src_new/tests/test_<module>.py`). Prefer parametrized cases and shared fixtures when touching data samplers or losses. For RL updates, add deterministic smoke tests that mock reward functions under `src_post/tests/` (create it if needed) and track expected metrics deltas. Use `python -m pytest src_new/tests/ -k <keyword>` for focused runs and ensure group-level simulations complete without exceptions before filing a PR.

## Commit & Pull Request Guidelines
Follow Conventional Commit prefixes (`feat(training): …`, `fix(rl): …`). Keep commits scoped to coherent changes across code, configs, and docs. Pull requests should link issues, describe SFT/RL impact, list validation commands (data conversion, SFT, RL, pytest, Ruff), and attach sample outputs or reward traces when behaviour shifts. Request review from maintainers for the stage you touched (SFT vs RL).

## Configuration & Data Handling
Stage intermediate artifacts in `outputs/` or `model_cache/` rather than Git. Note any non-default config knobs in the PR description so downstream runs stay reproducible. Use sanitized assets from `test_data/` or `demo/` for repros, and keep raw customer data external.

## Coordinate Tokens
Coordinate tokens are deprecated and not consistently enabled across checkpoints. Treat them as optional; inference and evaluation code must succeed without relying on coordinate token support.

