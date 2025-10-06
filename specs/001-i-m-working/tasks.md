# Tasks: TRL→Manual GRPO Trainer Migration

**Input**: Design documents from `/data3/Qwen2.5-VL-main/specs/001-i-m-working/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Phase 3.1: Setup
- [X] T001 Verify environment: run ruff/pyright/pytest locally
  - Command: `cd /data3/Qwen2.5-VL-main && ruff check . && pyright && pytest -q || true`
  - Status: Environment verified, minor line length warnings (non-blocking)
- [X] T002 [P] Create example debug RL YAML snippet in docs referencing absolute paths and no defaults (document required keys)
  - File: Documented in `/data3/Qwen2.5-VL-main/configs/dense_rl/debug.yaml` with comprehensive comments

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
- [X] T003 [P] Unit test: runner rejects missing `model.attn_implementation` in YAML (no defaults allowed)
  - File: `/data3/Qwen2.5-VL-main/tests/rl/test_yaml_validation.py`
- [X] T004 [P] Unit test: runner rejects unset `model.image_max_pixels` (must be explicit)
  - File: `/data3/Qwen2.5-VL-main/tests/rl/test_yaml_validation.py`
- [X] T005 [P] Unit test: loss weights missing any of `caption,grounding,formatting` → actionable error (RL has no teacher‑student pairing)
  - File: `/data3/Qwen2.5-VL-main/tests/rl/test_yaml_validation.py`
- [X] T006 [P] Unit test: `bf16:false` or absent → mandatory bf16 error
  - File: `/data3/Qwen2.5-VL-main/tests/rl/test_yaml_validation.py`
- [X] T007 [P] Unit test: logging emits required TB scalars and console lines (mock writer) including `reward`, `rewards/<name>/mean`, `grad_norm`, `learning_rate`, `eta_minutes`, `step`, `epoch`
  - File: `/data3/Qwen2.5-VL-main/tests/rl/test_logging_metrics.py`
- [X] T008 [P] Integration test: `--mode load` prints one-line JSON with device/dtype/vocab
  - File: `/data3/Qwen2.5-VL-main/tests/rl/test_runner_load_mode.py`
- [X] T009 [P] Integration test: short train with debug.yaml (small mock dataset) writes TB scalars and a checkpoint
  - File: `/data3/Qwen2.5-VL-main/tests/rl/test_runner_train_debug.py`

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [X] T010 [P] Enforce explicit config (no defaults) in `src_new/rl/runner.py` and related loaders per contracts
  - Edits: `/data3/Qwen2.5-VL-main/src_new/rl/runner.py`
- [X] T011 [P] Implement console+TB parity: add `grad_norm`, `learning_rate`, `eta_minutes`, `step`, `epoch` scalars
  - Edits: `/data3/Qwen2.5-VL-main/src_new/rl/grpo_trainer.py`
- [X] T012 [P] Add per-reward `{mean,std}` scalars under `rewards/<name>/...` and world-aggregate when cross-rank enabled
  - Edits: `/data3/Qwen2.5-VL-main/src_new/rl/grpo_trainer.py` (already implemented with world aggregation)
- [X] T013 Compute and log ETA minutes based on elapsed time and remaining steps; expose in console and TB
  - Edits: `/data3/Qwen2.5-VL-main/src_new/rl/grpo_trainer.py`
- [X] T014 Ensure `TrainingStateManager` records `train/loss` and integrates with added scalars without duplication
  - Edits: `/data3/Qwen2.5-VL-main/src_new/rl/grpo_trainer.py` (TensorBoard writer added with train/loss tag)
- [X] T015 Verify checkpoint saver includes tokenizer/processor and minimal processor config
  - Edits: `/data3/Qwen2.5-VL-main/src_new/training/checkpoint_saver.py` (implemented; YAML path embedded in checkpoints)

## Phase 3.4: Configuration (edit only configs/dense_rl)
- [X] T016 Update `/data3/Qwen2.5-VL-main/configs/dense_rl/dense_base.yaml` to contain only universal constants and strictly no runnable defaults (keep REQUIRED comments)
  - Edits: `/data3/Qwen2.5-VL-main/configs/dense_rl/dense_base.yaml`
- [X] T017 Create or refine `/data3/Qwen2.5-VL-main/configs/dense_rl/debug.yaml` with all required keys explicitly set (no derived defaults), minimal short-run values; validate with load mode
  - Edits: `/data3/Qwen2.5-VL-main/configs/dense_rl/debug.yaml`
- [X] T018 [P] Removed: Disallow env-var expansion and CLI-config overrides; configs must be YAML-only
  - Edits: `/data3/Qwen2.5-VL-main/src_new/rl/runner.py`, `/data3/Qwen2.5-VL-main/src_new/utils/hf_components.py`, `/data3/Qwen2.5-VL-main/specs/001-i-m-working/contracts/rl_runner.md`, `/data3/Qwen2.5-VL-main/configs/dense_rl/README.md`, `/data3/Qwen2.5-VL-main/specs/001-i-m-working/quickstart.md`
- [X] T019 Document RL single‑turn requirement (no teacher‑student), single‑image batch assumption in spec and configs README
  - Edits: `/data3/Qwen2.5-VL-main/specs/001-i-m-working/spec.md`, `/data3/Qwen2.5-VL-main/configs/dense_rl/README.md`

## Phase 3.5: Polish
- [X] T020 [P] Update `/data3/Qwen2.5-VL-main/specs/001-i-m-working/quickstart.md` with verified console/TB snippets
  - Edits: `/data3/Qwen2.5-VL-main/specs/001-i-m-working/quickstart.md`
- [X] T021 [P] Finalize contracts docs with exact error messages observed in tests
  - Edits: `/data3/Qwen2.5-VL-main/specs/001-i-m-working/contracts/rl_runner.md`
- [X] T022 [P] Remove any dead code and comments per simplicity principle
  - Edits: Removed legacy `_map_yaml_to_grpo_config` stub from `src_new/rl/runner.py`

## Phase 3.6: Optional Next
- [ ] T023 Cross-rank K sampling PoC behind `distributed.cross_rank_sampling` with `k_split_mode`

## Dependencies
- Setup before tests; tests before implementation; config edits after core logging/validation in code.
- Parallel [P] tasks touch different files and can run concurrently.

## Parallel Example
```bash
# Run validation tests in parallel
/task T003 && /task T004 && /task T005 && /task T006

# After tests fail, implement logging+parity in parallel
/task T011 && /task T012 && /task T013
```

## Validation Checklist
- [X] `--mode load` prints valid JSON
- [X] TB contains required scalars and per-reward metrics (implementation complete, unit test covers logging API)
- [X] Console mirrors TB scalar keys; includes ETA, step/epoch, LR, grad_norm
- [X] Checkpoint saved & reloadable; contains tokenizer/processor (unit test verified)
- [X] YAML with any missing required key fails with actionable error
- [X] Only configs/dense_rl/{dense_base.yaml,debug.yaml} are edited for configuration; no defaults in code
- [X] YAML-only policy enforced: no env-var expansion or CLI overrides; configs are source of truth
- [X] RL path is single‑turn only; no teacher pairing occurs in RL dataset or runner
