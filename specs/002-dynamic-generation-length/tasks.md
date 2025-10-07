# Tasks: Dynamic GT‑Aware Generation Length for Dense RL

**Input**: Design documents from `/specs/002-dynamic-generation-length/`
**Prerequisites**: plan.md (required)

## Execution Flow (main)
```
1. Load plan.md from feature directory
2. Generate tasks by category with TDD ordering
3. Number tasks sequentially (T001, T002...)
4. Create dependency notes and parallel examples
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup
- [ ] T001 Ensure branch is `002-dynamic-generation-length` and entrypoint is `scripts/run_dense_grpo.sh` (update run docs if needed)
- [ ] T002 Add YAML knobs in configs: `grpo.dynamic_length` (incl. `mask_overflow_only`) and `rewards_config.length_vs_gt` in `configs/dense_rl/dense_base.yaml`, `debug.yaml`, `standard.yaml`
- [ ] T003 [P] Pin tokenizer and environment determinism notes in `scripts/run_dense_grpo.sh` (ensure identical tokenizer across ranks)

## Phase 3.2: Tests First (TDD)
- [ ] T004 Create config contract tests for dynamic_length keys in `specs/002-dynamic-generation-length/contracts/config_contract.md` (keys, types, required/optional)
- [ ] T005 Create logging contract tests in `specs/002-dynamic-generation-length/contracts/logging_contract.md` (metric tags: `dynamic_length/*`, completion lengths, truncation flags)
- [ ] T006 [P] Integration test plan in `specs/002-dynamic-generation-length/quickstart.md`: steps to enable dynamic cap and verify logs + no slow-rank timeout under multi-GPU

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T007 Implement tokenizer-based GT length estimation (builder reuse) in `src_new/processing/coordinate_converter.py` usage path notes (no code change here; used by buffer)
- [ ] T008 Implement per-sample cap in `src_new/rl/buffer.py`:
  - Compute GT assistant text from `meta.objects` via `CoordinateTokenConverter.convert_objects_to_tokens`
  - Tokenize once to get `gt_len`; derive `cap = clamp(round(alpha*gt_len + eos_margin), min_cap, max_cap)`
  - Pass `cap` as `max_new_tokens` to `generation.sample_k` for that sample
  - Collect `dynamic_length/cap` for logging
- [ ] T009 Adjust masking policy in `src_new/rl/buffer.py` to keep truncated completions in loss; optionally mask only tokens beyond the cap (guarded by YAML)
- [ ] T010 Add `length_vs_gt` reward in `src_new/rl/rewards/format_rewards.py` using tokenizer-based estimator and exponential overflow penalty; register in `src_new/rl/rewards/registry.py`
- [ ] T011 Plumb `rewards_config.length_vs_gt` params through `src_new/rl/runner.py` wrapper to the reward function
- [ ] T012 Add logging of dynamic caps in trainer step in `src_new/rl/grpo_trainer.py` (e.g., `dynamic_length/mean_cap|min_cap|max_cap`)

## Phase 3.4: Distributed Consistency (Accelerate)
- [ ] T013 Ensure shared dataset index and synchronized buffer refresh already present in `src_new/rl/grpo_trainer.py` (verify gather/reduce flow)
- [ ] T014 Pad/trim variable-length tensors before cross-rank collectives (rewards/advantages lengths) — verify paths in `src_new/rl/grpo_trainer.py`
- [ ] T015 [P] Add per-rank generation timing + cap diagnostics; resample guard remains active; ensure identical caps across ranks by computing from `meta.objects`

## Phase 3.5: YAML & Docs
- [ ] T016 Update `configs/dense_rl/dense_base.yaml`, `debug.yaml`, `standard.yaml` with dynamic_length defaults (alpha, margins, bounds, masking policy) and `length_vs_gt` defaults
- [ ] T017 [P] Update `src_new/rl/DYNAMIC_GENERATION_LENGTH.md` with final knobs and log tags summary

## Phase 3.6: Polish & Validation
- [ ] T018 [P] Add unit tests for `length_vs_gt` scoring (short/within/overflow/tail-numeric cases)
- [ ] T019 [P] Add small integration test to assert cap is applied and logs contain `dynamic_length/*` keys (single-GPU smoke)
- [ ] T020 Run multi-GPU smoke via `scripts/run_dense_grpo.sh` with 2 GPUs and debug config; confirm no slow-rank timeouts; capture logs in `run_dense.log`
- [ ] T021 Finalize docs: update `quickstart.md` with troubleshooting tips (cap vs tokenizer drift, sync guards)
- [ ] T022 Implement disabled-mode fallback in `src_new/rl/buffer.py` and `src_new/rl/grpo_trainer.py`: when `grpo.dynamic_length.enabled=false`, bypass per-sample cap and use fixed `grpo.max_new_tokens`; add an integration check in logs
- [ ] T023 [P] Add regression test to ensure formatting rewards (wrappers/coords/separators) remain computed/logged unchanged after enabling dynamic length (single-GPU smoke)

## Dependencies
- T001→T002→T004/T005/T006→T008/T009/T010/T011→T012→T013/T014/T015→T016/T017→T018/T019/T020/T021
- [P] Parallelizable: T003, T006, T015, T017, T018, T019 can run alongside others when not touching same files

## Parallel Example
```
# Parallel batch after T008 is in place:
Task: "T015 Add per-rank generation timing + cap diagnostics"  # grpo_trainer.py
Task: "T017 Update DYNAMIC_GENERATION_LENGTH.md with final knobs"
Task: "T018 Unit tests for length_vs_gt"
Task: "T019 Integration test for dynamic_length logs"
```

## Notes
- Tests before implementation where applicable; contract docs first
- Keep tokenizer/version consistent across ranks to ensure equal caps
- Avoid adding new modules unless required; reuse existing utilities
