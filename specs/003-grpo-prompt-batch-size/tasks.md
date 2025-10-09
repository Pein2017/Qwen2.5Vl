# Tasks: GRPO Prompt Batch Size Refactoring

**Input**: Design documents from `/specs/003-grpo-prompt-batch-size/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are constitutionally MANDATORY. Each user story lists test tasks before implementation tasks.

**Organization**: Tasks are grouped by user stories so each slice is independently deliverable.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no shared edits)
- **[Story]**: Story label (Setup, Foundational, US1, US2, Polish)
- Include precise file paths

---

## Phase 1: Setup (Shared Infrastructure)

- [x] T001 [P] [Setup] Verify Accelerate is installed and log its version in `configs/dense_rl/README.md`
- [x] T002 [Setup] Create smoke config `configs/dense_rl/grpo_prompt_batch_smoke.yaml` with `prompt_batch_size=4`, `sample_k=8`
- [x] T003 [P] [Setup] Add pytest marker `@requires_gpu` helper in `tests/conftest.py` to skip GPU-only tests cleanly

---

## Phase 2: Foundational (Blocking Prerequisites)

- [x] T004 [Foundational] Extend `src_new/config/rl_config.py` dataclasses to validate `prompt_batch_size` and `sample_k`
- [x] T005 [Foundational] Add guard in `src_new/rl/runner.py` to apply `drop_last` and skip `optimizer.step()` when collected trajectories < `prompt_batch_size × sample_k`

**Checkpoint**: Core batching constraints enforced; user story work can begin.

---

## Phase 3: User Story 1 – Configure Stable GRPO Prompt Batches (Priority: P1)

**Goal**: Sequentially broadcast prompts across ranks, accumulate `prompt_batch_size × sample_k` trajectories, and update once per batch.

**Independent Test**: Smoke run with fake model completes one optimizer step after 32 trajectories, applies `drop_last` to any leftovers, and records reward logs.

### Tests (MANDATORY) ⚠️

- [x] T006 [US1] Add unit tests in `tests/rl/test_prompt_batch_config.py` covering invalid `prompt_batch_size`, negative `sample_k`, and mismatch between config and CLI flags
- [x] T007 [US1] Add unit tests in `tests/rl/test_prompt_batch_accumulation.py` using a stub policy to ensure `optimizer.step()` fires once per batch and fails fast on shortfall
- [x] T008 [US1] Add distributed smoke test `tests/rl/test_prompt_batch_smoke.py` using `accelerate launch --num_processes 2` with dummy model and synthetic prompts (CPU only)
- [x] T009 [US1] Add unit tests ensuring `grpo.sample_k` divisible by world size when `sample_k_per_rank=false` and documenting split behavior when true

### Implementation

- [x] T010 [US1] Update `src_new/rl/runner.py` to broadcast prompts, drive sequential accumulation, apply `drop_last` for incomplete batches, and surface shortfall logs
- [x] T011 [US1] Adjust `src_new/rl/grpo_trainer.py` to accumulate losses per cycle, assign zero reward when sub-rewards fail, and clear gradients post-step
- [x] T012 [US1] Document the prompt batching workflow in `docs/rl/PROMPT_BATCHING_GUIDE.md`

**Checkpoint**: Prompt batching confirmed via tests and docs.

---

## Phase 4: User Story 2 – Monitor Trajectory Aggregation Health (Priority: P2)

**Goal**: Provide operators with live telemetry (fill ratio, smoothed reward, lagging ranks) and NCCL watchdog visibility.

**Independent Test**: Simulated run outputs `rl/prompt_batch/*` metrics and warns on injected lag.

### Tests (MANDATORY) ⚠️

- [x] T013 [US2] Add telemetry unit tests in `tests/rl/test_prompt_batch_telemetry.py` asserting fill ratio, smoothed reward (`reward_average`), invalid fraction, trajectories collected, and dropped prompts logging
- [x] T014 [US2] Extend `tests/rl/test_runner_train_debug.py` to simulate lagging rank (sleep) and assert WARN log + watchdog behavior

### Implementation

- [x] T015 [US2] Instrument `src_new/rl/prompt_batch_telemetry.py` helper to compute smoothed reward and fill ratio per cycle
- [x] T016 [US2] Update `src_new/rl/grpo_trainer.py` logging to emit telemetry metrics and console summaries (NOTE: `--simulate-rank-lag` CLI flag already exists in runner.py lines 638-643, 681-694)
- [x] T017 [US2] Publish telemetry to TensorBoard via `src_new/rl/grpo_trainer.py` under `rl/prompt_batch/*` namespace
- [x] T019 [US2] Prefer cross-rank global metrics (reward mean/std, advantage std/max_abs) in console and TensorBoard when available; add TB tags: `rl_global/advantages_std`, `rl_global/advantages_max_abs`, `rl_global/reward_std`

**Checkpoint**: Telemetry validated and documented.

---

## Phase N: Polish & Cross-Cutting Concerns

- [x] T018 [Polish] Add GPU smoke test `tests/rl/test_prompt_batch_gpu.py` (marked `@requires_gpu`) running 1 accumulation cycle with real tokenizer/model weights subset; ensure skip when GPUs unavailable

---

## Dependencies & Execution Order

- Setup → Foundational → US1 → US2 → Polish
- US2 depends on telemetry hooks introduced in US1
- Polish GPU test can run after US1 (feature complete) but before final docs

### Parallel Opportunities

- T001 and T003 in Setup
- After T010 lands, T011 and T012 may proceed in parallel
- US2 tasks T015–T017 can execute concurrently (different files) once tests T013–T014 exist

---

## Implementation Strategy

### MVP (User Story 1)
1. Finish Setup + Foundational
2. Deliver US1 with unit + distributed smoke tests
3. Validate reward trend improvement manually

### Incremental Delivery
1. MVP (US1)
2. Add telemetry enhancements (US2)
3. Polish with reward analysis script and optional GPU validation

### Team Parallelization
1. Core engineer: US1 implementation/tests
2. Observability engineer: US2 telemetry tasks once US1 merged
3. Ops engineer: Polish reward analysis + GPU smoke setup
