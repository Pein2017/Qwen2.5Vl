# Implementation Plan: GRPO Prompt Batch Size Refactoring

**Branch**: `003-grpo-prompt-batch-size` | **Date**: 2025-10-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-grpo-prompt-batch-size/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Refactor the dense captioning GRPO pipeline so each optimizer update aggregates `sample_k × prompt_batch_size` sequential trajectories across eight GPUs, reducing reward volatility while honoring memory constraints. The plan introduces configurable accumulation parameters, leverages official distributed tooling (PyTorch DDP + Hugging Face Accelerate) to orchestrate gradient syncing, and enhances basic telemetry so operators can verify steadily improving reward trends without adding new peripheral features.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.12 (conda env `ms`)  
**Primary Dependencies**: PyTorch 2.x distributed + NCCL, Hugging Face Accelerate, Transformers, existing `src_new/rl` utilities  
**Storage**: File-based datasets (JSONL) and SafeTensors checkpoints under `/data3/Qwen2.5-VL-main`  
**Testing**: pytest (unit + integration), ruff, pyright, targeted smoke GRPO runs with prompt batching (via `prompt_batch_size`) assertions  
**Target Platform**: Linux multi-GPU node (8× NVIDIA GPUs) with CUDA-aware NCCL backend  
**Project Type**: Single repository (`src_new/` primary) with RL module  
**Performance Goals**: Produce a clear upward trend in primary reward metrics compared with the single-prompt baseline while keeping sequential rollout throughput close to current levels  
**Constraints**: Sequential generation per GPU due to memory limits; bf16 policy; fail-fast validation required; `grpo.sample_k` divisible by world size when `sample_k_per_rank=false`; NCCL watchdogs must guard against sync deadlocks; no ad-hoc scripts outside builders/processors  
**Scale/Scope**: 8 GPU ranks per node with sequential generation; prompt_batch_size tuned per available memory

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Fill the table below with `PASS`, `PASS WITH ACTION`, or `NEEDS ACTION`, plus a short remediation note. Reference the constitution (`.specify/memory/constitution.md`) when in doubt.

| Principle / Rule | Status | Notes |
|------------------|--------|-------|
| Single Source of Truth & Explicit Contracts | PASS WITH ACTION | Extend GRPO configs to include prompt_batch_size while keeping YAML schema validated. |
| Fail-Fast Validation & Observability | PASS WITH ACTION | Need new checks for trajectory counts and expanded telemetry dashboards/logging. |
| Configuration & Reproducibility | PASS | YAML-driven configs already in place; extend dataclasses accordingly. |
| Separation of Concerns | PASS | Refactor stays within `src_new/rl` runners/trainers without leaking responsibilities. |
| Testing & TDD | PASS WITH ACTION | Add pytest coverage for accumulation math, reward trend checks, and lagging-rank handling. |
| Simplicity & Minimal Surface Area | PASS | Reuse Accelerate/DDP wrappers instead of bespoke orchestration. |
| Decoupling & Reuse | PASS | Shared builders/validators reused; new accumulation helpers made modular. |
| Versioning (No Backward Compatibility Guarantees) | PASS | Document SemVer impact (likely MINOR) and update configs accordingly. |
| Security & Compliance | PASS | Work stays within approved workspace; no new data ingestion. |
| Documentation & Traceability | PASS WITH ACTION | Update quickstart, spec references, and telemetry docs post-implementation. |
| Performance & Resource Stewardship | PASS WITH ACTION | Benchmark accumulation throughput and guard VRAM consumption. |
| Code Review & CI Gates | PASS | Maintain lint/type/test gate enforcement. |
| Execution Environment & Tooling Rules | PASS | Continue using `ms` conda environment, Accelerate integrates with existing setup. |
| Operational Workflow & Artifact Expectations | PASS WITH ACTION | Ensure plan/spec/tasks remain in sync and Constitution Checks revisited after design.

If a principle requires follow-up, create tasks in `/specs/[###-feature]/tasks.md` and link them in the Notes column.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```
src_new/
├── config/
├── data/
├── processing/
├── rl/
│   ├── grpo_trainer.py
│   ├── runner.py
│   ├── utils.py
│   ├── logprobs.py
│   ├── rewards/
│   └── telemetry/
└── tools/

tests/
├── rl/
│   ├── test_runner_train_debug.py
│   ├── test_logging_metrics.py
│   └── fixtures/
└── integration/
```

**Structure Decision**: Work stays within the existing `src_new/rl` package (trainer, runner, utils, telemetry) and corresponding `tests/rl` suites; additional helper modules or telemetry exports will be added under these directories to preserve separation of concerns.

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |

## Phase 0: Outline & Research (Completed)
- Review Accelerate + PyTorch DDP capabilities for sequential gradient accumulation.
- Define deterministic prompt batching (via `prompt_batch_size`) strategy that reuses existing sampling mechanisms while respecting memory limits.
- Specify lightweight telemetry expansions (reward trend tracking, fill ratio, lagging ranks) and NCCL watchdog expectations.
- Outcome: [research.md](./research.md) capturing decisions, rationale, and alternatives; no outstanding clarifications.

## Phase 1: Design & Contracts (In Progress)
- Model entities for prompt batch settings and reward trend snapshots ([data-model.md](./data-model.md)).
- Define CLI, YAML schema updates, telemetry tags (`rl/prompt_batch/reward_average` with a 5-cycle reward_average_window, fill ratio, lag warnings) ([contracts/prompt_batching.md](./contracts/prompt_batching.md)).
- Produce quickstart guide covering config prep, smoke run (CPU/GPU), telemetry inspection, and lagging-rank simulation ([quickstart.md](./quickstart.md)).
- Update agent context (Codex) so future automation reflects Python 3.12, Accelerate, and dataset storage conventions.

## Phase 2: Implementation Preparation (Next)
- Enumerate execution tasks in `tasks.md` covering config/dataclass updates, drop-last guard, telemetry instrumentation, tests, and documentation adjustments.
- Ensure Constitution Check PASS WITH ACTION items translate into explicit tasks (validation, observability, testing, documentation, performance benchmarking).
