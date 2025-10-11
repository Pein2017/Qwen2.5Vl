# Implementation Plan: GRPO Post-Training Diagnostics and Error Detection

**Branch**: `004-grpo-post-training` | **Date**: 2025-10-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/data3/Qwen2.5-VL-main/specs/004-grpo-post-training/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This diagnostic feature addresses training instability in GRPO post-training for dense captioning by systematically identifying hidden algorithmic or tensor mismatch errors. The primary observed symptom is zero advantage standard deviation after some iterations, suggesting the SFT checkpoint produces overly deterministic outputs within each K-completion group, causing reward collapse and preventing learning. The implementation will instrument the GRPO pipeline with comprehensive validation checkpoints, compare diversity across different SFT fine-tuning degrees (Phase 2 vs Phase 3), and test temperature sweeps to diagnose sampling diversity issues. All findings will be captured in diagnostic configs, code comments, and this plan artifact (per Constitution v4.1.1 prohibition on temporary docs).

## Technical Context

**Language/Version**: Python 3.12 (conda env: `ms`)  
**Primary Dependencies**: PyTorch 2.x, Transformers (Qwen2.5-VL), Accelerate, TensorBoard, SciPy (Hungarian assignment for detection rewards)  
**Storage**: JSONL datasets, SafeTensors checkpoints, YAML configs  
**Testing**: pytest, ruff (lint), pyright (type-check)  
**Target Platform**: Linux server (8×GPU A100/H100, CUDA)  
**Project Type**: Single project (multimodal ML training pipeline)  
**Performance Goals**: Identify root cause of GRPO training instability within 50 training steps; diagnostic overhead <10% of baseline training time  
**Constraints**: 
- Sequential processing mandate (Constitution v4.1.1): `batch_size=1` for all `model.generate()`/`model.forward()` calls
- Peak GPU memory <80GB per device
- All findings captured in spec/plan/code comments (no temporary summary docs)
- Diagnostic configs must run to completion in <15 minutes on 8 GPUs

**Scale/Scope**: 
- 6 user stories (3×P1, 2×P2, 1×P3)
- Instrumentation spans 5 pipeline stages (dataset → buffer → generation → reward → loss)
- Checkpoint comparison: 2 SFT degrees × 3 temperatures = 6 test configurations
- Expected deliverables: 2 diagnostic configs, 10 instrumentation points, actionable error identification

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Fill the table below with `PASS`, `PASS WITH ACTION`, or `NEEDS ACTION`, plus a short remediation note. Reference the constitution (`.specify/memory/constitution.md`) when in doubt.

| Principle / Rule | Status | Notes |
|------------------|--------|-------|
| Single Source of Truth & Explicit Contracts | PASS | All diagnostic instrumentation uses existing `ConversationBuilder` and `ConversationProcessor`; no bypassing of typed builders |
| Fail-Fast Validation & Observability | PASS WITH ACTION | Core feature—adds validation checkpoints at 5 pipeline stages; task: ensure validators raise actionable errors (not warnings) |
| Configuration & Reproducibility | PASS WITH ACTION | FR-008 mandates new diagnostic YAML configs; task: ensure all temperature/checkpoint sweep params are explicit (no in-code defaults) |
| Separation of Concerns | PASS | Diagnostics layer cleanly in `src_new/rl/diagnostics/` module; reuses existing validators, doesn't leak into losses or training loop |
| Testing & TDD | PASS WITH ACTION | User Story 1-6 each have independent tests; task: write failing pytest cases before implementing instrumentation points |
| Simplicity & Minimal Surface Area | PASS | Diagnostic configs are OFF by default; enabled only via explicit `--config diagnostics.yaml`; no new CLI flags |
| Decoupling & Reuse | PASS | Reuses `validators.py`, `tensor_validation.py`, reward registry; new diagnostic helpers are composable and stateless |
| Versioning (No Backward Compatibility Guarantees) | PASS | No changes to checkpoint format or JSONL schema; diagnostics are observability-only |
| Security & Compliance | PASS | Stays within `/data3/Qwen2.5-VL-main`; validates object types via existing allowlist; no new data ingestion |
| Documentation & Traceability | PASS | All findings captured in plan.md (this file) and code comments; spec references Constitution v4.1.1; no temporary docs created |
| Performance & Resource Stewardship | PASS WITH ACTION | Diagnostic overhead target <10%; task: profile instrumentation impact at 50 steps and log timing breakdowns |
| Sequential Processing & GPU Resource Constraints | PASS | User Story 5 explicitly validates `batch_size=1` compliance; no batching introduced |
| Code Review & CI Gates | PASS WITH ACTION | Task: add diagnostic config smoke test to CI (runs 10 steps, <5min timeout, zero errors) |
| Execution Environment & Tooling Rules | PASS | Requires `conda activate ms`; all paths absolute from `/data3/Qwen2.5-VL-main`; pytest/ruff/pyright gates enforced |
| Operational Workflow & Artifact Expectations | PASS | Follows spec→plan→tasks workflow; plan.md (this file) includes Constitution Check; tasks.md will map to user stories |

If a principle requires follow-up, create tasks in `/specs/[###-feature]/tasks.md` and link them in the Notes column.

## Project Structure

### Documentation (this feature)

```
specs/004-grpo-post-training/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0: GRPO diversity analysis & checkpoint comparison methodology
├── data-model.md        # Phase 1: Diagnostic entities schema (TrustRegionDiagnostic, RewardProfile, etc.)
├── quickstart.md        # Phase 1: Running diagnostic configs & interpreting output
├── contracts/           # Phase 1: Validation checkpoint interfaces
└── tasks.md             # Phase 2: Task breakdown by user story (/speckit.tasks command)
```

### Source Code (repository root)

```
/data3/Qwen2.5-VL-main/
├── src_new/
│   └── rl/                          # Existing GRPO implementation
│       ├── diagnostics/             # NEW: Diagnostic instrumentation module
│       │   ├── __init__.py
│       │   ├── trust_region.py      # TrustRegionDiagnostic tracker
│       │   ├── multimodal.py        # MultimodalAlignmentCheck validators
│       │   ├── rewards.py           # RewardProfile tracker (within/between variance)
│       │   ├── gradients.py         # GradientFlowSnapshot hooks
│       │   ├── sequential.py        # SequentialProcessingMonitor
│       │   ├── checkpoint_diversity.py  # CheckpointDiversityComparison
│       │   └── exporters.py         # Diagnostic artifact generation (histograms, heatmaps)
│       ├── grpo_trainer.py          # MODIFIED: Instrumentation hook points
│       ├── buffer.py                # MODIFIED: Multimodal validation checkpoints
│       ├── completion_loss.py       # MODIFIED: Trust region logging
│       ├── generation.py            # MODIFIED: Sequential processing assertions
│       └── validators.py            # REUSED: Existing multimodal validators
├── configs/dense_rl/
│   ├── diagnostic.yaml              # NEW: Minimal 10-sample, 20-step config with maximal logging
│   └── checkpoint_diversity_test.yaml  # NEW: Phase 2 vs Phase 3 checkpoint comparison config
└── tests/
    └── rl/
        └── diagnostics/             # NEW: Test suite for diagnostic features
            ├── test_trust_region.py
            ├── test_multimodal_alignment.py
            ├── test_reward_variance.py
            ├── test_gradient_flow.py
            ├── test_sequential_processing.py
            └── test_checkpoint_diversity.py
```

**Structure Decision**: Single-project structure. Diagnostic instrumentation lives in a dedicated `src_new/rl/diagnostics/` module following Separation of Concerns principle. Reuses existing validators and integrates at minimal hook points in `grpo_trainer.py`, `buffer.py`, and `completion_loss.py`. Test-first approach with independent test files per user story.

## Complexity Tracking

*No constitutional violations requiring justification. All "PASS WITH ACTION" items are standard follow-up tasks tracked in Phase 2.*

## Progress Tracking

| Phase | Status | Artifacts | Notes |
|-------|--------|-----------|-------|
| 0 - Research | ✅ COMPLETE | research.md | GRPO diversity collapse analysis, checkpoint comparison methodology, reference implementations review |
| 1 - Design | ✅ COMPLETE | data-model.md, contracts/, quickstart.md | Diagnostic entity schemas, validation interfaces, usage guide |
| 2 - Tasks | ⏳ PENDING | tasks.md | Awaiting `/speckit.tasks` command |

---

## Phase 0: Research

**Output**: `research.md` in `/data3/Qwen2.5-VL-main/specs/004-grpo-post-training/`

### Objectives

1. **GRPO Diversity Collapse Diagnosis**: Analyze why zero advantage std occurs when all K completions within a group have identical/similar rewards
2. **Checkpoint Comparison Methodology**: Design experiments to compare Phase 2 vs Phase 3 SFT checkpoints with temperature sweeps
3. **Reference Implementation Review**: Document algorithmic differences between official GRPO (Qwen2-VL-Finetune, ms-swift) and current `src_new/rl` implementation
4. **Failure Mode Catalog**: Enumerate known GRPO failure modes (ratio degeneracy, multimodal drift, reward collapse) with detection strategies

### Research Questions

- **RQ1**: What is the expected ratio of within-group variance to between-group variance for healthy GRPO training?
- **RQ2**: How does SFT fine-tuning degree affect generation diversity at fixed temperature? (Hypothesis: Phase 2 < Phase 3 fine-tuning → more diversity)
- **RQ3**: What are the critical differences in advantage normalization between official references and `src_new/rl/advantage_normalizer.py`?
- **RQ4**: Which reward functions are most sensitive to diversity collapse (formatting vs. detection rewards)?
- **RQ5**: What are the canonical validation checkpoints in official GRPO implementations for trust region health?

### Deliverables

- **Diversity Analysis Report**: Mathematical derivation of advantage collapse when rewards are identical; recommended variance thresholds (std > 0.05 justification)
- **Checkpoint Comparison Protocol**: Experimental design for testing Phase 2/3 checkpoints with temperatures {0.7, 0.9, 1.1} across 50 prompts; success criteria
- **Algorithm Parity Matrix**: Side-by-side comparison of generation kwargs, ratio computation, advantage normalization, KL regularization between references and `src_new/rl`
- **Instrumentation Strategy**: Identified hook points in GRPO pipeline for trust region, multimodal, reward, gradient, sequential diagnostics

---

## Phase 1: Design

**Outputs**: 
- `data-model.md` in `/data3/Qwen2.5-VL-main/specs/004-grpo-post-training/`
- `contracts/` directory with interface definitions
- `quickstart.md` for running diagnostics

### Objectives

1. **Diagnostic Entity Schemas**: Define dataclasses for `TrustRegionDiagnostic`, `MultimodalAlignmentCheck`, `RewardProfile`, `GradientFlowSnapshot`, `SequentialProcessingMonitor`, `CheckpointDiversityComparison`
2. **Validation Interfaces**: Specify contracts for pipeline checkpoints (dataset→buffer→generation→reward→loss)
3. **Configuration Design**: Create diagnostic YAML schemas with explicit logging/export parameters
4. **Quickstart Guide**: Document how to run diagnostic configs, interpret TensorBoard logs, and identify issues from exported artifacts

### Design Decisions

- **Instrumentation Strategy**: Lightweight decorators/context managers at 5 hook points; <10% overhead via lazy evaluation and rank-0-only exports
- **Storage Format**: Diagnostic artifacts export to `{output_dir}/diagnostics/{step:06d}/` as JSON (scalars/distributions) and PNG (histograms/heatmaps)
- **Failure Detection**: Automated warning/error thresholds (e.g., ratio std < 0.1 → WARNING, image token mismatch → ERROR)
- **Checkpoint Diversity Testing**: Separate runner script `scripts/run_checkpoint_diversity_test.sh` that sweeps temperatures and logs diversity ratio

### Deliverables

- **data-model.md**: Typed schemas for all diagnostic entities with examples
- **contracts/**: 
  - `validation_checkpoint.py`: Interface for `validate_at_stage(stage: str, **tensors) -> Dict[str, Any]`
  - `diagnostic_export.py`: Interface for `export_artifacts(step: int, output_dir: Path) -> None`
- **quickstart.md**: Step-by-step guide with example commands and expected outputs

---

## Phase 2: Task Breakdown

**Output**: `tasks.md` generated by `/speckit.tasks` command

### Task Organization

Tasks will be grouped by user stories (US1-US6) with explicit dependencies and parallel-safety markers `[P]`:

**US1 - Trust Region Validation (P1)**:
- Write failing tests for generation_logps storage, retrieval, ratio computation
- Implement `TrustRegionDiagnostic` tracker in `diagnostics/trust_region.py`
- Add instrumentation hooks in `buffer.py` (generation) and `completion_loss.py` (loss computation)
- Export ratio histograms and degeneracy warnings

**US2 - Multimodal Alignment Verification (P1)**:
- Write failing tests for image token/pixel_values/THW consistency across pipeline stages
- Implement `MultimodalAlignmentCheck` in `diagnostics/multimodal.py`
- Add validation checkpoints at dataset emission, buffer generation, loss computation
- Integrate with existing `validators.py::assert_patches_match_thw`

**US3 - Reward Function Sanity Check (P1)**:
- Write failing tests for within-group/between-group variance tracking and checkpoint diversity
- Implement `RewardProfile` and `CheckpointDiversityComparison` in `diagnostics/rewards.py` and `diagnostics/checkpoint_diversity.py`
- Create `configs/dense_rl/checkpoint_diversity_test.yaml` with Phase 2/3 checkpoint paths and temperature sweep
- Add reward variance logging to `rl/reward_logger.py`

**US4 - Gradient Flow Verification (P2)**:
- Write failing tests for gradient norm tracking and accumulation scaling
- Implement `GradientFlowSnapshot` with gradient hooks in `diagnostics/gradients.py`
- Add per-layer gradient logging in `grpo_trainer.py::_compute_loss_for_single_completion`
- Export gradient flow heatmaps

**US5 - Sequential Processing Compliance (P2)**:
- Write failing tests for batch_size assertions and memory monitoring
- Implement `SequentialProcessingMonitor` in `diagnostics/sequential.py`
- Add assertions in `generation.py::sample_k` and `completion_loss.py::compute_streaming_loss`
- Log peak GPU memory per training step

**US6 - Reference vs. Current Policy Alignment (P3)**:
- Write failing tests for ref_model freezing and KL computation
- Add KL breakdown logging in `grpo_trainer.py::train`
- Compare advantage normalization formulas against reference implementations

### Cross-Cutting Tasks

- Create `configs/dense_rl/diagnostic.yaml` (minimal 10-sample, 20-step config)
- Implement `diagnostics/exporters.py` for histogram/heatmap generation
- Add CI smoke test job for diagnostic config
- Update `src_new/rl/GRPO_README.md` with diagnostics section

---

## Next Steps

1. **Run `/speckit.tasks`** to generate tasks.md with granular task breakdown
2. **Phase 0 Execution**: Create `research.md` with diversity analysis and reference algorithm comparison
3. **Phase 1 Execution**: Create `data-model.md`, `contracts/`, and `quickstart.md`
4. **Implementation**: TDD workflow per user story (write failing tests → implement → refactor)
5. **Validation**: Run diagnostic configs on 8 GPUs, identify root cause of training instability, capture findings in code comments and plan updates
