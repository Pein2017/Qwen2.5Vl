---
description: "Task list for GRPO Post-Training Diagnostics and Error Detection"
---

# Tasks: GRPO Post-Training Diagnostics and Error Detection

**Input**: Design documents from `/data3/Qwen2.5-VL-main/specs/004-grpo-post-training/`
**Prerequisites**: spec.md, plan.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Tests are constitutionally MANDATORY and follow TDD workflow (write failing tests FIRST).

**Organization**: Tasks grouped by user story (US1-US6) for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1=Trust Region, US2=Multimodal, US3=Rewards, US4=Gradients, US5=Sequential, US6=KL)
- Paths are absolute from `/data3/Qwen2.5-VL-main/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create diagnostic module structure and test framework

- [X] T001 Create `src_new/rl/diagnostics/` directory structure with `__init__.py`
- [X] T002 Create `tests/rl/diagnostics/` directory structure  
- [X] T003 [P] Create `configs/dense_rl/diagnostic.yaml` stub (minimal 10 samples, 20 steps)
- [X] T004 [P] Create `configs/dense_rl/checkpoint_diversity_test.yaml` stub (Phase 2/3 comparison)
- [X] T005 [P] Add pytest fixtures in `tests/rl/diagnostics/conftest.py` for mock model/tokenizer/processor

---

## Phase 2: Foundational (Contracts & Exporters)

**Purpose**: Core interfaces that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 [P] Copy `specs/004-grpo-post-training/contracts/validation_checkpoint.py` to `src_new/rl/diagnostics/validation_checkpoint.py` *(Note: Files are copied as-is from specs/contracts/; do not modify. Future edits must update source in specs/contracts/)*
- [X] T007 [P] Copy `specs/004-grpo-post-training/contracts/diagnostic_export.py` to `src_new/rl/diagnostics/diagnostic_export.py` *(Note: Files are copied as-is from specs/contracts/; do not modify. Future edits must update source in specs/contracts/)*
- [X] T008 [P] Implement `StandardDiagnosticExporter` in `src_new/rl/diagnostics/exporters.py` with JSON/PNG export methods
- [X] T009 [P] Implement `TensorBoardExporter` wrapper in `src_new/rl/diagnostics/exporters.py`
- [X] T010 [P] Add unit tests for exporters in `tests/rl/diagnostics/test_exporters.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Trust Region Validation (Priority: P1) 🎯

**Goal**: Detect degenerate GRPO ratios (π_current / π_generation ≈ 1.0) caused by missing generation_logps

**Independent Test**: Run 10-step GRPO loop, verify ratio statistics (mean ∈ [0.8, 1.5], std > 0.1)

### Tests for User Story 1 (MANDATORY) ⚠️

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T011 [P] [US1] Write failing test `test_trust_region_diagnostic_creation` in `tests/rl/diagnostics/test_trust_region.py`
  - Test: Create `TrustRegionDiagnostic` with mock ratio tensors, verify fields populated correctly
  
- [X] T012 [P] [US1] Write failing test `test_trust_region_ratio_computation` in `tests/rl/diagnostics/test_trust_region.py`
  - Test: Mock generation_logps and current_logps, compute ratios, verify mean/std/percentiles match expected

- [X] T013 [P] [US1] Write failing test `test_trust_region_degeneracy_detection` in `tests/rl/diagnostics/test_trust_region.py`
  - Test: Feed all-1.0 ratios, verify `is_degenerate=True` flag set

- [X] T014 [P] [US1] Write failing test `test_trust_region_fallback_warning` in `tests/rl/diagnostics/test_trust_region.py`
  - Test: Simulate missing generation_logps, verify fallback_count > 0 and warning logged

### Implementation for User Story 1

- [X] T015 [P] [US1] Implement `TrustRegionDiagnostic` dataclass in `src_new/rl/diagnostics/trust_region.py`
  - Fields: step, ratio statistics (mean/std/min/max/percentiles), degeneracy flags, fallback counter
  - Methods: `to_tensorboard()`, `export_histogram()`

- [X] T016 [US1] Add `compute_trust_region_diagnostic()` helper in `src_new/rl/diagnostics/trust_region.py`
  - Input: step, ratios tensor, generation_logps_present flag
  - Output: `TrustRegionDiagnostic` instance
  - Logic: Compute statistics, check degeneracy threshold (std < 0.1)

- [ ] T017 [US1] Add instrumentation hook in `src_new/rl/completion_loss.py::_get_generation_logprobs`
  - After ratio computation, call `compute_trust_region_diagnostic()`
  - Store diagnostic in trainer state
  - Log warning if fallback used or degeneracy detected

- [ ] T018 [US1] Add instrumentation hook in `src_new/rl/buffer.py::generate_and_score`
  - Verify generation_logps tensor shape and non-zero values
  - Log generation_logps_present status

- [ ] T019 [US1] Add TensorBoard logging in `src_new/rl/grpo_trainer.py::train` loop
  - After each training step, log `diagnostic.to_tensorboard()`
  - Export histogram every 10 steps via exporter

- [ ] T020 [US1] Run pytest `tests/rl/diagnostics/test_trust_region.py` - all tests MUST pass

**Checkpoint**: User Story 1 complete - trust region diagnostics functional

---

## Phase 4: User Story 2 - Multimodal Alignment Verification (Priority: P1) 🎯

**Goal**: Validate image tokens, pixel_values, and image_grid_thw consistency across pipeline stages

**Independent Test**: Process sample with 1 image, verify `<|image_pad|>` count == THW-derived patch count at 3 stages

### Tests for User Story 2 (MANDATORY) ⚠️

- [ ] T021 [P] [US2] Write failing test `test_multimodal_alignment_check_creation` in `tests/rl/diagnostics/test_multimodal_alignment.py`
  - Test: Create `MultimodalAlignmentCheck` with mock tensors, verify all fields populated

- [ ] T022 [P] [US2] Write failing test `test_image_token_count_validation` in `tests/rl/diagnostics/test_multimodal_alignment.py`
  - Test: Mock image_grid_thw, compute expected tokens, verify match with decoded count

- [ ] T023 [P] [US2] Write failing test `test_pixel_values_row_count_validation` in `tests/rl/diagnostics/test_multimodal_alignment.py`
  - Test: Mock pixel_values and THW, verify row count matches sum(t*h*w)

- [ ] T024 [P] [US2] Write failing test `test_multimodal_mismatch_raises_error` in `tests/rl/diagnostics/test_multimodal_alignment.py`
  - Test: Feed mismatched tensors, verify `ImageTokenMismatchError` raised

### Implementation for User Story 2

- [ ] T025 [P] [US2] Implement `MultimodalAlignmentCheck` dataclass in `src_new/rl/diagnostics/multimodal.py`
  - Fields: stage, sample_idx, expected vs actual counts, shape validation flags
  - Methods: `raise_on_mismatch()`, `to_log_dict()`

- [ ] T026 [P] [US2] Implement `MultimodalAlignmentValidator` in `src_new/rl/diagnostics/multimodal.py`
  - Implements `ValidationCheckpoint` protocol from contracts
  - Methods: `validate_at_stage()`, `get_expected_image_tokens()`, `assert_patches_match_thw()`

- [ ] T027 [US2] Add validation checkpoint in `src_new/rl/data/dataset.py::__getitem__`
  - Stage: "dataset"
  - Validate image tokens/pixel_values/THW after conversation builder
  - Call `check.raise_on_mismatch()` for fail-fast

- [ ] T028 [US2] Add validation checkpoint in `src_new/rl/buffer.py::generate_and_score`
  - Stage: "buffer_generation"
  - Validate after vision tensor packing
  - Integrate with existing `validators.py::assert_patches_match_thw`

- [ ] T029 [US2] Add validation checkpoint in `src_new/rl/completion_loss.py::compute_streaming_loss`
  - Stage: "loss_computation"
  - Validate before forward pass

- [ ] T030 [US2] Run pytest `tests/rl/diagnostics/test_multimodal_alignment.py` - all tests MUST pass

**Checkpoint**: User Story 2 complete - multimodal alignment validated at 3 pipeline stages

---

## Phase 5: User Story 3 - Reward Function Sanity Check (Priority: P1) 🎯

**Goal**: Track within-group/between-group reward variance and compare Phase 2 vs Phase 3 checkpoint diversity

**Independent Test**: Generate 50 completions from 2 checkpoints, verify within-group std > 0.05 for ≥60% prompts

### Tests for User Story 3 (MANDATORY) ⚠️

- [ ] T031 [P] [US3] Write failing test `test_reward_profile_creation` in `tests/rl/diagnostics/test_reward_variance.py`
  - Test: Create `RewardProfile` with mock rewards, verify within/between variance computed

- [ ] T032 [P] [US3] Write failing test `test_within_group_variance_tracking` in `tests/rl/diagnostics/test_reward_variance.py`
  - Test: Mock reward tensor [P, K], compute within-group std per prompt, verify mean/median

- [ ] T033 [P] [US3] Write failing test `test_diversity_ratio_computation` in `tests/rl/diagnostics/test_reward_variance.py`
  - Test: Mock within/between variance, compute ratio, verify < 0.1 triggers degenerate flag

- [ ] T034 [P] [US3] Write failing test `test_checkpoint_diversity_comparison` in `tests/rl/diagnostics/test_checkpoint_diversity.py`
  - Test: Mock 2 checkpoints with different diversity, verify comparison metrics

- [ ] T035 [P] [US3] Write failing test `test_checkpoint_viability_assessment` in `tests/rl/diagnostics/test_checkpoint_diversity.py`
  - Test: Verify `is_viable_for_grpo` flag based on advantage_std and diversity_ratio thresholds

### Implementation for User Story 3

- [ ] T036 [P] [US3] Implement `RewardProfile` dataclass in `src_new/rl/diagnostics/rewards.py`
  - Fields: raw stats, within/between variance, diversity ratio, collapsed flags
  - Methods: `to_tensorboard()`, `to_json()`

- [ ] T037 [P] [US3] Implement `compute_reward_profile()` helper in `src_new/rl/diagnostics/rewards.py`
  - Input: step, reward_name, reward_tensor [P, K], threshold
  - Output: `RewardProfile` instance
  - Logic: Compute within-group std per prompt, between-group std, diversity ratio

- [ ] T038 [P] [US3] Implement `CheckpointDiversityComparison` dataclass in `src_new/rl/diagnostics/checkpoint_diversity.py`
  - Fields: checkpoint info, within/between variance, advantage stats, viability flag
  - Methods: `compare_to()`, `to_json()`

- [ ] T039 [US3] Add reward variance logging in `src_new/rl/grpo_trainer.py::train` loop
  - After reward computation, call `compute_reward_profile()` for each reward function
  - Log to TensorBoard with `rewards/{name}/*` prefix

- [ ] T040 [US3] Integrate with existing `src_new/rl/reward_logger.py`
  - Add within-group/between-group variance to reward logs
  - Flag collapsed rewards (std < 0.01)

- [ ] T040a [US3] Add EOS termination, completion length, and cap-hit ratio logging to `RewardProfile`
  - Compute EOS termination ratio (count of completions ending with `<|im_end|>` / total completions)
  - Compute completion length distribution (mean, std, min, max) relative to ground-truth length
  - Compute cap-hit ratio (count of completions hitting max_new_tokens / total completions)
  - Log to TensorBoard with `generation/*` prefix
  - Add assertions: EOS ratio > 80%, mean length ∈ [0.8×GT, 1.5×GT], cap-hit < 30%

- [ ] T041 [US3] Create checkpoint diversity test script `scripts/run_checkpoint_diversity_test.sh`
  - Load Phase 2 and Phase 3 checkpoints
  - Sweep temperatures {0.7, 0.9, 1.1}
  - Generate 50 completions × 2 checkpoints
  - Compute diversity metrics and save to `outputs/checkpoint_diversity/comparison_report.json`

- [ ] T042 [US3] Populate `configs/dense_rl/checkpoint_diversity_test.yaml` with full config
  - Checkpoint paths for Phase 2 and Phase 3
  - Temperature sweep parameters
  - Output directory settings

- [ ] T043 [US3] Run pytest `tests/rl/diagnostics/test_reward_variance.py` and `test_checkpoint_diversity.py` - all tests MUST pass

**Checkpoint**: User Story 3 complete - reward variance tracking and checkpoint diversity comparison functional

---

## Phase 6: User Story 4 - Gradient Flow Verification (Priority: P2)

**Goal**: Monitor gradient norms per layer to detect vanishing/exploding gradients

**Independent Test**: Run 5 training steps, verify gradients in top-8 LLM layers have norm ∈ [0.01, 10.0]

### Tests for User Story 4 (MANDATORY) ⚠️

- [ ] T044 [P] [US4] Write failing test `test_gradient_flow_snapshot_creation` in `tests/rl/diagnostics/test_gradient_flow.py`
  - Test: Create `GradientFlowSnapshot` from mock parameter, verify norm computed

- [ ] T045 [P] [US4] Write failing test `test_gradient_vanishing_detection` in `tests/rl/diagnostics/test_gradient_flow.py`
  - Test: Feed parameter with grad_norm < 1e-6, verify `has_zero_gradients=True`

- [ ] T046 [P] [US4] Write failing test `test_gradient_exploding_detection` in `tests/rl/diagnostics/test_gradient_flow.py`
  - Test: Feed parameter with grad_norm > 100, verify `has_exploding_gradients=True`

- [ ] T047 [P] [US4] Write failing test `test_gradient_heatmap_export` in `tests/rl/diagnostics/test_gradient_flow.py`
  - Test: Mock snapshots for multiple layers, verify heatmap PNG exported

### Implementation for User Story 4

- [ ] T048 [P] [US4] Implement `GradientFlowSnapshot` dataclass in `src_new/rl/diagnostics/gradients.py`
  - Fields: step, layer_name, grad norms (mean/max/min), health flags
  - Methods: `from_parameter()`, `export_heatmap()`

- [ ] T049 [US4] Add gradient hooks in `src_new/rl/grpo_trainer.py::_compute_loss_for_single_completion`
  - Register hooks on top-8 LLM layers
  - After backward pass, capture gradient norms
  - Create `GradientFlowSnapshot` per layer

- [ ] T050 [US4] Add per-layer gradient logging in `src_new/rl/grpo_trainer.py::train` loop
  - Log gradient norms to TensorBoard with `gradients/{layer_name}/norm` prefix
  - Aggregate snapshots across micro-steps

- [ ] T051 [US4] Add heatmap export every 10 steps
  - Call `export_heatmap()` with all layer snapshots
  - Save to `{output_dir}/diagnostics/{step:06d}/gradient_flow.png`

- [ ] T052 [US4] Run pytest `tests/rl/diagnostics/test_gradient_flow.py` - all tests MUST pass

**Checkpoint**: User Story 4 complete - gradient flow monitoring functional

---

## Phase 7: User Story 5 - Sequential Processing Compliance (Priority: P2)

**Goal**: Verify batch_size=1 compliance and GPU memory < 80GB per constitutional mandate

**Independent Test**: Run GRPO with sample_k=4, assert 4 sequential generate calls and 4 sequential forward passes

### Tests for User Story 5 (MANDATORY) ⚠️

- [ ] T053 [P] [US5] Write failing test `test_sequential_processing_monitor_creation` in `tests/rl/diagnostics/test_sequential_processing.py`
  - Test: Create `SequentialProcessingMonitor` with mock batch sizes, verify compliance check

- [ ] T054 [P] [US5] Write failing test `test_batch_size_violation_detection` in `tests/rl/diagnostics/test_sequential_processing.py`
  - Test: Feed batch_size > 1, verify `is_compliant=False` and error raised

- [ ] T055 [P] [US5] Write failing test `test_gpu_memory_tracking` in `tests/rl/diagnostics/test_sequential_processing.py`
  - Test: Mock CUDA memory allocation, verify peak tracking and OOM risk flag

### Implementation for User Story 5

- [ ] T056 [P] [US5] Implement `SequentialProcessingMonitor` dataclass in `src_new/rl/diagnostics/sequential.py`
  - Fields: step, batch_size lists (generation/forward), GPU memory peaks, compliance flags
  - Methods: `raise_on_violation()`, `to_tensorboard()`

- [ ] T057 [US5] Add batch_size assertions in `src_new/rl/generation.py::sample_k`
  - Before each `model.generate` call, assert `batch_size == 1`
  - Track batch sizes in monitor

- [ ] T058 [US5] Add batch_size assertions in `src_new/rl/completion_loss.py::compute_streaming_loss`
  - Before each forward pass, assert `input_ids.shape[0] == 1`
  - Track batch sizes in monitor

- [ ] T059 [US5] Add GPU memory monitoring in `src_new/rl/grpo_trainer.py::train` loop
  - Track `torch.cuda.max_memory_allocated()` per device
  - Check against 80GB threshold, set `has_oom_risk` flag

- [ ] T060 [US5] Add sequential compliance logging to TensorBoard
  - Log `sequential/is_compliant` and `sequential/peak_memory_mb`

- [ ] T061 [US5] Run pytest `tests/rl/diagnostics/test_sequential_processing.py` - all tests MUST pass

**Checkpoint**: User Story 5 complete - sequential processing validated, memory monitored

---

## Phase 8: User Story 6 - Reference vs. Current Policy Alignment (Priority: P3)

**Goal**: Validate KL regularization (ref_model frozen, KL term computed correctly)

**Independent Test**: Enable beta=0.05, run 10 steps, verify ref_model frozen and KL ∈ [0.01, 0.1]

### Tests for User Story 6 (MANDATORY) ⚠️

- [ ] T062 [P] [US6] Write failing test `test_ref_model_freezing` in `tests/rl/diagnostics/test_kl_regularization.py`
  - Test: Initialize trainer with beta > 0, verify ref_model.requires_grad=False for all params

- [ ] T063 [P] [US6] Write failing test `test_kl_divergence_computation` in `tests/rl/diagnostics/test_kl_regularization.py`
  - Test: Mock current/ref logits, compute KL, verify positive and bounded < 1.0

- [ ] T064 [P] [US6] Write failing test `test_loss_breakdown_logging` in `tests/rl/diagnostics/test_kl_regularization.py`
  - Test: Mock GRPO loss and KL term, verify total_loss = grpo_loss + beta * KL_loss

### Implementation for User Story 6

- [ ] T065 [US6] Add ref_model parameter freeze verification in `src_new/rl/grpo_trainer.py::__init__`
  - If beta > 0, verify all ref_model parameters have `requires_grad=False`
  - Log warning if any parameter is trainable

- [ ] T066 [US6] Add KL divergence logging in `src_new/rl/grpo_trainer.py::train` loop
  - Log `kl_divergence`, `beta`, and loss breakdown to TensorBoard
  - Track KL statistics (mean, max, min)

- [ ] T067 [US6] Compare advantage normalization against reference implementations
  - Add code comment in `src_new/rl/advantage_normalizer.py` documenting formula match with Qwen2-VL-Finetune
  - Log comparison metrics if divergence > 5%

- [ ] T068 [US6] Run pytest `tests/rl/diagnostics/test_kl_regularization.py` - all tests MUST pass

**Checkpoint**: User Story 6 complete - KL regularization validated

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final configs, documentation, and CI integration

- [ ] T069 [P] Populate `configs/dense_rl/diagnostic.yaml` with full configuration
  - 10 samples, 20 steps, maximal logging enabled
  - All diagnostic flags turned on
  - Export interval = 10 steps

- [ ] T069a [P] Add reward weight validation logic in `src_new/rl/diagnostics/rewards.py`
  - Validate config: sum of reward weights > 0
  - Validate at least one detection reward (bbox_giou, quad_l1, line_l1, coverage, geometry_sanity) has non-zero weight
  - Raise actionable error with missing weights listed if validation fails
  - Add unit test in `tests/rl/diagnostics/test_reward_variance.py`

- [ ] T070 [P] Update `src_new/rl/GRPO_README.md` with diagnostics section
  - Document all diagnostic entities and their purpose
  - Add troubleshooting guide referencing quickstart.md
  - Link to diagnostic configs

- [ ] T071 [P] Create smoke test script `scripts/test_diagnostic_config.sh`
  - Run diagnostic.yaml for 20 steps
  - Verify zero errors and artifacts exported
  - Timeout < 10 minutes on 8 GPUs

- [ ] T072 [P] Add CI job `.github/workflows/diagnostic_smoke_test.yml` (if CI exists)
  - Run smoke test on each PR
  - Fail if diagnostic config crashes or produces errors

- [ ] T073 [P] Run full validation per quickstart.md
  - Execute diagnostic config end-to-end
  - Verify TensorBoard metrics logged correctly
  - Check exported artifacts (JSON, PNG) in diagnostics/ directory

- [ ] T074 Code cleanup and type-check
  - Run `pyright src_new/rl/diagnostics/` - fix all type errors
  - Run `ruff check src_new/rl/diagnostics/` - fix all lint errors
  - Add missing docstrings to all public functions

- [ ] T075 [P] Final integration test across all user stories
  - Run diagnostic.yaml with all features enabled
  - Verify all 6 user stories produce expected diagnostics
  - Capture findings in plan.md updates (no temporary docs)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-8)**: All depend on Foundational completion
  - Can proceed in parallel (if staffed) or sequentially by priority (P1 → P2 → P3)
- **Polish (Phase 9)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (Trust Region)**: Independent - can start after Foundational
- **US2 (Multimodal)**: Independent - can start after Foundational  
- **US3 (Rewards)**: Independent - can start after Foundational
- **US4 (Gradients)**: Independent - can start after Foundational
- **US5 (Sequential)**: Independent - can start after Foundational
- **US6 (KL Regularization)**: Independent - can start after Foundational

All user stories are independently testable and can be implemented in parallel.

### Within Each User Story

1. Tests MUST be written and FAIL before implementation (TDD)
2. Dataclass models before helper functions
3. Helper functions before instrumentation hooks
4. Instrumentation hooks before logging integration
5. All tests MUST pass before story marked complete

### Parallel Opportunities

**Phase 1 (Setup)**: All tasks [P] can run in parallel
**Phase 2 (Foundational)**: T006-T010 all marked [P] can run in parallel
**Phase 3-8 (User Stories)**: Once Foundational complete, all 6 user stories can start in parallel
**Phase 9 (Polish)**: T069-T073, T075 all marked [P] can run in parallel

---

## Parallel Example: User Story 1 (Trust Region)

```bash
# Launch all tests for US1 together:
Task: "Write failing test test_trust_region_diagnostic_creation in tests/rl/diagnostics/test_trust_region.py"
Task: "Write failing test test_trust_region_ratio_computation in tests/rl/diagnostics/test_trust_region.py"
Task: "Write failing test test_trust_region_degeneracy_detection in tests/rl/diagnostics/test_trust_region.py"
Task: "Write failing test test_trust_region_fallback_warning in tests/rl/diagnostics/test_trust_region.py"

# After tests fail, implement dataclass and helpers in parallel:
Task: "Implement TrustRegionDiagnostic dataclass in src_new/rl/diagnostics/trust_region.py"
Task: "Add compute_trust_region_diagnostic() helper in src_new/rl/diagnostics/trust_region.py"
```

---

## Parallel Example: All P1 User Stories

```bash
# Once Foundational (Phase 2) is complete, launch all P1 stories:
Task: "Implement User Story 1 (Trust Region Validation)"
Task: "Implement User Story 2 (Multimodal Alignment Verification)"
Task: "Implement User Story 3 (Reward Function Sanity Check)"
```

---

## Implementation Strategy

### MVP First (P1 Stories Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3-5: User Stories 1-3 (all P1 priority)
4. **STOP and VALIDATE**: Run diagnostic.yaml, verify P1 diagnostics working
5. Analyze findings, identify root cause of GRPO training instability

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 (Trust Region) → Test independently → Diagnose ratio degeneracy
3. Add US2 (Multimodal) → Test independently → Verify alignment
4. Add US3 (Rewards) → Test independently → Compare checkpoints, identify diversity collapse
5. Add US4 (Gradients) → Test independently → Check gradient flow
6. Add US5 (Sequential) → Test independently → Validate compliance
7. Add US6 (KL Regularization) → Test independently → Optional feature validation
8. Each story adds diagnostic capability without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (Phase 1-2)
2. Once Foundational done:
   - Developer A: US1 (Trust Region)
   - Developer B: US2 (Multimodal)
   - Developer C: US3 (Rewards)
   - Developer D: US4 (Gradients)
   - Developer E: US5 (Sequential)
   - Developer F: US6 (KL Regularization)
3. Stories complete independently, integrate via shared exporters/TensorBoard

---

## Success Validation

After completing all tasks, verify success criteria from spec.md:

- **SC-001**: Trust region ratio mean ∈ [0.8, 1.5], std > 0.15 for 90% of steps ✓
- **SC-002**: 100% multimodal alignment validation pass rate ✓
- **SC-003**: ≥3 reward functions with within-group std > 0.05 for ≥60% prompts ✓
- **SC-003b**: Checkpoint diversity comparison identifies viable SFT checkpoint ✓
- **SC-004**: Non-zero gradients in top-8 LLM layers for 95% of steps ✓
- **SC-005**: 100% batch_size=1 compliance, peak memory < 80GB ✓
- **SC-006**: Diagnostic artifacts exported within 5 min ✓
- **SC-007**: <5% divergence vs. official GRPO reference ✓
- **SC-008**: Debug config completes in <10 min with actionable report ✓
- **SC-009**: Reward weight validation prevents config errors ✓
- **SC-010**: EOS termination >80%, mean length ∈ [0.8×GT, 1.5×GT] ✓

---

## Notes

- **Total Tasks**: 77 (75 original + T040a for FR-010 + T069a for FR-009)
- [P] tasks = different files, no dependencies - can run in parallel
- [Story] label (US1-US6) maps task to user story for traceability
- TDD MANDATORY: Tests written and failing BEFORE implementation
- Each user story independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All findings captured in plan.md updates (no temporary docs per Constitution v4.1.1)
- Diagnostic overhead target: <10% of baseline training time
- GPU memory threshold updated to <80GB per device (Constitutional compliance)
