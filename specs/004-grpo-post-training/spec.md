# Feature Specification: GRPO Post-Training Diagnostics and Error Detection

**Feature Branch**: `004-grpo-post-training`  
**Created**: 2025-10-09  
**Status**: Draft  
**Constitution Version**: 4.1.1  
**Input**: User description: "GRPO post-training diagnostics and error detection for dense captioning task with focus on finding hidden algorithmic or tensor mismatch errors causing training instability"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Trust Region Validation (Priority: P1)

As an ML engineer debugging GRPO training, I need to verify that the trust region mechanism is working correctly by confirming that generation-policy log-probs are properly stored and used in ratio computation, so that the policy gradient signals are not degenerate.

**Why this priority**: This is the most critical failure mode documented in the codebase. If the GRPO ratio degenerates to ≈1.0 due to missing or incorrect generation log-probs, the entire training loop becomes ineffective regardless of other components.

**Independent Test**: Run a minimal 10-step GRPO training loop with extensive logging of: (1) generation_logps storage at buffer creation, (2) old_logps retrieval during loss computation, (3) computed ratios exp(cur_logps - old_logps), (4) ratio statistics (mean, std, min, max). Success = ratios show meaningful variance (std > 0.1) and are not centered at 1.0.

**Acceptance Scenarios**:

1. **Given** a trained SFT checkpoint and validation dataset, **When** GRPO training runs for 10 steps with `sample_k=4`, **Then** `generation_logps` tensor in buffer has shape `[P×K, max_len]` with non-zero values
2. **Given** stored generation_logps in buffer, **When** computing loss for completion k, **Then** `old_logps_k` matches `generation_logps[k, :]` and ratio distribution shows mean ≠ 1.0 with std > 0.1
3. **Given** multi-rank training (8 GPUs), **When** generation occurs, **Then** all ranks produce identical generation_logps for the same prompt/seed

---

### User Story 2 - Multimodal Alignment Verification (Priority: P1)

As an ML engineer, I need to validate that image tokens, pixel_values, and image_grid_thw maintain consistency throughout the GRPO pipeline (generation → reward → loss), so that vision-language alignment doesn't drift and cause training instability.

**Why this priority**: The dense captioning task is multimodal. Any mismatch between `<|image_pad|>` count and `image_grid_thw` causes forward pass errors or silent corruption. This is a documented critical contract.

**Independent Test**: Instrument the pipeline with validators at: (1) dataset emission, (2) buffer.generate_and_score, (3) completion_loss.compute_streaming_loss. Log image token counts, THW shapes, pixel_values row counts. Success = all three checkpoints report identical counts with no warnings.

**Acceptance Scenarios**:

1. **Given** a sample with 1 image at resolution 1568×1176, **When** processed through conversation builder, **Then** decoded prompt contains exactly `(⌈1568/28⌉ × ⌈1176/28⌉) // 4 = 1225` image_pad tokens
2. **Given** packed vision tensors in buffer, **When** slicing for per-completion forward, **Then** `pixel_values.shape[0] == sum(t*h*w for t,h,w in image_grid_thw)` holds
3. **Given** cross-rank buffer split, **When** vision tensors are chunked, **Then** `assert_patches_match_thw` passes without warnings

---

### User Story 3 - Reward Function Sanity Check (Priority: P1)

As an ML engineer, I need to verify that reward functions return meaningful signals (not all zeros, not all NaNs, with reasonable variance) and that the weighted combination produces advantages that drive learning, so that the policy has a clear optimization signal.

**Why this priority**: If rewards are degenerate (all equal, all zero, or dominated by noise), advantages will be meaningless and the policy won't improve. **Critical observation**: Zero advantage std after some iterations suggests the SFT checkpoint produces overly deterministic outputs within each K-completion group, causing reward collapse. This is testable independently and critical for diagnosing "no improvement" symptoms.

**Checkpoint Comparison Strategy**: Test diversity across different SFT fine-tuning degrees:
- **Phase 3 (fully fine-tuned)**: `outputs/7B-all_tokens/phase_3/9-23-phase_3-all_tokens-lower_grounding_weight-last_vision_4-with_text_only-resume/checkpoint-2000` (current default in `configs/dense_rl/standard.yaml`)
- **Phase 2 (less fine-tuned)**: `outputs/7B-all_tokens/phase_2/9-21-phase_2-last_blocks_6-all_tokens/best-200-eval_loss0.7006`
- **Hypothesis**: Phase 2 checkpoint may produce more diverse outputs → non-zero advantages → effective GRPO training

**Independent Test**: Generate 50 completions from **both checkpoints** with varying temperatures (0.7, 0.9, 1.1), compute per-function rewards without standardization, log: (1) within-group reward variance (std across K completions per prompt), (2) between-group variance (std across prompts), (3) advantage std. Success = at least one checkpoint shows within-group std > 0.05 for 60% of prompts at temperature=0.9.

**Acceptance Scenarios**:

1. **Given** K=4 completions per prompt from Phase 2 and Phase 3 checkpoints at temperature=0.9, **When** computing rewards within each group, **Then** at least one checkpoint shows ≥60% of prompts with within-group reward std > 0.05 (identifies which SFT degree works)
2. **Given** ground-truth objects in meta, **When** comparing Phase 2 vs Phase 3 checkpoints, **Then** log diversity ratio: `Phase2_within_std / Phase3_within_std` to quantify over-fitting effect
3. **Given** K=4 completions per prompt, **When** forming advantages A = (r - mean_K) / std_K for both checkpoints, **Then** report which checkpoint produces non-zero advantage std and viable GRPO gradients

---

### User Story 4 - Gradient Flow Verification (Priority: P2)

As an ML engineer, I need to confirm that gradients flow correctly through the GRPO loss computation (no vanishing/exploding gradients, proper accumulation across K completions and prompt batches), so that optimizer steps actually update the model.

**Why this priority**: Even with correct ratios and rewards, if gradients are zero or explode, training won't work. This is secondary to P1 issues but critical if P1 checks pass.

**Independent Test**: Add gradient hooks to LLM layers, run 5 training steps, log gradient norms per layer and per micro-step. Success = gradients are non-zero (norm > 1e-6), bounded (norm < 100), and consistent across steps.

**Acceptance Scenarios**:

1. **Given** GRPO loss computation for one completion, **When** calling `accelerator.backward(loss_k / K)`, **Then** LLM top-layer gradients have norm ∈ [0.01, 10.0]
2. **Given** prompt_batch_size=4 and sample_k=4, **When** accumulating gradients over 16 completions, **Then** final gradient norm scales roughly as sqrt(16) compared to single-completion norm
3. **Given** gradient clipping with max_norm=1.0, **When** optimizer step occurs, **Then** clipped norm logged matches expectation and loss decreases over 10 steps

---

### User Story 5 - Sequential Processing Compliance (Priority: P2)

As an ML engineer with limited GPU memory, I need to verify that the implementation respects the constitutional requirement for sequential `model.generate()` and `model.forward()` calls (no batching), so that training runs without OOM errors on constrained hardware.

**Why this priority**: The constitution (v4.1.1) mandates sequential processing for GRPO. Violations cause OOM. This is architectural compliance rather than algorithmic correctness.

**Independent Test**: Run GRPO training with GPU memory monitoring, confirm that: (1) `sample_k` calls to `model.generate` occur sequentially with batch_size=1, (2) per-completion forward passes occur one-by-one in CompletionLossComputer, (3) peak memory stays below threshold.

**Acceptance Scenarios**:

1. **Given** sample_k=4, **When** generating completions in buffer.py, **Then** 4 sequential `model.generate` calls occur with batch_size=1 each
2. **Given** K=4 completions in CompletionLossComputer, **When** computing loss, **Then** 4 sequential forward passes occur with input_ids.shape[0]=1
3. **Given** 8 GPUs and prompt_batch_size=4, **When** training, **Then** peak GPU memory per device < 80GB and no OOM errors

---

### User Story 6 - Reference vs. Current Policy Alignment (Priority: P3)

As an ML engineer using KL regularization, I need to verify that the reference model (if enabled) is properly frozen and that KL divergence is computed correctly, so that the policy doesn't drift too far from the initial checkpoint.

**Why this priority**: Optional feature (beta > 0). Only relevant if using KL regularization. Lower priority than core GRPO mechanism.

**Independent Test**: Enable beta=0.05, run 10 steps, log: (1) reference model parameter updates (should be zero), (2) KL term values, (3) total loss breakdown. Success = ref model frozen, KL term ≈ 0.01-0.1 range.

**Acceptance Scenarios**:

1. **Given** beta=0.05 in config, **When** initializing trainer, **Then** ref_model parameters are frozen (requires_grad=False)
2. **Given** frozen ref_model, **When** computing KL divergence, **Then** KL values are positive and bounded < 1.0
3. **Given** KL-enabled training, **When** logging loss breakdown, **Then** total_loss = grpo_loss + beta * KL_loss with both terms logged separately

---

### Edge Cases

- **What happens when** a completion contains zero `<|im_end|>` tokens (truncated generation)?
  - **Expected**: Completion mask flags as truncated; if `mask_truncated_completions=true`, loss contribution is zero; otherwise full sequence contributes.
  
- **How does system handle** NaN rewards from a single reward function?
  - **Expected**: `torch.nan_to_num` converts to 0.0; reward is clipped and logged; training continues but warning issued.

- **What happens when** all K completions for a prompt have identical rewards (zero within-group variance)?
  - **Expected**: Advantage computation yields NaN (0/0) or zeros; std_K clamped to 1e-4 as fallback; policy receives no learning signal for that prompt; logged as "degenerate advantage" warning. This is the **observed issue** causing training failure.
  
- **What validation guards catch** `pixel_values` row count != THW-derived patch count?
  - **Expected**: `assert_patches_match_thw` in validators.py raises explicit error with expected vs. actual counts; training halts immediately.
  
- **Are there environment constraints** (conda activation, workspace paths)?
  - **Expected**: Constitution mandates `conda activate ms` and paths rooted at `/data3/Qwen2.5-VL-main`; violations logged as warnings; absolute paths in config avoid ambiguity.

- **What happens when** `generation_logps` are not stored (buffer implementation bug)?
  - **Expected**: `completion_loss.py` falls back to current policy log-probs (line 243), logs warning "degenerates to ratio~1.0", training continues but ineffective.

- **How does system handle** distributed training with uneven dataset sizes (not divisible by world_size)?
  - **Expected**: Drop-last behavior at epoch boundary; incomplete prompt batches discarded with logged warning; counters reset.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST instrument GRPO pipeline with checkpoints that log generation_logps storage, retrieval, and ratio computation at each training step
- **FR-002**: System MUST validate multimodal alignment (image tokens, pixel_values, THW) at three pipeline stages: dataset emission, buffer generation, loss computation
- **FR-003**: System MUST compute and log per-reward-function statistics (mean, std, min, max, NaN count) before and after standardization for every logging interval, with separate tracking of **within-group variance** (across K completions per prompt) and **between-group variance** (across prompts)
- **FR-004**: System MUST capture gradient norms per LLM layer at each micro-step and log aggregate statistics (mean, max, zero-gradient count)
- **FR-005**: System MUST verify sequential processing compliance by asserting batch_size=1 for all `model.generate()` and `model.forward()` calls in GRPO loop
- **FR-006**: System MUST export diagnostic artifacts (trust region ratio histograms, reward correlation matrices, gradient flow heatmaps) to output_dir after 50 steps
- **FR-007**: System MUST compare GRPO implementation against official reference (Qwen2-VL-Finetune, ms-swift) for divergences in: generation kwargs, loss computation, advantage normalization
- **FR-008**: System MUST provide a debug config (`configs/dense_rl/diagnostic.yaml`) that runs minimal training (10 samples, 20 steps) with maximal logging enabled, and a checkpoint comparison config (`configs/dense_rl/checkpoint_diversity_test.yaml`) that tests Phase 2 vs Phase 3 checkpoints with temperature sweeps
- **FR-009**: System MUST validate that reward weights in config sum to > 0 and that at least one detection reward is non-zero (avoid pure formatting collapse); flag rewards with std < 0.01 as collapsed. *Note: See research.md RQ4 for detection vs. formatting reward categorization.*
- **FR-010**: System MUST log EOS termination ratio, completion length distribution, and cap-hit ratio to detect pathological generation (runaway tails, premature truncation)

### Key Entities

- **TrustRegionDiagnostic**: Stores per-step ratio statistics (mean, std, percentiles), generation_logps existence flags, fallback usage counts
- **MultimodalAlignmentCheck**: Records image token counts, THW shapes, pixel_values row counts at each pipeline stage; raises errors on mismatch
- **RewardProfile**: Per-function reward statistics (raw and standardized), correlation matrix across functions, temporal trends (moving average), **checkpoint-specific diversity metrics** (within-group vs between-group variance)
- **GradientFlowSnapshot**: Layer-wise gradient norms, zero-gradient flags, exploding gradient counts, optimizer step validity
- **SequentialProcessingMonitor**: Batch size assertions, GPU memory peaks, OOM event counts, accumulation boundary correctness
- **ComparisonBaseline**: Reference implementation metrics (official GRPO loss formula, advantage computation, reward aggregation) for validation
- **CheckpointDiversityComparison**: Stores diversity measurements across SFT checkpoints (Phase 2 vs Phase 3), temperature sweep results, advantage std correlations, recommended checkpoint selection

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Trust region diagnostic shows ratio mean ∈ [0.8, 1.5] with std > 0.15 for 90% of training steps (confirms non-degenerate ratios)
- **SC-002**: Multimodal alignment checks pass 100% of validations with zero warnings across 100 training steps (confirms vision-language consistency)
- **SC-003**: At least 3 reward functions show raw **within-group** std > 0.05 for ≥60% of prompts and correlation < 0.95 with each other (confirms diverse signals within K-completion groups, not redundant or collapsed)
- **SC-003b**: Checkpoint diversity comparison identifies which SFT checkpoint (Phase 2 vs Phase 3) produces higher within-group variance, with diversity ratio logged and actionable recommendation provided
- **SC-004**: Gradient flow snapshot shows non-zero gradients in top-8 LLM layers with norm ∈ [0.01, 10.0] for 95% of steps (confirms learning signal)
- **SC-005**: Sequential processing monitor confirms batch_size=1 for 100% of generate/forward calls and peak GPU memory < 80GB per device (constitutional compliance)
- **SC-006**: Diagnostic artifacts (ratio histograms, reward trends, gradient heatmaps) exported to `{output_dir}/diagnostics/` within 5 minutes of training start
- **SC-007**: Comparison with official reference shows <5% divergence in loss computation and advantage normalization formulas (algorithmic parity)
- **SC-008**: Debug config runs to completion in <10 minutes on 8 GPUs with zero errors and produces actionable report identifying top 3 suspected issues
- **SC-009**: Reward weights validation prevents configuration errors (catches all-zero formatting rewards or missing detection rewards) with clear error messages
- **SC-010**: EOS termination ratio > 80%, mean completion length ∈ [0.8×GT, 1.5×GT], cap-hit ratio < 30% for healthy generation patterns
