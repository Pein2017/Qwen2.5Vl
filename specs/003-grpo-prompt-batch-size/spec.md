# Feature Specification: GRPO Prompt Batch Size Refactoring

**Feature Branch**: `003-grpo-prompt-batch-size`  
**Created**: 2025-10-08  
**Status**: Draft  
**Constitution Version**: 4.0.0  
**Input**: User description: "I now want to refactor my current post- training framework of GRPO based on Qwen2.5VL-7B model on my dense captioning task. Before, I have fine-tuned the model by `sft` in `src_new/`. And have implemented the `src_new/rl/` module but suffering the extremly fluctuating rewards, which indicates poor learning. I finally realized that I should increase the amount of samples before `optimizer.step()` to update the models parameters.

The current workflow is:
```
1. Use 8 GPUs available in this node, launch a DDP-type training with a distributed data sampler.
2. Iterate on the train dataset, spread one prompt/sample (image) to 8 ranks.
3. Ask each rank to generate one response respectively, since my limited GPU memory can only afford one generation at a time, and one `loss.backward`. Therefore, every heavy computation must be Iterative instead of `batch operation`.
4. Then, each response from each rank gets a reward value.
5. Sync this reward across 8 ranks to conduct and calculate the GRPO advantage of each response.
6. Each rank gets the `loss` after sync and then call `loss.backward()` to accumulate the gradient.
7. After one full generation on this sample/prompt, we can call the `optimizer.step()` to update the models parameter with respect to 8 responses on 1 sample prompt.
```
However, I want to further extend the above procedure and introduce a varaible like `prompt_batch_size`, similar to `effective batch size`. I want to gather the information across multiple samples instead of just one sample, which can provide more information and robust estimation. That means, I may want: sample_k=8 and prompt_batch_size=16. And we need to manage to get 8*16 `trajectories` for averaging the gradients before calling the `optimizer.step()`. However, we need to manage to find an efficient and elegant way to `rollout` the samplings, gather GRPO advantages, and accumulate the `loss.backward` on 8 GPUs, subjective to limited GPU constrain of one step inference at a time. Please help me convert to my objective into the spec document with proper organization and polishment. Help me name this `spec` as `GRPO-prompt-batch-size-refactoring`."

## Clarifications

### Session 2025-10-08
- Q: How should prompts be distributed across ranks during each accumulation cycle? → A: Broadcast the same prompt to all ranks so each rank generates one trajectory; move to the next prompt after collecting sample_k trajectories.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Stable GRPO Prompt Batches (Priority: P1)

An RL researcher configures a dense captioning GRPO run to accumulate rewards across multiple prompts before any optimizer step, reducing reward volatility while staying within GPU memory limits.

**Why this priority**: Without stable batches, the team cannot trust reward trends, causing misinformed experiments and wasted GPU time.

**Independent Test**: Launch a smoke GRPO run with `prompt_batch.prompt_batch_size > 1` and `grpo.sample_k = 8`; confirm the system completes one optimizer step only after accumulating `prompt_batch_size × sample_k` trajectories, discards any leftover prompts via `drop_last`, and records the batch statistics.

**Acceptance Scenarios**:

1. **Given** a run configured with `prompt_batch_size` = 16 and sample_k = 8, **When** the researcher starts training, **Then** the system executes sequential rollouts until 128 trajectories are gathered and only then performs `optimizer.step()` once.
2. **Given** a researcher enters a `prompt_batch_size` that exceeds available prompts for the epoch, **When** training starts, **Then** the system fails fast with an actionable message describing the shortfall and how to adjust the setting.

---

### User Story 2 - Monitor Trajectory Aggregation Health (Priority: P2)

A training operator monitors live telemetry to ensure prompt batches are progressing, rewards remain stable, and no GPU rank lags behind during sequential rollouts.

**Why this priority**: Operators must be able to halt or adjust runs before GPU hours are wasted on divergent gradients or stalled workers.

**Independent Test**: Stream metrics for a pilot run and verify the dashboard/logs report per-accumulation-cycle progress (fill ratio, smoothed reward average, lagging ranks) and that global, cross-rank advantage metrics are reported (and preferred) when multi-GPU is enabled.

**Acceptance Scenarios**:

1. **Given** a long-running GRPO job, **When** the operator inspects logs or dashboards, **Then** they see per-step metrics including prompt_batch_size achieved, total trajectories accumulated, global reward mean/std, and global advantage std/max_abs (preferred over local per-chunk metrics when available).
2. **Given** one GPU rank falls behind during sequential generation, **When** the lag exceeds the runtime’s latency threshold, **Then** the system surfaces a warning with rank ID and recommended mitigation (e.g., reduce `prompt_batch_size`).

---

### Edge Cases

- GPU memory drops mid-batch because cumulative activations exceed the sequential budget; the run must pause with instructions for lowering `prompt_batch_size` or adjusting `grpo.sample_k`.
- Dataset provides fewer prompts than required for an accumulation cycle; the runner MUST discard the incomplete set via `drop_last` and continue without applying gradients.
- Reward aggregation receives NaNs or extreme outliers; the system must surface diagnostics and treat missing sub-rewards as reward value `0`.
- Environment misconfiguration (missing `ms` conda activation or wrong workspace path) prevents sequential rollouts from loading checkpoints; the run must fail fast before consumption of GPU cycles.
- NCCL synchronization stalls; a watchdog MUST surface a warning (or abort when configured) if any rank remains idle past the communication backend timeout, preventing deadlocks unrelated to model generation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The training configuration MUST allow operators to set prompt_batch_size and sample_k, with validation ensuring both are positive integers and that prompt_batch_size × sample_k trajectories define each optimizer step.
- **FR-002**: The system MUST accumulate trajectories sequentially until the configured prompt_batch_size and sample_k product is reached, broadcasting each prompt to all ranks so each rank contributes one trajectory before advancing to the next prompt, then execute exactly one optimizer update per accumulation cycle.
- **FR-003**: If a prompt batch cannot be filled (e.g., insufficient prompts remaining in the epoch), the runner MUST apply a `drop_last` policy that discards the partial set, logs the discard, and skips the optimizer step.
- **FR-004**: Runtime telemetry MUST expose per-accumulation-cycle metrics including prompt_batch_size achieved, total trajectories accumulated, fill ratio, smoothed primary reward (`rl/prompt_batch/reward_average` computed over the last 5 cycles by default), lagging rank warnings, trajectories collected, dropped prompt counts, and elapsed seconds.
- **FR-005**: When multi-GPU is enabled and `normalization.cross_rank_advantages=true`, the system MUST compute and prefer global cross-rank metrics for reward mean/std and advantage std/max_abs in both console logs and TensorBoard. When unavailable (e.g., single-GPU), fall back to local metrics.

### Batching Workflow Summary

1. Sample `prompt_batch.prompt_batch_size` prompts using the deterministic sampler.
2. Broadcast the first prompt to all ranks; each rank generates one trajectory (trajectory = single response).
3. Repeat step 2 until `grpo.sample_k` trajectories are obtained per prompt, updating the reward trend after each prompt.
4. Continue until `prompt_batch_size × sample_k` trajectories are collected (one accumulation cycle).
5. If insufficient prompts remain, apply `drop_last` and record the discard count.
6. Perform `optimizer.step()`, zero gradients, and log fill ratio, trajectories collected, dropped prompts, and global metrics (preferred) with local fallback.
7. Configuration is YAML-only. Operators set `sampling.prompt_batch_size` and `grpo.sample_k` in YAML. No CLI overrides are supported.

### Key Entities *(include if feature involves data)*

- **Prompt Batch Settings**: Captures `sampling.prompt_batch_size` and `grpo.sample_k` for each run.
- **Reward Trend Snapshot**: Per-cycle record of total trajectories collected and the smoothed reward metric streamed to telemetry, augmented with global reward/advantage metrics when available.

## Assumptions & Dependencies

- All training continues within the sanctioned `ms` conda environment and the existing GRPO pipeline; no new hardware orchestration layer is introduced.
- Sequential generation per GPU rank remains the only feasible approach given memory limits; scheduling improvements focus on accumulation logic rather than parallel decoding.
- GPU validation is preferred for realism while CPU-only mocks support rapid smoke testing; both paths must be documented.
- `grpo.sample_k` applies to the entire process group; when `grpo.sample_k_per_rank=false` the value MUST be divisible by the world size, otherwise documentation MUST describe the split behavior.
- EOS remains `<|im_end|>` only, and bf16 (`bf16: true` at the YAML root) is required per repository policy.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training runs show a steady upward trend in the primary reward metric (moving-average baseline) without the sharp oscillations observed in the single-prompt baseline, demonstrating that prompt batching delivers progressively improving rewards. Final judgment of the trend is performed manually by the feature owner.

## Testing Guidance

- CPU-only distributed smoke tests MUST use `accelerate launch --num_processes=2` with stubbed policies to validate prompt broadcast and accumulation loops.
- GPU smoke tests SHOULD cover at least one accumulation cycle with real tokenizer/model weights when hardware is available and must be clearly marked to skip when GPUs are absent.
- Lagging-rank scenarios MUST be simulated (e.g., artificial sleeps) to ensure telemetry and NCCL watchdog warnings behave as specified.
- Smoke tests are expected to cover only a few accumulation cycles; there is no strict runtime limit, but NCCL watchdog alerts MUST prevent indefinite hangs.
- Recommended environment safeguards: `NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_BLOCKING_WAIT=1`, and an explicit timeout (e.g., `NCCL_TIMEOUT=180`) should be set unless overridden by operations policy.
