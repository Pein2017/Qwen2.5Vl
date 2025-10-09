# Data Model: GRPO Prompt Batch Size Refactoring

**Date**: 2025-10-08  
**Source Spec**: [spec.md](./spec.md)  
**Scope**: Minimal data abstractions needed to configure prompt batching and track reward trends.

---

## Entities

### PromptBatchSettings
- **Purpose**: Captures operator-facing parameters that define how many prompts and trajectories compose a single accumulation cycle.
- **Fields**:
- `sampling.prompt_batch_size` (int > 0): Number of prompts accumulated per accumulation cycle.
- `grpo.sample_k` (int > 0): Number of trajectories generated per prompt (global; split across ranks when `sample_k_per_rank=false`).
- **Validation Rules**:
  - Both fields MUST be positive integers.
  - `prompt_batch_size × sample_k` MUST be attainable under current memory budget; validation occurs during training launch.
  - When `sample_k_per_rank=false`, `grpo.sample_k` MUST be divisible by the distributed world size; otherwise documentation MUST describe per-rank sampling.

### RewardTrendSnapshot
- **Purpose**: Represents the telemetry signal emitted after each accumulation cycle so operators can confirm steadily improving rewards.
- **Fields**:
  - `cycle_id` (int): Monotonic counter of completed accumulation cycles.
  - `trajectories_collected` (int): Total valid trajectories gathered in the cycle.
  - `smoothed_reward` (float): Moving-average reward value computed from the last `reward_average_window` cycles.
  - `fill_ratio` (float): Collected trajectories divided by expected `prompt_batch_size × sample_k`.
  - `invalid_fraction` (float): Invalid trajectories divided by total trajectories in the cycle.
  - `dropped_prompts` (int): Number of prompts skipped due to `drop_last` in the cycle.
  - `elapsed_seconds` (float): Wall-clock seconds spent on the accumulation cycle.
  - `reward_average_window` (int): Number of cycles used to compute `smoothed_reward` (default 5).
  - `reward_global_mean` (float): Cross-rank, per-prompt global reward mean.
  - `reward_global_std` (float): Cross-rank, per-prompt global reward std (unbiased=False).
  - `adv_global_std` (float): Cross-rank, per-prompt global advantage std (post scaling if enabled).
  - `adv_global_max_abs` (float): Cross-rank, per-prompt max absolute advantage.
- **Validation Rules**:
  - `trajectories_collected` MUST equal `sampling.prompt_batch_size × grpo.sample_k` (global) when `fill_ratio` is 1.0; otherwise the runner MUST skip `optimizer.step()` for that cycle (drop_last) and continue.
  - `smoothed_reward` MUST be finite; NaN/inf values trigger warnings and skip trend updates while zeroing missing rewards.

---

## Relationships
- Each training run has a single `PromptBatchSettings` instance.
- A `RewardTrendSnapshot` is generated after every optimizer step and references the same `PromptBatchSettings` for context.

---

## Persistence Notes
- `PromptBatchSettings` is sourced from YAML configuration; no additional persistence required.
- `RewardTrendSnapshot` values are streamed through existing console/TensorBoard logging pipelines; when available, global metrics are preferred over per-chunk local metrics.

---

**Open Issues**: None.
