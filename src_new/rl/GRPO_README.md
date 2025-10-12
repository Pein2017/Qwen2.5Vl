## GRPO Post-Training (Dense Captioning) for Qwen2.5‑VL

### What this is
- **Goal**: Reinforcement post-training for dense captioning using GRPO, tightly aligned with the SFT processing stack (HF-first) and the Qwen2.5‑VL DetectionModel validations.
- **Design**: TRL-free, manual trainer built on Accelerate; strict contracts for multimodal alignment, tokenization, and geometry formatting.
- **Scope**: Single-turn, image-conditioned dense-caption generation with reward shaping that balances format correctness and detection quality.

### RL refactor v2 (merged, concise)
- **Status**: COMPLETE (2025-10-08)
- **Key changes**: Strict YAML (no defaults); typed `RLConfig` (`src_new/config/rl_config_v2.py`); clean separation of `sampling`/`generation`/`grpo` (algorithm-only); runner/trainer consume typed config only; no `raw_config`/`.get(...)` in critical paths.
- **Files updated**: `configs/dense_rl/{dense_base.yaml,debug.yaml,standard.yaml,README.md}`, `src_new/config/rl_config_v2.py`, `src_new/rl/{runner.py,grpo_trainer.py}`.
- **Required YAML**: 14 sections (~80+ required fields) including `paths`, `experiment`, `model`, `sampling`, `generation`, `grpo`, `normalization`, `training`, `optimizer`, `layer_config`, `logging`, `checkpointing`, `rewards` + `rewards_config`, `evaluation`. Optional only: `paths.ref_model_path`, `grpo.max_advantage_magnitude`, `grpo.beta_anneal` (when `beta_start=0`).
- **Trainer highlights**: New `__init__(..., rl_config: RLConfig, output_dir: str)`; rewritten `_build_manual_cfg(rl_config)`; evaluation via `config.evaluation.*`; dynamic-length and buffer calls use `config.generation.dynamic_length.*`.
- **Usage**:
  - Validate: `python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode load`
  - Train: `python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode train`
- **Benefits**: Fail-fast validation with clear errors; no hidden defaults; IDE-friendly typed access; reproducible configs; HF-first dataflow unchanged.

### Quality consolidation (2025-10-12)
- **Status**: COMPLETE
- **Key additions (no tensor changes)**:
  - Dataclasses for payloads in `src_new/rl/types.py` (`GenerationResult`, `CompletionSlice`, `ClipDiagnostics`, `PromptBatchMetrics`, `BufferTelemetry`, `TrainingLogs`).
  - Central metric keys in `src_new/rl/metrics_keys.py` to avoid string drift.
  - Diagnostics toggles in `src_new/rl/diagnostics/settings.py` (e.g., `DEBUG_TIMING`, `GENERATION_WARN_THRESHOLD`, `RL_CLEAR_CACHE`).
  - Memory policy wrapper `src_new/rl/memory_policy.py` (`maybe_clear_gpu_cache`) with defaults preserving current behavior.
  - Unified generation args helper `generation.build_generation_args()` used in train/eval paths to align optional kwargs.
  - Trainer uses `MetricsAggregator` for cross-rank stats and clarifies buffer reuse logging (`buffer/reuse_active`).


## Modular Architecture

The trainer has been refactored from a monolithic 2052-line file into specialized components (~1200 lines core + 5 modules), improving maintainability and testability.

### Core Modules

1. **`metrics_aggregator.py`** (170 lines) - Cross-rank tensor gathering
   - `gather_rewards_stats()`, `gather_advantage_stats()`, `gather_per_reward_components()`
   - `gather_termination_ratio()`, `gather_flag()`, `synchronize_index()`
   - Eliminates ~150 lines of repetitive gather/reduce patterns

2. **`tensorboard_logger.py`** (160 lines) - Structured logging
   - `log_training_scalars()`, `log_completion_metrics()`, `log_dynamic_length_metrics()`
   - `log_advantage_metrics()`, `log_per_reward_components()`, `log_eval_metrics()`
   - Eliminates ~100 lines of repetitive `add_scalar` calls

3. **`completion_loss.py`** (290 lines) - Per-completion GRPO loss
   - `compute_streaming_loss()` - handles vision extraction, logprob computation, generation logprob alignment, KL, clipping diagnostics, backward pass, cleanup
   - Eliminates ~180 lines of complex nested loops

4. **`console_formatter.py`** (90 lines) - Console output formatting
   - `format_training_summary()`, `format_prompt_batch_summary()`
   - Eliminates ~40 lines of manual string building

5. **`advantage_normalizer.py`** (190 lines) - Cross-rank advantage normalization
   - `normalize_advantages_cross_rank()` - handles padding, global stats, magnitude clipping
   - Eliminates ~130 lines of complex multi-rank logic

### Shared types & utilities (consolidation)
- **`types.py`** - Runtime dataclasses (`GenerationResult`, `CompletionSlice`, etc.).
- **`metrics_keys.py`** - Centralized scalar/logging key names.
- **`diagnostics/settings.py`** - Env-driven debug toggles (read once).
- **`memory_policy.py`** - Optional cache clearing policy wrapper.

### Integration

```python
# Initialization in BBUGRPOTrainer.__init__
self._metrics_aggregator = MetricsAggregator(self.accelerator, self._device)
self._loss_computer = CompletionLossComputer(accelerator, device, epsilon_low, epsilon_high, loss_type)
self._advantage_normalizer = AdvantageNormalizer(accelerator, device, scale_rewards, max_advantage_magnitude)
self._tb_logger = TensorBoardLogger(...)  # initialized in train() after writer creation

# Usage examples
reward_stats = self._metrics_aggregator.gather_rewards_stats(generation_result)
loss_result = self._loss_computer.compute_streaming_loss(model, ref_model, generation_result, current_beta)
self._advantage_normalizer.normalize_advantages_cross_rank(chunks, debug_timing, global_step)
self._tb_logger.log_training_scalars(step, reward_mean, reward_std, learning_rate, grad_norm, epoch, temperature, beta, eta_minutes)
```

**Total reduction**: ~850 lines eliminated through deduplication and separation of concerns.


## End-to-End Flow (Overview)
1. **Bootstrap (HF-first)**
   - Load tokenizer/processor/model via the shared HF loader.
   - Validate special tokens and geometry wrappers at startup.
2. **Dataset & Conversation**
   - RL dataset reads strict JSONL and images.
   - Build prompts with `ConversationBuilder` (same chat template as SFT), producing `input_ids`, `attention_mask`, optional `pixel_values` and `image_grid_thw`.
3. **Per-sample generation**
   - For each sample, generate `K` iid completions (`sample_k`) using `model.generate` through `generation.sample_k`.
   - Optional dynamic per-sample cap (GT-aware) chooses `max_new_tokens` from ground-truth objects; EOS is `<|im_end|>`.
4. **Token masks & sequence extraction**
   - Slice completions from the full sequence (prompt + completion) and build a mask that stops after the first `<|im_end|>`; optionally allow overflow tokens to carry gradients.
5. **Trust-region data (critical)**
   - Recompute per-token log-probs under the policy that actually generated each completion and store them in the buffer. These are used for the GRPO ratio denominator to avoid degenerate ratio≈1.
6. **Reward computation**
   - Compute reward components via the registry, optionally standardize, clip, and sum with YAML-provided weights. Observe-only rewards are always logged if requested.
7. **Advantages**
   - Within each sample's `K` completions: center rewards by the group mean; optionally scale by the group std and clamp the magnitude. Optionally compute cross-rank normalization for distributed runs.
8. **Training step (streamed)**
   - For each completion, compute current per-token log-probs, build the ratio to stored generation log-probs, apply clipped GRPO loss, optional KL to a frozen reference model, and backprop. Average across `K` within the optimizer step boundary.
9. **Logging, checkpointing, evaluation**
   - Unified reward logging to console/TensorBoard, best-checkpoint rotation, and a light evaluation harness that generates exactly one sample per rank and aggregates scalar metrics.


## Forward Processing (Detailed)

### 1) Conversation building and multimodal inputs
- Prompts are built with the same HF chat template as SFT; typed user turn includes image placeholders.
- Processor maps images to packed `pixel_values` with per-image THW metadata (`image_grid_thw`).
- Strict invariants:
  - Decoded prompt's `<|image_pad|>` count must match the number of images.
  - Packed `pixel_values` rows equal `∑(t*h*w)` derived from `image_grid_thw`.
  - Model forward/generate is forbidden with image tensors but zero image tokens.

### 2) Generation of K completions per sample
- Helper `generation.sample_k` wraps the HF `generate` path and returns `K` iid sequences on CPU to keep GPU peaks constant.
- EOS enforcement: `<|im_end|>` is resolved and passed to generation when available.
- Optional dynamic GT-aware cap (see below) adjusts `max_new_tokens` per sample before generation.
- Memory safeguards: K generations are right-padded and stacked; GPU cache is cleared between generations.

### 3) Building completion masks and text
- For each completion:
  - Slice completion tokens after the prompt length.
  - Build a binary mask; stop after the first `<|im_end|>` (tokens beyond can be zeroed or kept, depending on masking policy).
  - Decode completion text (without dropping special tokens) for reward functions that parse wrappers/geometry.

### 4) Generation-policy log-probs (trust region)
- For each completion, recompute per-token log-probs under the exact policy that generated it and store them in the buffer (`generation_logps`).
- During training, current-policy log-probs are compared against these stored log-probs to form the GRPO ratio π_current / π_generation. This avoids the degenerate ratio=1 issue when using current-policy log-probs in both numerator and denominator.

### 5) Rewards and aggregation
- Reward functions run over raw completion text (and optionally `meta` with GT structure) and return floats. The registry provides formatting rewards (wrappers, separators, vocab, pairing, duplication) and detection/geometry rewards (e.g., IoU, ordering, coverage).
- Results are optionally standardized online (per component) and then combined by a weighted sum to form a scalar reward per completion.
- For each sample, compute group mean/std across its `K` completions, then form advantages:
  - `A_i = (r_i − mean_K) / std_K` (if scaling enabled), optionally clamped by magnitude.
- Distributed option: Cross-rank normalization pads/trims variable-length tensors and aggregates statistics across processes to stabilize multi-rank training.

### 6) GRPO loss with clipping and optional KL
- For each completion (streamed micro-batch):
  - Compute current per-token log-probs over completion tokens only.
  - Form the ratio against stored generation log-probs; apply asymmetric clipping with `epsilon_low/high`.
  - Compute the GRPO objective using the masked completion region.
  - If KL regularization is enabled, forward the same tokens through a frozen reference model and add the TRL-style KL proxy on masked tokens.
- Losses are averaged across the `K` completions within the accumulation boundary and backpropagated.


## Dynamic Generation Length (GT-aware)

### Motivation
Dense scenes vary widely; a single global `max_new_tokens` either truncates rich scenes or lets small scenes drift into chaotic tails.

### Approach
For each sample, estimate a GT-aligned target length from its objects and derive a per-sample cap. The same cap is used for all `K` completions of that sample.

### Implementation
- **Estimator**: Tokenizer-based (recommended) or proxy (numbers + wrappers)
  - Build canonical assistant text from `meta.objects` using `CoordinateTokenConverter.convert_objects_to_tokens(objects)`
  - `gt_len = len(tokenizer(gt_text, add_special_tokens=False).input_ids)`
- **Cap formula**: `cap = clamp(round(alpha * gt_len + eos_margin), min_cap, max_cap)`
- **Masking**: Truncated completions contribute gradients (keep full masks or mask only beyond cap)

### Configuration (YAML)
```yaml
generation:
  dynamic_length:
    enabled: true
    estimator: tokenizer           # or: numbers_wrappers
    alpha: 1.1
    eos_margin: 16
    min_cap: 64
    max_cap: 1200
    hard_cap: true                 # enforce via per-sample max_new_tokens
```

### Optional Reward: `length_vs_gt`
Continuous reward scoring completion length against GT:
- Inside tolerance window `[lower, upper]`: 1.0
- Below lower: linear ramp `r/lower`
- Above upper: exponential decay `exp(-gamma * (r - upper))`
- Optional tail penalty: penalize numeric fraction in overflow region

```yaml
rewards:
  length_vs_gt: 0.0  # observe-only or assign weight

rewards_config:
  length_vs_gt:
    estimator: tokenizer           # match GRPO estimator
    lower: 0.7
    upper: 1.2
    gamma: 3.0
    tail_numeric_weight: 0.4
```

### Metrics
- TensorBoard: `dynamic_length/{mean_cap,min_cap,max_cap}`, completion length stats, EOS termination ratio, cap-hit fraction
- Console: cap statistics per step


## Prompt Batching

### Purpose
Accumulate rewards and gradients across multiple prompts before each optimizer update, reducing reward volatility and improving gradient estimates.

### Architecture
1. **Sample prompts**: Draw `sampling.prompt_batch_size` prompts (sequential broadcast)
2. **Per-prompt generation**: Each prompt generates `grpo.sample_k` trajectories (GLOBAL; split evenly across ranks)
3. **Accumulate**: Gradients accumulate via `loss.backward()` without `optimizer.step()`
4. **Optimizer step**: After collecting all prompts × trajectories

### Configuration
```yaml
sampling:
  prompt_batch_size: 4           # Prompts per optimizer update
  reward_average_window: 5       # Smoothing window for reward metrics
  sample_k: 8                    # GLOBAL trajectories per prompt (across all ranks)
```

### Modes
- **Global-K (default)**: `sample_k` divided evenly across ranks → total `sample_k` trajectories per prompt
  - Requirement: `sample_k % world_size == 0`

### Drop-Last Behavior
If dataset has fewer prompts than `prompt_batch_size`:
- Partial batch is dropped (no optimizer step)
- Gradients cleared, counters reset
- Warning logged with discard count

### Metrics
- TensorBoard: `rl/prompt_batch/{fill_ratio,reward_average,trajectories_collected,dropped_prompts,invalid_fraction}`
- Console: `fill_ratio`, trajectories per cycle, dropped prompts


## Distributed Execution & Sampling Window
- Accelerate manages process groups and device placement; the trainer infers world size/rank and configures a **sampling window** so that each optimizer step sees a consistent set of completions.
- Sampling mode:
  - Global-K only: `sample_k` is GLOBAL and split evenly across ranks.
- Shared-buffer resampling safeguards:
  - All ranks regenerate at the same time to keep collectives aligned.
  - A slow-generation guard resamples if any rank exceeds a generation-time threshold or yields zero completions.
- Cross-rank normalization:
  - Rewards are padded/trimmed to a common length before collective stats; advantages are rebuilt from global mean/std to avoid per-rank drift.
  - Synchronized via `AdvantageNormalizer.normalize_advantages_cross_rank()`


## Vision Packing & Validation
- Packed `pixel_values` and `image_grid_thw` are carried through generation and training.
- Slicing for per-sample forwards is done by computing cumulative image and patch counts.
- Two layers of checks:
  - Lightweight debug validator during training logs unusual tokens-per-image ratios or image-token mismatches in decoded prompts.
  - Strict assertion validates that the total packed rows equal the THW-derived sum and that chunked slices align across micro-steps.


## Logging, Checkpointing, and Evaluation
- **Logging (updated)**
  - Logging occurs exactly once per optimizer step (i.e., once per prompt-batch accumulation), avoiding duplicate logs within a cycle.
  - Console prints a concise one-liner with overall reward, raw reward, LR, grad, and ETA in hours, plus a second line with per-reward RAW means only.
    - Example:
      - `[step=189 reward=7.4314±1.0626 raw=6.1288±0.9441 lr=1.631e-07 grad=0.021 eta=0.12h]`
      - `[REWARDS_RAW] bbox_giou=0.674 grounding_acc=0.588 line_giou=0.644 ...`
  - TensorBoard logging per optimizer step includes:
    - Overall (normalized): `reward`, `reward_std`
    - Overall (raw): `raw_reward`, `raw_reward_std`
    - Per-reward components (normalized): `rewards/{name}/{mean,std}` and aliases `reward/mean/{name}`, `reward/std/{name}`
    - Per-reward components (raw): `raw_rewards/{name}/{mean,std}` and aliases `raw_reward/mean/{name}`, `raw_reward/std/{name}`
    - Overall aliases for easy dashboards: `reward/mean/average_reward`, `reward/std/average_reward`, `raw_reward/mean/average_reward`, `raw_reward/std/average_reward`
    - Core scalars: `train/learning_rate`, `train/grad_norm`, `train/epoch`, `train/eta` (hours)
  - Notes:
    - Normalized reward ("reward") is useful when reward standardization is enabled; otherwise prioritize raw metrics for comparability.
    - ETA is logged as hours at `train/eta`.
  - Buffer reuse metrics (clarified):
    - `buffer/steps_per_generation`, `buffer/reuse_count`, `buffer/generation_efficiency` (S as float), and `buffer/reuse_active` (0 for current behavior).
  - Keys are centralized in `src_new/rl/metrics_keys.py`.
- **Checkpointing**
  - Periodic step-based saves with a best-checkpoint manager keyed to reward; model/processor/tokenizer and minimal generation config are persisted.
- **Evaluation harness**
  - Lightweight single-image generation per rank with the same HF-first pipeline.
  - Computes reward metrics and additional absolute metrics (e.g., strict parse rate, duplication, vocab bans, numeric tail fraction, and approximate box Giou) and aggregates on rank 0.
  - Can dump example predictions next to the report for quick inspection.


## Failure Modes & Safeguards
- **Ratio degeneracy**: If generation log-probs are not stored, the GRPO ratio collapses to ~1. Use the buffer's `generation_logps` for the denominator.
- **Multimodal drift**: If `<|image_pad|>` counts do not match image grids or packed patch totals, fail fast and re-check processing/template.
- **Runaway tails**: Enable dynamic per-sample caps; inspect numeric-tail metrics.
- **Distributed hangs**: Keep per-sample completions consistent across ranks, resample on slow-rank detection, and pad variable-length tensors before collectives.
- **Non-finite loss**: The trainer guards by reducing temperature scale, clearing the buffer, and resampling.
- **NCCL timeouts**: Reduce `prompt_batch_size` to lower memory pressure; increase timeout (`NCCL_TIMEOUT=300`); enable async error handling.
- **Low fill ratio**: Check for generation failures (OOM), reward computation errors (NaN/Inf), or dataset access issues.


## Minimal Pseudocode (for orientation)
```python
# Generation + scoring (per sample)
ids, mask, pv, thw, meta = sample["input_ids"], sample["attention_mask"], sample.get("pixel_values"), sample.get("image_grid_thw"), sample.get("meta")
cap = dynamic_cap(meta) if dyn_enabled else global_max_new_tokens
seqs = sample_k(model, tokenizer, {"input_ids": ids, "attention_mask": mask, "pixel_values": pv, "image_grid_thw": thw}, k=K, max_new_tokens=cap)
completions, masks = slice_and_mask(seqs, prompt_len=len(ids), eos_id=IM_END_ID)
old_logps = per_token_logps(model, concat(ids, completions), logits_to_keep=len(completions), pixel_values=pv, image_grid_thw=thw)
rewards = weighted_sum([fn(text, meta) for text in decode(completions)])
A = normalize_within_group(rewards, group_size=K, cross_rank=distributed)
# buffer holds: prompt/completion ids+masks, advantages A, rewards, old_logps, packed vision
```

```python
# Training step (streamed over K, with prompt batching)
for prompt_idx in range(prompt_batch_size):
    generation_result = generate_and_score(next_sample())  # K completions
    for k in range(K):
        cur_logps = per_token_logps(model, concat(ids, comp[k]), logits_to_keep=len(comp[k]), pixel_values=pv, image_grid_thw=thw)
        ratio = exp(cur_logps - old_logps[k])
        loss_k = grpo_loss(ratio, A[k], mask=mask[k], epsilon_low, epsilon_high)
        if beta > 0:
            ref_logps = per_token_logps(ref_model, concat(ids, comp[k]), logits_to_keep=len(comp[k]), pixel_values=pv, image_grid_thw=thw, detach=True)
            loss_k += beta * kl_proxy(cur_logps.detach(), ref_logps)
        (loss_k / K).backward()  # accumulate gradients
optimizer.step()  # after all prompts × K
optimizer.zero_grad()
```


## Module Map (where each piece lives)
- `config/rl_config_v2.py`: Strict hierarchical RL config (v2). Typed dataclasses; no defaults; clear validation errors.
- `rl/runner.py`: Unified loader; builds HF components, datasets, reward functions, and instantiates the manual trainer. Applies phase-freeze policy to the model.
- `rl/grpo_trainer.py`: The manual GRPO trainer (Accelerate-based, ~1200 lines core). Orchestrates sampling windows, buffer regeneration, cross-rank normalization, streamed loss computation, logging, checkpointing, and evaluation. Uses modular components for metrics, logging, loss computation, and formatting.
- `rl/metrics_aggregator.py`: Cross-rank tensor gathering and metric computation (gather rewards, advantages, termination ratios, flags).
- `rl/tensorboard_logger.py`: Structured TensorBoard logging for training scalars, completion metrics, dynamic length, clip ratios, advantages, per-reward components, and evaluation.
- `rl/completion_loss.py`: Per-completion GRPO loss computation (vision extraction, logprob computation/alignment, clipping, KL, backward, cleanup).
- `rl/console_formatter.py`: Console output formatting for training summaries and prompt batch telemetry.
- `rl/advantage_normalizer.py`: Cross-rank advantage normalization with padding/alignment, global stats, and magnitude clipping.
- `rl/buffer.py`: Core "generate and score" logic. Generates `K` completions per sample, builds masks/text, computes and stores generation-policy log-probs, computes rewards, advantages, and prepares packed vision for training.
- `rl/generation.py`: Thin wrappers around HF `generate` with strict input sanitization and EOS handling; K-sampling with CPU offload and right-padding.
- `rl/logprobs.py`: Per-token log-prob computation with packed vision slicing and thorough GPU-memory cleanup; also provides slicing utilities for packed tensors.
- `rl/losses.py`: GRPO clipped objective and TRL-style KL proxy.
- `rl/validators.py`: Debug and strict validators for image-token alignment and packed vision integrity.
- `rl/schedules.py`: Temperature and beta schedules; simple curriculum helper.
- `rl/eval.py` & `rl/eval_utils.py`: Lightweight evaluation harness and utilities for saving sample predictions and converting objects to simple boxes.
- `rl/types.py`: Dataclasses for typed payloads across modules.
- `rl/metrics_keys.py`: Central constants for logging keys.
- `rl/diagnostics/settings.py`: Env-driven diagnostics toggles.
- `rl/memory_policy.py`: GPU cache clearing policy wrapper.


## Contracts & Invariants (checklist)
- **HF-first**: Always apply the chat template and processor; never craft raw `<|image_pad|>` tokens.
- **Token/image alignment**: `<|image_pad|>` count ⇔ `image_grid_thw`; packed patch rows equal THW sum.
- **EOS discipline**: Training uses the first `<|im_end|>` boundary; generation decodes only new tokens and trims at `<|im_end|>`.
- **Trust region**: Always compute and use generation-policy log-probs for the denominator.
- **Advantages**: Center within each K-group; optionally scale and clamp; cross-rank normalization pads/trims before collectives.
- **Prompt batching**: Accumulate gradients over `prompt_batch_size` prompts; drop incomplete batches at epoch boundaries.
- **Dynamic caps**: Per-sample `max_new_tokens` derived from GT length estimation; same cap for all K completions of a sample.
- **Safety**: Fail fast on shape mismatches; guard against slow-rank generation; resample on non-finite losses.


## How to extend
- **New rewards**: Register in the rewards registry; the loader discovers and logs them automatically.
- **Alternate caps**: Swap the dynamic-length estimator or tail penalties while preserving per-sample cap semantics.
- **Schedules**: Adjust temperature or KL beta schedules via the schedule helpers.
- **Evaluation**: Add absolute metrics or richer sample exports without changing training internals.
- **Modular components**: Extend or swap `MetricsAggregator`, `TensorBoardLogger`, `CompletionLossComputer`, `AdvantageNormalizer`, or `ConsoleFormatter` for different RL algorithms (PPO, DPO) or custom logging/metrics.

## Buffer Reuse via steps_per_generation (experimental)

The infrastructure for Swift-style buffer reuse is present (`GenerationBuffer` on CPU, streamed optimization), but full multi-step reuse is experimental and not the default. See `src_new/rl/BUFFER_REUSE_STATUS.md` for the latest status.

### Current behavior
- Default `grpo.steps_per_generation: 1` (generate each cycle, then optimize once).
- A single `GenerationBuffer` covers one prompt-batch cycle; completions are streamed per micro-step; optimizer steps once at the accumulation boundary.

### Guidance
- If you enable `steps_per_generation > 1`, expect experimental behavior; verify trust-region ratios and watch for overfitting. Prefer keeping `steps_per_generation: 1` unless you actively validate reuse.

### Submodule index (complete, for quick discovery)

- Core training & generation
  - `runner.py`: CLI entry; loads typed config, builds datasets/rewards/model, applies phase-freeze.
  - `grpo_trainer.py`: Manual GRPO trainer (Accelerate); sampling window, buffer, loss, logging, eval.
  - `buffer.py`: `generate_and_score`, dynamic caps, K sampling, reward compute, advantage build, `split_buffer`.
  - `generation.py`: Safe `generate` wrappers, `sample_k`, `build_generation_args` (shared by train/eval).
  - `logprobs.py`: Per-token log-probs over completion slices; packed vision slicing utilities; GPU cleanup.
  - `losses.py`: GRPO clipped objective and TRL-style KL proxy.
  - `validators.py`: Debug and strict validators for image-token alignment and packed vision integrity.
  - `schedules.py`: Temperature/beta schedules and simple curriculum.
  - `training_state.py`: Persistent state (global step, sampler pos, buffer reuse counters).
  - `tensor_utils.py`: Small tensor helpers (e.g., THW normalization).

- Data & prompting
  - `data/dataset.py`: RL dataset reconstruction via `ConversationBuilder` (single-turn, packed vision tensors, `meta`).
  - `prompting/conversation.py`: Conversation utilities for dense RL (training-matched prompts).

- Rewards (registry-backed)
  - `rewards/registry.py`: Name→function registry discovery and wiring.
  - `rewards/format_rewards.py`: Formatting/structure rewards (parse, wrappers, coords, separators, vocab, length/length_window).
  - `rewards/detection_rewards.py`: Geometry/detection rewards (coverage, ordering, bbox_giou, quad_l1, line_l1, geometry_sanity, caption_f1, grounding_acc).
  - `rewards/standardizer.py`: Online reward standardization helpers.
  - `rewards/sanitizer.py`: Reward sanitization and bounds.
  - `metrics_keys.py`: Centralized metric key names used across logging.

- Logging, diagnostics, and utilities
  - `tensorboard_logger.py`: Structured TB logging (training, completion, dynamic-length, clip ratios, per-reward, eval).
  - `console_formatter.py`: Concise console one-liners and detailed summaries.
  - `console_metrics_collector.py`: Consolidates scalar/histogram snippets for console/TB.
  - `metrics_aggregator.py`: Cross-rank gather/reduce for rewards/advantages/termination and per-reward components.
  - `prompt_batch_telemetry.py`: Prompt-batch accumulation telemetry and smoothing windows.
  - `reward_logger.py`: Per-reward means/std formatters for console/TB.
  - `diagnostics/settings.py`: Env-driven toggles (timing thresholds, cache policy, warnings).
  - `memory_policy.py`: Optional GPU cache clearing policy wrapper.
  - `text_dump.py`: Generation text sample dumping for inspection.
  - `distributed.py`: Thin distributed helpers used by the trainer.
  - `utils.py`: Misc small helpers (IM_END resolution, conversation builder, device utils, trimming).

- Buffer reuse (experimental)
  - `generation_buffer.py`: CPU-side buffer for multi-step reuse (`steps_per_generation`).
  - `BUFFER_REUSE_STATUS.md`: Current guidance and caveats for reuse.

- Evaluation
  - `eval.py`: Lightweight eval harness; generates one sample per rank and aggregates metrics.
  - `eval_utils.py`: Helpers for dumping samples and converting objects to simple boxes.

