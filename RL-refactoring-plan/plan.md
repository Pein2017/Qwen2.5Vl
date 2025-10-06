### Qwen2.5‑VL RL Refactoring Plan (Manual GRPO, no TRL)

- **Goal**: Fully remove dependency on TRL while maximizing reuse of `src_new` SFT stack. Keep HF‑first processing (chat template), packed vision tensors, strict validations, and checkpointing/logging. Change only the algorithmic core: sampling K completions → reward/advantage → GRPO loss + backprop.
- **Scope**: Dense captioning RL in `src_new/rl` only. `src_post` is out of scope.

---

### Status (2025‑xx)
- ✅ `BBUGRPOTrainer` implemented (`src_new/rl/grpo_trainer.py`) with generation buffering, advantage normalization, optional KL, bf16 autocast, reward standardization, and checkpointing.
- ✅ TRL dependency removed: runner now instantiates the manual trainer by default; `src_new/rl/trainer.py` and TRL configs/scripts were deleted.
- ✅ Module split landed (`generation.py`, `logprobs.py`, `buffer.py`, `losses.py`, `schedules.py`, `validators.py`, `distributed.py`).
- ✅ Runner, config plumbing, and feature-check script updated for the manual path; history hooks exposed for temperature/reward diagnostics.
- ✅ Reward standardizer, per-reward metrics, and clip/advantage logging wired into the trainer and documentation.
- ✅ Cross-rank advantage normalization available via `distributed.compute_global_advantages`; rank-aware seeding added.
- ⏳ Full cross-rank sampling (shared prompt shards across ranks) remains future work; current implementation assumes per-rank sampling with optional global advantage normalization.

### 1) Design principles and invariants (must keep)
- HF‑first conversation construction via `ConversationBuilder` and `Qwen2_5_VLProcessor.apply_chat_template()`.
- Vision alignment contracts:
  - `<|image_pad|>` count equals expected tokens from `image_grid_thw` and `merge_size`.
  - `pixel_values` packed 2D rows equal `∑(t*h*w)` (strict shape validation via `tensor_validation`).
  - `image_grid_thw` is `[num_images, 3]` per sample.
- EOS is `<|im_end|>`; mask completions at first EOS (optional truncate mask).
- No coordinate tokens; geometry is raw integers with canonical wrappers only.
- Checkpointing and logging via `CheckpointSaver` and `TrainingStateManager` (no custom distributed ops).
- Freezing/unfreezing via `PhaseFreezeManager` as in SFT.

---

### 2) Target architecture (high level)
- Replace TRL usage with a small, self‑contained RL trainer that mirrors SFT trainer style:
  - New: `src_new/rl/grpo_trainer.py` → `BBUGRPOTrainer`
  - Update: `src_new/rl/runner.py` to call BBUGRPOTrainer (and remove TRL routing)
  - Reuse: `src_new/rl/data/dataset.py` + `src_new/rl/prompting/conversation.py`
  - Reuse: `src_new/rl/rewards/registry.py` (formatting + detection rewards)
  - Reuse: `src_new/models/wrapper.py` (`DetectionModel`) for forward/generate and validations
  - Reuse: `src_new/utils/hf_components.py` loader, `PhaseFreezeManager`, `CheckpointSaver`, `TrainingStateManager`

### 2A) Canonical module map & ownership (single source of truth)
- Placement
  - Keep RL under `src_new/rl/` to preserve SFT/RL parity. Split to `src_post/` only when objectives diverge (e.g., group-level QC).
- Modules (RL)
  - `src_new/rl/grpo_trainer.py`: `BBUGRPOTrainer` (orchestrates; no inline generation/logprob/loss logic)
  - `src_new/rl/generation.py`: `prepare_generate_inputs`, `generate_completions`, `sample_k`, `concatenate_queries_and_responses`
  - `src_new/rl/logprobs.py`: `get_per_token_logps`, `slice_packed_vision_for_sample`
  - `src_new/rl/buffer.py`: `generate_and_score`, `split_buffer`
  - `src_new/rl/losses.py`: `compute_grpo_loss`, `compute_kl`
  - `src_new/rl/schedules.py`: `temperature_at`, `beta_at`, `curriculum_stage_at`
  - `src_new/rl/validators.py`: `debug_validate_image_alignment`, `assert_patches_match_thw`, `check_completion_masks`
  - `src_new/rl/distributed.py`: `broadcast_indices`, `broadcast_gen_cfg`, `all_gather_rewards`, `compute_global_advantages`, `barrier`, `seed_for_rank`
  - `src_new/rl/rewards/{registry.py, standardizer.py}`: pure reward functions; standardization separate
- Shared (import-only; single-source)
  - `src_new/utils/hf_components.py`, `src_new/training/{phase_freeze_manager.py, checkpoint_saver.py, training_state_manager.py}`, `src_new/models/wrapper.py`, `src_new/processing/*`
- Decoupling rules
  - Generation/logprob modules contain no reward/loss; DeepSpeed logic only in trainer/runner.
  - No in-code defaults; all knobs from YAML; validation centralized.
  - Vision slicing centralized; EOS/min_new_tokens handled in generation; runner sets config fields.

---

### 3) Modules and changes
- Core trainer (`src_new/rl/grpo_trainer.py`)
  - Builds distributed-aware dataloaders (optional `DistributedSampler`), seeds per rank, and reuses SFT parameter groups/LR schedulers.
  - Uses helper modules for generation (`generation.py`), scoring (`buffer.py`), log-prob slicing (`logprobs.py`), losses (`losses.py`), schedules (`schedules.py`), validators (`validators.py`), and distributed utilities (`distributed.py`).
  - Supports temperature schedules with automatic down-scaling on NaNs, optional KL (`beta_start` + anneal), reward standardization, advantage clipping, and cross-rank advantage normalization.
  - Logs per-step histories (temperature/reward/clip/advs) and exposes them for feature checks.
  - Checkpointing via `CheckpointSaver` mirrors SFT behaviour; rank 0 only.
- Runner (`src_new/rl/runner.py`)
  - Loads components/datasets, applies phase freezing, instantiates `BBUGRPOTrainer`, and drops the TRL fallback.
- CLI checks (`scripts/rl_feature_checks.py`)
  - Updated to drive the manual trainer, emit summary diagnostics, and honour override flags without TRL callbacks.

- Runner wiring (done): `src_new/rl/runner.py` now builds components/datasets, applies phase freeze, instantiates `BBUGRPOTrainer`, and honours `resume_from_checkpoint`.
    - Save final checkpoint via `CheckpointSaver`

- Keep: `src_new/rl/eval.py` (optional small helper to use BBUGRPOTrainer for eval in future, but not needed now)

- Remove/Deprecate (RL only):
  - TRL imports/usage from `src_new/rl/trainer.py` and `scripts/rl_feature_checks.py` (replace checks to call BBUGRPOTrainer)

---

### 4) Training loop blueprint (manual GRPO)

```python
for global_step in range(max_steps):
    # 1) Get next generation batch (local) of size: per_device_train_batch_size * steps_per_generation
    gen_batch = next(loader)

    # 2) Periodically generate and buffer (K completions each prompt)
    if step % (steps_per_generation * num_iterations) == 0 or buffer is None:
        buffer = generate_and_score(gen_batch)  # returns dict of tensors & advantages
        # split into steps_per_generation equal chunks
        chunks = split_buffer(buffer, steps_per_generation)

    # 3) Select current chunk (size = per_device_train_batch_size)
    inputs = chunks[step % steps_per_generation]

    # 4) Compute per-token log-probs (with graph) on completions only
    per_token_logps = _get_per_token_logps(model, ..., detach=False)

    # 5) Compute GRPO loss (with optional KL)
    loss = grpo_loss(per_token_logps, old_per_token_logps, advantages, completion_mask, beta, eps_low, eps_high, loss_type)

    # 6) Backward + grad accumulation
    (loss / grad_accum_steps).backward()
    if (micro_step + 1) % grad_accum_steps == 0:
        clip_grad_norm(model.parameters(), max_grad_norm)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        log/save/eval via TrainingStateManager + CheckpointSaver
```

Notes:
- Generation uses `DetectionModel.generate(...)` with `eos_token_id = <|im_end|>`; per‑sample loop to minimize memory.
- Per‑token log‑probs computed strictly sequentially per sample, slicing packed vision tensors by patch offsets (`image_grid_thw` cumulative sums), identical to current RL helper.
- Advantages computed group‑wise over K completions (mean/std normalize), optional standardization/clipping.

---

### 5) Memory and performance controls
- Generation batch size = `per_device_train_batch_size * steps_per_generation` → buffer results → split into `steps_per_generation` micro‑batches for backward.
- Per‑sample sequential log‑prob pass minimizes VRAM usage (only 1 sample live).
- Enable gradient checkpointing, set `model.config.use_cache=False` for training.
- Recommend `attn_implementation: eager` for RL if instability observed.
- Honor `mask_truncated_completions`: zero out truncated sequences in `completion_mask` to avoid gradients from incomplete outputs.

---

### 6) Distributed and DeepSpeed
- Keep DS ZeRO‑2 via environment flags already used by runner (`BBU_DEEPSPEED_ENABLED`, `BBU_DEEPSPEED_CONFIG`).
- If DS enabled, initialize engine/model wrapping similarly to SFT initialization path; otherwise, use plain `torch.optim.AdamW` param groups with LR scheduling.
- No custom cross‑rank ops; rely on DS optimizer sharding only. Local logging via `TrainingStateManager`.

### 6A) Cross-rank K sampling for same prompt (Distributed)
*Status*: cross-rank advantage normalization is implemented via `distributed.compute_global_advantages`; shared prompt sampling/gather remains future work (tracked here for completeness).
- Objective: split K completions across GPUs for the same prompt/image, then aggregate rewards to compute global advantages.
- Preconditions:
  - `sample_k` divisible by `world_size` (recommended), or define remainder policy; set `k_per_rank = sample_k // world_size`.
  - `per_device_train_batch_size = 1` for backprop; generation can buffer via `steps_per_generation`.
- Implementation toggle:
  - `BBUGRPOTrainer` consults `cfg.distributed.cross_rank_sampling`; when disabled we fall back to the existing local-only K sampling path.
- Coordination (new `src_new/rl/distributed.py`):
  - `broadcast_indices(indices: Tensor[int]) -> List[int]`: rank 0 selects dataset indices via `RLDenseJSONLDataset.__getitem__`; all ranks fetch the same samples locally (paths resolved per-rank).
  - `broadcast_gen_cfg(cfg)`: keep generation params identical; derive per-rank seed via `seed_for_rank(base_seed, global_step, rank, gen_idx)`.
  - `all_gather_rewards(local_rewards: Tensor[B*k_per_rank]) -> Tensor[B*sample_k]`: gather per-completion scalar rewards across ranks (metadata stays local).
  - `compute_global_advantages(rewards, group_size=sample_k) -> Tensor`: mean/std per prompt across all completions; return normalized advantages; split back per rank.
  - `barrier(label)`: synchronization at generation, reward, and advantage points.
- Data flow per step:
  1) Rank 0 picks next indices; broadcast to all ranks. Each rank resolves the same samples locally (conversation builder + images).
  2) Each rank generates `k_per_rank` completions per prompt (sequentially per sample to fit memory).
  3) Compute local rewards; `all_gather` to form global rewards; compute global advantages; scatter local slices back.
  4) Each rank computes per-token logps for its own completions and backprops with its local advantages.
- Seeding for diversity (per-rank sampling):
  - `seed_for_rank = base_seed_from_yaml + global_step*sample_k + rank_id` (persisted for resume) ensures distinct sampling; log effective seeds on rank 0.
- Buffering and accumulation:
  - Maintain buffer splitting by `steps_per_generation` after advantages are computed; keep vision THW/patch slicing per chunk.
  - Ensure `sample_k == world_size * k_per_rank`; otherwise round-robin fill the remainder and log a warning.
- YAML (distributed block):
  - `distributed:`
    `  cross_rank_sampling: true`
    `  gather_rewards_only: true`        # gather only scalar rewards/advantages
    `  k_split_mode: divisible|round_robin`
    `  reduce_timeout_s: 600`
  - Runner refactor: manual YAML parsing now surfaces temperature/beta schedules and distributed knobs directly to `BBUGRPOTrainer` (no `_map_yaml_to_grpo_config`).
- Validation:
  - Assert consistency of prompt text hashes and THW sums across ranks for a small sample periodically.
  - Check that gathered reward tensor shape equals `[B*sample_k]` and no NaNs appear before normalization.
- DeepSpeed notes:
  - Use `torch.distributed` collectives alongside ZeRO-2 safely; avoid long gaps between collectives (set `NCCL_ASYNC_ERROR_HANDLING=1`).
  - Keep optimizer stepping identical; all_reduce is handled by DS; our gather/reduce is rewards/advantages only.

---

### 7) Config surface (YAML, strict)
- Reuse `EnhancedRLConfig` keys already enforced in `runner.py`:
  - Top‑level required: `model_path`, `train_data_path`, `val_data_path`, `data_root`, `output_dir`, `tb_dir`, `run_name`, `bf16: true`.
  - Training: `per_device_train_batch_size`, `update_steps` (grad accumulation), `learning_rate`, `weight_decay`, `max_steps`, `warmup_steps`, `logging_steps`, `save_steps`, `seed`.
  - Generation: `sample_k`, `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`.
  - GRPO block: `epsilon_low`, `epsilon_high`, `beta`, `loss_type`, `scale_rewards`, `mask_truncated_completions`.
  - Layer control: `layer_config` (applied via `PhaseFreezeManager`).
- Optional additions (already anticipated in runner):
  - `steps_per_generation` (default: `update_steps` if not set)
  - `temperature_schedule: {constant|linear_decay|cosine}`
  - `standardize_rewards: bool`
  - `max_advantage_magnitude: float` (|adv| clip)

---

### 8) Optimizer & schedulers
- Parameter groups (like SFT): vision, merger, LLM; learning rates surfaced for logging.
- Scheduler: cosine or from YAML; warmup via `warmup_steps`.
- Grad norm clipping (match SFT defaults).

---

### 9) Logging & metrics (parity)
- Scalars (train):
  - `loss`, `num_tokens`, `completions/mean_length`, `completions/min_terminated_length`, `completions/max_terminated_length`, `completions/clipped_ratio`
  - `reward`, `reward_std`, per‑reward `rewards/<name>/{mean,std}`
  - `kl` when `beta>0`
- Textual (optional): sampled `prompt`, `completion`, `advantages`, rewards per function.
- TensorBoard directory: `tb_dir/run_name` (created like runner does now).

---

### 10) Migration plan
- Phase 0 (landside, no behavior change):
  - Add `src_new/rl/grpo_trainer.py` (ported helpers + blank train loop) and unit tests for generation/log‑prob slices.
- Phase 1 (wire up):
  - Update `src_new/rl/runner.py` to instantiate `BBUGRPOTrainer` (feature flag `use_manual_trainer: true`, default true). Keep TRL path behind `false` for fallback during bring‑up.
  - Adapt `scripts/rl_feature_checks.py` to call BBUGRPOTrainer (or create `scripts/manual_rl_feature_checks.py`).
- Phase 2 (remove TRL):
  - Delete TRL‑based `src_new/rl/trainer.py` or move to `_legacy/`.
  - Drop TRL dependency from env after validation.

---

### 11) Validation checklist
- Loader smoke: `python -m src_new.rl.runner --config ... --mode load` prints device/dtype and vocab parity.
- Short run (100 samples, 1 epoch):
  - Rewards finite; advantages non‑zero variance for K≥2
  - Completion masks consistent; truncation masking toggles work
  - Logs written to `tb_dir/run_name`
  - Checkpoint saved (best + last); contains tokenizer/processor
- Alignment checks:
  - `<|image_pad|>` tokens match THW expectation
  - Packed `pixel_values` rows match THW patch products
  - EOS trimming at first `<|im_end|>`
- Distributed:
  - ZeRO‑2 run across 8 GPUs with `per_device_train_batch_size=1`, `update_steps≥K` to meet memory constraints
  - Cross-rank smoke (≥2 GPUs, `distributed.cross_rank_sampling=true`) covering broadcast indices, reward gather, and resumed-seed determinism

---

### 12) Risk & mitigations
- Risk: memory spikes during log‑prob compute → Mitigation: strictly per‑sample sequential; optional micro‑batching inside `_get_per_token_logps`.
- Risk: reward scaling instability → Standardization + advantage clip, schedule temperature.
- Risk: DS configuration mismatch → honor existing env gates; surface clear logs for effective DS state.

---

### 13) Deliverables
- `src_new/rl/grpo_trainer.py` with documented interfaces
- `src_new/rl/distributed.py` housing broadcast/gather helpers consumed by the trainer
- Updated `src_new/rl/runner.py` to use BBUGRPOTrainer
- Updated feature checks script (manual path)
- Short README blurb in `src_new/UNIFIED_DOCUMENTATION.md` referencing manual RL trainer

---

### 14) Timeline (suggested)
- Day 1–2: Implement trainer helpers (generation, masks, log‑probs), trivial train loop on single GPU
- Day 3: Wire optimizer/scheduler/DS; add checkpointing/logging; add temperature schedule/standardization/clip
- Day 4: Integrate runner; run debug YAML (100 samples); fix stability
- Day 5: Multi‑GPU validation; remove TRL usage; finalize docs

---

### 15) Accuracy & Stability Refinements (high-priority)

1) Generation controls (quality over speed)
- Enforce EOS and provide length floors:
  - Always set `eos_token_id = <|im_end|>`.
  - Add optional `generation.min_new_tokens` (e.g., 24–64) to reduce early truncation.
  - Prefer `mask_truncated_completions: false` for the first N steps so truncated samples still contribute gradients; switch to `true` after formatting stabilizes.
- Sampling knobs (default‑stable): `temperature ∈ [0.9, 1.0]`, `top_p ≈ 0.95`, `repetition_penalty ∈ [1.05, 1.10]`.
- Optional constrained decoding (format‑first): introduce a `GeometryConstrainedLogitsProcessor` (gated by YAML `constrained_generation: true`) that:
  - Encourages geometry wrappers (`<|box_start|>...<|box_end|>`, etc.) and object ref tokens when completion begins.
  - Penalizes banned vocab and illegal punctuation inside geometry spans.
  - Is soft: down‑weights logits rather than hard masks to preserve exploration.

2) Reward shaping & calibration
- Use the existing `RewardStandardizer` with EMA (per‑key) to stabilize scale across training.
- Robustify with Huber‑style clipping before standardization (e.g., clamp to ±5σ) to damp outliers.
- Curriculum for accuracy:
  - Stage‑A (stabilize formatting): emphasize formatting rewards, e.g., `parse:0.40, wrappers:0.20, coords:0.20, separators:0.10, vocab:0.10`, `length_window: 0.10`.
  - Stage‑B (introduce geometry): increase `coords`, add `geometry_sanity`, `ordering`, small `coverage`.
  - Stage‑C (quality): introduce `bbox_giou/quad_l1/line_l1` gradually (start at 0.05–0.20 total weight).
- Length/window rewards:
  - Use `length_window` to discourage too short/too long outputs and reduce degenerate `<|im_end|>`‑at‑once behavior.

3) Advantage normalization & grouping
- Normalize advantages per prompt over K completions using mean/std with ε=1e‑4 (already planned).
- Optional (accuracy over simplicity): compute rewards aggregation across ranks for identical prompts to reduce normalization bias when K > local batch capacity:
  - Gather rewards for a generation batch across ranks (via `torch.distributed.all_gather_object`) before computing group mean/std; fallback to local when not available.

4) KL regularization (optional but helps stability)
- Add an optional KL term (`beta > 0`) for the first N steps (e.g., `beta_start=0.02` → cosine/linear anneal to 0 over 10–20% of steps).
- Implementation: reuse current model as reference by temporarily disabling PEFT adapters (if any) during ref pass; otherwise clone a frozen reference (memory permitting).

5) Freezing schedule tuned for RL
- Prefer `phase_3` with `vision_freeze_patch_embed=true`, `vision_trainable_top_k_blocks=0–2`, `llm_trainable_top_k_blocks=4–8`.
- If drift or instability observed, fallback to `phase_2` (unfreeze only the last K LLM blocks; vision remains mostly frozen).

6) Optimizer & numerics
- AdamW with parameter groups (vision≪LLM): base_lr≈5e‑6, merger_lr≈5e‑5, vision_lr≈1e‑7; weight_decay in [0.0, 1e‑3].
- Gradient clipping `max_grad_norm=1.0`.
- Enforce bf16; fall back to fp32 if bf16 not supported.
- Detect NaN/Inf in loss or grads; on detection: skip step, zero grads, lower temperature by a small factor for the next generation cycle, and log an event.

7) Buffer regeneration & scheduling
- Keep `steps_per_generation * num_iterations` policy; regenerate only when needed to reduce target drift across accumulation window.
- Shuffle buffer slices before splitting to avoid correlation between prompts and micro‑steps.
- Set and log per‑rank seeds for reproducibility; disable dropout in model for training (policy learning is sensitive to noise).

8) Validation & guardrails
- Periodic eval (e.g., every `save_steps`) using `src_new/rl/eval.py` on the val JSONL:
  - Report: `valid_parse_rate`, wrapper/coords/separators/vocab pass rates, mean reward, detection metrics (GIoU/L1/order/coverage/sanity).
- Train‑time health checks:
  - Assert `sample_k` divides `per_device_train_batch_size * world_size * update_steps` (or log a warning and auto‑adjust `steps_per_generation`).
  - Verify `<|image_pad|>` tokens vs THW after decode for a small sample each logging interval; warn on drift.

9) YAML additions (optional)
- `generation: { min_new_tokens: int, constrained_generation: bool }`
- `grpo: { beta_start: float, beta_anneal: {type: cosine|linear, steps: int} }`
- `rewards: { curriculum: {stage: A|B|C, switch_steps: [int,int]} }`
- `normalization: { cross_rank_advantages: bool }`

10) Defaults (stable starting point)
- `temperature: 1.0`, `top_p: 0.95`, `repetition_penalty: 1.05`
- `sample_k: 4`, `min_new_tokens: 32`, `max_new_tokens: 512–1024`
- `mask_truncated_completions: false` for first 500–1,000 steps, then `true`
- `beta_start: 0.02 → 0.0` over first 10% steps (optional)
- Freezing: `phase_3` with `llm_trainable_top_k_blocks: 6`, `vision_trainable_top_k_blocks: 0`, `vision_freeze_patch_embed: true`

---

### 16) Immediate code-aligned tactics (drop-in, low risk)

- Strict per-sample log-prob pass (already present): reuse `src_new/rl/logprobs.get_per_token_logps` for sequential evaluation.
- Completion mask visibility: keep debug of `completion_mask.sum()` so zero-loss windows are visible when `mask_truncated_completions=true`.
- Advantage clipping toggle: expose `grpo.max_advantage_magnitude` in YAML and clamp symmetrically in the trainer (default off).

---

### 17) Reuse & Decoupling (single sources of truth)

- src_new/rl/generation.py
  - `prepare_generate_inputs(batch)`
  - `generate_completions(model, tokenizer, batch, generation_config|kwargs)`
  - `sample_k(batch, k, max_new_tokens, temperature, repetition_penalty)`
  - `concatenate_queries_and_responses(queries, responses)`

- src_new/rl/logprobs.py
  - `get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, pixel_values, image_grid_thw, images_per_sample, temperature, detach=False)`
  - `slice_packed_vision_for_sample(pixel_values, image_grid_thw, images_cumsum, patch_offset)` → `(pv_slice, grid_slice, new_patch_offset)`

- src_new/rl/buffer.py
  - `generate_and_score(model, tokenizer, inputs, reward_fns, reward_weights, gen_cfg)` → dict of tensors incl. `prompt_ids`, `prompt_mask`, `completion_ids`, `completion_mask`, `old/ref_logps` (optional), `advantages`, and aligned vision tensors
  - `split_buffer(buffer, steps_per_generation)` → List[dict] with correct THW/patch slicing per chunk

- src_new/rl/losses.py
  - `compute_grpo_loss(per_token_logps, old_per_token_logps, advantages, completion_mask, epsilon_low, epsilon_high, loss_type, beta=0.0, per_token_kl=None, delta=None)`
  - `compute_kl(ref_logps, logps)` → per-token KL

- src_new/rl/schedules.py
  - `temperature_at(step, schedule, base)`
  - `beta_at(step, beta_start, anneal)`
  - `curriculum_stage_at(step, switch_steps)`

- src_new/rl/validators.py
  - `debug_validate_image_alignment(prompt_ids, image_grid_thw, tokenizer)`
  - `assert_patches_match_thw(pixel_values, image_grid_thw, images_per_sample, chunk_ranges)`
  - `check_completion_masks(completion_mask)`

- src_new/rl/rewards/standardizer.py (reuse existing)
  - `RewardStandardizer` (EMA)
  - `standardize_rewards(rewards_per_func, names, clip_sigma=5.0)`

- src_new/rl/runner.py
  - Stays thin: load config, build components/datasets, materialize reward functions + weights + schedules; instantiate `BBUGRPOTrainer`.

- src_new/rl/grpo_trainer.py
  - `BBUGRPOTrainer` composes only the above modules; no inline duplication of generation/logprob/loss/schedules.

Decoupling rules
- Generation/log-prob modules contain no reward or loss logic.
- Rewards are pure functions from `rewards/registry.py`; standardization is a separate step.
- No in-code defaults; all knobs come from YAML and are validated centrally.
- DeepSpeed details live only in trainer/runner; utilities are DS-agnostic.
- Vision slicing logic is centralized so both trainer and eval reuse it.
- EOS/min_new_tokens handling lives in generation module; runner sets config fields.
