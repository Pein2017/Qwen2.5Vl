## Refactoring Plan for `src_post/` (GRPO, Group-Level QC)

### Goals
- Reduce complexity in `grpo_runner.py` by decomposing into cohesive modules with single responsibilities.
- Make configuration explicit, validated, and strongly typed; remove hidden defaults for core knobs.
- Decouple I/O (dataset, logging, checkpoints), processing (prompts, TF/decoding), optimization (GRPO update), and orchestration (epochs, DDP).
- Keep public APIs small, documented, and easy to test. Ensure rank-synchronized logging/checkpointing stays robust.

### Current structure (key files)
- `src_post/grpo_runner.py` – monolithic entrypoint: config parse, DDP setup, data, convo building, sampling, TF, KL, loss, optimizer, logging, ETA.
- `src_post/conversation.py` – Stage‑A/Stage‑B prompt builders and mission hints.
- `src_post/dataset_group_qc.py` – group‑level dataset.
- `src_post/rewards/*` – modular rewards; registry in `rewards/__init__.py`.
- `src_post/logits_processors.py` – decode‑time masking.
- `src_post/span_parser.py` – parse Stage‑B outputs.
- `src_post/reward_sampling_agg.py` – small helpers (currently unused in runner).
- `src_post/utils/text.py` – formatting/keyword utilities used by rewards.

### What stays as-is
- `conversation.py`: Keep `GroupQCConversationBuilder` API; we will reuse it directly.
- `dataset_group_qc.py`: Keep dataset semantics and label normalization.
- `span_parser.py`: Keep `parse_stage_b_output` logic and regexes.
- `logits_processors.py`: Keep `GeometryCoordMaskLogitsProcessor` unchanged.

### Concrete extraction map (function-by-function)
- Move from `grpo_runner.py` to `src_post/generation.py`:
  - `_decode_to_text(processor, token_ids)`
  - `_to_device_and_cast(enc, device)`
  - `_sft_style_preprocess_image(img)`
  - `build_best_stage_a_lines(images)` (rename to `build_stage_a_context_lines(policy, processor, images, gen_cfg, logits_processors, stopping)`) – keep greedy behavior and sanitization hook.
  - Stop criteria class `StopOnTokens` and construction of `stopping_stage_a` (as a factory `build_stage_a_stopping(processor)`).

- Move from `grpo_runner.py` to `src_post/teacher_forcing.py`:
  - `_compute_logprobs(logits, target_ids)`
  - `_tf_sum_logprob_over_response(model, enc, response_ids, length_norm)`
  - `_tf_sum_logprob_and_logits_over_response(model, enc, response_ids, length_norm)`
  - `_kl_to_ref_over_response(model, ref_model, enc, response_ids)`
  - `_kl_to_ref_with_cur_logits(ref_model, enc, response_ids, cur_logits, prompt_len)`
  - `_tf_decision_probs(model, enc, processor, length_norm=True)`
  - `_maybe_autocast(enabled: bool)` (keep device dtype bf16 policy here for TF paths).

- Move from `grpo_runner.py` to `src_post/data_loader.py`:
  - `_build_epoch_indices(dataset_len, world_size, rank, limit_groups, seed, epoch)`
  - `iter_batches(indices, batch_size, drop_last)` (new): yields index slices per mini-batch.

- Move from `grpo_runner.py` to `src_post/rewards/compose.py`:
  - `compose_reward(gt_label, pred_label, summary_lines, reason, tf_p_pass, tf_p_fail)`; inject `checklist_lines` via a parameter instead of reading inside to avoid hidden deps. Keep finite checks and descriptive errors.

- Move from `grpo_runner.py` to `src_post/logging_utils.py`:
  - `compute_eta(dt_global, epoch_round, ptr, indices_len, batch_size, drop_last, epochs)` (uses exact formula now in runner).
  - `reduce_metrics_across_ranks(vec_local) -> vec_global` (encapsulate all_reduce SUM/MAX where needed).
  - `rank0_log(logger, tb_writer, metrics_writer, step, scalars: Dict[str, float], lrs: Dict[str, float])`.

- Introduce `src_post/models.py`:
  - `load_processor(path) -> Qwen2VLProcessor` (migrate `_load_processor`).
  - `load_detection_model(checkpoint, processor, device) -> DetectionModel` (migrate `_load_detection_model` with Qwen2.5 fixes and dtype policy).
  - `apply_training_freeze(policy, processor, cfg) -> List[Dict]]` (extract PhaseFreezeManager usage and param-group discovery for aligner/LLM top-K/vision top-K).
  - `wrap_ddp_if_needed(policy, device) -> Optional[DDP]`.

- Introduce `src_post/checkpoints.py`:
  - `save_if_rank0(policy, processor, output_dir, tag, skip_save)`.

### Configuration and validation
- New `src_post/config.py`:
  - `@dataclass(frozen=True) RLRunnerConfig` (move current `RunnerConfig` here; keep required fields first; add explicit types; keep `epochs`, `drop_last`).
  - `load_and_validate_config(path: str) -> RLRunnerConfig`:
    - Verify: `checkpoint`, `processor`, `output_dir` required.
    - Verify: `epochs>=1`, `batch_size>=1`, `K_B>=1`, `K_A>=1`, `train_stage_a_mode in {off,conditional,joint}`.
    - If `use_ref_kl=True`, require `ref_checkpoint` path exists.
    - Disallow silent defaults on core knobs; raise `ValueError` with remediation hints.

- Strengthen rewards registry validation in `src_post/rewards/__init__.py`:
  - Update `build_reward_fns` to raise on unknown names (aligns with current validation already done in runner), e.g.:
    - If `name not in REGISTRY`: raise `KeyError(f"Unknown reward name: {name}")`.

### Orchestrator class
- New `src_post/runner.py` with `class RLRunner`:
  - Constructor accepts `cfg: RLRunnerConfig`.
  - `run()` performs: seeding → DDP init → build processor/policy/ref → apply freezing/groups/optimizer/scheduler → dataset → epoch/batch loop.
  - Calls out to generation/TF/reward/compose modules; no inlined helpers in orchestrator beyond control flow.
  - Preserves existing distributed behavior: rank-0 only logging/checkpointing, all_reduce for metrics, MAX over dt for ETA.

### Behavior-preserving notes (grounded in your code)
- Keep bf16 autocast usage only inside our generation/TF helpers; no external autocast conflicting with DDP.
- Preserve current sanitization logic `_sanitize_stage_a_text` and apply it inside `generation.build_stage_a_context_lines` when `sanitize_stage_a=True`.
- Keep Stage‑B/Stage‑A generation configs intact, including `min_new_tokens=1`, repetition/no-repeat knobs, and decode-time logits processor wiring.
- Maintain streamed-backward pattern: call `.backward()` per candidate term to avoid graph retention; keep KL inline to prevent double‑reduce issues.
- Maintain micro-step equalization via `max_images_tf` for conditional Stage‑A.
- Preserve current ETA computation formula (moved to `logging_utils.compute_eta`).

### Incremental phases (no behavior change)
1) Extract modules and rewire imports (no logic changes). Add unit-safe wrappers where necessary.
2) Move config dataclass + loader; replace dict parsing; keep CLI args.
3) Introduce `RLRunner` and make `grpo_runner.py` a thin CLI delegator.
4) Tighten rewards registry validation; add docstrings and types for public APIs.
5) Add basic tests for compose/reducer/ETA/epoch indices.

### File-by-file task checklist
- src_post/generation.py
  - [ ] Port `_decode_to_text`, `_to_device_and_cast`, `_sft_style_preprocess_image`
  - [ ] Add `build_stage_a_context_lines(...)`
  - [ ] Add `build_stage_a_stopping(processor)`
- src_post/teacher_forcing.py
  - [ ] Port `_compute_logprobs`, TF functions, KL helpers, `_maybe_autocast`
- src_post/data_loader.py
  - [ ] Port `_build_epoch_indices`; add `iter_batches`
- src_post/rewards/compose.py
  - [ ] Port `compose_reward` (take `checklist_lines` as param)
- src_post/logging_utils.py
  - [ ] Add `reduce_metrics_across_ranks`, `compute_eta`, `rank0_log`
- src_post/models.py
  - [ ] Port processor/model loaders; add freeze/param-group builder; add DDP wrapper
- src_post/checkpoints.py
  - [ ] Add `save_if_rank0`
- src_post/config.py
  - [ ] Add `RLRunnerConfig`; `load_and_validate_config`
- src_post/rewards/__init__.py
  - [ ] Make `build_reward_fns` raise on unknown names
- src_post/runner.py
  - [ ] Implement `class RLRunner` using extracted APIs
- src_post/grpo_runner.py
  - [ ] Replace body with thin CLI delegator to `RLRunner`

### Risks and mitigations
- Risk: Subtle dtype/device mismatches after extraction → Mitigate by centralizing device casts in `generation._to_device_and_cast` and TF helpers; add asserts/logging in failure messages.
- Risk: DDP behavior regression → Keep `_build_epoch_indices` semantics identical; unit-test sharding.
- Risk: Reward inputs mismatch → `compose_reward` raises explicit errors listing available keys.
- Risk: ETA regressions → Unit-test `compute_eta` against synthetic progress.

### Coding standards alignment
- Fail fast with actionable messages in config load and compose_reward.
- Strong typing on public APIs; avoid implicit Any.
- No wildcard imports; define `__all__` in new modules where appropriate.
- No top-level I/O or GPU init beyond orchestrator entry.

### Migration path
- Backwards compatible CLI remains at `python -m src_post.grpo_runner --config ...`.
- Config keys unchanged for users; internals switch to dataclass loader.
- Subsequent PR can delete legacy helpers from `grpo_runner.py` after stabilization.

### Expected impact
- Clear separation of concerns; smaller files; easier to test and evolve.
- Safer distributed training via centralized sharding/ETA and rank‑0 logging.
- Faster iteration on rewards and prompts without touching orchestration.
