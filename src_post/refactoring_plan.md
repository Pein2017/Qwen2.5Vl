## Refactoring Plan for `src_post/` (GRPO, Group-Level QC)

### Objectives (aligned with Qwen2.5‑VL-main/coding-rule)
- Enforce explicit, validated configuration (no silent defaults for core knobs) via a single frozen dataclass schema.
- Decouple orchestration from processing, teacher‑forcing, rewards, logging, and checkpointing; keep modules small with single responsibilities.
- Strong typing, clear public APIs, fail‑fast validations with actionable messages.
- Deterministic and robust DDP behavior; rank‑0 logging and safe checkpointing.
- Preserve current features and metrics; move code to testable units with docstrings and types.
- Preserve JSONL/TensorBoard schemas and console log fields for comparability.

---

### Current structure (summary)
- `src_post/grpo_runner.py`: Monolithic runner (config parse, DDP setup, dataset, generation, TF, GRPO, KL, freezing, optimizer, scheduler, logging, ETA, diagnostics).
- `src_post/conversation.py`: Stage‑A/Stage‑B builders and mission hints.
- `src_post/dataset_group_qc.py`: dataset and label normalization.
- `src_post/rewards/*`: registry and functions (label_match, decision_prob, violations, coverage, formatting, cleanliness, taxonomy, consistency, repetition penalties).
- `src_post/logits_processors.py`: geometry/coordinate masking at decode time.
- `src_post/span_parser.py`: Stage‑B parsing.
- `src_post/utils/text.py`: formatting/keyword utilities.
- `src_post/reward_sampling_agg.py`: small helpers (currently unused by runner).

---

### Target architecture (modules and contracts)

1) Configuration and validation
- File: `src_post/config.py`
- API:
  - `@dataclass(frozen=True) class RLRunnerConfig: ...` (required fields first; `Optional[...]` explicit; strong types). Include: paths, runtime, generation, sampling, training, selective‑unfreeze, KL, rewards, Stage‑A mode, diagnostics toggles.
  - `def load_and_validate_config(path: str) -> RLRunnerConfig`
    - Read YAML/JSON → dict → dataclass; validate core keys and ranges; fail fast with actionable error messages including expected vs actual and remediation hints.
    - Require core: `checkpoint`, `processor`, `output_dir`, and one of `train_data_dir|eval_data_dir`.
    - When `train: true`: `epochs ≥ 1`, `batch_size ≥ 1`.
    - Sampling: `K_B ≥ 1`, `K_A ≥ 1`; if `train_stage_a_mode == joint` then `K_set ≥ 1`. `train_stage_a_mode ∈ {off, conditional, joint}`.
    - Rewards: `len(reward_fns) == len(reward_weights)`; all names present in registry; raise with list of valid names.
    - KL: if `use_ref_kl: true`, require `ref_checkpoint`.
    - Normalize `limit_groups`: accept `None|-1|0` as “no limit”; store canonical `-1`.
    - No silent defaults for core knobs. Derived only for non‑core paths: default `results_jsonl` → `<output_dir>/results.jsonl`, `metrics_jsonl` → `<output_dir>/metrics.jsonl`, and TB `run_name` when missing.
    - Back‑compat: accept legacy key `processor_path` as alias of `processor`.

2) Models and freezing
- File: `src_post/models.py`
- API:
  - `load_processor(path: str) -> Qwen2VLProcessor`
    - Preserve tokenizer chat template.
  - `load_detection_model(checkpoint: str, processor: Qwen2VLProcessor, device: str) -> DetectionModel`
    - Call `apply_comprehensive_qwen25_fixes()`; set dtype like current runner; set `use_cache` toggles matching TF/generation semantics.
  - `apply_freeze_and_param_groups(policy: DetectionModel, tokenizer, cfg: RLRunnerConfig) -> List[Dict[str, Any]]`
    - Use `PhaseFreezeManager` phase_3 with `top_k_llm_layers`, `top_k_vision_blocks`, `freeze_vision_patch_embed`.
    - Discover aligner via current candidate paths; return three named groups (`aligner`, `llm_topk`, `vision_topk`) with per‑group LRs resolved from config.
  - `wrap_ddp_if_needed(model: torch.nn.Module, device: str) -> Optional[DDP]`
    - Use `find_unused_parameters=True`, `broadcast_buffers=False`; expose helper to `_sync_params_and_buffers(authoritative_rank=0)` before TF passes.

3) Data loading and sharding
- File: `src_post/data_loader.py`
- API:
  - `build_epoch_indices(dataset_len: int, world_size: int, rank: int, limit_groups: int, seed: int, epoch: int) -> List[int]`
    - Deterministic shuffle by `(seed + epoch)` → apply limit → trim to multiple of `world_size` → position‑shard by rank (matches current behavior).
  - `iter_batches(indices: List[int], batch_size: int, drop_last: bool) -> Iterable[List[int]]`

4) Conversation building and text rendering
- File: `src_post/conversation.py` (keep)
- Add minimal docstrings and type hints on public methods.
- Encourage calling `GroupQCConversationBuilder.validate_image_placeholder_count(rendered_text, expected_num_images)` before encoding Stage‑A to enforce `<image>` placeholder count.

5) Generation utilities
- File: `src_post/generation.py`
- API:
  - `to_device_and_cast(enc: Dict[str, Any], device: str) -> Dict[str, Any>`
  - `decode_to_text(processor: Qwen2VLProcessor, token_ids: List[int]) -> str`
  - `sft_style_preprocess_image(img: Any) -> PIL.Image`
  - `build_stage_a_stopping(processor: Qwen2VLProcessor) -> Optional[StoppingCriteriaList]` (newline/`。` when single‑token)
  - `build_stage_a_context_lines(policy, processor, images: List[PIL.Image], gen_cfg: GenerationConfig, logits_processors: LogitsProcessorList, stopping: Optional[StoppingCriteriaList], sanitize: bool) -> List[str]`
    - Mirrors current greedy flow; optional `sanitize` applies the current `_sanitize_stage_a_text` logic.

6) Teacher‑forcing and KL helpers
- File: `src_post/teacher_forcing.py`
- API:
  - `compute_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor`
  - `tf_sum_logprob_over_response(model, enc, response_ids: List[int], length_norm: bool) -> torch.Tensor`
  - `tf_sum_logprob_and_logits_over_response(model, enc, response_ids: List[int], length_norm: bool) -> Tuple[torch.Tensor, torch.Tensor, int]`
  - `kl_to_ref_over_response(model, ref_model, enc, response_ids: List[int]) -> torch.Tensor`
  - `kl_to_ref_with_cur_logits(ref_model, enc, response_ids: List[int], cur_logits: torch.Tensor, prompt_len: int, length_norm: bool=True) -> torch.Tensor`
  - `tf_decision_probs(model, enc, processor, length_norm: bool=True) -> Tuple[float, float]`
  - `_maybe_autocast(enabled: bool)` context manager for bf16/gpu.

7) Rewards composition
- File: `src_post/rewards/compose.py`
- API:
  - `compose_reward(gt_label: str, pred_label: Optional[str], summary_lines: List[str], reason: Optional[str], checklist_lines: List[str], reward_names: List[str], reward_weights: List[float], tf_p_pass: float, tf_p_fail: float) -> float`
    - Uses `REGISTRY`, validates names, lengths, and finiteness; raises explicit errors. No hidden globals.
- Strengthen registry validation: update `rewards/__init__.py::build_reward_fns` to raise on unknown names (list valid names) and keep config‑time validation.

8) Logging, metrics reduction, ETA
- File: `src_post/logging_utils.py`
- API:
  - `reduce_sum(t: torch.Tensor, world_size: int) -> torch.Tensor`; `reduce_max(...)`
  - `aggregate_training_metrics(local_vec: torch.Tensor, world_size: int) -> Dict[str, float]` (wrap current vector layout; typed keys; docs)
  - `compute_eta(dt_max_sec: float, processed_so_far_g: int, epoch_total_g: int, epoch_idx: int, epochs: int, items_per_update_g: int) -> float` (hours)
  - `rank0_log(logger, tb_writer, metrics_writer, step: int, prefix: str, scalars: Dict[str, float], lrs: Dict[str, float]) -> None`
  - Preserve current printed fields and TB scalar names; append Phase‑A diagnostics and optional `kl_b` when present.

9) Checkpointing
- File: `src_post/checkpoints.py`
- API:
  - `save_if_rank0(policy, processor, output_dir: str, tag: str, skip_save: bool) -> None`
    - Safe mkdirs; save to `<output_dir>/checkpoints/<tag>/` and `processor/`; barrier on multi‑GPU; fail fast with clear message.

10) Phase‑A diagnostics (toggleable)
- File: `src_post/diagnostics.py`
- API:
  - `compute_phase_a_diagnostics(summary_lines: List[str], checklist_lines: List[str]) -> Dict[str, float]` returning `{formatting, coverage, cleanliness, taxonomy, consistency}` in [0,1].
  - Pure function; no device deps; unit testable. Control via config flag `enable_phase_a_diagnostics` (defaults off unless set).

11) Orchestrator
- File: `src_post/runner.py`
- API:
  - `class RLRunner:`
    - `__init__(self, cfg: RLRunnerConfig)`
    - `run(self) -> None`
  - Encapsulates: seeding → DDP init → processor/model/ref → freezing/groups → optimizer/scheduler → dataset → epoch/batch loop. Uses only extracted helpers; no large inline blocks. Behavior mirrors current runner (streamed backward per candidate, per‑sample inline KL, Stage‑A conditional/joint, `max_images_tf`, rank‑0 logging, JSONL/TB write, final checkpoint save and barrier).

12) CLI entry (thin)
- File: `src_post/grpo_runner.py`
- Behavior: parse `--config`, call `load_and_validate_config`, instantiate `RLRunner(cfg).run()`.

---

### Contract updates and coding‑rule compliance
- Strong typing: annotate all public functions/classes; avoid implicit `Any`.
- Fail‑fast validations: raise with explicit remediation; include expected vs actual values.
- No silent defaults for core hyperparameters (checkpoint, processor, output_dir, epochs, batch_size, K_A/K_B, train_stage_a_mode). Optional toggles may have explicit safe defaults.
- Import hygiene: absolute imports; no wildcard imports; define `__all__` for modules with public APIs.
- Determinism: seed setting centralized; document trade‑offs; best‑effort reproducibility.
- No top‑level GPU init or I/O in libraries; orchestrator/entry only.
- Error handling: no swallowing; use `try/finally` for deterministic cleanup; re‑raise.
- Immutability: prefer frozen dataclasses and pure functions; avoid mutating inputs.
- Preserve decode‑time masking via `GeometryCoordMaskLogitsProcessor`.
- Preserve JSONL record schema and TB scalar names.

---

### Migration steps (incremental, no behavior change)
1) Extract helpers (from `grpo_runner.py`):
- To `generation.py`: `to_device_and_cast`, `decode_to_text`, `sft_style_preprocess_image`, `build_stage_a_stopping`, `build_stage_a_context_lines` (greedy summaries; optional sanitize).
- To `teacher_forcing.py`: `compute_logprobs`, `tf_sum_logprob_over_response`, `tf_sum_logprob_and_logits_over_response`, `kl_to_ref_over_response`, `kl_to_ref_with_cur_logits`, `tf_decision_probs`, `_maybe_autocast`.
- To `models.py`: `load_processor`, `load_detection_model`, `apply_freeze_and_param_groups`, `wrap_ddp_if_needed`.
- To `data_loader.py`: `build_epoch_indices`, `iter_batches`.
- To `logging_utils.py`: metrics vector reduction/aggregation, ETA, rank‑0 logging.
- To `checkpoints.py`: final save (+processor) and barriers.
- To `rewards/compose.py`: `compose_reward` using registry.
- To `diagnostics.py`: the five Phase‑A diagnostics built from existing reward utilities.

2) Introduce config dataclass:
- Implement `RLRunnerConfig` + loader; replace dict accesses with dataclass fields.
- Validate on load; remove in‑runner ad‑hoc defaults for core knobs. Accept `processor_path` alias and normalize `limit_groups`.

3) Introduce `RLRunner` orchestrator:
- Port control flow; keep logged fields/format exactly; centralize DDP init/destroy; call helpers for TF, KL, generation, and masking.

4) Strengthen rewards registry validation:
- Make `rewards/__init__.py::build_reward_fns` raise on unknown names (list valid names) and keep config‑time validation.

5) Add tests:
- Unit tests for: `compose_reward`, TF helpers on small tensors, ETA computation, epoch indices sharding, diagnostics functions.

6) Documentation:
- Update README/tutorial to reference new modules; add API docstrings.

---

### Additional improvements (optional but recommended)
- Metrics schema: create a small `TypedDict` for aggregated metrics to avoid positional vector coupling.
- Pluggable rewards: move weight/name parsing into `compose` with clearer error messages; allow YAML lists as well as comma‑sep strings (already partially supported).
- Sanitize hook: move `_sanitize_stage_a_text` to `generation.py` and guard by config flag.
- Logging prefixes: keep `[epoch/epochs][processed/total]` prefix; add optional per‑rank local counters behind a debug flag.
- Diagnostics toggles: allow enabling/disabling Phase‑A diagnostics in config (`enable_phase_a_diagnostics: true`).

---

### Risks and mitigations
- DDP regressions: keep `_sync_params_and_buffers` behavior; add tests; manually verify multi‑GPU logging and barriers.
- Device/dtype mismatches after extraction: centralize `.to(device)` and autocast in `generation`/`teacher_forcing`; add guards.
- Reward inputs mismatch: `compose_reward` should raise clear KeyErrors listing required keys.
- ETA correctness: test with synthetic traces; ensure `reduce_max` is used for per‑update wall‑time.
- Back‑compat: keep `processor_path` alias and `limit_groups` normalization; preserve JSONL/TB schemas.

---

### Acceptance criteria
- `python -m src_post.grpo_runner --config <yaml>` produces identical console/JSONL/TB metrics (plus the new diagnostics grouping where enabled).
- Config parse fails fast on missing core keys with clear messages.
- Unit tests pass (ETA, reducers, TF, compose, diagnostics, sharding).
- Public functions are typed and documented; no wildcard imports; no top‑level side effects.
