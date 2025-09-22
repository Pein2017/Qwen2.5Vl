# Tasks Document

- [x] 1. Extend config schema with validated keys
  - Files: src_post/config.py, src_post/README.md
  - Add keys: enable_clipped_grpo, epsilon_low, epsilon_high, loss_type_stage_b, enable_entropy_mask_stage_b, entropy_top_quantile_stage_b, entropy_min_threshold_stage_b, max_resample_times, stage_a_top_m, uncertainty_decay_factor, pairwise_select, log_all_candidates, soft_overlong_penalty_enabled, soft_overlong_penalty_weight
  - Conditional requirements: when enable_clipped_grpo=true → require epsilon_low∈(0,1] and loss_type_stage_b∈{grpo,bnpo,dr_grpo}; when enable_entropy_mask_stage_b=true → require exactly one of entropy_top_quantile_stage_b∈(0,1] or entropy_min_threshold_stage_b>0; when soft_overlong_penalty_enabled=true → require soft_overlong_penalty_weight≥0
  - Update RLRunnerConfig dataclass (frozen) and fail‑fast validation with remediation text referencing YAML keys
  - Document new keys in README with examples
  - _Requirements: Config/Validation, Coding‑Rule
  - _Prompt: Role: Senior Python Engineer (ML systems) | Task: Extend src_post/config.py with frozen dataclass fields and strict validators for the listed keys, adding conditional requirements and actionable error messages; update README with a concise example YAML | Restrictions: No silent defaults for core knobs; optional toggles default to safe false; no wildcard imports | Success: New keys parsed and validated; invalid inputs raise clear errors; README shows correct usage

- [x] 2. Implement GRPO loss utilities
  - New File: src_post/tf/grpo_loss.py
  - Functions: build_reply_mask, per_token_logps_from_logits, compute_ratio_and_clip, apply_entropy_mask_from_logits, reduce_loss
  - All functions check shapes/dtypes; raise on NaN/Inf; unit‑testable in isolation
  - _Leverage: src_post/tf/teacher_forcing.py::compute_logprobs, sequence_entropy_from_logits
  - _Requirements: Stage‑B Loss
  - _Prompt: Role: ML Systems Developer | Task: Create grpo_loss.py with the listed utilities implementing token‑wise ratio clipping and entropy masking compatible with Qwen2.5‑VL logits | Restrictions: Keep API pure; no global state; thorough docstrings; strict type hints | Success: Utilities pass unit tests and integrate without changing current behavior when disabled

- [x] 3. Integrate clipped GRPO + entropy mask in runner
  - File: src_post/runner.py
  - For each reply, get (sum_logp, logits, prompt_len); compute per‑token logps over reply; set old_logps=detach(cur) when on‑policy; compute coef_1/coef_2 with (epsilon_low, epsilon_high); per‑token loss = −min(coef_1*A, coef_2*A) + beta*KL; optional entropy mask; reduce by loss_type_stage_b; backward with accum scale
  - Track metrics: sb_clip_low_mean/min, sb_clip_high_mean/max, sb_clip_region_mean, sb_entropy_threshold, sb_entropy_mask_ratio
  - Guarded by enable_clipped_grpo / enable_entropy_mask_stage_b
  - _Requirements: Stage‑B Integration
  - _Prompt: Role: Training Loop Engineer | Task: Modify runner Stage‑B loss path to call grpo_loss utilities and compute clipped GRPO with optional entropy mask and metrics | Restrictions: Keep existing behavior when toggles are false; preserve DDP and KL logic; fail on NaN/Inf | Success: Feature toggles switch behavior; metrics appear; training remains stable on smoke test

- [x] 4. Add std==0 dynamic resampling
  - File: src_post/runner.py
  - If rewards std==0, attempt up to max_resample_times additional samples for the group; accept first std>0 else skip and count
  - Metric: std0_resample_count
  - _Requirements: Sampling Stability
  - _Prompt: Role: ML Engineer | Task: Implement bounded resampling for degenerate groups to avoid wasted updates | Restrictions: Do not alter RNG seeding logic; bounded compute; clear logs when resampling | Success: Resampling triggers when std==0; otherwise behavior unchanged

- [x] 5. Stage‑A credit focusing and gating
  - File: src_post/credit/credit_assignment.py
  - Compute per‑image Δ_i; sort and backprop only top M when stage_a_top_m>0; gate uncertainty: zero or decay advantage by uncertainty_decay_factor; pairwise selection strategy (heuristic|entropy|delta); split pairwise advantage proportional to token count of each summary
  - Batch encodings per image where safe
  - _Requirements: Stage‑A Enhancements
  - _Prompt: Role: Algorithm Engineer | Task: Add top‑M focusing, improved pairwise selection, and uncertainty gating into ConditionalAssigner/PairwiseFallbackAssigner | Restrictions: Preserve existing interfaces; keep fail‑fast checks; avoid regressions when M=0 | Success: Unit tests pass; deltas and gating behave as specified; no regressions

- [x] 6. Reward shaping and diagnostics
  - Files: src_post/rewards/soft_overlong.py (new), src_post/rewards/__init__.py, src_post/rewards/compose.py, src_post/runner.py
  - Implement optional soft_overlong_penalty (trigger when reply hits max_new_tokens_stage_b without EOS); register in REGISTRY; add per‑function reward diagnostics capture for the selected/best reply; include diagnostics in logs/JSONL when enabled
  - _Requirements: Rewards/Diagnostics
  - _Prompt: Role: NLP Engineer | Task: Add optional soft overlong penalty reward and per‑function diagnostics capture | Restrictions: Keep compose normalization; fail on non‑finite reward; do not alter existing reward semantics | Success: New reward available; diagnostics logged; compose continues to validate inputs

- [x] 7. Logging and metrics wiring
  - Files: src_post/runner.py, src_post/logging/logging_utils.py
  - Emit new scalar metrics; ensure rank0_log handles additional keys; avoid breaking vector aggregation (keep new stats as direct scalars)
  - JSONL: if log_all_candidates, write all K_B replies with per‑fn rewards and total reward; preserve existing result schema when disabled
  - _Requirements: Observability
  - _Prompt: Role: Observability Engineer | Task: Wire new metrics into rank‑0 logging and optional JSONL payload | Restrictions: Do not change existing vector reduction protocol; keep outputs backward‑compatible | Success: Scalars appear in TB/JSONL; old consumers unaffected

- [x] 8. Unit tests for utilities and algorithms
  - Files: tests_post/tf/test_grpo_loss.py, tests_post/credit/test_credit_assignment.py, tests_post/config/test_config_validation.py
  - Cover clipping math, entropy mask selection, top‑M selection, pairwise split, invalid configs
  - _Requirements: Testing
  - _Prompt: Role: QA Engineer | Task: Create unit tests covering happy paths and edge cases for new utilities and config validators | Restrictions: No GPU dependence; deterministic seeds; assert error messages contain remediation hints | Success: Tests pass locally in CI‑like env

- [-] 9. Integration smoke tests
  - Files: tests_post/integration/test_runner_smoke.py
  - Short run with tiny dataset and K_A=K_B∈{2,3}; features off (baseline) and on (clipped+entropy); assert finite loss/metrics and improved stability signals
  - _Requirements: Integration
  - _Prompt: Role: Integration Engineer | Task: Implement smoke tests to validate stability and toggles | Restrictions: Keep runtime short; no external network | Success: Tests run in <2 minutes and pass

- [ ] 10. Docs update
  - Files: src_post/README.md
  - Document new config keys, examples, and notes about stability and toggles
  - _Requirements: Docs
  - _Prompt: Role: Technical Writer | Task: Update README with concise sections for new features and YAML examples | Restrictions: Keep minimal; link to code where necessary | Success: Docs render cleanly and are accurate