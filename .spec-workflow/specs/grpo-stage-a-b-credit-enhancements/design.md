# Design Document

## Overview
Enhance `src_post/` GRPO with opt-in Stage‑B ratio‑clipped GRPO and entropy masking, Stage‑A top‑M credit and improved pairwise attribution, std==0 dynamic resampling, reward diagnostics, and richer logging. All changes use explicit config with fail‑fast validation and preserve current behavior when new toggles are disabled.

## Coding-Rule Compliance (Qwen2.5-VL-main/coding-rule)
- Explicit, frozen dataclasses; no silent defaults for core knobs. Optional features gated by explicit booleans with safe defaults (false).
- Boundary validation with actionable errors (expected vs actual, remediation hints).
- HF‑first; reuse existing `teacher_forcing.py`, `generation.py`; no monkey‑patching.
- Strong typing for public APIs; no wildcard imports.

## Code Reuse
- `tf_sum_logprob_and_logits_over_response()` to obtain `(sum_logp, logits, prompt_len)` for replies.
- `compute_logprobs()` and `sequence_entropy_from_logits()` for token‑wise GRPO and entropy.
- `compose_reward()` and reward registry for diagnostics.
- `credit/credit_assignment.py` as the extension point for Stage‑A.

## Architecture
- Runner loop unchanged in shape: sample → rewards → z‑score → Stage‑B loss (+KL) → Stage‑A loss → step.
- New utilities in `src_post/tf/grpo_loss.py` handle per‑token ratios, clipping, entropy masks, and reductions.
- Config extended in `src_post/config.py` with strict validation; toggles off keeps current semantics.

## Configuration (src_post/config.py)
Add new validated keys to `RLRunnerConfig` (required only when their feature is enabled):
- Stage‑B (clipped GRPO)
  - `enable_clipped_grpo: bool` (default false; safe)
  - `epsilon_low: float` (required if enabled; range (0, 1])
  - `epsilon_high: float` (required if enabled; range [0, 1]; effective upper slack; set 0 for two‑sided symmetric if desired)
  - `loss_type_stage_b: str` in {`grpo`,`bnpo`,`dr_grpo`} (required if enabled)
- Stage‑B (entropy mask)
  - `enable_entropy_mask_stage_b: bool` (default false)
  - One of:
    - `entropy_top_quantile_stage_b: float` in (0, 1] (mask tokens below top‑q entropy), or
    - `entropy_min_threshold_stage_b: float` in (0, +∞) (absolute threshold)
- Sampling stability
  - `max_resample_times: int` (≥0; default 0)
- Stage‑A focusing & gating
  - `stage_a_top_m: int` (≥0; default 0 = all)
  - `uncertainty_decay_factor: float` in [0,1] (default 0.0 means zero out when gate fails)
  - `pairwise_select: str` in {`heuristic`,`entropy`,`delta`} (default `heuristic`)
- Logging & shaping
  - `log_all_candidates: bool` (default false)
  - `soft_overlong_penalty_enabled: bool` (default false); `soft_overlong_penalty_weight: float` (required if enabled; ≥0)

Validation rules (fail‑fast examples):
- If `enable_clipped_grpo`, then `epsilon_low` and `loss_type_stage_b` must be set; raise: "When enable_clipped_grpo=true, set epsilon_low∈(0,1] and loss_type_stage_b∈{grpo,bnpo,dr_grpo}."
- If `enable_entropy_mask_stage_b`, require exactly one of quantile or threshold; raise if both/none set.
- If `soft_overlong_penalty_enabled`, require weight≥0.

## Utilities (new: src_post/tf/grpo_loss.py)
- `build_reply_mask(prompt_len: int, targets: Tensor) -> BoolTensor`
- `per_token_logps_from_logits(logits: Tensor, target_ids: Tensor) -> Tensor`
- `compute_ratio_and_clip(cur_logps: Tensor, old_logps: Tensor, eps_low: float, eps_high: float) -> tuple[Tensor,Tensor]`
- `apply_entropy_mask_from_logits(logits: Tensor, reply_mask: Tensor, *, top_quantile: Optional[float], min_threshold: Optional[float]) -> BoolTensor`
- `reduce_loss(per_token_loss: Tensor, reply_mask: Tensor, loss_type: str) -> Tensor` supporting {grpo,bnpo,dr_grpo}

All functions raise on shape mismatch, invalid ranges, or NaN/Inf.

## Runner Integration (src_post/runner.py)
Stage‑B (per reply):
1) Call `tf_sum_logprob_and_logits_over_response()` to get `(sum_logp, cur_logits, prompt_len)`.
2) Build reply mask; compute `per_token_cur_logps` over reply tokens. Set `old_logps = per_token_cur_logps.detach()` when effectively on‑policy (documented condition: grad_accum % steps_per_generation == 0 and num_iterations==1), else keep stored old_logps (future‑ready).
3) Ratios and clipping: `coef_1 = exp(cur − old)`, `coef_2 = clamp(coef_1, 1−ε_low, 1+ε_high)`.
4) Per‑token GRPO term: `per_token_loss = −min(coef_1*A, coef_2*A)`; add `+ beta*KL` using existing cached‑logits KL.
5) If `enable_entropy_mask_stage_b`, compute entropy on reply tokens and mask tokens below threshold/quantile.
6) Reduce by `loss_type_stage_b`, multiply by accum scale, and backward.
7) Metrics: track low/high/region clip ratios and entropy threshold used; log in scalars and JSONL.

Std==0 resampling:
- If reward std == 0 for K_B, attempt up to `max_resample_times` to re‑sample replies; on first std>0, use; else skip and count.

Logging:
- If `log_all_candidates`, emit all K_B replies with per‑function rewards and total reward.

Stage‑A (credit_assignment.py):
- Compute per‑image Δ_i = best_margin_variant − baseline_margin; sort; if `stage_a_top_m>0`, only backprop top‑M images.
- Uncertainty gate: if gate fails, set advantage to 0 or decay by `uncertainty_decay_factor`.
- Pairwise fallback: choose pairs by `pairwise_select` strategy; split advantage proportional to reply token counts, not fixed 0.5/0.5.
- Batch encodings per image where safe to reduce overhead.

## Reward Diagnostics & Shaping
- Continue normalization by sum(|w|). Add per‑function reward values into metrics for the selected best reply.
- Optional `soft_overlong_penalty`: detect when reply length reaches `max_new_tokens_stage_b` without EOS and add negative reward scaled by weight.

## Metrics
Add to existing aggregated metrics:
- `sb_clip_low_mean/sb_clip_low_min`, `sb_clip_high_mean/sb_clip_high_max`, `sb_clip_region_mean`
- `sb_entropy_threshold` (if mask enabled), `sb_entropy_mask_ratio`
- `std0_resample_count`
- Stage‑A: `sa_top_m_ratio`, `sa_pairwise_trigger_ratio`

## Error Handling
- Raise on invalid config combos, shape mismatches, NaN/Inf in rewards/loss. Preserve DDP barriers and keep fail‑fast semantics.

## Testing
- Unit: ratio clipping math, entropy mask selection, top‑M selection, pairwise split by token count, std==0 resampling.
- Integration: small synthetic dataset toggling features on/off; metrics present and finite.
- E2E: short smoke with K_A=K_B∈{2,3}, features disabled (baseline parity) and enabled (stability gains).

## Implementation Plan
1) Config keys + validation in `config.py` (explicit errors)
2) New `tf/grpo_loss.py` utilities + tests
3) Runner Stage‑B integration (clipping/masking, metrics, logging)
4) Credit assignment improvements (top‑M, pairwise split, gating)
5) Reward diagnostics + soft overlong penalty
6) Docs: update `src_post/README.md` and example YAML keys
