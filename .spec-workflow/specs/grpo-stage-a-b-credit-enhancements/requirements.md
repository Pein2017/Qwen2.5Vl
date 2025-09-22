# Requirements Document

## Introduction
Enhance the existing GRPO post-training pipeline in `src_post/` for Qwen2.5-VL to better support downstream AI quality control (group-level pass/fail) by improving: (1) Stage‑B GRPO loss robustness (ratio clipping and entropy masking), (2) Stage‑A credit assignment focus and gating, (3) reward shaping and diagnostics, (4) stability (std=0 resampling) and efficiency, and (5) observability.

## Alignment with Product Vision
Reinforce robustness and interpretability of group-level QC decisions while preserving the repo's HF‑first and fail‑fast invariants; minimize hard-coded rules; keep geometry/coord handling safe; and maintain selective unfreeze and simple ops for reproducibility.

## Requirements

### Requirement 1: Stage‑B clipped GRPO with optional entropy mask
**User Story:** As a researcher, I want Stage‑B to use clipped importance ratios and optional entropy masking so that updates are stable and focus on informative tokens.

#### Acceptance Criteria
1. WHEN computing Stage‑B loss THEN the system SHALL compute per‑token log-prob ratios r = exp(cur − old) and apply clip bounds [1−ε, 1+ε].
2. IF `entropy_mask_stage_b` is enabled THEN the system SHALL mask out tokens below an entropy threshold (quantile or absolute) on reply positions.
3. WHEN `loss_type` is set to `grpo|bnpo|dr_grpo` THEN the system SHALL aggregate per‑token loss accordingly.

### Requirement 2: Dynamic resampling for std==0 groups
**User Story:** As a practitioner, I want degenerate groups with zero reward variance to be resampled a limited number of times so that training steps are not wasted.

#### Acceptance Criteria
1. IF group reward std == 0 THEN the system SHALL attempt up to `max_resample_times` extra samples and accept the first batch where std > 0.
2. IF still std == 0 after retries THEN the system SHALL skip updates and increment a counter.

### Requirement 3: Stage‑A focused credit with advanced gating
**User Story:** As a practitioner, I want Stage‑A to update only the most impactful images and to gate low-confidence candidates so that gradients target useful evidence.

#### Acceptance Criteria
1. WHEN evaluating K_A candidates per image THEN the system SHALL compute Δ_i (margin change vs baseline) per image and expose per‑image Δ_i diagnostics.
2. IF `stage_a_top_m` > 0 THEN the system SHALL backprop only for top‑M images by Δ_i per group.
3. IF `uncertainty_gate` is enabled THEN the system SHALL zero the advantage when entropy gate fails (configurable decay factor allowed).
4. Pairwise fallback SHALL split advantage proportionally to token counts and support a `pairwise_select` strategy (heuristic|entropy|delta).

### Requirement 4: Reward shaping + diagnostics
**User Story:** As a maintainer, I want additional shaping options and per‑function diagnostics so that behavior can be tuned safely.

#### Acceptance Criteria
1. The system SHALL support an optional `soft_overlong_penalty` for overlong Stage‑B completions.
2. The system SHALL log per‑reward function values for the selected completion.
3. The system SHALL keep weight normalization by sum(|w|) and validate finite reward outputs.

### Requirement 5: Observability and outputs
**User Story:** As a user, I want richer logs/jsonl so I can understand and debug decisions.

#### Acceptance Criteria
1. Results JSONL SHALL optionally include all K_B replies with their rewards (guarded by `log_all_candidates`).
2. Metrics SHALL include clip ratios (low/high/region) and entropy thresholds when features are enabled.

## Non-Functional Requirements

### Code Architecture and Modularity
- Keep changes localized to `src_post/runner.py`, `src_post/credit/credit_assignment.py`, `src_post/tf/teacher_forcing.py`, and `src_post/rewards/*` with minimal cross-coupling.
- Add new config keys to `src_post/config.py` with explicit validations; no hidden defaults.

### Performance
- Batch TF forwards where feasible (Stage‑A candidate scoring) and avoid redundant processor work.
- Guard resampling with small caps to prevent runaway compute.

### Security
- No external network calls; deterministic RNG seeding preserved.

### Reliability
- Fail-fast on non-finite rewards/loss; preserve DDP sync barriers.

### Usability
- Clear YAML keys with help comments; logs show thresholds and counts for new features.