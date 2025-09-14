# Requirements Document

## Introduction
Diagnose and fix severe Stage‑A summary degradation and zero/near‑zero GRPO variance in `src_post/` (Stage‑A poor summaries; Stage‑B appears reasonable). Ensure prompts, image flow, encoding, and rewards match SFT summary variant and Qwen2.5‑VL processor expectations.

## Alignment with Product Vision
Supports robust post‑training (GRPO) for BBU group QC by restoring high‑quality per‑image summaries and meaningful training signals.

## Requirements

### Requirement 1: Prompt Alignment (Stage‑A)
**User Story:** As a researcher, I want Stage‑A to use the exact SFT summary prompts, so that generated lines stay on‑domain and concise.

#### Acceptance Criteria
1. WHEN Stage‑A builds messages THEN system prompt SHALL equal `SUMMARY_SYSTEM_PROMPT` and user text SHALL equal `SUMMARY_USER_PROMPT` from `src_new/processing/templates.py`.
2. WHEN encoded THEN exactly one image SHALL be present (typed messages validator passes; no teacher turns).
3. WHEN decoding THEN one short Chinese line is produced without special tokens/coords.

### Requirement 2: Vision Token Flow Validation
**User Story:** As an engineer, I want internal checks guaranteeing image tokens and grid counts are non‑zero and consistent, so that vision features are used.

#### Acceptance Criteria
1. WHEN encoding Stage‑A THEN decoded prompt SHALL contain non‑zero `<|image_pad|>` and `image_grid_thw.size(0)==1`.
2. WHEN mismatch THEN the system SHALL log warnings with values and sample identifiers.

### Requirement 3: Reward Variance & Logging
**User Story:** As a trainer, I want non‑zero variance across Stage‑B rewards per prompt, so that GRPO computes meaningful advantages.

#### Acceptance Criteria
1. WHEN computing rewards for `K_B≥2` samples THEN std(rewards) SHALL be > 0.001 in at least 80% of updates on a small dataset; otherwise log `skip_std0` and inputs.
2. Reward composition SHALL depend on per‑reply text (`pred_label`, `reason`) and TF probabilities (`tf_p_pass/fail`).

### Requirement 4: Teacher‑Forcing Routing
**User Story:** As a developer, I want TF helpers to bypass `DetectionModel` loss assertions when labels are absent, so that forwards succeed.

#### Acceptance Criteria
1. TF helpers SHALL call `model.base_model` for logits when labels are not provided.
2. Stage‑B TF SHALL be length‑norm consistent with reward margin.

### Requirement 5: Config‑Driven Prompt Bias
**User Story:** As a user, I want to toggle minimal vs checklist Stage‑B prompts, so that bias can be controlled.

#### Acceptance Criteria
1. `use_mission_checklist` config toggles Stage‑B builder branch; JSONL records whether minimal prompt was used.

## Non-Functional Requirements

### Code Architecture and Modularity
- Keep changes localized to `src_post/` with clean interfaces.
- Maintain strict validations; fail fast on invalid states.

### Performance
- No more than 5% wall‑time overhead from added checks and logging.

### Reliability
- Deterministic seeds honored; no NaNs/Inf; tensors validated before generation/TF.

### Usability
- Clear, actionable error and warning messages with remediation hints.
