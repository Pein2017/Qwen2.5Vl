# Feature Specification: Dynamic GT‑Aware Generation Length for Dense RL

**Feature Branch**: `002-dynamic-generation-length`  
**Created**: [DATE]  
**Status**: Draft  
**Input**: User description: "I want to implement the plan in @DYNAMIC_GENERATION_LENGTH.md"

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors, actions, data, constraints
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## User Scenarios & Testing (mandatory)

### Primary User Story
As an RL practitioner training dense captioning with GRPO, I want generation length to adapt per sample based on ground‑truth (GT) content so that small scenes don’t produce long, chaotic outputs, while dense scenes are allowed sufficient length—improving stability, reward quality, and distributed performance.

### Acceptance Scenarios
1. Given a sample with few GT objects, When generating K completions, Then the system caps max length near the GT‑estimated length and no completion exceeds this cap.
2. Given a sample with many GT objects, When generating K completions, Then the system allows a larger cap proportional to GT length, avoiding premature truncation.
3. Given enabled dynamic length, When a completion exceeds the plausible window, Then the length‑vs‑GT reward decreases sharply and the run logs the applied cap and completion lengths.
4. Given truncated completions (no EOS), When computing loss, Then tokens are not fully masked out; gradients still flow (optionally masking only tokens beyond the dynamic cap).
5. Given multi‑GPU training, When generating for a shared sample across ranks, Then all ranks use the same per‑sample cap and complete collectives without slow‑rank timeouts.

### Edge Cases
- What happens when GT length is extremely small (near zero)?
  - The cap still respects a configured minimum; reward uses safe denominators.
- How does the system behave if tokenization variability across ranks is possible?
  - [NEEDS CLARIFICATION: enforce deterministic tokenizer settings across ranks and versions]

## Requirements (mandatory)

### Functional Requirements
- **FR-001**: The system MUST compute a per‑sample generation length cap from GT (configurable: estimator, alpha, margins, min/max bounds).
- **FR-002**: The system MUST apply the computed cap as the `max_new_tokens` for all completions of that sample within the step.
- **FR-003**: The system MUST provide a continuous `length_vs_gt` reward that returns high scores within a tolerance window and penalizes overflows strongly.
- **FR-004**: The system MUST allow truncated completions (no EOS) to contribute gradients; it MUST NOT zero the entire completion mask by default.
- **FR-005**: The system MUST provide configuration to optionally mask only tokens beyond the dynamic cap, while keeping earlier tokens trainable.
- **FR-006**: The system MUST log per‑iteration stats for completion lengths and dynamic caps (min/mean/max), and expose `length_vs_gt` in console and TensorBoard.
- **FR-007**: The system MUST maintain distributed determinism for cap computation so all ranks use identical caps for the same sample.
- **FR-008**: The system MUST keep existing formatting rewards (wrappers/coords/separators) compatible and unaffected by the new cap.
- **FR-009**: The system MUST provide YAML configuration to enable/disable dynamic length and tune its parameters without code changes.
- **FR-010**: The system MUST degrade gracefully to existing fixed‑length behavior when dynamic length is disabled.

### Key Entities (include if feature involves data)
- **Sample Metadata (meta)**: Contains `objects` used to reconstruct canonical assistant text for GT‑based length estimation.
- **Dynamic Length Config**: User‑facing knobs (enabled, estimator, alpha, eos_margin, min_cap, max_cap, hard_cap, masking policy).
- **Rewards Config**: User‑facing knobs for `length_vs_gt` (estimator=tokenizer, lower, upper, gamma, tail_numeric_weight).

---

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous  
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified

---

## Execution Status

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

---

## Clarifications

### Session 2025-10-07
- Q: Which length estimator should we use for both the per-sample cap and the length_vs_gt reward? → A: Tokenizer token count (strict alignment)

