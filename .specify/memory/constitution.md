<!--
Sync Impact Report
- Version change: 2.0.0 → 3.0.0
- Modified principles:
  - Added "Simplicity & Minimal Surface Area (NON‑NEGOTIABLE)"
  - Added "Decoupling & Reuse"
  - Renamed "Versioning & Backward Compatibility" → "Versioning (No Backward Compatibility Guarantees)"
  - Clarified "Separation of Concerns" to avoid unnecessary file proliferation
- Added sections: None beyond new principles
- Removed sections: Backward‑compatibility requirements and migration obligations
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md (footer reference updated to v3.0.0)
  - ✅ .specify/templates/spec-template.md (no changes required)
  - ✅ .specify/templates/tasks-template.md (no changes required)
- Follow-up TODOs: None
-->

# Qwen2.5‑VL Multi-modal AI for Engineering Quality Inspection Constitution

## Core Principles

### I. Single Source of Truth & Explicit Contracts (NON‑NEGOTIABLE)
- Behavior is defined by explicit, versioned contracts (APIs, schemas, configs). Hidden defaults are prohibited.
- Validate inputs at module boundaries with actionable errors (expected vs actual, where it failed, how to fix).
- Deprecations (when used) MUST include migration notes and a removal timeline; otherwise deletion is acceptable per Versioning policy below.
- Rationale: Prevents ambiguity, enables safe integration, and supports automated checks.

### II. Fail‑Fast Validation & Observability
- Fail early on invalid inputs, states, or configuration; never swallow errors.
- Use structured logging and measurable health checks across critical paths.
- Error messages MUST include remediation hints and key parameters (sanitized).
- Rationale: Reduces mean‑time‑to‑diagnose and avoids silent drift.

### III. Configuration & Reproducibility
- All runtime behavior MUST be controlled by validated configuration objects (no implicit environment coupling).
- Project provides a single, deterministic setup path; scripts are non‑interactive and repeatable.
- Record relevant versions/seeds in outputs when determinism matters; document trade‑offs explicitly.
- Rationale: Ensures consistent results across machines and over time.

### IV. Separation of Concerns
- Keep clear boundaries between data I/O, processing, core logic, and CLI surfaces.
- Avoid top‑level side effects (no hidden global state, no I/O or GPU init at import time).
- Maintain decoupled boundaries without unnecessary file proliferation; prefer internal modules/namespaces over splitting across many files when clarity permits.
- Rationale: Improves readability, testing, and safe change isolation.

### V. Testing & TDD Discipline (NON‑NEGOTIABLE)
- Write tests before implementation for new behavior. Contract and integration tests precede core code.
- CI MUST run tests, lint, and type checks; any failure blocks merge.
- Tests MUST be deterministic where feasible and validate observable behavior, not implementation details.
- Rationale: Prevents regressions and enforces correctness as the system evolves.

### VI. Simplicity & Minimal Surface Area (NON‑NEGOTIABLE)
- Prefer the simplest solution that meets the requirement: minimal lines of code and minimal number of files.
- Remove dead code and speculative abstractions; no "just‑in‑case" features (YAGNI).
- Co‑locate related functionality when it improves clarity; keep public surfaces as small as possible.
- Rationale: Smaller surfaces are easier to maintain, review, and reason about.

### VII. Decoupling & Reuse
- Implementations should be disentangled and reusable: pure functions where possible, stable small interfaces, and no hidden coupling.
- Prefer reuse over duplication when it does not introduce disproportionate abstraction cost; keep utilities lightweight.
- Rationale: Encourages consistent behavior while preserving simplicity.

### VIII. Versioning (No Backward Compatibility Guarantees)
- Version tags and releases MAY follow semantic labels, but code/API backward compatibility is NOT guaranteed.
- Breaking changes can be introduced without deprecation when they materially simplify the system.
- When practical, provide brief migration notes; they are recommended but not required.
- Rationale: Prioritizes simplicity and maintainability over long‑term compatibility.

### IX. Security & Compliance
- Never commit secrets; use secure configuration channels and least‑privilege access.
- Sanitize and validate all external inputs; avoid unsafe eval/exec and insecure defaults.
- Track third‑party licenses and restrict usage to compliant terms; document data handling policies when applicable.
- Rationale: Protects users, data, and the project’s legal posture.

### X. Documentation & Traceability
- Public APIs and complex modules MUST have concise docstrings explaining purpose, inputs/outputs, and constraints.
- Significant decisions include a short rationale (ADR or equivalent) linked from relevant modules.
- Keep docs concise and accurate; prefer brevity over verbosity.
- Rationale: Maintains shared understanding with minimal overhead.

### XI. Performance & Resource Stewardship
- Establish baselines for critical paths; measure before optimizing and validate after changes.
- Favor linear‑time algorithms and minimal allocations in hot paths; surface limits and trade‑offs.
- Avoid premature optimization; prioritize clarity unless performance targets demand otherwise.
- Rationale: Delivers reliable performance without sacrificing maintainability.

### XII. Code Review & CI Gates
- All changes go through PR review with checklists covering principles in this Constitution.
- No direct pushes to the default branch; protected branches enforce required status checks.
- Reviews focus on correctness, clarity, boundary validation, contract compatibility, and unnecessary complexity.
- Rationale: Ensures consistent quality and knowledge sharing.

## Additional Constraints & Standards

- Coding Standards
  - Strong, explicit typing on public APIs; clear naming; guard clauses for error/edge cases.
  - Import hygiene (prefer absolute imports); avoid magic numbers; prefer immutable data structures.
  - Error messages include expected vs actual and remediation hints.

- Platform & Environment
  - Provide clear setup instructions and non‑interactive scripts; avoid user‑specific paths or machine‑local assumptions.
  - Scripts and CLIs accept explicit parameters and produce machine‑readable outputs when appropriate.

- Dependency Management
  - Keep dependency surface minimal; pin or constrain critical dependencies; include lockfiles/constraints where supported.
  - Document upgrade policy and test compatibility regularly.

- Observability
  - Prefer structured logs; expose minimal metrics for critical components.
  - Keep logging levels consistent; avoid noisy defaults in libraries.

## Development Workflow, Review Process & Quality Gates

- Constitution Check is a mandatory gate during planning and post‑design. Violations require justification
  (Complexity Tracking) or design simplification.
- TDD order is enforced: tests → implementation → refactor. Contract/integration tests come first.
- Reviews verify: explicit contracts, boundary validation, configuration strictness, reproducibility, documentation,
  CI status, and minimality (no unnecessary files/LOC). Any principle changes require documentation updates.

## Governance

- Supremacy: This Constitution supersedes other practices. Module docs may elaborate but MUST not contradict it.
- Amendments: Proposed via PR modifying `.specify/memory/constitution.md` with a Sync Impact Report.
  - Required approvals: at least one maintainer.
  - All dependent templates MUST be synced in the same PR.
- Versioning: Semantic versioning applies to this Constitution document only.
  - MAJOR/MINOR/PATCH indicate scope of governance changes, not backward compatibility promises for code.
- Compliance Reviews: Periodic audits verify adherence to contracts, validation, testing discipline, documentation
  accuracy, protected‑branch CI gates, and simplicity/minimality.

**Version**: 3.0.0 | **Ratified**: 2025-10-05 | **Last Amended**: 2025-10-05