# Qwen2.5-VL Multi-modal AI for Engineering Quality Inspection Constitution
<!--
Sync Impact Report
- Version: 3.0.0 → 4.0.0
- Modified Principles:
  - Single Source of Truth & Explicit Contracts (HF-first enforcement added)
  - Fail-Fast Validation & Observability (expanded multimodal checks and logging parity)
  - Configuration & Reproducibility (clarified YAML schemas and seeding duties)
  - Separation of Concerns (refined module boundaries and reuse contracts)
  - Testing & TDD (codified fail-first test workflow with concrete tooling)
  - Simplicity & Minimal Surface Area (tightened toggle and dependency policy)
  - Decoupling & Reuse (explicitly linked to ConversationBuilder and dataset reuse)
  - Versioning (No Backward Compatibility Guarantees) (tied to SemVer governance)
  - Security & Compliance (added environment isolation and data handling rules)
  - Documentation & Traceability (mandated spec/plan/tasks synchronization)
  - Performance & Resource Stewardship (enumerated VRAM/bf16 expectations)
  - Code Review & CI Gates (linked to lint/type/test guardrails)
- Added Sections:
  - Execution Environment & Tooling Rules
  - Operational Workflow & Artifact Expectations
- Removed Sections: None
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/tasks-template.md
  - ⚠ Pending: Confirm whether `.specify/templates/commands/` should exist; create command templates or update guidance if repository intends to rely on cursor command manifests only.
- Follow-up TODOs:
  - TODO(RATIFICATION_DATE): Original ratification date not documented in repository history; maintainers must supply the canonical adoption date.
-->

## Core Principles

### Single Source of Truth & Explicit Contracts
All conversational data, span annotations, geometry metadata, and configs MUST flow through the typed builders in `src_new/processing` and `src_new/data`. `ConversationBuilder` and `ConversationProcessor` are the only sanctioned prompt generators; every run MUST apply `Qwen2_5_VLProcessor.apply_chat_template()` to map typed messages to tensors without hand-crafting `<|image_pad|>`. JSONL datasets, YAML configs, and SafeTensors checkpoints are authoritative; ad-hoc scripts that bypass builders or mutate tensors in place are prohibited. Rationale: enforcing a single contract surface keeps multimodal alignment verifiable and prevents drift across SFT, RL, and inference pipelines.

### Fail-Fast Validation & Observability
Every stage MUST validate tensor shapes, token counts, span coverage, geometry wrappers, and image token budgets before compute-intensive steps. Use the validators in `src_new/utils/tensor_validation.py`, `models/wrapper.py`, and augmentation sanity checks; raise actionable errors that identify expected vs. actual values. Training, inference, and RL runs MUST emit console and TensorBoard metrics in lockstep (loss, grad_norm, LR, ETA, reward components) so regressions surface immediately. Rationale: deterministic validation plus mirrored observability prevents silent corruption of multimodal data streams.

### Configuration & Reproducibility
Runtime behavior MUST be governed by explicit YAML files that conform to the frozen dataclasses in `src_new/config`. In-code defaults for hyperparameters, paths, dtype, or attention settings are forbidden. CLI entry points MUST require `--config` paths, document derived values, and log any auto-resolved paths. Seeds (`torch`, CUDA, numpy, python) MUST be set where deterministic replay is feasible; deviations require rationale in docs. Rationale: reproducible experiments and deployments depend on immutable, reviewable configuration artifacts.

### Separation of Concerns
Processing, data loading, modeling, losses, training loops, RL runners, augmentation, and utilities MUST stay in their dedicated modules with contracts expressed via typed dataclasses or small interfaces. Cross-module imports that bypass published APIs are disallowed. Any new feature MUST integrate at the correct layer (e.g., geometry transforms in `augmentation/`, loss weighting in `losses/token_grouping.py`) without bleeding responsibilities across packages. Rationale: clean boundaries preserve testability and unlock targeted performance or RL work without regressions.

### Testing & TDD
Implementations MUST begin with failing tests that capture expected behavior, covering unit cases (e.g., span extraction, geometry parsing), integration flows (ConversationBuilder → collator → DetectionModel), and regression harnesses when bugs are fixed. Tests MUST run under `pytest` with strict lint (`ruff`) and type (`pyright`) gates; CI merges are blocked until the full suite passes. Red-Green-Refactor discipline applies: write test, observe failure, implement minimal fix, then clean up. Rationale: TDD keeps geometry and multimodal invariants from regressing across rapid iteration.

### Simplicity & Minimal Surface Area
Every addition MUST justify itself against existing abstractions; new toggles, CLI flags, or datasets default to OFF until a concrete use case is documented. Prefer smallest viable feature set, avoid speculative support for unused object types or legacy coordinate tokens, and remove dead code promptly. Features that are merely “nice-to-have” (e.g., resumability toggles, tolerance knobs, auxiliary tooling) MUST be deferred until the core workflow proves value or explicitly demands them in the spec. Rationale: the multimodal stack is complex enough—minimal surfaces reduce cognitive load and validation burden while keeping implementation focused on primary outcomes.

### Decoupling & Reuse
Shared logic (conversation building, geometry text formatting, validation, reward shaping) MUST live in reusable helpers with stable interfaces, not duplicated inline. RL regeneration, SFT training, and inference pipelines MUST consume the same builders and validator stack to guarantee consistent behavior. When new capabilities are added, prioritize composable helpers over monolithic scripts. Rationale: reusable components accelerate iteration and keep guardrails consistent across modalities.

### Versioning (No Backward Compatibility Guarantees)
Artifacts follow SemVer. MAJOR bumps are required for breaking principle changes, altered geometry/label formats, or governance rewrites. MINOR covers new principles or material expansions; PATCH is restricted to clarifications. The project explicitly disclaims backward compatibility for intermediate checkpoints or JSONL schemas without a documented migration. Rationale: explicit versioning prevents accidental reliance on unstable interfaces.

### Security & Compliance
Keep development inside the curated workspace (`/data3/Qwen2.5-VL-main`) and approved dependencies. No secrets, tokens, or proprietary datasets may be committed. External data ingestion MUST validate geometry bounds and object type allowlists (`{bbu, bbu_shield, connect_point, label, fiber, wire}`) before use. Rationale: consistent handling of industrial imagery and annotations minimizes leakage risk.

### Documentation & Traceability
Every feature MUST ship with synchronized `spec.md`, `plan.md`, and `tasks.md` artifacts, each referencing the constitution version used during approval. Changes to training loops, conversation formats, or validators MUST update the relevant docs in `docs/` (e.g., Implementation Guide, Troubleshooting) so operators can trace rationale. Rationale: traceable documentation keeps cross-functional teams aligned and audit-ready.

### Performance & Resource Stewardship
Training and RL configs MUST specify expected VRAM, precision (`bf16` default, `fp16=false`), dataset size limits, and debug-step budgets. Performance regressions (throughput, memory spikes) MUST be detectable via logs and, when possible, guarded by tests or benchmarks. Rationale: controlled resource usage keeps experimentation safe on shared infrastructure.

### Code Review & CI Gates
All changes MUST undergo peer review with explicit confirmation that constitution principles are satisfied. PRs MUST pass lint, type-check, unit, and integration suites locally before CI. Reviewers MUST block merges that lack Constitution Check updates in plan/spec or that bypass mandated tests. Rationale: disciplined review ensures principles are continuously enforced.

## Execution Environment & Tooling Rules

- All development shells MUST activate the `ms` conda environment (`source ~/.bashrc && conda activate ms`) before running Python, training scripts, or notebooks. Running outside this env is prohibited.
- Use absolute or workspace-relative paths rooted at `/data3/Qwen2.5-VL-main`. Avoid ad-hoc temp directories that bypass repository layout.
- Standard entry points MUST remain the documented scripts: `bash scripts/run_new_train.sh` for training, `python -m src_new.inference ...` for inference, and `bash data_conversion/convert_dataset.sh` for dataset processing. Deviations require documented rationale.
- Tooling stack: Python from `/root/miniconda3/envs/ms/bin/python`, `ruff`, `pyright`, and `pytest` for quality gates. Introducing new global tooling requires governance approval.
- Never patch environment variables globally or modify shared conda installations without coordination; project-specific overrides live in scripts or configs.

## Operational Workflow & Artifact Expectations

- Feature work MUST originate from a filled `spec.md` (prioritized, independently testable user stories), proceed through an approved `plan.md` (with Constitution Check outcomes), and conclude with `tasks.md` that map tasks to user stories. Skipping artifacts violates traceability.
- `plan.md` Constitution Check MUST explicitly address every principle above, noting PASS, PASS with action, or NEEDS ACTION, along with remediation tasks.
- `tasks.md` MUST group tasks by user stories, flag parallel-safe items `[P]`, and include test tasks first when constitution requires coverage. Sample tasks in the template MUST be replaced.
- Docs under `docs/` and `specs/` MUST be updated when principles affect operations (e.g., new validation logs, environment shifts). Quickstarts MUST remain executable under the mandated environment rules.
- Runtime guidance (CLI help, README snippets) MUST reference the current constitution version and highlight any principle-driven constraints (e.g., bf16-only policy).

## Governance

- The constitution supersedes conflicting guidelines. Amendments require (1) documented rationale, (2) review by maintainers responsible for training, data, and tooling, and (3) an acknowledged migration or rollout plan when principles affect active workstreams.
- Semantic Versioning Policy: determine MAJOR/MINOR/PATCH impact before edits, document reasoning in the Sync Impact Report, and update the version line below accordingly.
- Compliance Reviews: Every new feature or major refactor MUST record the constitution version in specs and plans, rerun the Constitution Check after design, and resolve any NEEDS ACTION items before implementation merges.
- Enforcement: PR reviewers MUST block merges that violate principles, skip environment rules, or neglect required documentation updates. Automated CI SHOULD include lint, type-check, unit, integration, and (where feasible) smoke-training jobs aligned with the constitution.
- Exception Handling: Temporary deviations require written approval, a target rollback date, and explicit TODO entries referencing the responsible owner.

**Version**: 4.0.0 | **Ratified**: TODO(RATIFICATION_DATE): Original adoption date unknown—maintainers to supply official record. | **Last Amended**: 2025-10-08
