
# Implementation Plan: TRL→Manual GRPO Trainer Migration

**Branch**: `001-i-m-working` | **Date**: 2025-10-05 | **Spec**: /data3/Qwen2.5-VL-main/specs/001-i-m-working/spec.md
**Input**: Feature specification from `/specs/001-i-m-working/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (and generate tasks.md for this feature)
9. STOP - Ready for execution
```

**IMPORTANT**: This plan uses absolute paths under `/data3/Qwen2.5-VL-main/specs/001-i-m-working`.

## Summary
- Migrate RL from TRL to a manual GRPO trainer with HF-first parity (already implemented under `src_new/rl/`).
- Complete observability parity with console+TensorBoard metrics (avg reward, per-reward, grad_norm, ETA, epoch/step, LR).
- Validate explicit YAML config surfaces (bf16, model.attn_implementation, model.image_max_pixels, loss weights, layer_config, paths) with fail-fast errors.
- Provide short-run debug config guidance and smoke procedures; defer cross-rank K sampling to a follow-up.

## Technical Context
**Language/Version**: Python 3.12 (conda env `ms`)  
**Primary Dependencies**: PyTorch, transformers, TensorBoard, (no TRL)  
**Storage**: Filesystem (checkpoints, TB logs, JSONL datasets)  
**Testing**: pytest (unit + small integration), pyright/ruff for hygiene  
**Target Platform**: Linux (CUDA available, multi-GPU optional)  
**Project Type**: single  
**Performance Goals**: Stable VRAM (per-sample log-prob path), bf16 numeric stability, no OOMs on debug runs  
**Constraints**: bf16 enforcement; explicit YAML required; no backward compatibility guarantees; simplicity/minimality  
**Scale/Scope**: Single-node 1–8 GPUs; `sample_k` small for smoke; debug ≤ 20 steps

## Constitution Check
- Single Source of Truth & Explicit Contracts: PASS — YAML + CLI contracts; fail-fast validation required.
- Fail‑Fast Validation & Observability: PASS with action — add console+TB parity (grad_norm, ETA, epoch/step, LR).
- Configuration & Reproducibility: PASS — deterministic seeds and explicit config; document debug limits.
- Separation of Concerns: PASS — helpers are decoupled (`generation`, `buffer`, `logprobs`, `losses`, `validators`).
- Testing & TDD: NEEDS ACTION — add unit coverage for logging tags and YAML validation errors.
- Simplicity & Minimal Surface Area: PASS — keep changes additive within `src_new/rl/` and existing trainer state manager.
- Decoupling & Reuse: PASS — preserve reusable helpers; avoid duplication.
- Versioning (No Backward Compatibility Guarantees): PASS — note in docs; no BC promises.
- Security & Compliance: PASS — no secrets; local paths only.
- Documentation & Traceability: NEEDS ACTION — quickstart + logging contract docs.
- Performance & Resource Stewardship: PASS — per-sample log-prob, short debug runs.
- Code Review & CI Gates: PASS with action — ensure lint/type/test checks pass locally.

## Project Structure
```
/specs/001-i-m-working/
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (entities & config surfaces)
├── quickstart.md        # Phase 1 output (load/train/TB)
├── contracts/
│   ├── rl_runner.md     # CLI contract & required YAML keys
│   └── logging_metrics.md # TB tag map + console parity
└── tasks.md             # Phase 2 output (execution tasks)
```

**Structure Decision**: Single repo, single package (`src_new/rl`) with decoupled helpers. Docs and plans live under `specs/001-i-m-working`.

## Phase 0: Outline & Research
1) Unknowns / decisions to finalize:
- Cross-rank K sampling workflow: remainder policy, API toggles, gather semantics.
- Logging tag map parity with HF Trainer conventions (`train/loss`, `learning_rate`, `grad_norm`, `step`, `epoch`, ETA expression).
- Minimal debug YAML: concrete values for steps, sample_k, rewards, and dataset sizes.
2) Research tasks captured in `research.md` with decisions/rationales.

## Phase 1: Design & Contracts
1) Entities (`data-model.md`): `RLLoaderConfig`, `ManualTrainerConfig`, metrics schema, and logging keys.  
2) Contracts (`contracts/rl_runner.md`): CLI flags + required YAML keys with error messages.  
3) Logging contract (`contracts/logging_metrics.md`): TB tags and console parity checklist.  
4) Quickstart (`quickstart.md`): Loader smoke, short train, TB path verification; expected console/TB samples.

## Phase 2: Task Planning Approach
- Generate `tasks.md` with test-first items, logging parity, config validation tests, debug run scripts, and (optional) cross-rank sampling follow-up.
- Mark [P] for tasks that touch different files and can run in parallel.

## Phase 3+: Future Implementation
- Execute `tasks.md` in order; prioritize logging parity and validation tests before extended runs.

## Complexity Tracking
| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | — | — |

## Progress Tracking
**Phase Status**:
- [x] Phase 0: Research complete (skeleton created)
- [x] Phase 1: Design complete (skeleton created)
- [x] Phase 2: Task planning complete (tasks.md written)
- [ ] Phase 3: Tasks generated → in execution
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution v3.0.0 - See `.specify/memory/constitution.md`*
