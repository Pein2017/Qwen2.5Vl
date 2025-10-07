
# Implementation Plan: Dynamic GT‑Aware Generation Length for Dense RL

**Branch**: `002-dynamic-generation-length` | **Date**: 2025-10-07 | **Spec**: specs/002-dynamic-generation-length/spec.md
**Input**: Feature specification from `/specs/002-dynamic-generation-length/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
3. Fill the Constitution Check section based on the constitution document.
4. Evaluate Constitution Check section
5. Execute Phase 0 → research.md
6. Execute Phase 1 → contracts, data-model.md, quickstart.md
7. Re-evaluate Constitution Check section
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

## Summary
Implement a GT‑aware, per‑sample generation length cap and a `length_vs_gt` reward using tokenizer token counts for strict alignment. Allow truncated completions to contribute gradients (optionally mask only beyond the cap). Ensure distributed determinism and synchronization with Accelerate by computing identical caps across ranks, synchronizing buffer refresh, and padding/trimming tensors before collectives. Entrypoint: `scripts/run_dense_grpo.sh`.

## Technical Context
**Language/Version**: Python 3.12 (bf16 for model)  
**Primary Dependencies**: PyTorch, Hugging Face Transformers, Accelerate  
**Storage**: N/A (in‑memory training)  
**Testing**: pytest for contract/integration tests (to be added)  
**Target Platform**: Linux + CUDA (multi‑GPU)  
**Project Type**: single library/repo  
**Performance Goals**: Prevent slow‑rank timeouts; bound generation latency; stable memory footprint  
**Constraints**: Deterministic tokenizer across ranks; per‑sample cap computed from `meta.objects` only; no RNG in cap logic  
**Scale/Scope**: Multi‑GPU GRPO with K completions per sample

## Constitution Check
- Single Source of Truth & Contracts: Add explicit YAML knobs (`grpo.dynamic_length`, `rewards_config.length_vs_gt`) and logging contract keys.  
- Fail‑Fast & Observability: Validate config keys, log cap stats (`dynamic_length/*`), lengths, truncation flags.  
- Configuration & Reproducibility: No hidden defaults; behavior driven by YAML; environment pinned in `run_dense_grpo.sh`.  
- Separation of Concerns: Compute cap in RL buffer; rewards remain in rewards module; no CLI coupling.  
- TDD Discipline: Define contracts and quickstart; add tests in next phase.  
- Simplicity: Minimal new knobs; reuse existing modules.  
Status: PASS (no violations requiring justification).

## Project Structure

### Documentation (this feature)
```
specs/002-dynamic-generation-length/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
src_new/
├── rl/
│   ├── buffer.py                 # per-sample cap, masks, logging
│   ├── generation.py             # sampling K completions
│   ├── grpo_trainer.py           # trainer loop, cross-rank logic
│   ├── rewards/
│   │   ├── format_rewards.py     # add length_vs_gt
│   │   └── registry.py           # register reward
│   ├── runner.py                 # config plumb, reward wrapping
│   ├── validators.py             # tensor validations
│   └── utils.py                  # tokenizer EOS, builder
├── processing/
│   ├── coordinate_converter.py   # build GT canonical assistant text
│   └── conversation/*.py
configs/
└── dense_rl/*.yaml               # add dynamic_length knobs

scripts/
└── run_dense_grpo.sh             # entrypoint & env for distributed runs
```

**Structure Decision**: Single project; augment existing RL modules (`buffer.py`, `rewards/*`, `runner.py`, `grpo_trainer.py`) and configs. No new top‑level projects.

## Phase 0: Outline & Research
1. Unknowns / Decisions:
   - Deterministic tokenizer across ranks (version, settings) → enforce and document.  
   - Exact masking policy when EOS missing → keep tokens in loss; optional overflow‑only masking.  
   - Distributed sync with Accelerate → shared sample index, identical cap, synchronized buffer refresh, pad/trim before gather.  
   - Logging surface → namespacing `dynamic_length/{cap,cap_min,cap_max}`, truncation ratio, per‑rank timing.
2. Research Tasks:
   - Best practices for Accelerate collectives with variable lengths.  
   - Tokenizer determinism and environment pinning in multi‑GPU.  
   - Reward shaping for exponential overflow decay (`gamma`).
3. Consolidate in `research.md`.

**Output**: research.md generated.

## Phase 1: Design & Contracts
1. Data Model (`data-model.md`):
   - DynamicLengthConfig: enabled, estimator=tokenizer, alpha, eos_margin, min_cap, max_cap, hard_cap.  
   - MaskingPolicy: `mask_truncated_completions=false` (overflow masking knob removed).  
   - RewardsConfig.length_vs_gt: estimator=tokenizer, lower, upper, gamma, tail_numeric_weight.  
   - Logging keys: `dynamic_length/{mean_cap,min_cap,max_cap}`, `completions/{mean_length,min_length,max_length}`, `terminated_with_eos`, `truncated_flags`.
2. Contracts (`contracts/`):
   - Config contract (YAML keys & validation rules).
   - Logging contract (metric keys, meanings, units).
3. Quickstart (`quickstart.md`):
   - Enable dynamic_length in YAML; run with `scripts/run_dense_grpo.sh`; verify logs; watch NCCL stability.  
   - Accelerate tips: ensure same tokenizer and env variables; confirm identical caps across ranks.

**Output**: data-model.md, contracts/*, quickstart.md generated.

## Phase 2: Task Planning Approach
- Use TDD: author config/logging contracts and failing tests first; implement buffer cap + reward + logging; then distributed validations.  
- Parallelizable: rewards/registry update and docs; buffer.masks/cap logic; runner config plumb.

## Phase 3+: Future Implementation
- See template notes; beyond /plan scope.

## Complexity Tracking
(none)

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v3.0.0 - See `.specify/memory/constitution.md`*
