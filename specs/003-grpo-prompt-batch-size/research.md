# Research: GRPO Prompt Batch Size Refactoring

**Date**: 2025-10-08  
**Participants**: Training/RL maintainers, tooling engineers  
**Objective**: Determine standards-aligned strategies for multi-prompt GRPO accumulation while respecting the project constitution.

---

## Decision 1: Orchestration Framework for Multi-Prompt Accumulation
- **Decision**: Use Hugging Face Accelerate on top of PyTorch DDP to coordinate gradient accumulation across sequential trajectories.
- **Rationale**: Accelerate provides battle-tested gradient accumulation, optimizer synchronization, and mixed-precision utilities without replacing our existing Trainer abstractions. It integrates with the current `ms` environment and supports `torch.distributed` under the hood, keeping code concise and avoiding bespoke communication primitives.
- **Alternatives Considered**:
  - **Pure PyTorch DDP (Manual)**: Requires custom gradient buckets, manual all-reduce for every trajectory, and bespoke error handling. Higher maintenance burden.
  - **DeepSpeed ZeRO**: Powerful but introduces additional configuration, optimizer partitioning, and potential conflicts with sequential rollout logic; overkill for current scale.

## Decision 2: Prompt Batch Scheduling Strategy
- **Decision**: Implement a prompt batching coordinator that pulls prompts in deterministic chunks (size = prompt_batch_size) using the existing sampler, without introducing additional data balancing requirements.
- **Rationale**: Reusing established sampling keeps sequencing deterministic and resume-friendly while avoiding new data dependencies; the focus remains on runtime accumulation and gradient stability.
- **Alternatives Considered**:
  - **Independent prompt pool per rank**: Risks category drift and duplicated prompts.
  - **Global random sampling without deterministic ordering**: Undermines reproducibility and complicates resume logic.

## Decision 3: Telemetry and Observability Enhancements
- **Decision**: Extend the existing RL logging utilities to emit per-accumulation metrics (prompt batch fill percentage, smoothed reward average, lagging ranks) to both console and TensorBoard.
- **Rationale**: Satisfies constitution’s fail-fast observability requirement and supports User Story 2. Leveraging current logging keeps implementation concise while adding new scalar tags.
- **Alternatives Considered**:
  - **External monitoring stack (e.g., Prometheus)**: Would add deployment/runtime complexity outside current scope.
  - **Ad-hoc print statements**: Inadequate for long-running jobs and violates observability standards.

## Decision 4: Handling Invalid Trajectories
- **Decision**: Introduce a validation layer that inspects reward tensors for NaN/Inf/outlier values before aggregation, assigns reward `0` when sub-rewards fail, and records the reason in telemetry plus a side report.
- **Rationale**: Aligns with fail-fast validation and ensures corrupt data does not bias gradients. Explicit zeroing keeps reward trends meaningful and trustworthy.
- **Alternatives Considered**:
  - **Silently clamping rewards**: Hides data issues and conflicts with transparency principles.
  - **Aborting entire accumulation on first invalid trajectory**: Too aggressive; controlled quarantine keeps job running while surfacing actionable warnings.

## Decision 5: Partial Batch Handling
- **Decision**: Apply a `drop_last` policy so incomplete prompt sets are discarded without affecting optimizer steps or gradients.
- **Rationale**: Prevents partial updates, simplifies testing, and removes the need for resumability metadata. Logging the discard count keeps operators informed.
- **Alternatives Considered**:
  - **Abort entire run on partial batch**: Excessively disruptive near epoch boundaries.
  - **Pad with duplicate prompts**: Risks biasing reward trends and complicates traceability.

---

**Open Questions**: None (all clarifications resolved).  
**Next Step**: Proceed to Phase 1 design deliverables (data-model, contracts, quickstart).
