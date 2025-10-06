# Feature Specification: TRL→Manual GRPO Trainer Migration

**Feature Branch**: `001-i-m-working`  
**Created**: 2025-10-05  
**Status**: Draft  
**Input**: User description: "I'm working on the migration from TRL based GRPO training to manual trainer, referring to @plan.md ."

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

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies  
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## Current Status (review of src_new/rl)

- Manual trainer in place (TRL removed):
  - `src_new/rl/grpo_trainer.py`: BBUGRPOTrainer with distributed-aware dataloaders, generation buffering, optional KL, advantage normalization, temperature schedule, reward standardization, advantage clipping, and checkpointing via shared saver. Logs per-step histories (loss/reward/temperature/clip/advantages).
  - `src_new/rl/runner.py`: defaults to manual trainer; TRL path removed; "load" mode prints device/dtype/vocab JSON; requires explicit YAML sections (bf16 true, model.attn_implementation, model.image_max_pixels, loss weights, layer_config, paths).
- Decoupled helper modules (reusable):
  - `generation.py` (prepare inputs, sample_k), `logprobs.py` (sequential per-sample logps), `buffer.py` (generate_and_score, split_buffer), `losses.py` (GRPO, KL), `schedules.py` (temperature/beta), `validators.py` (image/token alignment, masks), `distributed.py` (barrier/broadcast/gather advantages).
- Datasets & eval:
  - Runner builds `RLDenseJSONLDataset` and `RLConversationContext` (HF-first pipeline parity).
  - `src_new/rl/eval.py` evaluates completions with reward registry and emits a JSON report.
- Feature checks:
  - `scripts/rl_feature_checks.py` updated to exercise manual trainer and emit JSON summaries; no TRL callbacks/imports.
- Cross-rank: advantage normalization implemented; shared cross-rank K sampling remains future work per plan.

---

## User Scenarios & Testing (mandatory)

### Primary User Story
As a maintainer of Qwen2.5‑VL training, I need a TRL‑free RL training path that preserves the end‑to‑end GRPO workflow (generation → rewards → advantages → update) so that I can run dense‑caption RL without the TRL dependency and with clear, fail‑fast validations and logging.

### Acceptance Scenarios
1. Given a valid RL YAML config on disk with required sections (bf16 true, model.attn_implementation, model.image_max_pixels, loss weights, layer_config, paths), when I run `python -m src_new.rl.runner --config /abs/config.yaml --mode load`, then the process exits successfully and prints a JSON line including device, dtype, and tokenizer/model vocab counts.
2. Given a short debug RL YAML with absolute data paths and `bf16: true`, when I run `python -m src_new.rl.runner --config /abs/config.yaml --mode train` for 20 steps/1 epoch, then TensorBoard logs are written under `tb_dir/run_name` and a checkpoint (with tokenizer/processor) is saved to `output_dir`.
3. Given the feature‑check harness, when I run `python scripts/rl_feature_checks.py --config /abs/config.yaml`, then it executes the manual trainer path and emits a JSON summary of rewards/metrics without importing or requiring TRL.
4. Given a multi‑GPU run where advantage normalization is enabled in YAML, when I run training, then logs indicate cross‑rank advantage normalization (combined stats) and the run completes without distributed errors.
5. Given an invalid or incomplete YAML (e.g., missing required loss weights or model.attn_implementation), when I run load/train, then it fails fast with actionable error messages naming the missing key(s).
6. During a short debug train run, console logs at each logging step display step/epoch, ETA, learning rate, grad_norm, loss, reward mean/std, and per‑reward aggregates; TensorBoard contains the same scalars under conventional tags (e.g., `train/loss`, `reward`, `rewards/<name>/mean`, `grad_norm`, `learning_rate`, `eta_minutes`, `step`, `epoch`).

### Smoke commands
```bash
# 1) Module import/type check
python -m py_compile $(ls src_new/rl/*.py) && python -m py_compile scripts/rl_feature_checks.py

# 2) Loader smoke (prints JSON)
python -m src_new.rl.runner --config /abs/configs/dense_rl/debug.yaml --mode load

# 3) Short train (debug) → logs + checkpoint
python -m src_new.rl.runner --config /abs/configs/dense_rl/debug.yaml --mode train

# 4) Feature checks (JSON summary)
python scripts/rl_feature_checks.py --config /abs/configs/dense_rl/debug.yaml
```

### Edge Cases
- Rewards degenerate (e.g., `sample_k < 2` or all completions identical) → training still runs; logs show non‑crashing behavior and clearly indicate zero/near‑zero variance.
- Missing/invalid YAML keys (paths, loss weights, generation knobs) → process fails fast with actionable error messages.
- Resource limits (GPU OOM, disabled bf16) → process fails fast with clear remediation guidance.
- Distributed on/off and DeepSpeed toggles → runs must either proceed successfully or fail with explicit, actionable diagnostics.

## Requirements (mandatory)

### Functional Requirements
- **FR‑001**: Provide a TRL‑free RL training path as the default; legacy TRL code paths are removed/disabled.
- **FR‑002**: Support a "load" mode that validates components and prints a one‑line JSON with device, dtype, and vocab parity.
- **FR‑003**: Produce TensorBoard logs under `tb_dir/run_name` and save HF‑compatible checkpoints (including tokenizer/processor) to `output_dir` during short runs.
- **FR‑004**: Feature‑check harness runs end‑to‑end using the manual trainer and outputs a JSON summary of core metrics without TRL callbacks/imports.
- **FR‑005**: Configuration is explicit and validated; missing required keys (data paths, loss weights, generation knobs, layer config, model attn/image limits, bf16) fail fast with actionable messages.
- **FR‑006**: Support K‑sampling, temperature/top‑p, optional KL (beta), optional advantage normalization, and masking for truncated completions; expose histories for diagnostics.
- **FR‑007**: Enforce bf16 policy when requested; provide clear errors when unsupported.
- **FR‑008**: Log minimum metrics to console and TensorBoard: loss, reward mean/std, per‑reward aggregates (`rewards/<name>/{mean,std}`), completion length stats, clipped ratio, temperature, grad_norm, learning_rate, eta_minutes, global_step, epoch.
- **FR‑009**: Keep helper modules decoupled and reusable (generation, scoring, log‑prob, losses, schedules, distributed utilities) with no hidden coupling.
- **FR‑010**: Prioritize simplicity and minimality per constitution: minimal files/LOC consistent with clarity; no backward compatibility guarantees required.
- **FR‑011**: Distributed runs provide at least advantage normalization across ranks; cross‑rank prompt K‑split is future work.
- **FR‑012**: Console logs MUST mirror core TensorBoard scalars each logging step using conventional tags compatible with HF Trainer dashboards.
- **FR‑013**: Provide ETA in hours:minutes computed from elapsed time and remaining steps; include step/epoch counters in console and TB.
- **FR‑014**: RL is single‑turn only (no teacher‑student pairing); assume standard single‑image batches per sample in RL modules.

### Key Entities (include if feature involves data)
- **RL Training Run**: A single execution configured by a YAML file producing logs and checkpoints.
- **Reward Function**: A named scalar function per completion whose weighted sum contributes to advantages.
- **Generation Buffer**: A cached set of K completions per prompt with masks and per‑token metadata used for loss computation.
- **Checkpoint Artifact**: A saved model bundle including tokenizer/processor and minimal metadata to reload for inference/eval.
- **Metrics Telemetry**: Logged scalars and short textual previews sufficient to diagnose training health.

### Observability & Logging
- Console and TensorBoard MUST include at minimum:
  - `train/loss`, `reward` (mean), `reward_std`
  - `rewards/<name>/mean`, `rewards/<name>/std` for each enabled reward
  - `completions/mean_length`, `completions/min_length`, `completions/max_length`, `completions/clipped_ratio`
  - `temperature`, `grad_norm`, `learning_rate`, `eta_minutes`, `step`, `epoch`
- Logs are emitted at `logging_steps` cadence and reflect distributed aggregation policy where applicable (e.g., reward mean across ranks).

### Non‑Goals (this phase)
- Cross‑rank K sampling for a shared prompt across GPUs (advantage normalization only is included).
- TRL fallback paths (fully removed).
- Backward compatibility guarantees for RL configs (simplicity first).

---

## Clarifications

**Q: Which reward metrics must be non-zero for smoke test acceptance?**
A: Only formatting rewards (parse, wrappers, coords, separators, vocab) must be non-zero. Detection/geometry rewards (bbox_giou, quad_l1, line_l1, etc.) can be zero for smoke testing.

**Q: What are the concrete step/epoch limits for debug runs?**
A: 20 steps, 1 epoch maximum for smoke testing.

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

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
*Updated by main() during processing*

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

---
