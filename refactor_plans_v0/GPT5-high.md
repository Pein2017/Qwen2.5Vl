## Qwen2.5‑VL — High‑Level Plan to Merge `src_rl` into `src_new/rl`

### Objectives
- **Unify pipelines**: Consolidate RL GRPO code (`src_rl`) into `src_new/rl` to remove duplication with SFT (`src_new`).
- **Preserve behavior**: Keep current CLIs, configs, and scripts working; provide import shims and deprecation notices.
- **Enforce invariants**: Centralize the HF‑first invariants (placeholder counts, THW/token alignment, assistant span masking) and validation logic in `src_new`.

### Non‑Goals
- No new algorithms or feature additions; parity first.
- No changes to dataset format or conversation templates.

### Guiding Principles
- **HF‑first**: typed messages → template render → processor tensors. Never hand‑craft `<|image_pad|>`.
- **Single source of truth**: conversation building, model wrapper, generation, and validations live in `src_new` and are imported by both SFT and RL paths.
- **No hidden defaults**: configs remain explicit; missing keys fail fast with clear errors.
- **Dynamic geometry**: keep geometry tokens and coordinate ranges dynamic; avoid hard‑coded IDs.

---

### Target Repository Structure (post‑merge)
- `src_new/`
  - `processing/` (authoritative conversation builders, span detection)
  - `data/` (datasets, collation; emits `pixel_values`, `image_grid_thw`)
  - `models/`
    - `wrapper.py` (authoritative `DetectionModel`, validations, loss plumbing)
  - `losses/` (grouped CE; shared utilities)
  - `training/` (SFT trainer)
  - `inference/` (training‑matched generation CLI and library helpers)
  - `rl/`
    - `runner.py` (entrypoint; imports shared components from `src_new`)
    - `config.py` (RL config schema + adapters)
    - `rewards/` (registry, group‑QC rewards, z‑score/advantage)
    - `prompting/` (Stage‑A/B prompts; reuse `processing` builders)
    - `algorithms/` (GRPO core, KL/reference policy optional)
    - `utils/` (RL‑specific helpers only; no duplicates of `src_new/utils`)
- `scripts/`
  - `run_group_qc_rl.sh` (points to `python -m src_new.rl.runner`)
  - `run_new_train.sh` (unchanged)
- `configs/rl/` (unchanged file locations; may add deprecation headers where needed)

Compatibility shims for deprecation period:
- `src_rl/__init__.py`, `src_rl/runner.py`, etc. → thin wrappers importing from `src_new.rl.*` with warnings.
- If `src_post` also exists for RL, mirror shims there to the same `src_new.rl` entrypoints.

---

### Module Mapping (source → destination)
- `src_rl/runner.py` → `src_new/rl/runner.py`
- `src_rl/config.py` → `src_new/rl/config.py` (or `config_schema.py` if split)
- `src_rl/rewards/*` → `src_new/rl/rewards/*`
- `src_rl/prompting/*` → `src_new/rl/prompting/*`
- `src_rl/algorithms/*` → `src_new/rl/algorithms/*`
- `src_rl/utils/*` → merge into `src_new/utils` only if not duplicating; else keep under `src_new/rl/utils`.

Authoritative shared imports in RL after the move:
- Conversation building: `from src_new.processing.conversation_processor import ConversationProcessor`
- Dataset/collation: `from src_new.data.dataset import ...`
- Model wrapper: `from src_new.models.wrapper import DetectionModel`
- Generation helpers: `from src_new.inference import ...` (same decoding path as SFT)

---

### Configs and CLI Compatibility
- Keep existing RL configs in `configs/rl/*.yaml` and scripts:
  - Recommended: `bash scripts/run_group_qc_rl.sh /abs/path/to/configs/rl/group_qc_grpo.yaml`
  - Direct: `/root/miniconda3/envs/ms/bin/python -m src_new.rl.runner --config /abs/path/to/configs/rl/group_qc_grpo.yaml`
- Maintain key names; add an adapter layer in `src_new/rl/config.py` to map legacy keys to unified internal dataclasses.
- Validate: `attn_implementation="eager"`, `bf16`, batch size 1 by default (or enforced via config), and explicit paths. No in‑code defaults.
- Provide a clear error if both RL and SFT try to override shared settings inconsistently.

---

### Shared Invariants (centralized in `src_new`)
- Placeholder count in rendered text equals number of images; error otherwise.
- `pixel_values` rows equal `sum_i (t_i*h_i*w_i)`; enforce `image_grid_thw` shape `[num_images, 3]`.
- `<|image_pad|>` count equals expected `(t*h*w)//(merge_size**2)` totals.
- Assistant spans include `<|im_end|>` and labels outside spans are `-100`.
- Coordinate‑token mode is legacy only; disabled by default in `src_new`.

RL reuses the same validators through the `DetectionModel` forward path and dataset collation.

---

### Training Paths (after merge)
- SFT: `src_new.training` uses grouped CE (caption/grounding/formatting masks).
- RL: `src_new.rl.algorithms.grpo` computes rewards, z‑scores to advantages, optimizes `-A * logp(y)` with optional KL to reference.
  - Stage‑A summaries (optional); Stage‑B decision; decode‑time masking for geometry is optional.
  - RL sampling and evaluation call the same generation helpers as SFT to ensure parity.

---

### Inference and Generation (shared)
- Single set of generation utilities under `src_new.inference` used by both SFT eval and RL sampling.
- CLI preserved:
  ```bash
  /root/miniconda3/envs/ms/bin/python -m src_new.inference \
    --model_path /abs/path/to/checkpoint \
    --data_root /abs/path/to/data \
    --dataset {train|val} \
    --output_file /abs/path/to/out.jsonl \
    [--config_path /abs/path/to/config.yaml] \
    [--generation_variant {dense|summary}] \
    [--force_eager_attention]
  ```

---

### Telemetry, Logging, and Checkpointing
- Use `src_new/utils` logging across both paths; unify log levels and formats.
- Save processor/tokenizer with checkpoints to preserve HF expectations.
- Keep loss diagnostics finite and grouped component reporting consistent.

---

### Test & Health‑Check Plan (must pass before deprecation)
- Unit tests:
  - Conversation building parity: `src_rl` prompts vs `src_new` builders yield identical rendered text and image placeholder counts.
  - Dataset/Collator: shape checks (`pixel_values`, `image_grid_thw`), span masks, `<|im_end|>` inclusion.
  - DetectionModel: validator failures on intentionally malformed inputs.
  - RL rewards: registry wiring and deterministic small cases.
- Integration tests:
  - Short SFT sanity run (50–100 steps) using `configs/phase_3/standard.yaml`.
  - Short RL sanity run (100–200 updates) using `configs/rl/group_qc_grpo.yaml` via `scripts/run_group_qc_rl.sh`.
  - Inference CLI parity check across SFT and RL models.

---

### Migration Phases and Rollout
- **Phase 0 — Inventory (Day 0)**
  - Freeze `main`. List overlapping modules between `src_rl` and `src_new`.
  - Decide on any small config key adapters required.
- **Phase 1 — Scaffolding (Day 1)**
  - Create `src_new/rl/*` package and move `runner`, `config`, `rewards`, `prompting`, `algorithms`, minimal `utils`.
  - Wire imports to shared `src_new` components (processing, data, models, inference).
- **Phase 2 — Script & Config Wiring (Day 1–2)**
  - Update `scripts/run_group_qc_rl.sh` to call `python -m src_new.rl.runner`.
  - Keep `configs/rl/*` unchanged; add config adapters and explicit validation.
- **Phase 3 — Testing & Parity (Day 2–3)**
  - Run unit + integration tests; fix any invariants/regressions.
  - Benchmark sampling/eval parity between old RL and unified RL.
- **Phase 4 — Shims & Deprecation (Day 3)**
  - Add deprecation shims under `src_rl/*` (and `src_post/*` if present) that import from `src_new.rl.*` with warnings.
  - Announce deprecation window and cut a release tag.
- **Phase 5 — Cleanup (Post‑window)**
  - Remove shims after deprecation window; update docs to point exclusively to `src_new/rl`.

---

### Risks and Mitigations
- **Hidden duplicated utilities**: risk of diverging helpers. Mitigation: forbid duplicates; centralize in `src_new/utils` and reference from RL.
- **Config drift**: legacy keys may not map 1:1. Mitigation: adapter + validation layer; document mapping.
- **Generation parity**: subtle decoding diffs. Mitigation: single generation module + golden outputs.
- **Import cycles**: moving into `src_new` can introduce cycles. Mitigation: keep RL‑specific code under `src_new/rl/*` and import only from `src_new` leaf modules.

---

### Acceptance Criteria
- `bash scripts/run_new_train.sh` works unchanged.
- `bash scripts/run_group_qc_rl.sh /abs/path/to/configs/rl/group_qc_grpo.yaml` launches and trains using `src_new.rl.runner`.
- All HF‑first invariants enforced identically across SFT and RL.
- RL and SFT share a single `DetectionModel`, conversation builder, dataset/collator, and generation utilities.
- Existing configs and checkpoints load without modification.

---

### Owner and Timeline
- **Owner**: Core training team (SFT owner + RL owner as reviewers).
- **ETA**: 3 days to parity (Phases 0–4), 1–2 weeks deprecation window.
