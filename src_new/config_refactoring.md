## Qwen2.5‑VL `src_new` Configuration Refactor

### 1. Purpose
Unify every configuration entry point behind a layered, sectioned schema that is explicit, testable, and ergonomic for experiments. The refactor removes implicit defaults, keeps the HF-first invariants intact, and makes cross-phase overrides predictable.

---
### 2. Design Principles
- **Layered, not duplicated**: inherit from a global base, optional phase base, then the experiment file. Layers are explicit and ordered.
- **Gated sections**: a feature block validates its fields only when `enabled: true`; stray sub-fields while disabled are rejected.
- **Fail-fast diagnostics**: every validation error names the JSON-pointer path (`features.augmentation.smart_resize.factor`).
- **No silent defaults**: all critical knobs live in YAML; code only computes convenience properties (e.g., `run_output_dir`).
- **HF parity preserved**: image token alignment, span masks, and grouping loss behaviour remain unchanged.

---
### 3. Target Layout & Layering
```
configs/
  base.yaml                  # global defaults
  phase_1/
    base.yaml                # phase defaults (optional)
    standard.yaml            # experiment overrides
  phase_2/
    ...
```
Layering order (left→right, later overrides earlier):
1. `configs/base.yaml`
2. Auto-discovered `configs/phase_{n}/base.yaml` when `features.phase.name` is set
3. Any paths listed in a local `extends:` array
4. The current YAML passed on the CLI (`scripts/train_new.py --config ...`)

Rules:
- All referenced files must exist; layering graph must be acyclic.
- If a phase name is declared but the phase base file is missing, abort with a path-qualified message.
- CLI stays backward compatible (`phase_3/standard` resolves to `configs/phase_3/standard.yaml`).

---
### 4. Module Structure
| Module | Responsibility |
| --- | --- |
| `config/schema.py` | `@dataclass` sections mirroring YAML (model, data, training, loss, features, output, runtime, advanced). Only non-critical helpers get defaulted values. |
| `config/validators.py` | Declarative predicates (`requires_if`, `forbids_when`, `exactly_one`, `in_set`, range checks) shared across modules. |
| `config/loader.py` | Layer resolution, YAML merge, path normalization, dataclass construction, error aggregation. |

---
### 5. Section Overview & Gating
Each section gate (`enabled: true/false`) controls validation of its payload. Required fields are listed as they become active.

- **model**: `model_path`, `attn_implementation`, `torch_dtype`, `use_cache`, `merge_size`, `max_pixels`.
- **data**: `data_root` (always required). `train_data_path`, `val_data_path`, and `teacher_pool_file` may be omitted if derivable via `DataResolver`.
- **training**: epochs, batch sizes, accumulation, optim LRs, warmup, weight decay, grad norm, scheduler, precision flags, collator, dataloader knobs, optional `conversation_variant_ratios`, optional LR overrides.
- **loss**: teacher/student/group weights (non-negative, sum > 0) and geometry bounds (`max_coord_value`).
- **features.augmentation**: requires `rng_seed`, `apply_to_teachers`, `lines_policy`, `debug_visualization`. If `smart_resize.enabled`, require `factor|min_pixels|max_pixels|max_ratio`. Optional sub-blocks (`image_geom`, `photometric`, `lines`, `criteria`, `type_policies`).
- **features.augmentation_schedule**: list of `{start_epoch, preset}`; optional per-step `smart_resize` block validated like the main one.
- **features.teacher_augmentation**: enabled → require `photometric` sub-block.
- **features.teacher_pairing**: enabled → require `teacher_ratio`, `dynamic_pairing_enabled`; if dynamic pairing is on, require `dynamic_pair_cross_bucket_explore_prob`.
- **features.phase**: enabled → require `name`. Enforce phase-specific constraints (e.g., `phase_2` needs `llm_top_k_block > 0 or -1`; `phase_1` forces `llm_top_k_block = 0` when provided).
- **features.checkpoint**: enabled → require `save_steps`, `save_total_limit`, `metric_for_best_model`, `greater_is_better`, `save_strategy`, `load_best_model_at_end`, plus exactly one of `min_interval_steps` or `interval_multiplier_of_eval_steps`.
- **features.logging**: enabled → require `logging_steps`, `report_to`, `disable_tqdm`.
- **features.evaluation**: enabled → require `strategy`, `eval_steps`.
- **output**: always require `run_name`, `output_dir`, `tb_dir`. Dataclass exposes computed `run_output_dir`, `tensorboard_dir`, `log_file_dir`.
- **runtime**: optional `seed` (non-negative).
- **advanced**: optional dev toggles (`span_include_im_end_in_labels`, `debug_alignment`, `packed_segment_isolation`, `augmentation_smart_resize_defaults`).

---
### 6. Validation Semantics
Implement predicates once in `validators.py`:
- `requires_if(path, value, fields)`
- `forbids_when(path, value, fields)`
- `exactly_one(paths)`
- `in_set(path, allowed)`
- Range helpers (`positive_int`, `probability`, etc.)

All errors are aggregated and surfaced as a single exception with bullet-listed issues. Examples:
- `features.phase.llm_top_k_block must be >0 or -1 when phase is phase_2`
- `features.augmentation.smart_resize.factor is required when smart_resize.enabled=true`

---
### 7. Integration Touch Points
- `scripts/train_new.py`: replace current loader call with `load_layered_config`, keep CLI stable.
- `src_new/data/dataset.py`: feature gates (`augmentation`, `teacher_pairing`), derived data paths.
- `src_new/models/wrapper.py`: train-time geometry bounds and dtype/attention settings from `model` + `loss.geometry`.
- `src_new/models/loss_manager.py`: grouped loss weights.
- `src_new/training/phase_freeze_manager.py`: read from `features.phase`.
- `src_new/training/callbacks.py`: checkpoint/logging/eval cadence from gated sections.
- `src_new/training/state_manager.py`: honour new checkpoint policy semantics.
- `src_new/inference.py`: ensure checkpoint configs still round-trip; layering applies when reloading from disk.

---
### 8. Migration Plan
1. **Bootstrap schema**: introduce the sectioned dataclasses, validators, and layered loader behind a feature flag; add unit tests for each gate.
2. **Author canonical YAMLs**: create `configs/base.yaml` and phase bases by lifting the shared content from today’s `phase_1/base.yaml`; update experiment YAMLs to `extends` the new bases.
3. **Dual-load period**: allow old flat configs for a short window (deprecated warning). Add a conversion script to help authors migrate.
4. **Flip default**: make the layered loader the default path; drop support for flat schema once the repo YAMLs are migrated.
5. **Cleanup**: remove legacy key paths and redundant validation logic scattered across modules.

Dependencies: update documentation (`UNIFIED_DOCUMENTATION.md`, training READMEs) to show the new layout and gating semantics.

---
### 9. Quality Strategy
- **Unit tests**: validator coverage for every gate, layering order, `extends` cycle detection, path normalization, phase-specific constraints, augmentation smart-resize rules.
- **Integration tests**: load minimal + augmentation-heavy configs; instantiate `Dataset`, `DetectionModel`, and run a dry Trainer init.
- **Regression**: run existing training smoke tests with the new schema; ensure checkpoint reload and inference config discovery keep working.

---
### 10. Risk & Mitigation
- **Config drift during migration** → enforce CI validation of every YAML in `configs/` using the new loader.
- **Experiment friction** → provide `scripts/convert_config.py` to upgrade flat files and document the layering recipe.
- **Hidden coupling** → inventory every `config.xxx` access before coding; replace with section-aware helpers to avoid partial migration.

---
### 11. Checklist Before Deprecating Legacy Loader
- [ ] All YAMLs under `configs/` use the layered schema and pass validation.
- [ ] Unit + integration suites cover the new loader and section gates.
- [ ] Documentation updated (training guide, inference notes, augmentation docs).
- [ ] Conversion script and migration notes committed.
- [ ] Legacy loader path removed and CI set to fail on old schema usage.

This refined plan keeps the configuration surface concise, makes layering intention explicit, and creates a single source of truth for validation and defaults across training, evaluation, and inference flows.
