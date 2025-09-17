# Qwen2.5‑VL JSON Training Pipeline — Architecture Details (src_new_json)

**Complete Reference for the JSON‑first fine‑tuning pipeline (no coordinate or wrapper tokens)**

---

## Executive Snapshot (2025‑09)

- Entrypoints:
  - Training: `/root/miniconda3/envs/ms/bin/python scripts/train_new_json.py --config <name> --log_level {INFO|DEBUG}`
  - Inference: `/root/miniconda3/envs/ms/bin/python -m src_new_json.inference --config_path /abs/config.yaml --model_path /abs/checkpoint --data_root /abs/data --output_file /abs/out.jsonl`

- Critical invariants:
  - JSON‑only conversations: assistant output must be a valid JSON array; no special wrapper or coordinate tokens are used in content.
  - Fast tokenizer required with `offset_mapping`; a valid chat template must be available (tokenizer or processor).
  - Image alignment: decoded `<|image_pad|>` count equals number of images; shapes are validated.
  - Labels/spans: assistant spans come from regex+offsets; include immediate `<|im_end|>`; labels outside spans are `-100`.
  - Data collator: only `standard` is supported; packed collation is disabled (rejected by config validation).
  - Dynamic pairing: per‑epoch bucketed random pairing; eval/inference are forced single‑turn.

- Data → Model → Loss → Checkpoint (HF‑first):
  - JSONL dataset → `ConversationBuilder` (JSON variants) → Qwen2.5‑VL processor → `DetectionModel` wrapper (validations) → grouped CE over JSON (teacher/student split) → local aggregation → SafeTensors + tokenizer/processor saved; best checkpoint rotation enabled.

- Freezing phases (separate runs):
  - `phase_1`: train merger (+ optional last‑K LLM blocks); vision largely frozen.
  - `phase_2`: unfreeze last‑K LLM (and optionally last‑K vision); merger trainable.
  - `phase_3`: broaden unfreezing conservatively; patch_embed typically frozen.

---

## Training Pipeline (step‑by‑step)

1) Parse & validate config
- Load `configs/` YAML; deep‑merge base + override via `load_config`.
- Normalize/validate paths; set rank‑aware logging; seed RNGs.
- Enforce JSON‑mode constraints: collator must be `standard`; coordinate/auxiliary token toggles are removed.

2) Tokenizer/processor & model
- Load fast tokenizer and `Qwen2VLImageProcessor`; apply `max_pixels` from YAML.
- Build `Qwen2_5_VLForConditionalGeneration` with dtype/attention settings; prefer `flash_attention_2` for training.
- Wrap with `DetectionModel` for strict validations and training‑matched behavior.

3) Dataset & conversations (JSON)
- `Dataset` reads JSONL; validates per‑sample structure; applies optional augmentation.
- Dynamic pairing: per‑epoch `_episode_map` is (re)built; in `__getitem__`, inject a single teacher from the same major‑type bucket when selected.
- `ConversationBuilder` renders JSON‑first messages; processor applies the chat template; text/tensors are produced for training.

4) Collation & shapes
- Only `collator_standard` is supported; batches are padded to max length.
- Emit `input_ids`, `attention_mask`, `labels`, and when images exist `pixel_values`, `image_grid_thw`.

5) Forward & validations (wrapper)
- Validate text/image shapes and token/image counts; enforce `<|image_pad|>` count equals number of images.
- Use grouped CE path with explicit teacher/student span masks; bypass base loss when spans are provided.

6) Loss computation (JSON grouping)
- Single CE pass; role masks (teacher/student) split assistant spans.
- JSON grouping engine parses assistant JSON to build char spans, maps to token spans via offset mapping, and creates group masks:
  - caption (string values under `label`)
  - grounding (numeric literals within `box_points|quadrilateral_points|line_points` arrays)
  - formatting (JSON braces/brackets/colons/commas/quotes, keys, whitespace, and residual assistant tokens)
- Per‑role, per‑group CE means are weighted by `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight`.
- Report grouped metrics with `teacher_*/student_*` prefixes; verify decomposition (group sums ≈ role LLM loss; roles sum to total).

7) Phases & optimizer
- `PhaseFreezeManager` applies `phase_name` policy; optional last‑K unfreeze for LLM/vision; keep patch_embed frozen by default.
- Differential LRs recommended (vision < LLM < merger); enable gradient checkpointing for memory.

8) Logging, evaluation, checkpointing
- `TrainingStateManager` aggregates losses locally; metrics logged in a concise, stable schema.
- `CheckpointSaver` writes SafeTensors, tokenizer/processor; best checkpoint via atomic copy with rotation.

---

## Dynamic Contrastive Pairing (training‑time feature)

- Purpose: Provide stronger gradients via contextual teacher examples without changing the core loss path.
- Policy (per epoch):
  - Every sample acts as the student once.
  - With probability `teacher_ratio`, the student receives one teacher from the same major‑type bucket (by `objects[*].desc` prefix → `{bbu,bbu_shield,connect_point,fiber,wire,label}`; stable tie‑breaks; `misc` fallback).
  - If the bucket is empty, fallback to uniform over all samples (excluding self).
  - Optional cross‑bucket exploration: sample from all samples with a small probability.
  - Conversation order: teacher → student; only the last (student) turn is trained.
- Eval/inference: forced single‑turn (`teacher_ratio = 0.0`).

---

## Data Schema & Rules (authoritative summary)

- Input JSONL per sample:
  - `images: [str]` (usually one path per sample)
  - `objects: [dict]` with exactly one geometry per object: `bbox_2d | quad | line`, and `desc` (hierarchical, slash/comma conventions)
  - `width`, `height`, optional `meta`
- Conversation assistant JSON (rendered text):
  - `[{"box_points": [[x1,y1],[x2,y2]], "label": "..."}, {"quadrilateral_points": [...]}, {"line_points": [...]}]`
- Validation:
  - One geometry per object; geometry matches object type; desc follows template rules.
  - Remarks are free text within `desc` (positional last segment) but never leak `<|...|>` or coordinates.

---

## Configuration (strict, JSON mode)

- Model/runtime: `model_path`, `attn_implementation`, `torch_dtype`, `use_cache`.
- Training: `num_train_epochs`, `per_device_{train,eval}_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `vision_lr`, `merger_lr`, `llm_lr`, `warmup_ratio`, `weight_decay`, `max_grad_norm`, `lr_scheduler_type`, `gradient_checkpointing`, `bf16|fp16`.
- Data: `train_data_path`, `val_data_path`, `data_root`, `max_total_length`, `num_teacher_samples`, `max_dataset_size`.
- Collator: `collator_type: standard` (packed is disabled by validation).
- Dynamic pairing:
  - `teacher_ratio: float [0,1]` (applies to training only; eval forced to 0)
  - `dynamic_pairing_enabled: bool`
  - `dynamic_pair_target_assignment: {current|random|opposite}` (current is used by the engine)
  - `dynamic_pair_cross_bucket_explore_prob: float [0,1]`
  - `dynamic_pair_temperature`
- Vision processing: `merge_size`, `max_pixels`.
- Loss weights: `teacher_loss_weight`, `student_loss_weight`, `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight` (≥ 0; not all zeros).
- Augmentation: `use_aug`, `augmentation` (preset), `teacher_augmentation` (photometric‑only), `augmentation_schedule`.
- Logging & Trainer: `eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`, `save_total_limit`, `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`, `logging_steps`, `report_to`, `disable_tqdm`.
- Dataloader: `dataloader_num_workers`, `pin_memory`, `prefetch_factor`, `remove_unused_columns`.

---

## Loss Computation & Diagnostics (JSON)

- Inputs: logits, labels, teacher/student spans, JSON grouping masks (shifted).
- Grouped CE per role:
  - Compute CE once; slice by role and group masks; aggregate with configured weights.
  - Emit role totals and per‑group means + counts.
- Reported keys include (stable for dashboards):
  - `teacher_llm_loss`, `student_llm_loss`
  - `teacher_caption_loss`, `teacher_grounding_loss`, `teacher_formatting_loss`
  - `student_caption_loss`, `student_grounding_loss`, `student_formatting_loss`
- Decomposition checks:
  - Group sums per role approximate role LLM loss; roles sum to total loss within tolerance.

---

## Collation

- Standard only: pad sequences to max length in the batch; emit image tensors when present.
- Packed collation is disabled in JSON mode and rejected by config validation.

---

## Inference (training‑matched, JSON)

- Loads tokenizer/processor/model; enforces image token/tensor alignment.
- Renders simple or teacher‑student (num_teachers>0) conversations; batch size is forced to 1 if teachers used.
- Parses assistant JSON strictly; normalizes to visualization/training formats.
- Troubleshooting aids: comprehensive logging for image token alignment and JSON parsing diagnostics.

---

## Troubleshooting (compact)

| Issue | Root cause | Fix |
|---|---|---|
| Invalid collator_type | Packed collation not supported | Set `collator_type: standard` |
| Image tokens/text mismatch | Wrong image count/template propagation | Verify chat template; ensure `<|image_pad|>` equals number of images |
| Empty or malformed JSON output | Model drift or bad prompt/template | Tighten prompts; ensure variant ratios; add JSON parsing diagnostics |
| Span loss mismatch warnings | Group sums vs role totals diverge | Inspect grouping masks; enable `debug_alignment` |

---

## Performance & Ops

- Training: prefer `flash_attention_2`; `bf16` for memory/perf balance; batch size 1 with gradient accumulation.
- Typical 7B settings: `per_device_train_batch_size=1`, `gradient_accumulation_steps≈8`, VRAM ≈24GB, throughput ≈2–3 samples/s on A100.
- Save SafeTensors; shard size ~5GB recommended; rotate best checkpoints by metric.

---

## Module Map (src_new_json)

- `config/`: strict dataclass; JSON‑mode validation; path normalization.
- `data/`: dataset + augmentation; HF‑first conversations; offset‑based spans; dynamic pairing.
- `processing/`: templates, JSON formatter, variant registry (builds assistant/user JSON text).
- `models/`: wrapper + validations; grouped CE over JSON; diagnostics via role/group metrics.
- `losses/`: JSON grouping utilities.
- `training/`: BBUTrainer; TrainingStateManager; CheckpointSaver; callbacks.
- `inference.py`: training‑matched inputs; strict JSON parsing.
- `utils/`: path/data resolvers; rank‑aware logging; validation helpers.

---

## Example Snippets

### Minimal training command
```bash
python scripts/train_new_json.py --config phase_1/standard --log_level INFO
```

### Minimal config excerpt (JSON mode)
```yaml
collator_type: standard
teacher_ratio: 0.6

dynamic_pairing_enabled: true
dynamic_pair_target_assignment: current
dynamic_pair_cross_bucket_explore_prob: 0.0

caption_loss_weight: 1.2
grounding_loss_weight: 1.5
formatting_loss_weight: 0.3
```
