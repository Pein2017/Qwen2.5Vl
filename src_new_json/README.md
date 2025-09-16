# Qwen2.5‑VL JSON Training Framework (src_new_json)

A production‑ready fine‑tuning pipeline that uses strict JSON geometry in conversations (no coordinate or wrapper tokens). It mirrors the architecture of `src_new/` while replacing the special‑token conversation format with JSON‑first text.

## ✨ What’s different vs src_new
- Conversation format only: assistant/user content uses strict JSON arrays for geometry and labels (no `<|coord_*|>`, no `<|object_ref_*|>`, no `<|box_*|>` tokens).
- Grouped losses over JSON: caption/grounding/formatting are computed by parsing the assistant JSON and mapping char spans → token spans via offset mapping.
- No coordinate auxiliary losses or token‑range logic.
- Dynamic contrastive pairing is identical to `src_new` (same buckets, per‑epoch rebuild, last turn = student).
- Data collator: only `standard` is supported; packed collation is disabled (not yet verified in JSON mode).

## 🚀 Quick Start

### Training (JSON mode)
```bash
# Activate environment
source ~/.bashrc && conda activate ms

# Run with a JSON-mode config
python scripts/train_new_json.py \
  --config phase_1/standard \
  --log_level INFO
```

### Inference (JSON mode)
```bash
source ~/.bashrc && conda activate ms
python -m src_new_json.inference \
  --config_path configs/phase_2/base.yaml \
  --model_path /abs/path/to/checkpoint \
  --data_root /abs/path/to/data \
  --dataset val \
  --output_file /abs/path/to/out.jsonl
```

## 🔧 Configuration (authoritative surface)
Key sections (see `src_new_json/config/config.py` for full schema and validation):
- Model/runtime
  - `model_path`, `attn_implementation {flash_attention_2|eager|sdpa}`, `torch_dtype {bfloat16|float16|float32}`, `use_cache` (false for training).
- Training
  - `num_train_epochs`, `per_device_{train,eval}_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `vision_lr`, `merger_lr`, `llm_lr`, `warmup_ratio`, `weight_decay`, `max_grad_norm`, `lr_scheduler_type`, `gradient_checkpointing`, `bf16|fp16`.
- Data
  - `train_data_path`, `val_data_path`, `data_root`, `max_total_length`, `num_teacher_samples`, `max_dataset_size`.
  - `collator_type: standard` (packed is rejected).
- Dynamic contrastive pairing (training only)
  - `teacher_ratio: 0.0~1.0` (eval forced to 0.0)
  - `dynamic_pairing_enabled: true`
  - `dynamic_pair_target_assignment: current`
  - `dynamic_pair_cross_bucket_explore_prob: 0.0`
  - `dynamic_pair_candidate_pool_size: 128`, `dynamic_pair_temperature: 0.7` (compatibility knobs; sampling is uniform in‑bucket with hardness weighting)
- Geometry/vision
  - `merge_size`, `max_pixels`.
- Loss weights
  - `teacher_loss_weight`, `student_loss_weight`
  - `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight` (non‑negative; sum > 0)
- Augmentation (optional)
  - `use_aug`, `augmentation` (preset‑based), `teacher_augmentation` (photometric‑only), `augmentation_schedule`.
- Logging/Trainer
  - `eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`, `save_total_limit`, `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`, `logging_steps`, `report_to`, `disable_tqdm`.
- Dataloader
  - `dataloader_num_workers`, `pin_memory`, `prefetch_factor`, `remove_unused_columns`.

## 🧠 Dynamic Contrastive Pairing (training)
- Per‑epoch, all samples act as the student once. With probability `teacher_ratio`, a teacher is added.
- Teacher selection is uniform within the same major‑type bucket derived from `objects[*].desc` prefixes (`bbu`, `bbu_shield`, `connect_point`, `fiber`, `wire`, `label`), excluding self.
- If the bucket is empty, fallback to uniform over all samples; optional cross‑bucket exploration introduces small random variety.
- Conversation order is always teacher → student; only the last turn (student) is trained as generation target.
- Eval/inference: forced single‑turn (`teacher_ratio = 0.0`).

## 📐 Data and JSON schema
- Training input JSONL (unchanged from `src_new`):
  - `images: [str]`, `objects: [{desc, bbox_2d|quad|line}]`, `width`, `height`, optional `meta`.
- Conversation output (assistant JSON) is strict and used only in the rendered text:
  - `[{"box_points": [[x1,y1],[x2,y2]], "label": "..."}, {"quadrilateral_points": ...}, {"line_points": ...}]`
- The dataset still validates input objects using `bbox_2d|quad|line` (one geometry per object) and converts them to JSON strings for dialogues.

## 🧮 Losses & Grouping (JSON mode)
- Caption: string values of `label` fields (quotes excluded).
- Grounding: numeric literals inside `box_points|quadrilateral_points|line_points` arrays.
- Formatting: JSON syntax/keys and the residual assistant tokens.
- We use tokenizer offset mapping to convert JSON char spans → token spans; masks are shifted to match next‑token CE.
- No coordinate auxiliary losses; grouped CE only. Reported keys include:
  - `teacher_llm_loss`, `student_llm_loss`
  - `teacher_caption_loss`, `teacher_grounding_loss`, `teacher_formatting_loss`
  - `student_caption_loss`, `student_grounding_loss`, `student_formatting_loss`

## 🏗️ Architecture Components
- Data: `src_new_json/data/dataset.py` (HF‑first, offset spans, dynamic pairing)
- Processing: `src_new_json/processing/` (templates, variants, JSON formatter)
- Model: `src_new_json/models/wrapper.py` (HF wrapper, validations)
- Losses: `src_new_json/models/loss_manager.py`, `src_new_json/losses/*` (JSON grouping)
- Training: `src_new_json/training/bbu_trainer.py`, `training_state_manager.py`, `checkpoint_saver.py`
- Inference: `src_new_json/inference.py` (JSON parsing, visualization normalization)

## 🧭 Example: minimal JSON config (excerpt)
```yaml
# configs/phase_1/standard.yaml
num_train_epochs: 40
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4
collator_type: standard
teacher_ratio: 0.6

dynamic_pairing_enabled: true
dynamic_pair_target_assignment: current
dynamic_pair_cross_bucket_explore_prob: 0.0

model_path: model_cache/Qwen/Qwen2.5-VL-7B-Instruct
run_name: "json-phase_1"
output_dir: outputs/json/phase_1

use_aug: false
teacher_loss_weight: 0.6
student_loss_weight: 1
caption_loss_weight: 1.2
grounding_loss_weight: 1.5
formatting_loss_weight: 0.3

conversation_variant_ratios:
  dense_caption: 0.5
  coords_to_desc: 0.25
  desc_to_coords: 0.25
```

## 🧪 Health checks & invariants
- Image tokens: decoded `<|image_pad|>` count equals number of images.
- Assistant spans include immediate `<|im_end|>` and labels outside spans are `-100`.
- Grouping masks are disjoint and cover assistant spans; role‑group sums approximate role llm losses.
- Eval/inference are single‑turn.

## 🛠️ Troubleshooting
- “Invalid collator_type”: Only `standard` is supported; remove `packed` from YAML.
- “Image features and image tokens do not match”: ensure the builder receives the correct images and chat template is preserved.
- Empty or malformed assistant JSON: check variant ratios and ensure outputs conform to the schema; enable `debug_alignment: true` for span diagnostics.

## 📊 Metrics & checkpoints
- Logging uses `TrainingStateManager`; grouped CE metrics with `teacher_*/student_*` prefixes are emitted for dashboards.
- Checkpoints include model + tokenizer + processor via `CheckpointSaver`; best checkpoint selection matches `src_new`.

## ⚙️ Performance notes
- Prefer `flash_attention_2` for training, `eager` for inference stability.
- Typical 7B settings: `per_device_train_batch_size=1`, `gradient_accumulation_steps≈8`, ~24GB VRAM, ~2–3 samples/s on A100.

## License & versioning
- This module follows the same license as the repository. Versioning mirrors `src_new` semantics with JSON‑mode deviations noted above.
