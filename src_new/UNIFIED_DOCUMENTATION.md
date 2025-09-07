# Qwen2.5-VL BBU Detection Training Pipeline - Unified Documentation

**Complete Reference for Vision-Language Model Fine-tuning with Coordinate Token System**

---

## Executive Snapshot (2025-09)

- **Entrypoints**:
  - Training: `/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config <name> --log_level {INFO|DEBUG}`
    - Optional DeepSpeed via env: `BBU_DEEPSPEED_ENABLED=true`, `BBU_DEEPSPEED_CONFIG=scripts/zero2.json`
  - Inference: `/root/miniconda3/envs/ms/bin/python -m src_new.inference --config /abs/config.yaml --model_path /abs/checkpoint --data_root /abs/data --output_file /abs/out.jsonl`

- **Critical invariants**:
  - Fast tokenizer required with `offset_mapping` available; chat template must exist (tokenizer or processor).
  - If `coordinate_tokens_enabled=true`, checkpoint must be pre‑expanded: vocab > 151665; `<|line_start|>`, `<|line_end|>` present; full `<|coord_0|>.. <|coord_{max}|>` contiguous; embeddings padded to multiple of 128.
  - Image alignment: decoded `<|image_pad|>` count equals `sum_i (t*h*w) // (merge_size**2)`; `pixel_values` rows equal `sum_i (t*h*w)`.
  - Labels/spans: assistant spans from regex+offsets; include immediate `<|im_end|>`; `<|image_pad|>` always masked.

- **Data → Model → Loss → Checkpoint** (HF‑first): JSONL → `ConversationProcessor.apply_chat_template` → official processor tensors → `DetectionModel` wrapper (validations) → single‑pass CE (teacher/student masks) + optional coord aux losses → local aggregation (no custom distributed ops) → SafeTensors + tokenizer/processor saved; best checkpoint via atomic copy, rotation enabled.

- **Freezing phases**: `phase_1` train `visual.merger` (+ coord‑slice rows if coords); `phase_2` = phase_1 + last‑K LLM blocks; `phase_3` unfreeze all (keep `visual.patch_embed` frozen by default; optional last‑K vision blocks).

---

## Training Pipeline (step‑by‑step)

1) Parse & validate
- Load `configs/phase_1/base.yaml` merged with experiment override; auto‑resolve dataset paths via `data_root` when missing (`config/config.py::load_config`).
- Initialize rank‑aware logging; apply Qwen2.5‑VL patches; seed RNGs.

2) Tokenizer/processor & model
- Fast tokenizer and `Qwen2VLImageProcessor` from `config.model_path`; `image_processor.max_pixels` overridden from YAML.
- `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)` with dtype/attention/low‑CPU settings; device_map=`auto` when single GPU.
- Validate coordinate pre‑expansion if `coordinate_tokens_enabled=true` (vocab/content/128‑padding).

3) Dataset & conversations
- `Dataset` reads JSONL; validates per‑sample structure; optional augmentation; dynamic teacher assignment (ratio, softmax sampling).
- `ConversationProcessor.create_conversation(...)` builds teacher‑student or simple conversations, applies HF chat template, threads `conversation_text` + `offset_mapping`, enforces image token consistency.
- Labels/spans built via offset mapping; assistant spans include immediate `<|im_end|>`, all other regions masked; `<|image_pad|>` masked.

4) Collation & packing
- `collator_standard` pads to max length.
- `collator_packed` concatenates sequences into one row; emits `segment_lengths` when `packed_segment_isolation=true`; first token of each subsequent sample is masked to prevent cross‑sample transitions.

5) Forward & validations (`models/wrapper.py`)
- Validate text/image tensor shapes and token/image counts; compute expected image token count from THW grid and merge size; raise on mismatch.
- When spans provided, bypass base loss and use SOLUTION‑1 CE path; optionally build block‑diagonal attention for packed rows.

6) Loss computation (`models/loss_manager.py`)
- Single‑pass CE once; apply teacher/student masks; grouped LLM losses (caption/grounding/formatting) via `losses/token_grouping.py` with weights from config (teacher/student weighted separately).
- Coordinate auxiliary losses (optional): Kernelized‑KL + Unlikelihood on coordinate‑labeled targets; diagnostics (window mass, coord slice mass, gt prob, expected MAE bins, top‑k, outside‑window mass, non‑coord top‑k mass, entropy, margin, mean offset, coord‑pos count). Strictly fail if enabled but no coord targets within spans.

7) Phases & optimizer
- `PhaseFreezeManager` applies `phase_name`; coord‑slice grad masks on embeddings/head when applicable; param groups: `vision`, `merger`, `top_layers`, `coord_slice`, `llm`. LR logging reports true group names.

8) Logging, evaluation, checkpointing
- `TrainingStateManager` aggregates loss components locally; `_maybe_log_save_evaluate` overridden to avoid NCCL conflicts; evaluation accumulates components too.
- `CheckpointSaver` writes SafeTensors, tokenizer/processor, `coordinate_config.json` (coord mode); best checkpoint via atomic folder copy; rotation respects `save_total_limit`.

---

## Data Schema & Business Rules (authoritative)

- JSONL fields per sample: `images: [str]`, `objects: [dict]`, `width: int`, `height: int`.
- Supported geometry keys: `bbox_2d` (4 ints), `quad` (8 ints), `line` (even ≥ 4 ints); coordinates clamped to `[0, max_coord_value]`.
- Coordinate tokens mode: geometry lists emitted as `<|coord_N|>` tokens; non‑token mode uses integers; object text wrapped with canonical tokens: `<|object_ref_start|>...<|object_ref_end|>` + geometry wrappers (`<|box_*|>`, `<|quad_*|>`, `<|line_*|>`).
- Object categories (6): `bbu`, `bbu_shield`, `connect_point`, `label`, `fiber`, `wire`.
- Description policy (Chinese): one object per line; concise but complete; no extra commentary; sort outputs top‑to‑bottom then left‑to‑right; line start is leftmost endpoint (break ties by y then x); quad ordering TL → TR → BR → BL.
- Business attributes (condensed, values fixed):
  - BBU设备: 品牌 {华为, 中兴, 爱立信}; 完整性 {显示完整, 只显示部分}; 挡风板需求 {无需安装, 机柜空间充足需要安装}; [挡风板配置符合性] {这个BBU设备按要求配备了挡风板, 这个BBU设备未按要求配备挡风板}; [备注文本]
  - 挡风板: 品牌 {华为, 中兴}; 完整性 {显示完整, 只显示部分}; 安装方向 {安装方向正确, 安装方向错误}; [备注文本]
  - 螺丝、光纤插头: 类型 {BBU安装螺丝, 机柜处接地螺丝, 地排处接地螺丝, ODF端光纤插头, BBU端光纤插头}; 完整性 {显示完整, 只显示部分}; 合规性 {符合要求, 不符合要求}; [具体问题] 从 {未拧紧, 露铜, 复接, 生锈}; [备注文本]
  - 标签: 文字内容（可为空字符串）
  - 光纤: 保护措施 {无保护措施, 有保护措施}; 弯曲半径 {弯曲半径合理, 弯曲半径不合理（弯曲半径<4cm或者成环）}; [保护类型] {蛇形管, 铠装, 同时有蛇形管和铠装}; [备注文本]
  - 电线: 整齐度 {捆扎整齐, 分布散乱}; [备注文本]

---

## Configuration (strict)

- Model/runtime: `model_path`, `attn_implementation {flash_attention_2|eager|sdpa}`, `torch_dtype {float16|bfloat16|float32|bf16|fp16}`, `use_cache`.
- Training: `num_train_epochs`, `per_device_{train,eval}_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `vision_lr`, `merger_lr`, `llm_lr`, `warmup_ratio`, `weight_decay`, `max_grad_norm`, `lr_scheduler_type`, `gradient_checkpointing`, `bf16|fp16`.
- Data: `train_data_path`, `val_data_path`, `data_root`, `teacher_pool_file`, `max_total_length`, `num_teacher_samples`, `max_dataset_size`.
- Features: `coordinate_tokens_enabled`, `coordinate_init_mode {ms_mean|fourier_ramp}`, `max_coord_value`, `merge_size`, `max_pixels`.
- Loss: `teacher_loss_weight`, `student_loss_weight`, `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight`, `coordinate_loss_weight`, `teacher_ratio`.
- Collator: `collator_type {packed|standard}`, `packed_segment_isolation` (bool).
- HF Trainer: `eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`, `save_total_limit`, `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`.
- Logging: `logging_steps`, `report_to`, `disable_tqdm`.
- Dataloader: `dataloader_num_workers`, `pin_memory`, `prefetch_factor`, `remove_unused_columns`.
- LRs (optional): `lr_merger`, `lr_coord_slice`, `lr_top_layers`, `lr_full_model`.
- Variants: `conversation_variant_ratios` and/or `conversation_variant_schedule`.
- Spans & debug: `span_include_im_end_in_labels`, `debug_alignment`.
- Augmentation: `use_aug` (bool), optional `augmentation` or `augmentation_schedule` (preset schedule supported); teacher augmentation optional.

---

## Loss Computation & Diagnostics (single source)

- LLM loss: single CE pass (shifted), masks per role; grouped LLM masks from `TokenGroupingPlugin` (caption/grounding/formatting) with teacher/student role weights.
- Coordinate aux losses (optional):
  - Kernelized‑KL over Gaussian window centered at GT bin; Unlikelihood on non‑coord slice (top‑K).
  - Config: `coord_aux_enabled`, `coord_aux_tau`, `coord_aux_sigma_bins`, `coord_aux_window_bins`, `coord_aux_topk`, `coord_aux_lambda_kce`, `coord_aux_lambda_unlike`.
  - Diagnostics per role: `window_mass`, `coord_slice_mass`, `gt_prob`, `expected_mae_bins`, `top1_acc`, `top5_acc`, `outside_window_mass`, `noncoord_topk_mass`, `window_entropy`, `margin_top1_top2`, `mean_bin_offset`, `coord_pos_count`.

---

## Freezing Phases (separate runs)

- `phase_1`: Train `visual.merger`; if coord tokens present, enable coord‑slice grad masks on `embed_tokens.weight` and `lm_head.weight`; freeze LLM & vision.
- `phase_2`: Phase 1 + unfreeze last‑K LLM decoder blocks (`top_k_layers`, default 6); vision remains frozen; merger trainable.
- `phase_3`: Unfreeze all; keep `visual.patch_embed` frozen by default; optionally unfreeze only last‑K vision blocks (`vision_top_k_blocks`).

---

## Collation

- Standard: pad sequences to max length; emit `pixel_values` and `image_grid_thw` when present; validate THW vs `pixel_values` rows.
- Packed: concatenate to one row; mask first token at each boundary; optionally emit `segment_lengths` for block‑diagonal attention isolation.

---

## Inference (training‑matched)

- Loads tokenizer/processor/model with the same validations as training; enforces image token/tensor alignment.
- Teacher‑guided mode (num_teachers > 0) replicates dynamic pairing; batch size forced to 1.
- Parsing priority: coordinate token blocks (strict) → geometry tokens with raw numbers → JSON best‑effort; optional prefix constraints allow only coordinate slice + punctuation inside geometry sections.
- Stability recommendations: `attn_implementation="eager"`, `bf16`, batch size 1.

---

## Troubleshooting (compact)

| Issue | Root cause | Fix |
|---|---|---|
| NCCL timeouts during logging/eval | Conflicts in distributed ops | Use `BBUTrainer` (local aggregation) which overrides `_maybe_log_save_evaluate`; no custom distributed ops |
| Checkpoint save: `'Qwen2VLProcessor' has no get_vocab` | `processing_class` set to processor | Set `trainer.processing_class = tokenizer`; save processor separately |
| Span misalignment / NaNs | Char vs token mismatch; missing EOS in spans | Use offset mapping; include immediate `<|im_end|>` in assistant spans |
| Empty student spans | Incomplete teacher‑student conversations | Ensure builder emits full student responses; dataset always supplies student turn |
| Image features/tokens mismatch | `<|image_pad|>` count not equal to THW expansion | Validate with `utils/tensor_validation.py`; ensure merge size and image grid consistent |

---

## Performance & Ops

- Prefer `flash_attention_2` for training; `eager` for inference stability; `bf16` for memory/perf balance.
- 7B model typical settings: `per_device_train_batch_size=1`, `gradient_accumulation_steps≈8`, VRAM ≈24GB, throughput ≈2–3 samples/s on A100.
- Differential LRs recommended (vision < LLM; merger highest); enable gradient checkpointing for memory.
- Always save SafeTensors; shard `max_shard_size ~ 5GB`.

---

## Module Map (refactored `src_new`)

- `config/`: strict dataclass; path normalization; schedule & variant validation.
- `data/`: dataset + augmentation + teacher pool; HF‑first conversations; offset‑based spans; `<|image_pad|>` masking.
- `processing/`: conversation builder; coordinate/string conversion; canonical tokens; templates; variant registry.
- `models/`: wrapper + validations; SOLUTION‑1 CE path; optional coord aux; diagnostics.
- `losses/`: `coord_aux.py` and `token_grouping.py`.
- `training/`: BBUTrainer; PhaseFreezeManager; TrainingStateManager; CheckpointSaver.
- `inference.py`: engine with strict parsing and training‑matched prep.
- `utils/`: rank‑aware logging; path/data resolvers; tensor validation; debug logging.

---
