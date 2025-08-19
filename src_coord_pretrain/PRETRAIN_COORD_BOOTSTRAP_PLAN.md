## Qwen2.5‑VL Coord Bootstrap Pre‑Training Plan (Separate Module)

Purpose: a clean, separate pre‑training module that teaches a new “coordinate language” mapping integers and simple arithmetic to `<|coord_*|>` tokens, producing a checkpoint that downstream `src_new` fine‑tuning can consume.

---

### Scope & Outcomes
- Identity mapping: `N → <|coord_N|>` for `0 ≤ N ≤ max_coord` (currently 1024)
- Small arithmetic: `a + b`, `a − b`, `a × b`, `a ÷ b` (integer division) with result in `[0, max_coord]`
- Strict output: assistant replies are exactly one coord token (e.g., `<|coord_123|>`), no raw digits
- No geometry yet: focus on token usage alignment; short sequences for speed
- Deterministic: reproducible data + init (`coordinate_init_mode: fourier_ramp`)

---

### Best‑Practice Implementation (HF Transformers + src_new alignment)
- Use HuggingFace Qwen2‑VL components directly:
  - `transformers.Qwen2VLForConditionalGeneration`
  - `transformers.Qwen2VLProcessor` (for unified tokenizer/image processor)
  - `tokenizer.apply_chat_template(...)` to serialize ChatML messages for text‑only samples
- Keep compatibility expectations from `src_new`:
  - Coordinate tokens must exist in tokenizer vocab: `<|coord_0|>` … `<|coord_{max_coord}|>`
  - Model embedding matrix must include rows for those tokens; IDs should remain stable across stages
- Prefer starting from an expanded model cache that already contains coord tokens, then bootstrap teach usage. If expansion is needed, perform it once and persist.

---

### Tokenizer + Model Bring‑Up
Two supported paths (choose one and persist the result for reproducibility):

1) Pre‑expanded model path (recommended)
- Use an expanded checkpoint that already contains the coord tokens with deterministic init (e.g., produced by `/data3/Qwen2.5-VL-main/scripts/migrate_to_expanded_cache.py`).
- Load with:
  - `processor = Qwen2VLProcessor.from_pretrained(<expanded_model_path>)`
  - `model = Qwen2VLForConditionalGeneration.from_pretrained(<expanded_model_path>)`
- Validate:
  - Scan `tokenizer.get_vocab()` for all `<|coord_*>` tokens
  - Assert contiguous coverage `0..max_coord` and ID stability (save the token IDs list as an artifact)

2) On‑the‑fly expansion (only if you cannot use a pre‑expanded path)
- Build the coord token list: `[f"<|coord_{i}|>" for i in range(max_coord+1)]`
- Add as additional special tokens: `tokenizer.add_special_tokens({"additional_special_tokens": coord_tokens})`
- Resize embeddings: `model.resize_token_embeddings(len(tokenizer))`
- Initialize the new rows deterministically (Fourier ramp or mean‑projection). Document exact scheme and persist a JSON manifest of token → row index.
- Save the expanded tokenizer+model to an absolute path and reuse it for all runs.

Fail‑fast gates (both paths):
- Missing or partial coord range → raise with actionable message
- Model/Tokenizer size mismatch after resize → raise
- Persist a JSON with `{token: id}` mapping for coord tokens in the run dir

---

### Dataset (Text‑Only, ChatML)
- Rationale: for coord‑token usage alignment, images are unnecessary; HF chat template supports text‑only messages.
- JSONL line schema (one per sample):
  ```json
  {
    "messages": [
      {"role": "user", "content": "What is `123` in coordinate space?"},
      {"role": "assistant", "content": "<|coord_123|>"}
    ],
    "meta": {"task": "identity", "result": 123}
  }
  ```
- Sampling
  - Identity: uniform 0..max_coord with edge emphasis {0, 1, max−1, max}
  - Arithmetic: sample operands so result ∈ `[0, max_coord]`; include `+ − × ÷`
- Conversation serialization
  - Use `tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True, return_tensors="pt")`
  - Do not pass images for this module; no `<|image_pad|>` should be present

---

### Labeling (Assistant‑Only CE, src_new‑style)
- Objective: supervise only the assistant span; mask everything else with `-100`
- Approach: replicate `src_new` behavior using offset mapping
  1) Decode input ids to text (skip_special_tokens=False)
  2) Re‑tokenize with `return_offsets_mapping=True`
  3) Locate assistant content via regex on ChatML: `<|im_start|>assistant\n(.*?)<|im_end|>`
  4) Convert char spans to token spans using offsets; extend to include the immediate `<|im_end|>` token
  5) Set labels outside the span to `-100`; inside span equal to input ids
- Validation: span bounds valid; exactly one `<|coord_` inside assistant content; no digits `[0-9]` present

---

### Data Pipeline & Collation
- Reader: simple JSONL loader with schema validation and range checks
- Collator: pads `input_ids`, `attention_mask`, and builds masked `labels` per sample as above
- Determinism: set seeds for Python/NumPy/PyTorch; fix sampling order

---

### Training Orchestration (HF Trainer)
- Prefer `transformers.Trainer` for simplicity
  - Custom `DataCollator` returns `input_ids`, `attention_mask`, `labels`
  - `TrainingArguments`: `output_dir`, `bf16`/`fp16`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `warmup_steps`, `max_steps`, `save_steps`, `logging_steps`, `eval_steps`, `lr_scheduler_type`, `learning_rate`
- Freezing schedule
  - Stage 0 (optional): freeze all but token embeddings + LM head (200–500 steps, LR=5e‑5)
  - Stage 1: unfreeze full model (1k–3k steps, LR=1e‑5..2e‑5)
- Losses
  - CE over assistant span is sufficient
  - Auxiliary coord losses can be kept off in this module; they are handled in `src_new` if desired

---

### Evaluation & Strictness
- Held‑out prompts for identity and arithmetic
- Decode assistant part only, then assert:
  - Contains exactly one `<|coord_*|>` token
  - Contains no raw digits `[0-9]`
  - Result is in `[0, max_coord]`
- Log running accuracies; save a metrics JSON alongside checkpoints

---

### Checkpoint Export & Handoff to `src_new`
- Save via `model.save_pretrained(<abs_output_dir>)` and `tokenizer.save_pretrained(<abs_output_dir>)`
- Persist coord token ID mapping JSON in the same dir
- In `src_new` configs, point `model_path` to the exported dir
- `src_new` will retain its own ConversationProcessor/vision rules; this module just improves coord token usage

---

### Config (YAML) — this module
File: `/data3/Qwen2.5-VL-main/src_coord_pretrain/config/coord_bootstrap.yaml`
- Required keys
  - `data_path`: `/data3/Qwen2.5-VL-main/src_coord_pretrain/data/coord_bootstrap.jsonl`
  - `max_dataset_size`: integer or `-1`
  - `coordinate_tokens_enabled`: true
  - `max_coord_value`: 1024
  - `coordinate_init_mode`: "fourier_ramp"
  - Optimization hyperparameters (learning rates, batch sizes, steps)
  - `model_path`: absolute path to expanded model/tokenizer (with coord tokens)
  - `output_dir`: absolute path to write checkpoints and metrics

---

### Scripts (examples)
- Generate data
  ```bash
  source ~/.bashrc && conda activate ms
  python /data3/Qwen2.5-VL-main/src_coord_pretrain/scripts/generate_coord_bootstrap.py \
    --output /data3/Qwen2.5-VL-main/src_coord_pretrain/data/coord_bootstrap.jsonl \
    --num_identity 50000 --num_arithmetic 20000 --max_coord 1024 --seed 1337
  ```
- Run training
  ```bash
  source ~/.bashrc && conda activate ms
  bash /data3/Qwen2.5-VL-main/src_coord_pretrain/scripts/run_coord_bootstrap.sh
  ```

---

### Validation Gates (Fail‑Fast)
- Startup
  - Tokenizer contains all `<|coord_0|>` … `<|coord_{max_coord}|>`; raise if missing
  - Model embedding size matches tokenizer length after any resize
  - Exactly one assistant span per sample; label mask includes `<|im_end|>`
- Training
  - Periodic eval; decode strictness (one `<|coord_*|>`, no digits)
  - Abort on non‑finite loss; persist last valid checkpoint

---

### Tests (TDD)
- `tests/test_vocab_presence.py`: coord token presence and contiguous range; IDs persisted
- `tests/test_dataset_and_spans.py`: assistant‑only labels; `<|im_end|>` included; no digits in assistant content
- `tests/test_generation_strictness.py`: decoded outputs contain exactly one coord token, no digits

---

### Example Commands
```bash
source ~/.bashrc && conda activate ms
python /data3/Qwen2.5-VL-main/src_coord_pretrain/scripts/generate_coord_bootstrap.py \
  --output /data3/Qwen2.5-VL-main/src_coord_pretrain/data/coord_bootstrap.jsonl \
  --num_identity 50000 --num_arithmetic 20000 --max_coord 1024 --seed 1337
bash /data3/Qwen2.5-VL-main/src_coord_pretrain/scripts/run_coord_bootstrap.sh
```

---

### Implementation Order
1) Add tests in `/data3/Qwen2.5-VL-main/src_coord_pretrain/tests/`
2) Create/validate expanded tokenizer+model path; persist token→ID map JSON
3) Implement generator, dataset, and custom collator (assistant‑only labels)
4) Wire HF Trainer; run short bootstrap; export checkpoint
5) Hand off to `src_new` for full grounding training
