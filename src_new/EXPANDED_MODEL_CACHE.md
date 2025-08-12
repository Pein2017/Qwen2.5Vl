### Loading and using an expanded model cache (Qwen2.5‑VL)

This note explains how to load a pre‑expanded checkpoint whose tokenizer and embeddings already include coordinate tokens, so training and inference skip per‑run expansion.

- Base model directory: `/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct`
- Example expanded directory: `/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024`
- Relevant modules in `src_new`:
  - `processing/token_processor.py` (tokenizer/embedding expansion logic)
  - `models/wrapper.py` (validation of coordinate token range, extended checkpoint detection)
  - `scripts/train_new.py` (pre-distributed expansion step that is skipped if already extended)
  - `inference.py` (fast path for extended checkpoints)

### Prerequisites
- Use the `ms` conda environment and absolute paths.
  ```bash
  source ~/.bashrc && conda activate ms
  ```
- The expanded directory must include model weights, extended tokenizer files, and image processor files:
  - `config.json`, `generation_config.json`, model weights (`*.safetensors`)
  - `tokenizer.json`, `tokenizer_config.json` (with added tokens and chat template)
  - `preprocessor_config.json` (image processor)
  - Optional: `coordinate_config.json` documenting `max_coord_value`

### Configure training to use the expanded checkpoint
Edit your config YAML (for example `/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml`) so it points to the expanded directory and matches the `max_coord_value`.

```yaml
model_path: "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024"
# ... other settings ...
coordinate_tokens_enabled: true
max_coord_value: 1024
```

Run training as usual; the loader detects the extended vocab and skips expansion:
```bash
cd /data3/Qwen2.5-VL-main
source ~/.bashrc && conda activate ms
bash /data3/Qwen2.5-VL-main/scripts/run_new_train.sh
```

### Use with the inference CLI
```bash
source ~/.bashrc && conda activate ms
python -m src_new.inference \
  --config /data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml \
  --checkpoint "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024" \
  --image /abs/path/to/sample.jpg
```
If the checkpoint is extended (tokenizer vocab size > 151665), `src_new/inference.py` will take the fast path and avoid re‑expansion.

### What “extended” means and key invariants
- Base tokenizer length is 151665 (no coordinate tokens).
- After extension, added tokens are:
  - Geometry: `<|line_start|>`, `<|line_end|>`
  - Coordinate: `<|coord_0|>` … `<|coord_{max_coord_value}|>`
- Coordinate token IDs must start at 151667 and end at `151667 + max_coord_value`.
- The model’s input/output embeddings are resized (padded to a multiple of 128) and initialized for the new tokens.

### Quick validation checklist
- Tokenizer length is greater than 151665 at the expanded path.
- `AutoConfig.from_pretrained(<expanded_path>).vocab_size` > 151665.
- `src_new/models/wrapper.py` validates coordinate token range equals `[151667, 151667 + max_coord_value]` inclusive.
- The image processor files are present (`preprocessor_config.json`, etc.).

### Notes on `max_coord_value`
- Works out‑of‑the‑box for `max_coord_value = 1024` (as set in `configs/bbu_v2_use_coord.yaml`).
- If you intend to use values other than 1024 (e.g., 512 or 2048), one internal validator in `processing/token_processor.py` currently assumes exactly 1025 coordinate tokens (0..1024). That check must be generalized to the configured `max_coord_value` before using non‑1024 variants.
- Superset idea (e.g., export once at 2048 and run with smaller `max_coord_value`) is not allowed by current strict validation in `models/wrapper.py` (it requires the end ID to equal `151667 + max_coord_value`). Relaxing that would require a small code change.

### Where this is enforced in code (for reference)
- Tokenizer and embedding expansion logic: `src_new/processing/token_processor.py`
- Training pre‑distributed expansion and skip path: `scripts/train_new.py`
- Extended checkpoint detection and coordinate range validation: `src_new/models/wrapper.py`
- Inference fast path for extended checkpoints: `src_new/inference.py`

That’s it. Point `model_path` at your expanded directory, keep `max_coord_value` in sync, and both training and inference will skip redundant tokenizer/embedding expansion. 