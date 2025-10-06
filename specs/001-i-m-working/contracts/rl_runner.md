# RL Runner Contract

## CLI (required flags)
- `--config <path>`: absolute path to RL YAML.
- `--mode {load|train}`
- No other CLI flags are accepted; all parameters are sourced strictly from YAML. No environment-variable expansion is supported.

## YAML (required top-level)
- `model_path: str`
- `train_data_path: str`
- `val_data_path: str`
- `data_root: str`
- `output_dir: str`
- `bf16: true`
- `model.attn_implementation: eager|flash_attention_2|sdpa`
- `model.image_max_pixels: int (>0)`
- `loss`: includes `caption_loss_weight`, `grounding_loss_weight`, `formatting_loss_weight` (RL single‑turn; no teacher/student)
- `layer_config`: `llm_trainable_top_k_blocks`, `vision_trainable_top_k_blocks`, `vision_freeze_patch_embed`

## Errors (fail-fast)
- Missing required key → `ValueError("Missing required '<section.key>' - must be explicitly set")`.
  - Examples: `Missing required 'model.attn_implementation' - must be explicitly set`, `Missing required 'model.image_max_pixels' - must be explicitly set`
- Invalid enum for `model.attn_implementation` → `ValueError("model.attn_implementation must be one of {'eager','flash_attention_2','sdpa'}")`.
- `bf16 != true` → `ValueError("bf16 is mandatory for RL runs. Set root-level 'bf16: true' in the YAML.")`.
- Missing `rewards` or empty → `ValueError("'rewards' section cannot be empty - must explicitly set reward weights")`.

## Outputs
- Load mode: prints one-line JSON `{ok, device, dtype, vocab_model, vocab_tokenizer}`.
- Train mode: writes TB under `tb_dir/run_name` (from YAML) and saves checkpoints into `output_dir`.
