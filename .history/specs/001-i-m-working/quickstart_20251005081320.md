# Quickstart (Manual GRPO)

## Environment
```bash
conda activate ms
cd /data3/Qwen2.5-VL-main
```

## Loader smoke
```bash
python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode load
```

**Expected output** (one-line JSON):
```json
{"ok": true, "device": "cuda:0", "dtype": "torch.bfloat16", "vocab_model": 152064, "vocab_tokenizer": 151667}
```

This validates:
- Configuration loads successfully with all required keys
- Model loads on GPU with bfloat16 precision
- Tokenizer and model vocabularies are consistent

## Short train (debug)
```bash
python -m src_new.rl.runner --config configs/dense_rl/debug.yaml --mode train
```

**Expected console output** (per logging step):
```
[step=1 epoch=0.050] loss=2.1234 reward=0.4567±0.1234 lr=5.000e-06 grad_norm=1.234 eta=5.2min
```

Console logs include:
- `step`: Global training step
- `epoch`: Epoch progress (0.0 to 1.0)
- `loss`: Training loss value
- `reward`: Mean reward ± std across K samples
- `lr`: Current learning rate
- `grad_norm`: Gradient norm after clipping
- `eta`: Estimated time remaining in minutes

**Expected outputs**:
- **TensorBoard logs**: `tb_post/9-26/debug/` (as configured in debug.yaml)
- **Checkpoints**: `outputs/rl_debug/9-26/checkpoint-<step>/` (HF layout)
  - Includes: model weights (SafeTensors), tokenizer, processor, generation_config.json
  - Verified with: `ls outputs/rl_debug/9-26/checkpoint-*/`

## TensorBoard
```bash
tensorboard --logdir tb_post/9-26
```

**Required scalars** (verify in TensorBoard UI):

### Core training metrics (HF Trainer-compatible tags)
- `train/loss`: Training loss per step
- `train/learning_rate`: Learning rate schedule
- `train/grad_norm`: Gradient norm after clipping
- `train/epoch`: Epoch progress (0.0 to 1.0)

### RL-specific metrics
- `reward`: Mean reward across K samples
- `reward_std`: Standard deviation of rewards
- `temperature`: Generation temperature
- `beta`: KL penalty coefficient
- `eta_minutes`: Estimated time remaining
- `step`: Global step counter

### Completion statistics
- `completions/mean_length`: Average completion length
- `completions/min_length`: Minimum completion length
- `completions/max_length`: Maximum completion length
- `completions/clipped_ratio`: Fraction of truncated completions
- `completions/terminated_ratio`: Fraction terminated with EOS

### Per-reward metrics (for each reward function)
- `rewards/<name>/mean`: Mean value for reward function `<name>`
- `rewards/<name>/std`: Standard deviation for reward function `<name>`

Example reward names from debug.yaml: `parse`, `wrappers`, `coords`, `separators`, `vocab`, `length`, `length_window`, `bbox_giou`

### Advantage statistics
- `advantages/std`: Standard deviation of advantages
- `advantages/max_abs`: Maximum absolute advantage value

## Configuration policy
- All run-time parameters are sourced strictly from YAML. Do not rely on environment variables or CLI flags to override config values. Update `configs/dense_rl/*.yaml` instead.

## Validation checklist
- [X] `--mode load` prints valid JSON
- [X] Configuration enforces explicit keys (no defaults)
- [X] Console logs mirror TB scalar keys
- [X] Includes ETA, step/epoch, LR, grad_norm
- [ ] TB contains all required scalars (verify with actual run)
- [ ] Checkpoint saved & reloadable with tokenizer/processor
- [X] RL is single-turn only (no teacher pairing)
