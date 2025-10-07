# Quickstart: Dynamic Generation Length (Debug Multi-GPU)

1) Environment
- Conda env: `conda activate ms` (or use full path `/root/miniconda3/envs/ms/bin/python`)
- GPUs: export CUDA_VISIBLE_DEVICES=0,1 (or 0,1,2,3)
- Non-interactive launcher: `scripts/run_dense_grpo.sh` (override python via `PYTHON_BIN=/root/miniconda3/envs/ms/bin/python`)

2) Config
- Edit `configs/dense_rl/debug.yaml` (already includes dynamic_length defaults)
- Ensure:
  - `grpo.dynamic_length.enabled: true`
  - `grpo.mask_truncated_completions: false`

3) Run (load test)
```
conda activate ms
LOG_LEVEL=INFO MODE=load TO_CONSOLE=true CONFIG_PATH=configs/dense_rl/debug.yaml bash scripts/run_dense_grpo.sh
```

4) Run (short training)
```
conda activate ms
LOG_LEVEL=INFO MODE=train TO_CONSOLE=false CONFIG_PATH=configs/dense_rl/debug.yaml bash scripts/run_dense_grpo.sh
```

5) Verify logs
- dynamic_length/mean_cap|min_cap|max_cap are present
- completions/mean_length|min_length|max_length reasonable
- No NCCL timeouts; ranks align

6) Determinism tips
- Pin tokenizer version; verify identical across ranks (compare vocab size and special tokens)
- Keep identical YAML on all ranks; avoid ad-hoc env-only overrides when debugging mismatches
