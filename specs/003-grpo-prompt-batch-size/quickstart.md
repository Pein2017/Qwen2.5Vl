# Quickstart: GRPO Prompt Batch Size Refactoring

**Goal**: Run a smoke GRPO job that aggregates multiple prompts per optimizer step, verifies telemetry, and exercises resume logic.

---

## Prerequisites
1. `source ~/.bashrc && conda activate ms`
2. Pull latest `003-grpo-prompt-batch-size` branch.
3. Ensure dataset JSONL files referenced in the config are accessible.
4. Confirm NCCL multi-GPU visibility: `echo $CUDA_VISIBLE_DEVICES` should list 8 GPUs.

---

## Step 1: Prepare Configuration
Open `configs/dense_rl/debug.yaml` (or `standard.yaml` for longer training) and ensure the sampling block is set explicitly:

```yaml
sampling:
  prompt_batch_size: 4      # Prompts per optimizer update
  sample_k: 8               # Trajectories per prompt (global)
  sample_k_per_rank: false  # Split sample_k across ranks
  reward_average_window: 5  # Cycles for smoothed reward

normalization:
  cross_rank_advantages: true   # REQUIRED to compute/log global advantages
```

- If you want non-degenerate advantage std with very small local_k, prefer either:
  - `normalization.cross_rank_advantages: true` (recommended), or
  - `grpo.sample_k_per_rank: true` (each rank generates K trajectories, larger local tensors).

Adjust `training.dataset_size`, `training.num_train_epochs`, and reward weights as needed. When `sample_k_per_rank=false`, make sure `sample_k % world_size == 0`.

Recommended NCCL safeguards for smoke testing:

```bash
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=180
```

---

## Step 2: Launch Smoke Training Run (YAML-only)
```bash
python -m src_new.rl.runner \
  --config configs/dense_rl/debug.yaml \
  --mode train
```
- All sampling knobs are YAML-only—no CLI overrides are supported.
- Verify console output logs per accumulation cycle and includes global metrics when available:
  - `advantages/std` and `advantages/max_abs` should reflect cross-rank global values
  - `reward` and `reward_std` prefer global mean/std over local-chunk values

---

## Step 3: Inspect Telemetry
1. Launch TensorBoard (use the `tb_dir` you set in the config):  
   `tensorboard --logdir tb_dense`
2. Confirm new scalar tags under global metrics:
   - `rl_global/advantages_std`, `rl_global/advantages_max_abs`, `rl_global/reward_std`
3. Existing prompt batch tags remain available under `rl/prompt_batch/*`.

---

## Step 4: (Optional) Notes
1. Ensure the run exits cleanly without hanging.

---

## Step 5: Optional GPU Smoke Run
1. On a GPU node, activate `ms`, export the NCCL safeguards above, and run the standard configuration with a reduced dataset size.
2. Use real tokenizer/model weights and confirm one accumulation cycle completes without OOM.
3. GPU-only tests (e.g., `tests/rl/test_prompt_batch_gpu.py`) are marked with `@requires_gpu` and will skip automatically when CUDA is unavailable.

---

## Expected Metrics
- Global advantage std is non-zero when using cross-rank normalization (or per-rank K mode).
- Reward moving average trends upward compared with the single-prompt baseline without severe oscillations.
- Fill ratio hits the expected target (100% of trajectories) on every smoke-cycle or aborts early with clear logs.
- Telemetry shows global metrics when available and falls back to local metrics on single-GPU.

---

- **Fill ratio < 1.0**: Reduce `prompt_batch_size`, inspect dataset accessibility, or rerun with smaller accumulation to diagnose bottlenecks.
- **Lagging rank warnings**: Reduce `prompt_batch_size`, inspect GPU health, or adjust accelerator gradient accumulation steps. If warnings persist, investigate NCCL timeout configuration.
- **Invalid rewards**: Inspect saved quarantine report under checkpoint directory; fix reward computation before continuing.
