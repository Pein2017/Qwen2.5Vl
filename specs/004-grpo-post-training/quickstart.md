# Quickstart: Running GRPO Diagnostics

**Feature**: 004-grpo-post-training  
**Date**: 2025-10-09  
**Prerequisites**: Conda env `ms` activated, `/data3/Qwen2.5-VL-main` as working directory

## Quick Commands

### 1. Run Minimal Diagnostic Config (10 samples, 20 steps)

```bash
conda activate ms
cd /data3/Qwen2.5-VL-main

# Single-GPU smoke test
python -m src_new.rl.runner \
  --config configs/dense_rl/diagnostic.yaml \
  --mode train

# Multi-GPU (8 GPUs)
accelerate launch --config_file configs/accelerate/8gpu.yaml \
  -m src_new.rl.runner \
  --config configs/dense_rl/diagnostic.yaml \
  --mode train
```

**Expected output**: Training completes in <10 minutes, diagnostics exported to `outputs/diagnostics_test/diagnostics/`

---

### 2. Run Checkpoint Diversity Test (Phase 2 vs Phase 3)

```bash
bash scripts/run_checkpoint_diversity_test.sh
```

**What it does**:
- Loads Phase 2 and Phase 3 checkpoints
- Generates 50 completions per checkpoint at temperatures {0.7, 0.9, 1.1}
- Logs within-group variance, between-group variance, diversity ratio
- Outputs recommendation: which checkpoint to use for GRPO

**Expected output**: JSON report in `outputs/checkpoint_diversity/comparison_report.json`

---

## Interpreting Diagnostic Output

### TensorBoard Metrics

Launch TensorBoard:
```bash
tensorboard --logdir outputs/diagnostics_test/tb_logs --port 6006
```

**Key plots to check**:

1. **Trust Region** (`trust_region/*`):
   - `ratio_mean`: Should hover around 1.0 (range 0.8-1.5 is healthy)
   - `ratio_std`: Should be > 0.1 (if < 0.1 → degenerate, policy not learning)
   - `is_degenerate`: Should be 0.0 (1.0 means ratios collapsed)
   - `fallback_count`: Should be 0 (non-zero → generation_logps not stored)

2. **Rewards** (`rewards/<name>/*`):
   - `within_group_std`: Should be > 0.05 for ≥60% of prompts
   - `diversity_ratio`: Should be 0.2-1.0 (< 0.1 → over-fitting from SFT)
   - `is_collapsed`: Should be 0.0 (1.0 means reward has no variance)

3. **Sequential Processing** (`sequential/*`):
   - `is_compliant`: Should be 1.0 (0.0 means batch_size > 1 detected)
   - `peak_memory_mb`: Should be < 42,000 (42GB threshold)

4. **Gradient Flow** (`gradients/*`):
   - Layer-wise gradient norms should be in [0.01, 10.0] range
   - Check for zeros (vanishing) or >100 (exploding)

---

### Exported Artifacts

Diagnostics are exported to `{output_dir}/diagnostics/{step:06d}/`:

```
outputs/diagnostics_test/diagnostics/
├── 000010/
│   ├── trust_region_diagnostic.json      # Ratio statistics
│   ├── ratios_histogram.png              # Visual distribution
│   ├── reward_profile_coverage.json
│   ├── reward_profile_bbox_giou.json
│   ├── gradient_flow.png                 # Heatmap of layer-wise norms
│   └── multimodal_alignment_checks.json
├── 000020/
│   └── ...
```

**How to read `trust_region_diagnostic.json`**:
```json
{
  "step": 10,
  "ratio_mean": 1.02,
  "ratio_std": 0.18,   // ✅ Healthy (> 0.1)
  "is_degenerate": false,
  "fallback_count": 0,  // ✅ generation_logps working
  "clip_fraction": 0.12
}
```

**How to read `reward_profile_*.json`**:
```json
{
  "reward_name": "coverage",
  "within_group_mean_std": 0.08,  // ✅ Above threshold (0.05)
  "diversity_ratio": 0.35,        // ✅ Healthy (> 0.1)
  "is_collapsed": false,
  "prompts_with_variance_above_threshold": 42,  // Out of 50 total
  "total_prompts": 50
}
```

---

## Diagnosing Common Issues

### Symptom: Zero Advantage Std

**Diagnostics to check**:
1. `rewards/*/within_group_std` → Is it < 0.01 for all reward functions?
2. `rewards/*/diversity_ratio` → Is it < 0.1?
3. `trust_region/ratio_std` → Is it < 0.1?

**Root cause**: SFT checkpoint produces overly deterministic outputs

**Solution**:
- Run checkpoint diversity test: `bash scripts/run_checkpoint_diversity_test.sh`
- If Phase 2 shows higher within-group variance, use that checkpoint
- Otherwise, increase temperature in `configs/dense_rl/standard.yaml` (0.7 → 0.9 or 1.1)

---

### Symptom: Training Loss Not Decreasing

**Diagnostics to check**:
1. `trust_region/fallback_count` → Non-zero? → generation_logps not stored
2. `gradients/*` → Are all norms < 1e-6? → Vanishing gradients
3. `sequential/is_compliant` → Is it 0.0? → Batching violation causing instability

**Root cause**: Could be trust region degeneracy, gradient flow issue, or sequential processing violation

**Solution**:
- If fallback_count > 0: Debug `buffer.py::generate_and_score` to ensure `generation_logps` is stored
- If gradients vanishing: Check learning rate, try increasing from 1e-6 to 1e-5
- If not compliant: Fix `generation.py::sample_k` or `completion_loss.py` to use batch_size=1

---

### Symptom: Multimodal Alignment Errors

**Diagnostics to check**:
1. `multimodal_alignment_checks.json` → Which stage failed? (dataset, buffer, loss)
2. Error message: "ImageTokenMismatchError" with expected vs actual counts

**Root cause**: `<|image_pad|>` count doesn't match `image_grid_thw`

**Solution**:
- Check conversation builder logs: Does placeholder count == len(images)?
- Verify smart resize in augmentation: Are factor-28 constraints met?
- Inspect `image_grid_thw` tensor: Is shape `[num_images, 3]`?

---

### Symptom: OOM Errors

**Diagnostics to check**:
1. `sequential/peak_memory_mb` → Exceeds 42,000?
2. `sequential/is_compliant` → Is batching disabled?

**Root cause**: Sequential processing not enforced, or K completions too large

**Solution**:
- Reduce `sample_k` in config (e.g., from 8 to 4)
- Verify generation calls: `assert input_ids.shape[0] == 1` in `generation.py`
- Enable gradient checkpointing in model config

---

## Configuration Files

### `configs/dense_rl/diagnostic.yaml`

Minimal config for fast diagnostic runs:

```yaml
# Inherits from standard.yaml, overrides for speed
parent: configs/dense_rl/standard.yaml

data:
  train_data_path: data/processed/dense_rl_2k/train.jsonl
  max_samples: 10  # Minimal dataset

training:
  max_steps: 20
  logging_steps: 5

grpo:
  sample_k: 4  # Fewer completions for speed

diagnostics:
  enabled: true
  export_interval: 10  # Export artifacts every 10 steps
  trust_region_threshold: 0.1
  reward_variance_threshold: 0.05
  gradient_flow_layers: [0, 4, 8, 12, 16, 20, 24, 27]  # Sample layers
```

### `configs/dense_rl/checkpoint_diversity_test.yaml`

Checkpoint comparison config:

```yaml
checkpoints:
  phase_2:
    path: outputs/7B-all_tokens/phase_2/9-21-phase_2-last_blocks_6-all_tokens/best-200-eval_loss0.7006
  phase_3:
    path: outputs/7B-all_tokens/phase_3/9-23-phase_3-all_tokens-lower_grounding_weight-last_vision_4-with_text_only-resume/checkpoint-2000

diversity_test:
  num_prompts: 50
  k_completions: 50
  temperatures: [0.7, 0.9, 1.1]
  seed: 42

output:
  output_dir: outputs/checkpoint_diversity
  export_comparison_report: true
```

---

## Next Steps

After running diagnostics:

1. **Review TensorBoard logs** to identify which user story (US1-US6) is failing
2. **Check exported artifacts** in `diagnostics/{step:06d}/` for detailed analysis
3. **If diversity collapse confirmed**: Run checkpoint diversity test, use recommended checkpoint
4. **If algorithmic issue found**: File bug with diagnostic JSON attached
5. **Update plan.md** with findings (per Constitution v4.1.1, capture findings in plan, not temporary docs)

---

## Troubleshooting Checklist

- [ ] Conda env `ms` activated?
- [ ] Working directory is `/data3/Qwen2.5-VL-main`?
- [ ] Config file exists and is valid YAML?
- [ ] Dataset paths are absolute or relative to project root?
- [ ] Checkpoint paths are correct and contain `model.safetensors`?
- [ ] TensorBoard directory writable?
- [ ] GPU memory <40GB per device available?

For additional help, see:
- `src_new/rl/GRPO_README.md` for GRPO implementation details
- `src_new/UNIFIED_DOCUMENTATION.md` for SFT/RL architecture
- `specs/004-grpo-post-training/research.md` for failure mode catalog
