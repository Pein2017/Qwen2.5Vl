# GRPO Prompt Batch Size Training Guide

**Complete guide to prompt batching in GRPO for stable reward aggregation and robust gradient updates**

## 🎯 Overview

The prompt batching system enables GRPO training to accumulate rewards and gradients across multiple prompts before each optimizer update, significantly reducing reward volatility and improving training stability.

### **Key Benefits**
- **Reduced Reward Volatility**: Aggregate rewards across multiple prompts for more stable learning signals
- **Better Gradient Estimates**: More robust policy updates from larger trajectory batches
- **Memory-Efficient**: Sequential generation within GPU memory constraints
- **Distributed-Friendly**: Synchronized accumulation across multiple GPUs with drop_last safety

## 🚀 Quick Start

### **Prerequisites**
1. **Environment**: Activate the conda environment:
   ```bash
   conda activate ms
   ```

2. **Configuration**: Ensure YAML config includes prompt batch settings:
   ```yaml
   prompt_batch:
     prompt_batch_size: 4        # Number of prompts per accumulation cycle
     reward_average_window: 5    # Window for smoothed reward metrics
   
   grpo:
     sample_k: 8                 # Trajectories per prompt
     sample_k_per_rank: false    # Split sample_k across ranks
   ```

### **Running GRPO with Prompt Batching**
```bash
python -m src_new.rl.runner \
  --config configs/dense_rl/grpo_config.yaml \
  --prompt-batch-size 4 \
  --sample-k 8
```

**Important**: CLI flags `--prompt-batch-size` and `--sample-k` must match YAML values. Mismatches cause immediate failure.

## 🏗️ System Architecture

### **Accumulation Workflow**

The training loop follows this pattern:

1. **Sample Prompts**: Draw `prompt_batch_size` prompts from the dataset
2. **Sequential Broadcast**: For each prompt:
   - Broadcast prompt to all ranks
   - Each rank generates `sample_k / world_size` trajectories (when `sample_k_per_rank=false`)
   - Compute rewards and advantages across ranks
   - Accumulate gradients via `loss.backward()`
3. **Optimizer Step**: After collecting `prompt_batch_size × sample_k` total trajectories:
   - Execute `optimizer.step()`
   - Zero gradients
   - Reset accumulation counters

### **Example: 8 GPUs, prompt_batch_size=4, sample_k=8**

```
Prompt 1: All 8 ranks generate 1 trajectory each → 8 trajectories
Prompt 2: All 8 ranks generate 1 trajectory each → 8 trajectories  
Prompt 3: All 8 ranks generate 1 trajectory each → 8 trajectories
Prompt 4: All 8 ranks generate 1 trajectory each → 8 trajectories
─────────────────────────────────────────────────────────────────
Total: 4 prompts × 8 trajectories = 32 trajectories
→ optimizer.step() executes
```

### **Configuration Modes**

#### **Mode 1: Split Sample K (`sample_k_per_rank=false`)**
- **Behavior**: `sample_k` is divided across ranks
- **Per-rank trajectories**: `sample_k / world_size`
- **Total trajectories per prompt**: `sample_k`
- **Requirement**: `sample_k` must be divisible by `world_size`

```yaml
grpo:
  sample_k: 8
  sample_k_per_rank: false

# With world_size=2:
# - Each rank generates 8/2 = 4 trajectories per prompt
# - Total per prompt: 8 trajectories
```

#### **Mode 2: Per-Rank Sample K (`sample_k_per_rank=true`)**
- **Behavior**: Each rank independently generates `sample_k` trajectories
- **Per-rank trajectories**: `sample_k`
- **Total trajectories per prompt**: `sample_k × world_size`
- **Requirement**: No divisibility constraint

```yaml
grpo:
  sample_k: 5
  sample_k_per_rank: true

# With world_size=3:
# - Each rank generates 5 trajectories per prompt
# - Total per prompt: 5 × 3 = 15 trajectories
```

## 🔧 Configuration

### **Key Parameters**

```yaml
# Prompt batch configuration
prompt_batch:
  prompt_batch_size: 4          # Prompts per optimizer update
  reward_average_window: 5      # Cycles for smoothed reward
  watchdog_timeout_seconds: 180 # Optional NCCL timeout

# GRPO sampling
grpo:
  sample_k: 8                   # Trajectories per prompt (or per rank)
  sample_k_per_rank: false      # Trajectory distribution mode
  max_new_tokens: 512
  temperature: 1.0
  top_p: 0.95
  repetition_penalty: 1.05

# Training parameters
training:
  max_steps: 1000
  warmup_steps: 100

# Optimizer
optimizer:
  learning_rates:
    llm: 5.0e-6
    vision: 1.0e-5
    merger: 1.0e-5
  weight_decay: 0.01

# Logging
logging:
  logging_steps: 1              # Log every accumulation cycle

# Checkpointing
checkpointing:
  save_steps: 50
```

### **Drop Last Behavior**

If the dataset has fewer prompts remaining than `prompt_batch_size`:
- ✅ Partial batch is **dropped** (no optimizer step)
- ✅ Gradients are cleared
- ✅ Counters are reset
- ✅ Warning is logged with discard count

```
WARNING: Dropping incomplete prompt batch (end of epoch). 
         prompts=2 completions=16 expected=32
```

## 📊 Telemetry & Monitoring

### **TensorBoard Metrics** (`rl/prompt_batch/*`)

| Metric | Description |
|--------|-------------|
| `fill_ratio` | Collected trajectories / expected trajectories (should be 1.0) |
| `reward_average` | Moving average reward over last N cycles (default 5) |
| `trajectories_collected` | Total valid trajectories in the cycle |
| `dropped_prompts` | Prompts discarded via drop_last |
| `steps` | Global accumulation cycle counter |
| `wall_clock_seconds` | Duration of current accumulation cycle |
| `invalid_fraction` | Invalid trajectories / total trajectories |

### **Console Logging**

**Normal operation**:
```
INFO: Prompt batching configured for 4 prompts × 8 trajectories (expected 32 trajectories per optimizer step)
INFO: Sampling window | sample_k=8 | world_size=2 | per_rank_k=4 | prompts_per_cycle=4 | completions_per_cycle=16
INFO: [Step 1] reward_average=0.45 fill_ratio=1.00 trajectories=32 dropped_prompts=0
```

**Drop last scenario**:
```
WARNING: Dropping incomplete prompt batch (incomplete accumulation before optimizer.step()). 
         prompts=3 completions=24 expected=32
```

## 🔍 Validation & Testing

### **Unit Tests**

```bash
# Test prompt batch configuration
conda run -n ms pytest tests/rl/test_prompt_batch_config.py -v

# Test accumulation logic
conda run -n ms pytest tests/rl/test_prompt_batch_accumulation.py -v

# Test distributed smoke scenario
conda run -n ms pytest tests/rl/test_prompt_batch_smoke.py -v
```

### **Smoke Run**

```bash
# Quick validation with debug config
python -m src_new.rl.runner \
  --config configs/dense_rl/grpo_prompt_batch_smoke.yaml \
  --prompt-batch-size 4 \
  --sample-k 8
```

Expected output:
- ✅ Exactly one `optimizer.step()` after 32 trajectories
- ✅ `fill_ratio=1.0` in telemetry
- ✅ No drop_last warnings (for smoke config with sufficient data)
- ✅ Smoothed reward trending upward

## ⚠️ Common Issues & Solutions

### **Issue 1: Mismatched CLI and YAML Values**

**Symptom**:
```
ValueError: CLI flag --prompt-batch-size must match prompt_batch.prompt_batch_size 
in YAML (expected 4, got 8)
```

**Solution**: Ensure CLI overrides match YAML:
```bash
# YAML: prompt_batch.prompt_batch_size: 4
python -m src_new.rl.runner \
  --config config.yaml \
  --prompt-batch-size 4  # Must match YAML
```

### **Issue 2: sample_k Not Divisible by world_size**

**Symptom** (when `sample_k_per_rank=false`):
```
WARNING: sample_k=7 not divisible by world_size=2; 
         local_sample_k will be 3 (floor division)
```

**Solution**: Use divisible sample_k or enable `sample_k_per_rank`:
```yaml
# Option 1: Make divisible
grpo:
  sample_k: 8  # 8 % 2 = 0 ✅
  sample_k_per_rank: false

# Option 2: Enable per-rank mode
grpo:
  sample_k: 7  # Any value OK
  sample_k_per_rank: true
```

### **Issue 3: Incomplete Cycles Due to Small Dataset**

**Symptom**:
```
WARNING: Dropping incomplete prompt batch (end of epoch). 
         prompts=1 completions=8 expected=32
```

**Solution**: Reduce `prompt_batch_size` or increase dataset size:
```yaml
prompt_batch:
  prompt_batch_size: 2  # Smaller batch for small datasets
```

### **Issue 4: NCCL Timeout / Lagging Ranks**

**Symptom**:
```
ERROR: NCCL operation timed out after 180 seconds
```

**Solution**: 
1. Reduce `prompt_batch_size` to lower memory pressure
2. Increase NCCL timeout:
   ```bash
   export NCCL_TIMEOUT=300
   export NCCL_ASYNC_ERROR_HANDLING=1
   export TORCH_NCCL_BLOCKING_WAIT=1
   ```

### **Issue 5: Low Fill Ratio**

**Symptom**:
```
WARNING: fill_ratio=0.75 (expected 1.0)
```

**Solution**: Check for:
- Generation failures (OOM, invalid outputs)
- Reward computation errors (NaN/Inf values)
- Dataset access issues

## 🔧 Advanced Configuration

### **Dynamic Length Caps**

Enable GT-aware per-sample generation caps:
```yaml
grpo:
  dynamic_length:
    enabled: true
    estimator: "tokenizer"     # or "objects"
    alpha: 1.1                 # Cap multiplier
    eos_margin: 16             # Safety margin
    min_cap: 32                # Minimum cap
    max_cap: 512               # Maximum cap
    hard_cap: true             # Enforce strict limit
```

### **Reward Observation**

Compute metrics even when weight=0:
```yaml
rewards:
  wrappers: 1.0
  coords: 0.5
  bbox_giou: 0.0  # Zero weight but still computed if in observe_rewards

observe_rewards:
  - bbox_giou       # Compute and log even with weight=0
  - quad_l1
```

### **NCCL Safeguards**

For distributed stability:
```bash
# Recommended environment variables
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=180
export NCCL_DEBUG=INFO  # For debugging only
```

## 📈 Performance Tuning

### **Prompt Batch Size Selection**

| Dataset Size | GPU Memory | Recommended `prompt_batch_size` |
|--------------|------------|----------------------------------|
| < 100 samples | 24GB | 2-4 |
| 100-1000 samples | 24GB | 4-8 |
| > 1000 samples | 24GB | 8-16 |
| > 1000 samples | 40GB+ | 16-32 |

### **Memory vs. Stability Tradeoff**

- **Larger `prompt_batch_size`**: More stable rewards, higher memory, slower iteration
- **Smaller `prompt_batch_size`**: Faster iteration, lower memory, more reward variance

Start with `prompt_batch_size=4` and scale based on:
1. Memory headroom (monitor GPU usage)
2. Reward smoothness (check `reward_average` trends)
3. Training speed (iterations per hour)

### **Expected Metrics**

✅ **Good Training Signs**:
- `fill_ratio` consistently 1.0
- `reward_average` trending upward
- `invalid_fraction` < 0.05
- `dropped_prompts` = 0 (except epoch boundaries)
- Smooth loss curves in TensorBoard

❌ **Warning Signs**:
- `fill_ratio` < 0.9 consistently
- `invalid_fraction` > 0.1
- Frequent NCCL timeout warnings
- Flat or declining `reward_average`

---

## 📚 Related Documentation

- **[RL GRPO Overview](../rl_grpo_analysis_and_improvements.md)**: Core GRPO implementation details
- **[Troubleshooting Guide](../TROUBLESHOOTING_GUIDE.md)**: Common issues and solutions
- **[Configuration Guide](../SETUP_AND_CONFIGURATION.md)**: Full YAML schema reference
- **[Spec](../../specs/003-grpo-prompt-batch-size/spec.md)**: Feature specification and requirements
- **[Quickstart](../../specs/003-grpo-prompt-batch-size/quickstart.md)**: Step-by-step smoke test guide

---

**Next Steps**: For implementation details, see the source code in `src_new/rl/grpo_trainer.py` and `src_new/rl/runner.py`.
