# Contract: GRPO Prompt Batching Controls

**Audience**: RL operators configuring dense captioning GRPO runs  
**Source Spec**: [spec.md](../spec.md)  
**Updated**: 2025-10-08

---

## 1. CLI / Launch Contract

### Entry Command
```
source ~/.bashrc && conda activate ms
python -m src_new.rl.runner \
  --config /abs/path/to/grpo_config.yaml \
  --prompt-batch-size 16 \
  --sample-k 8
```

### Required Flags
| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--config` | path | YES | Points to base YAML config (existing schema). Must include optimizer, dataset, reward settings. |
| `--prompt-batch-size` | int > 0 | YES | Overrides/validates `prompt_batch.prompt_batch_size` in YAML; mismatch causes failure. |
| `--sample-k` | int > 0 | YES | Overrides/validates `grpo.sample_k`. |
| `--simulate-rank-lag` | int ≥ 0 | NO | Optional test hook; when provided, rank `n` sleeps to trigger watchdog warnings. |

### Failure Modes
- Missing mandatory flags → immediate exit with constitution reference.
- Non-integer or ≤0 values → validation error before launching Accelerate/DDP.
- Run aborts if runtime accumulation falls short of the expected trajectory count before `optimizer.step()`.
- CLI overrides (`--sample-k`, `--prompt-batch-size`) MUST match YAML values; mismatches are fatal to prevent silent drift.
- `--simulate-rank-lag` emits WARN-level logs identifying the impacted rank and triggers NCCL watchdog messages; the run should exit gracefully without deadlock.

---

## 2. YAML Configuration Schema Additions

Extend existing `src_new/config/rl_training.yaml` schema.

```yaml
training:
  prompt_batch_size: 16        # int > 0
# `sample_k` continues to live under the `grpo` block.
```

### Validation Rules
1. `prompt_batch_size * sample_k` determines the expected trajectory count per cycle; if runtime accumulation falls short the runner MUST apply `drop_last` and skip `optimizer.step()` for that cycle.
2. Schema updates MUST be reflected in the frozen dataclass (no optional defaults beyond those listed).

---

## 3. Telemetry Contract

### Scalar Tags (`TensorBoard` + console)
| Tag | Type | Description |
|-----|------|-------------|
| `rl/prompt_batch/fill_ratio` | float | Ratio of collected trajectories to expected total per cycle. |
| `rl/prompt_batch/reward_average` | float | Moving-average reward over the last 5 cycles (configurable). |
| `rl/prompt_batch/invalid_fraction` | float | Invalid trajectories divided by total trajectories. |
| `rl/prompt_batch/trajectories_collected` | float | Count of valid trajectories in the cycle. |
| `rl/prompt_batch/dropped_prompts` | float | Number of prompts discarded via `drop_last` in the cycle. |
| `rl/prompt_batch/steps` | float | Accumulation cycle index (global counter). |
| `rl/prompt_batch/lagging_rank` | int | Latest rank ID exceeding latency threshold (only logged when WARN). |
| `rl/prompt_batch/wall_clock_seconds` | float | Duration of current accumulation cycle. |

### Logging Guarantees
- Console emits INFO on cycle completion and WARN on lagging ranks or shortfalls.
- WARN logs include remediation guidance (reduce prompt_batch_size, investigate slow ranks, etc.).
- ERROR emitted when accumulation aborts due to unmet trajectory count or persistent invalid trajectories.

---

**Compliance**: All commands must be executed within `conda activate ms` and respect bf16 policy. Any deviation requires governance approval documented in the feature tasks.
