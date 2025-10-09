# Trainer Refactoring Summary

## Critical Changes Made

### 1. Updated Import
```python
# OLD
from src_new.config.rl_config import EnhancedRLConfig

# NEW
from src_new.config.rl_config_v2 import RLConfig
```

### 2. Updated __init__ Signature
```python
# OLD
def __init__(
    self,
    *,
    model: nn.Module,
    tokenizer: Any,
    processor: Any,
    train_dataset: Any,
    val_dataset: Any,
    reward_functions: Sequence[Callable[..., Sequence[float]]],
    reward_names: Sequence[str],
    reward_weights: Sequence[float],
    enhanced_cfg: EnhancedRLConfig,
    raw_config: Dict[str, Any],
    output_dir: str,
)

# NEW
def __init__(
    self,
    *,
    model: nn.Module,
    tokenizer: Any,
    processor: Any,
    train_dataset: Any,
    val_dataset: Any,
    reward_functions: Sequence[Callable[..., Sequence[float]]],
    reward_names: Sequence[str],
    reward_weights: Sequence[float],
    rl_config: RLConfig,  # v2 config only
    output_dir: str,
)
```

### 3. _build_manual_cfg Method - MUST BE REWRITTEN

The current implementation has undefined variables (`raw_training`, `train_cfg`, `grpo_cfg`, etc.) because the signature was updated but the body wasn't.

**Required implementation:**
```python
@staticmethod
def _build_manual_cfg(rl_config: RLConfig) -> ManualTrainerConfig:
    """Build manual config from strict v2 config - NO defaults."""
    
    # All values from typed config
    sample_k = rl_config.sampling.sample_k
    prompt_batch_size = rl_config.sampling.prompt_batch_size
    trajectories_per_cycle = prompt_batch_size * sample_k  # Will be adjusted by world_size
    
    beta_start = rl_config.grpo.beta_start
    beta_schedule = None
    if beta_start > 0.0 and rl_config.grpo.beta_anneal is not None:
        beta_schedule = {
            "type": rl_config.grpo.beta_anneal.type,
            "steps": rl_config.grpo.beta_anneal.steps,
        }
    
    grad_accum = rl_config.training.gradient_accumulation_steps
    
    return ManualTrainerConfig(
        sample_k=sample_k,
        prompt_batch_size=prompt_batch_size,
        trajectories_per_cycle=trajectories_per_cycle,
        reward_average_window=rl_config.sampling.reward_average_window,
        max_new_tokens=rl_config.generation.max_new_tokens,
        min_new_tokens=rl_config.generation.min_new_tokens,
        temperature=rl_config.generation.temperature,
        temperature_schedule="constant",
        top_p=rl_config.generation.top_p,
        repetition_penalty=rl_config.generation.repetition_penalty,
        epsilon_low=rl_config.grpo.epsilon_low,
        epsilon_high=rl_config.grpo.epsilon_high,
        beta=beta_start,
        loss_type=rl_config.grpo.loss_type,
        scale_rewards=rl_config.grpo.scale_rewards,
        mask_truncated_completions=rl_config.grpo.mask_truncated_completions,
        max_advantage_magnitude=rl_config.grpo.max_advantage_magnitude,
        gradient_accumulation_steps=grad_accum,
        per_device_train_batch_size=rl_config.training.per_device_train_batch_size,
        steps_per_generation=grad_accum,
        update_steps=grad_accum,
        logging_steps=rl_config.logging.logging_steps,
        save_steps=rl_config.checkpointing.save_steps,
        max_steps=rl_config.training.max_steps,
        bf16=rl_config.training.bf16,
        standardize_rewards=False,
        cross_rank_advantages=rl_config.normalization.cross_rank_advantages,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
    )
```

### 4. All raw_config Access Must Be Replaced

**Pattern to replace (45+ instances):**

```python
# OLD
dyn = (self.raw_config.get("grpo") or {}).get("dynamic_length", {})
dyn_enabled = bool(dyn.get("enabled", True))

# NEW  
dyn_enabled = self.config.generation.dynamic_length.enabled

# OLD
eval_cfg = self.raw_config.get("evaluation") or {}
self._eval_enabled = bool(eval_cfg.get("enabled", False))

# NEW
self._eval_enabled = self.config.evaluation.enabled

# OLD
tb_dir = self.raw_config.get("tb_dir", "tb_logs")

# NEW
tb_dir = self.config.paths.tb_dir
```

### 5. Evaluation Configuration Access

```python
# OLD (lines ~140-330)
eval_cfg = self.raw_config.get("evaluation") or {}
self._eval_enabled = bool(eval_cfg.get("enabled", False))
self._eval_every_steps = int(eval_cfg.get("eval_every_steps", 100))
# ... etc

# NEW
self._eval_enabled = self.config.evaluation.enabled
self._eval_every_steps = self.config.evaluation.eval_every_steps
self._eval_rounds = self.config.evaluation.rounds
self._eval_per_rank_samples = self.config.evaluation.per_rank_samples
self._eval_save_samples = self.config.evaluation.save_samples
self._eval_log_snippets = self.config.evaluation.log_text_snippets
self._eval_seed = self.config.evaluation.seed
```

### 6. Buffer Generate Call (line ~1200)

```python
# OLD
generation_result = buffer.generate_and_score(
    # ... many raw_config.get() calls
)

# NEW
generation_result = buffer.generate_and_score(
    dyn_enabled=self.config.generation.dynamic_length.enabled,
    dyn_estimator=self.config.generation.dynamic_length.estimator,
    dyn_alpha=self.config.generation.dynamic_length.alpha,
    dyn_eos_margin=self.config.generation.dynamic_length.eos_margin,
    dyn_min_cap=self.config.generation.dynamic_length.min_cap,
    dyn_max_cap=self.config.generation.dynamic_length.max_cap,
    dyn_hard_cap=self.config.generation.dynamic_length.hard_cap,
    # ... all from typed config
)
```

## Status

✅ Import updated
✅ __init__ signature updated  
⚠️ _build_manual_cfg needs body rewrite (current has undefined variables)
❌ raw_config access needs replacement (45+ instances)
❌ Evaluation config access needs update
❌ Buffer calls need update

## Next Steps

1. Fix _build_manual_cfg body (CRITICAL - currently broken)
2. Replace all self.raw_config references with self.config
3. Update evaluation configuration access
4. Update buffer generation calls
5. Test with debug config

