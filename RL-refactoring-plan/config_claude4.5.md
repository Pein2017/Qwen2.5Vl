# RL Configuration System: Updated Refactoring Plan (Oct 2025)

## Executive Summary

After auditing the current state, **significant YAML restructuring has already been completed**, but the Python code (`src_new/config/rl_config.py`, `src_new/rl/runner.py`, `src_new/rl/grpo_trainer.py`) has not caught up. This updated plan focuses on:

1. **Completing remaining YAML migrations** (paths, experiment, generation separation)
2. **Building typed v2 config system** to consume the new YAML structure
3. **Eliminating raw_config drilling** in trainer and runner
4. **Gradual migration** with backward compatibility

---

## Current State Analysis

### ✅ Already Completed (YAML)

The YAML configs have been **partially migrated** with these sections working:

```yaml
# ✅ Prompt batching (DONE)
prompt_batch:
  prompt_batch_size: 8
  reward_average_window: 5

# ✅ Normalization (DONE)
normalization:
  cross_rank_advantages: true

# ✅ Rewards structure (DONE)
rewards:
  bbox_giou: 0.40
  # ... other weights

rewards_config:
  clip_sigma: 5.0
  tau_iou: 0.5
  length_vs_gt:
    estimator: tokenizer
    lower: 0.7
    upper: 1.2
    gamma: 3.0
    tail_numeric_weight: 0.4

# ✅ Nested GRPO structures (DONE)
grpo:
  dynamic_length:
    enabled: true
    estimator: tokenizer
    alpha: 1.1
    # ...
  beta_anneal:
    type: cosine
    steps: 2000

# ✅ Evaluation (DONE in standard.yaml)
evaluation:
  enabled: true
  eval_every_steps: 50
  per_rank_samples: 1
  save_samples: 20
  seed: 17
```

### ❌ Still Missing (YAML)

```yaml
# ❌ Paths are scattered at root
model_path: "..."          # Should be paths.model_path
train_data_path: "..."     # Should be paths.train_data_path
data_root: "..."           # Should be paths.data_root
tb_dir: "..."              # Should be paths.tb_dir

# ❌ Experiment metadata scattered
output:
  run_name: "..."          # Should be experiment.run_name
runtime:
  seed: 17                 # Should be experiment.seed

# ❌ Generation params mixed in grpo
grpo:
  max_new_tokens: 1500     # Should be generation.max_new_tokens
  temperature: 1.0         # Should be generation.temperature  
  top_p: 0.95              # Should be generation.top_p
  repetition_penalty: 1.05 # Should be generation.repetition_penalty
  sample_k: 8              # Should be sampling.sample_k
```

### ❌ Code Not Updated

**`src_new/config/rl_config.py`:**
- `EnhancedRLConfig` doesn't parse new structures properly
- Missing: `PromptBatchConfig`, `NormalizationConfig`, `RewardsConfig` (structured), `EvaluationConfig`
- No version detection or migration path

**`src_new/rl/runner.py` & `grpo_trainer.py`:**
- 45+ instances of `raw_config.get()` drilling
- Manual config building mixes `enhanced_cfg` and `raw_config`
- No typed access for new YAML sections

---

## Revised Architecture

### 1. Complete Config Hierarchy (v2)

```python
@dataclass(frozen=True)
class RLConfig:
    """Root RL configuration - fully typed, no raw dict drilling."""
    
    # NEW: All paths centralized
    paths: PathsConfig
    
    # NEW: Experiment tracking
    experiment: ExperimentConfig
    
    # Existing (enhanced)
    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    
    # NEW: Separated concerns
    training: TrainingConfig
    optimizer: OptimizerConfig
    grpo: GRPOConfig              # Algorithm only (epsilon, beta, loss_type)
    sampling: SamplingConfig       # sample_k, prompt_batch_size
    generation: GenerationConfig   # max_new_tokens, temperature, etc.
    
    # NEW: From existing YAML
    normalization: NormalizationConfig
    rewards: RewardsConfig
    evaluation: EvaluationConfig
    
    # Existing
    layer_freezing: LayerFreezingConfig
    logging: LoggingConfig
    checkpointing: CheckpointingConfig
    runtime: RuntimeConfig
```

### 2. New Config Classes

#### **PathsConfig** (NEW)
```python
@dataclass(frozen=True)
class PathsConfig:
    """All file paths centralized."""
    model_path: str
    ref_model_path: Optional[str] = None
    train_data_path: str
    val_data_path: str
    data_root: str
    output_dir: str
    tb_dir: str
```

#### **ExperimentConfig** (NEW)
```python
@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment metadata and reproducibility."""
    run_name: str
    seed: int
    tags: List[str] = field(default_factory=list)
```

#### **SamplingConfig** (NEW - extracted from grpo + prompt_batch)
```python
@dataclass(frozen=True)
class SamplingConfig:
    """Prompt batching and K-sampling."""
    prompt_batch_size: int
    sample_k: int
    sample_k_per_rank: bool = False
    reward_average_window: int = 5
```

#### **GenerationConfig** (NEW - extracted from grpo)
```python
@dataclass(frozen=True)
class GenerationConfig:
    """Generation-specific parameters."""
    max_new_tokens: int
    min_new_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    dynamic_length: DynamicLengthConfig = field(default_factory=lambda: DynamicLengthConfig())

@dataclass(frozen=True)
class DynamicLengthConfig:
    enabled: bool = True
    estimator: str = "tokenizer"
    alpha: float = 1.1
    eos_margin: int = 16
    min_cap: int = 64
    max_cap: int = 1200
    hard_cap: bool = True
```

#### **GRPOConfig** (REFACTORED - algorithm only)
```python
@dataclass(frozen=True)
class GRPOConfig:
    """Pure GRPO algorithm parameters (NO generation/sampling)."""
    epsilon_low: float
    epsilon_high: float
    beta_start: float = 0.0
    beta_anneal: Optional[BetaAnnealConfig] = None
    loss_type: str = "grpo"
    scale_rewards: bool = True
    max_advantage_magnitude: Optional[float] = None
    mask_truncated_completions: bool = False

@dataclass(frozen=True)
class BetaAnnealConfig:
    type: str  # "linear" or "cosine"
    steps: int
```

#### **NormalizationConfig** (NEW - from existing YAML)
```python
@dataclass(frozen=True)
class NormalizationConfig:
    cross_rank_advantages: bool = True
```

#### **RewardsConfig** (ENHANCED - from existing YAML)
```python
@dataclass(frozen=True)
class RewardsConfig:
    weights: Dict[str, float]
    observe_only: List[str] = field(default_factory=list)
    config: RewardParamsConfig = field(default_factory=lambda: RewardParamsConfig())

@dataclass(frozen=True)
class RewardParamsConfig:
    clip_sigma: float = 5.0
    tau_iou: float = 0.5
    tau_quad: float = 0.02
    tau_line: float = 0.02
    length_vs_gt: LengthVsGTConfig = field(default_factory=lambda: LengthVsGTConfig())

@dataclass(frozen=True)
class LengthVsGTConfig:
    estimator: str = "tokenizer"
    lower: float = 0.7
    upper: float = 1.2
    gamma: float = 3.0
    tail_numeric_weight: float = 0.4
```

#### **EvaluationConfig** (NEW - from existing standard.yaml)
```python
@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    eval_every_steps: int = 0
    rounds: int = 1
    per_rank_samples: int = 1
    save_samples: int = 20
    log_text_snippets: bool = True
    seed: int = 17
```

---

## Implementation Plan (Revised)

### Phase 1: Complete YAML Migration (1 day)

**Goal:** Finish restructuring YAML files to match the v2 schema

#### Step 1.1: Add Missing Sections to Base

Update `configs/dense_rl/dense_base.yaml`:

```yaml
# ===== PATHS (base provides none; all required in child) =====
# Children MUST provide: model_path, ref_model_path, train_data_path, 
# val_data_path, data_root, output_dir, tb_dir

# ===== EXPERIMENT (base provides defaults) =====
experiment:
  tags: []
  # Children MUST provide: run_name, seed

# ===== SAMPLING (extract from current structure) =====
sampling:
  sample_k_per_rank: false
  reward_average_window: 5
  # Children MUST provide: prompt_batch_size, sample_k

# ===== GENERATION (extract from current grpo) =====
generation:
  min_new_tokens: 0
  top_k: 50
  # Children provide: max_new_tokens, temperature, top_p, repetition_penalty
  dynamic_length:
    enabled: true
    estimator: tokenizer
    alpha: 1.1
    eos_margin: 16
    min_cap: 64
    max_cap: 1200
    hard_cap: true

# ===== GRPO (algorithm only - remove generation params) =====
grpo:
  epsilon_low: 0.2
  epsilon_high: 0.2
  beta_start: 0.0
  loss_type: grpo
  scale_rewards: true
  mask_truncated_completions: false
  # beta_anneal provided by children
  # max_advantage_magnitude provided by children
```

#### Step 1.2: Update debug.yaml

```yaml
extends: [./dense_base.yaml]

# ===== PATHS =====
paths:
  model_path: "outputs/7B-all_tokens/phase_3/.../checkpoint-2000"
  ref_model_path: "outputs/7B-all_tokens/phase_3/.../checkpoint-2000"
  train_data_path: "data/ds_v2_full/train.jsonl"
  val_data_path: "data/ds_v2_full/val.jsonl"
  data_root: "data/ds_v2_full"
  output_dir: "outputs/rl_debug/10-5"
  tb_dir: "tb_dense"

# ===== EXPERIMENT =====
experiment:
  run_name: "debug"
  seed: 17
  tags: ["debug", "fast-iteration"]

# ===== SAMPLING =====
sampling:
  prompt_batch_size: 2
  sample_k: 8
  sample_k_per_rank: false

# ===== GENERATION =====
generation:
  max_new_tokens: 1500
  temperature: 1.0
  top_p: 0.95
  repetition_penalty: 1.05
  dynamic_length:
    enabled: true
    max_cap: 800  # Override base

# ===== GRPO =====
grpo:
  beta_start: 0.05
  beta_anneal:
    type: cosine
    steps: 80
  max_advantage_magnitude: 5.0

# ... rest stays the same (normalization, rewards, evaluation, etc.)
```

#### Step 1.3: Update standard.yaml

Same pattern as debug.yaml but with production values.

### Phase 2: Create v2 Config Module (2 days)

**Goal:** Build typed config system without breaking existing code

#### Step 2.1: Create `src_new/config/rl_config_v2.py`

```python
"""RL Configuration v2 - Typed hierarchical config system."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import all the dataclass definitions from above
# ...

@dataclass(frozen=True)
class RLConfig:
    """Root v2 configuration."""
    paths: PathsConfig
    experiment: ExperimentConfig
    model: ModelConfig
    # ... all configs
    
    @staticmethod
    def from_yaml_dict(cfg: Dict[str, Any]) -> "RLConfig":
        """Factory from YAML dict - handles both old and new structure."""
        
        # Detect if using new structure (presence of 'paths' section)
        using_v2_structure = "paths" in cfg and isinstance(cfg["paths"], dict)
        
        if using_v2_structure:
            # NEW STRUCTURE - direct mapping
            paths = PathsConfig(
                model_path=cfg["paths"]["model_path"],
                ref_model_path=cfg["paths"].get("ref_model_path"),
                train_data_path=cfg["paths"]["train_data_path"],
                val_data_path=cfg["paths"]["val_data_path"],
                data_root=cfg["paths"]["data_root"],
                output_dir=cfg["paths"]["output_dir"],
                tb_dir=cfg["paths"]["tb_dir"],
            )
            
            experiment = ExperimentConfig(
                run_name=cfg["experiment"]["run_name"],
                seed=cfg["experiment"]["seed"],
                tags=cfg["experiment"].get("tags", []),
            )
            
            sampling = SamplingConfig(
                prompt_batch_size=cfg["sampling"]["prompt_batch_size"],
                sample_k=cfg["sampling"]["sample_k"],
                sample_k_per_rank=cfg["sampling"].get("sample_k_per_rank", False),
                reward_average_window=cfg["sampling"].get("reward_average_window", 5),
            )
            
            generation = GenerationConfig(
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                min_new_tokens=cfg["generation"].get("min_new_tokens", 0),
                temperature=cfg["generation"]["temperature"],
                top_p=cfg["generation"]["top_p"],
                top_k=cfg["generation"].get("top_k", 50),
                repetition_penalty=cfg["generation"]["repetition_penalty"],
                dynamic_length=DynamicLengthConfig(
                    **cfg["generation"].get("dynamic_length", {})
                ),
            )
            
        else:
            # OLD STRUCTURE - compatibility adapter
            paths = PathsConfig(
                model_path=cfg["model_path"],
                ref_model_path=cfg.get("ref_model_path"),
                train_data_path=cfg["train_data_path"],
                val_data_path=cfg["val_data_path"],
                data_root=cfg["data_root"],
                output_dir=cfg.get("output", {}).get("output_dir", "outputs/rl"),
                tb_dir=cfg.get("tb_dir", "tb_logs"),
            )
            
            experiment = ExperimentConfig(
                run_name=cfg.get("output", {}).get("run_name", "rl_run"),
                seed=cfg.get("runtime", {}).get("seed", 42),
                tags=cfg.get("experiment", {}).get("tags", []),
            )
            
            # Extract from old locations
            grpo_raw = cfg.get("grpo", {})
            prompt_batch_raw = cfg.get("prompt_batch", {})
            
            sampling = SamplingConfig(
                prompt_batch_size=prompt_batch_raw.get("prompt_batch_size", 4),
                sample_k=grpo_raw.get("sample_k", 4),
                sample_k_per_rank=grpo_raw.get("sample_k_per_rank", False),
                reward_average_window=prompt_batch_raw.get("reward_average_window", 5),
            )
            
            generation = GenerationConfig(
                max_new_tokens=grpo_raw.get("max_new_tokens", 128),
                min_new_tokens=grpo_raw.get("min_new_tokens", 0),
                temperature=grpo_raw.get("temperature", 1.0),
                top_p=grpo_raw.get("top_p", 0.95),
                top_k=grpo_raw.get("top_k", 50),
                repetition_penalty=grpo_raw.get("repetition_penalty", 1.05),
                dynamic_length=DynamicLengthConfig(
                    **grpo_raw.get("dynamic_length", {})
                ),
            )
        
        # Build GRPO (algorithm only)
        grpo_raw = cfg.get("grpo", {})
        beta_anneal_dict = grpo_raw.get("beta_anneal")
        beta_anneal = (
            BetaAnnealConfig(**beta_anneal_dict) if beta_anneal_dict else None
        )
        
        grpo = GRPOConfig(
            epsilon_low=grpo_raw.get("epsilon_low", 0.2),
            epsilon_high=grpo_raw.get("epsilon_high", 0.2),
            beta_start=grpo_raw.get("beta_start", 0.0),
            beta_anneal=beta_anneal,
            loss_type=grpo_raw.get("loss_type", "grpo"),
            scale_rewards=grpo_raw.get("scale_rewards", True),
            max_advantage_magnitude=grpo_raw.get("max_advantage_magnitude"),
            mask_truncated_completions=grpo_raw.get("mask_truncated_completions", False),
        )
        
        # Normalization (already in YAML)
        norm_raw = cfg.get("normalization", {})
        normalization = NormalizationConfig(
            cross_rank_advantages=norm_raw.get("cross_rank_advantages", True),
        )
        
        # Rewards (already in YAML)
        rewards = RewardsConfig(
            weights=cfg.get("rewards", {}),
            observe_only=cfg.get("observe_rewards", []),
            config=RewardParamsConfig(
                clip_sigma=cfg.get("rewards_config", {}).get("clip_sigma", 5.0),
                tau_iou=cfg.get("rewards_config", {}).get("tau_iou", 0.5),
                tau_quad=cfg.get("rewards_config", {}).get("tau_quad", 0.02),
                tau_line=cfg.get("rewards_config", {}).get("tau_line", 0.02),
                length_vs_gt=LengthVsGTConfig(
                    **cfg.get("rewards_config", {}).get("length_vs_gt", {})
                ),
            ),
        )
        
        # Evaluation (already in standard.yaml)
        eval_raw = cfg.get("evaluation", {})
        evaluation = EvaluationConfig(
            enabled=eval_raw.get("enabled", False),
            eval_every_steps=eval_raw.get("eval_every_steps", 0),
            rounds=eval_raw.get("rounds", 1),
            per_rank_samples=eval_raw.get("per_rank_samples", 1),
            save_samples=eval_raw.get("save_samples", 20),
            log_text_snippets=eval_raw.get("log_text_snippets", True),
            seed=eval_raw.get("seed", 17),
        )
        
        # Build other configs (model, data, loss, training, optimizer, etc.)
        # Re-use existing from_dict methods where they exist
        
        return RLConfig(
            paths=paths,
            experiment=experiment,
            model=model,
            data=data,
            loss=loss,
            training=training,
            optimizer=optimizer,
            grpo=grpo,
            sampling=sampling,
            generation=generation,
            normalization=normalization,
            rewards=rewards,
            layer_freezing=layer_freezing,
            logging=logging_cfg,
            checkpointing=checkpointing,
            evaluation=evaluation,
            runtime=runtime,
        )
```

#### Step 2.2: Keep `EnhancedRLConfig` for Compatibility

Keep existing `EnhancedRLConfig` alongside v2 for gradual migration.

### Phase 3: Update Runner (1 day)

**Goal:** Use v2 config in runner, eliminate raw_config

#### Step 3.1: Update `src_new/rl/runner.py`

```python
def build_components(config_path: str) -> Dict[str, Any]:
    """Build components using v2 config."""
    from src_new.config.rl_config_v2 import RLConfig
    
    cfg_dict = _load_yaml(config_path)
    rl_config = RLConfig.from_yaml_dict(cfg_dict)
    
    # Use typed access throughout
    tokenizer, processor = load_tokenizer_and_processor(
        rl_config.paths.model_path,
        attn_implementation=rl_config.model.attn_implementation,
    )
    
    model = load_model(
        rl_config.paths.model_path,
        torch_dtype=rl_config.model.torch_dtype,
        # ... use typed config
    )
    
    return {
        "config": rl_config,  # Pass v2 config
        "tokenizer": tokenizer,
        "processor": processor,
        "model": model,
    }

def train(config_path: str, **kwargs) -> None:
    comps = build_components(config_path)
    rl_config = comps["config"]  # Type: RLConfig
    
    # Build trainer with typed config (NO raw_config)
    trainer = BBUGRPOTrainer(
        model=comps["model"],
        tokenizer=comps["tokenizer"],
        processor=comps["processor"],
        train_dataset=comps["train_dataset"],
        val_dataset=comps["val_dataset"],
        reward_functions=comps["reward_functions"],
        reward_names=comps["reward_names"],
        reward_weights=comps["reward_weights"],
        rl_config=rl_config,  # Pass v2 config only
        output_dir=rl_config.paths.output_dir,
    )
    trainer.train()
```

### Phase 4: Update Trainer (2 days)

**Goal:** Remove all raw_config access, use typed config

#### Step 4.1: Update `BBUGRPOTrainer.__init__`

```python
class BBUGRPOTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        processor: Any,
        train_dataset: Any,
        val_dataset: Any,
        reward_functions: Sequence[Callable],
        reward_names: Sequence[str],
        reward_weights: Sequence[float],
        rl_config: RLConfig,  # v2 config only
        output_dir: str,
    ):
        self.config = rl_config
        
        # Clean typed access
        self.output_dir = rl_config.paths.output_dir
        self._run_name = rl_config.experiment.run_name
        
        # Build manual config from typed source
        self.manual_cfg = ManualTrainerConfig(
            sample_k=rl_config.sampling.sample_k,
            prompt_batch_size=rl_config.sampling.prompt_batch_size,
            reward_average_window=rl_config.sampling.reward_average_window,
            max_new_tokens=rl_config.generation.max_new_tokens,
            min_new_tokens=rl_config.generation.min_new_tokens,
            temperature=rl_config.generation.temperature,
            top_p=rl_config.generation.top_p,
            repetition_penalty=rl_config.generation.repetition_penalty,
            epsilon_low=rl_config.grpo.epsilon_low,
            epsilon_high=rl_config.grpo.epsilon_high,
            beta_start=rl_config.grpo.beta_start,
            scale_rewards=rl_config.grpo.scale_rewards,
            max_advantage_magnitude=rl_config.grpo.max_advantage_magnitude,
            # ... all from typed config
        )
        
        # NO raw_config anywhere
```

#### Step 4.2: Replace All Raw Config Access

```python
# OLD (raw_config drilling)
dyn = (self.raw_config.get("grpo") or {}).get("dynamic_length", {})
dyn_enabled = bool(dyn.get("enabled", True))

# NEW (typed access)
dyn_enabled = self.config.generation.dynamic_length.enabled

# OLD
eval_cfg = self.raw_config.get("evaluation") or {}
self._eval_enabled = bool(eval_cfg.get("enabled", False))

# NEW
self._eval_enabled = self.config.evaluation.enabled
self._eval_every_steps = self.config.evaluation.eval_every_steps
```

#### Step 4.3: Update Buffer Call

```python
generation_result = buffer.generate_and_score(
    # ... existing params
    # Dynamic length params from typed config
    dyn_enabled=self.config.generation.dynamic_length.enabled,
    dyn_alpha=self.config.generation.dynamic_length.alpha,
    dyn_eos_margin=self.config.generation.dynamic_length.eos_margin,
    dyn_min_cap=self.config.generation.dynamic_length.min_cap,
    dyn_max_cap=self.config.generation.dynamic_length.max_cap,
    dyn_estimator=self.config.generation.dynamic_length.estimator,
    dyn_hard_cap=self.config.generation.dynamic_length.hard_cap,
)
```

### Phase 5: Testing & Validation (1 day)

**Goal:** Ensure both old and new configs work

#### Validation Checklist

- [ ] Old YAML structure still works (backward compat in from_yaml_dict)
- [ ] New YAML structure works
- [ ] `debug.yaml` loads and runs (10 steps smoke test)
- [ ] `standard.yaml` loads and runs
- [ ] No `raw_config.get()` calls in trainer
- [ ] All typed config access works
- [ ] Evaluation harness works
- [ ] TensorBoard logging complete

### Phase 6: Cleanup (0.5 days)

**Goal:** Remove legacy code, finalize docs

1. Archive old configs to `configs/dense_rl/legacy_v1/`
2. Update `configs/dense_rl/README.md` with v2 structure
3. Update `src_new/rl/grpo_readme.md` to reference new config sections
4. Add migration guide for users with custom configs

---

## Updated YAML Structure (Final)

### Base Config
```yaml
# configs/dense_rl/dense_base.yaml

# ===== MODEL =====
model:
  torch_dtype: bfloat16
  use_cache: false
  trust_remote_code: true
  image_max_pixels: 401408

# ===== SAMPLING =====
sampling:
  sample_k_per_rank: false
  reward_average_window: 5

# ===== GENERATION =====
generation:
  min_new_tokens: 0
  top_k: 50
  dynamic_length:
    enabled: true
    estimator: tokenizer
    alpha: 1.1
    eos_margin: 16
    min_cap: 64
    max_cap: 1200
    hard_cap: true

# ===== GRPO (algorithm only) =====
grpo:
  epsilon_low: 0.2
  epsilon_high: 0.2
  beta_start: 0.0
  loss_type: grpo
  scale_rewards: true
  mask_truncated_completions: false

# ===== NORMALIZATION =====
normalization:
  cross_rank_advantages: true

# ... other universal sections
```

### Debug Config
```yaml
# configs/dense_rl/debug.yaml

extends: [./dense_base.yaml]

# ===== PATHS =====
paths:
  model_path: "outputs/.../checkpoint-2000"
  ref_model_path: "outputs/.../checkpoint-2000"
  train_data_path: "data/ds_v2_full/train.jsonl"
  val_data_path: "data/ds_v2_full/val.jsonl"
  data_root: "data/ds_v2_full"
  output_dir: "outputs/rl_debug/10-5"
  tb_dir: "tb_dense"

# ===== EXPERIMENT =====
experiment:
  run_name: "debug"
  seed: 17
  tags: ["debug", "fast-iteration"]

# ===== SAMPLING =====
sampling:
  prompt_batch_size: 2
  sample_k: 8

# ===== GENERATION =====
generation:
  max_new_tokens: 1500
  temperature: 1.0
  top_p: 0.95
  repetition_penalty: 1.05

# ===== GRPO =====
grpo:
  beta_start: 0.05
  beta_anneal:
    type: cosine
    steps: 80
  max_advantage_magnitude: 5.0

# ===== REWARDS =====
rewards:
  bbox_giou: 0.40
  # ... other weights

rewards_config:
  clip_sigma: 5.0
  tau_iou: 0.5
  length_vs_gt:
    estimator: tokenizer
    lower: 0.7
    upper: 1.2
    gamma: 3.0

# ===== EVALUATION =====
evaluation:
  enabled: false  # Debug: skip eval
  eval_every_steps: 0
```

---

## Timeline (Updated)

| Phase | Duration | Risk | Deliverable |
|-------|----------|------|-------------|
| **Phase 1: Complete YAML** | 1 day | Low | Updated base/debug/standard configs |
| **Phase 2: v2 Config Module** | 2 days | Medium | `rl_config_v2.py` with backward compat |
| **Phase 3: Update Runner** | 1 day | Medium | Typed runner, no raw_config |
| **Phase 4: Update Trainer** | 2 days | High | Typed trainer, no raw_config |
| **Phase 5: Testing** | 1 day | Low | Validation suite |
| **Phase 6: Cleanup** | 0.5 days | Low | Docs, migration guide |
| **Total** | **7.5 days** | - | Production-ready v2 |

---

## Benefits Summary

### Before (Current Mixed State)
- ✅ Some YAML structure exists (prompt_batch, normalization, rewards_config)
- ❌ Paths scattered at root
- ❌ Generation params mixed in grpo
- ❌ Extensive raw_config drilling (45+ .get() calls)
- ❌ No IDE autocomplete for config access

### After (v2)
```python
# Clean typed access everywhere
model_path = config.paths.model_path
temp = config.generation.temperature
sample_k = config.sampling.sample_k
dyn_enabled = config.generation.dynamic_length.enabled
eval_steps = config.evaluation.eval_every_steps

# NO raw_config anywhere
```

### Immediate Wins
1. **Full IDE autocomplete** on all config fields
2. **Type safety** - errors at load time, not runtime
3. **Clear separation** - generation/sampling/algorithm concerns separated
4. **Backward compatible** - old configs still work during migration
5. **Fail-fast validation** - missing keys surface immediately

---

## Next Steps

1. ✅ Review this updated plan
2. **Phase 1** (1 day): Complete YAML restructuring (paths, experiment, sampling, generation sections)
3. **Phase 2** (2 days): Create `rl_config_v2.py` with backward compat adapter
4. **Phase 3** (1 day): Update runner to use v2 config
5. **Phase 4** (2 days): Update trainer to eliminate raw_config
6. **Phase 5** (1 day): Validation and testing
7. **Phase 6** (0.5 days): Cleanup and docs

---

**Status**: ✅ **UPDATED - READY FOR IMPLEMENTATION**  
**Last Updated**: 2025-10-08 (Revised after YAML audit)
