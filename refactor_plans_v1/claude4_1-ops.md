# Qwen2.5-VL Unified Training Framework - Refactoring Plan

**Objective**: Merge `src_rl/` and `src_new/` into a single, modular training framework that treats SFT and GRPO as different training modes sharing core components.

---

## 1. Current Architecture Analysis

### Shared Components (Already Working)
- **ConversationBuilder**: Both use `src_new.processing.conversation.builder`
- **Model Loading**: Both use `DetectionModel` wrapper with same validation logic
- **Data Format**: Both consume same JSONL format with image alignment
- **Tokenizer/Processor**: Identical setup with HF-first parity

### Duplicated Components (Need Unification)
- **Configuration Systems**: `src_rl/config.py` vs `src_new/config/`
- **Dataset Classes**: `RLDenseJSONLDataset` vs `Dataset`
- **Trainer Logic**: `VisionGRPOTrainer` vs `BBUTrainer`
- **Logging/Checkpointing**: Separate implementations
- **Component Builders**: Duplicated model/tokenizer loading

### Key Differences (Preserve)
- **Loss Computation**: Grouped LLM losses (SFT) vs Reward functions (GRPO)
- **Training Loop**: HF Trainer vs TRL GRPOTrainer
- **Generation**: Single-pass (SFT) vs Multi-sample (GRPO)

---

## 2. Unified Architecture Design

```
src_unified/
├── core/                          # Shared core components
│   ├── config/                    # Unified configuration
│   │   ├── schema.py             # Training mode schemas (SFT/GRPO)
│   │   ├── loader.py             # Config loading and validation
│   │   └── validator.py          # Common validation logic
│   ├── data/                      # Unified data processing
│   │   ├── dataset.py            # Unified Dataset class
│   │   ├── collator.py           # Shared collation logic
│   │   └── conversation.py       # Conversation processing wrapper
│   ├── models/                    # Shared model components
│   │   ├── loader.py             # Model/tokenizer/processor loading
│   │   ├── wrapper.py            # DetectionModel wrapper (from src_new)
│   │   └── validator.py          # Image/text alignment validation
│   ├── training/                  # Shared training infrastructure
│   │   ├── manager.py            # Unified training manager
│   │   ├── freezing.py           # Phase freezing manager
│   │   ├── checkpointing.py      # Unified checkpoint saving
│   │   ├── logging.py            # Unified logging/metrics
│   │   └── callbacks.py          # Training callbacks
│   └── utils/                     # Common utilities
│       ├── paths.py              # Path resolution
│       └── devices.py            # Device/dtype utilities
├── sft/                           # SFT-specific components
│   ├── trainer.py                # SFT trainer (BBUTrainer logic)
│   ├── losses.py                 # Grouped LLM losses
│   └── metrics.py                # SFT-specific metrics
├── grpo/                          # GRPO-specific components
│   ├── trainer.py                # GRPO trainer (VisionGRPOTrainer)
│   ├── rewards/                  # Reward functions
│   │   ├── registry.py
│   │   └── format_rewards.py
│   └── metrics.py                # GRPO-specific metrics
├── inference.py                   # Unified inference engine
├── train.py                       # Unified training entry point
└── eval.py                        # Unified evaluation
```

---

## 3. Core Module Specifications

### 3.1 Unified Configuration (`core/config/`)

**Design**: Single config schema supporting both training modes with mode-specific sections.

```python
@dataclass
class UnifiedConfig:
    # Shared sections
    model: ModelConfig              # Model path, attention, dtype, etc.
    data: DataConfig               # Paths, data_root, teacher_pool
    runtime: RuntimeConfig         # Seed, device, precision
    logging: LoggingConfig         # TensorBoard, metrics, checkpointing
    freezing: FreezingConfig       # Phase-based parameter freezing
    
    # Mode-specific sections
    training_mode: Literal["sft", "grpo"]
    sft: Optional[SFTConfig]       # SFT-specific: loss weights, variants
    grpo: Optional[GRPOConfig]     # GRPO-specific: sampling, rewards
```

**Benefits**:
- Single source of truth for configuration
- Mode switching via `training_mode` flag
- Shared validation logic
- Backward compatibility with existing YAMLs

### 3.2 Unified Dataset (`core/data/dataset.py`)

**Design**: Single dataset class that adapts behavior based on training mode.

```python
class UnifiedDataset:
    def __init__(self, config: UnifiedConfig, split: str):
        self.mode = config.training_mode
        self.conversation_builder = ConversationBuilder(...)
        # Initialize based on mode
        
    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self._load_base_sample(idx)
        
        if self.mode == "sft":
            return self._process_sft_sample(sample)
        elif self.mode == "grpo":
            return self._process_grpo_sample(sample)
            
    def _process_sft_sample(self, sample) -> Dict[str, Any]:
        # SFT processing: teacher pairing, augmentation, spans
        
    def _process_grpo_sample(self, sample) -> Dict[str, Any]:
        # GRPO processing: simple generation inputs, metadata
```

**Benefits**:
- Single data loading pipeline
- Shared conversation building and image processing
- Mode-specific optimizations without duplication
- Consistent tensor formats

### 3.3 Unified Model Loading (`core/models/loader.py`)

**Design**: Extract common model loading logic from both modules.

```python
class ModelLoader:
    @staticmethod
    def load_components(config: UnifiedConfig) -> Dict[str, Any]:
        """Load tokenizer, processor, and model with HF-first parity."""
        # Apply Qwen2.5-VL patches
        # Load tokenizer with fast tokenizer validation
        # Load image/video processors
        # Create unified processor with chat template
        # Load DetectionModel wrapper
        # Return {"tokenizer", "processor", "model"}
```

**Benefits**:
- Single implementation of model loading
- Consistent validation across modes
- Shared patch application and device management

### 3.4 Unified Training Manager (`core/training/manager.py`)

**Design**: High-level orchestrator that delegates to mode-specific trainers.

```python
class TrainingManager:
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.components = ModelLoader.load_components(config)
        self.dataset = UnifiedDataset(config, "train")
        self.freezing_manager = FreezingManager(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.logger = UnifiedLogger(config)
        
    def train(self):
        if self.config.training_mode == "sft":
            trainer = SFTTrainer(self.config, self.components, self.dataset)
        elif self.config.training_mode == "grpo":
            trainer = GRPOTrainer(self.config, self.components, self.dataset)
        
        trainer.train()
```

---

## 4. Implementation Strategy

### Phase 1: Core Infrastructure (Week 1-2)
1. **Extract Shared Components**
   ```bash
   # Create core modules
   mkdir -p src_unified/core/{config,data,models,training,utils}
   
   # Move and unify configuration
   # Combine src_new/config/ and src_rl/config.py logic
   
   # Unify model loading
   # Extract from src_rl/runner.py:build_components() and src_new model logic
   
   # Unify data processing
   # Merge Dataset and RLDenseJSONLDataset with mode switching
   ```

2. **Create Unified Entry Points**
   ```python
   # src_unified/train.py
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", required=True)
       parser.add_argument("--mode", choices=["sft", "grpo"], required=True)
       
       config = UnifiedConfig.load(args.config, training_mode=args.mode)
       manager = TrainingManager(config)
       manager.train()
   ```

### Phase 2: Mode-Specific Adapters (Week 3)
1. **SFT Adapter**
   - Move `BBUTrainer` logic to `sft/trainer.py`
   - Move grouped loss computation to `sft/losses.py`
   - Integrate with unified infrastructure

2. **GRPO Adapter**
   - Move `VisionGRPOTrainer` to `grpo/trainer.py`
   - Move reward functions to `grpo/rewards/`
   - Ensure TRL integration works with unified components

### Phase 3: Testing and Migration (Week 4)
1. **Create Migration Scripts**
   ```bash
   # scripts/migrate_configs.py - Convert old configs to unified format
   # scripts/test_parity.py - Ensure identical behavior
   ```

2. **Comprehensive Testing**
   - Unit tests for each core component
   - Integration tests for both training modes
   - Parity tests against original implementations

### Phase 4: Cleanup and Documentation (Week 5)
1. **Remove Old Modules**
   - Archive `src_rl/` and `src_new/`
   - Update all references to use `src_unified/`

2. **Update Documentation**
   - Single README for unified framework
   - Configuration guide with mode switching
   - Migration guide for existing users

---

## 5. Configuration Migration Strategy

### Unified YAML Format
```yaml
# Base configuration
model:
  model_path: /path/to/checkpoint
  attn_implementation: eager
  torch_dtype: bfloat16

data:
  data_root: /path/to/data
  train_data_path: train.jsonl
  val_data_path: val.jsonl

# Mode-specific sections
training_mode: sft  # or "grpo"

sft:
  num_train_epochs: 3
  learning_rate: 5e-6
  loss_weights:
    teacher_loss_weight: 0.8
    student_loss_weight: 1.0
    caption_loss_weight: 1.0
    grounding_loss_weight: 1.0
    formatting_loss_weight: 0.5

grpo:
  sample_k: 4
  max_new_tokens: 128
  temperature: 0.9
  reward_weights:
    parse: 1.0
    wrappers: 1.0
    coords: 0.8
```

### Backward Compatibility
```python
# Migration utility
def migrate_config(old_config_path: str, mode: str) -> str:
    """Convert old format to unified format."""
    if "src_new" in old_config_path or mode == "sft":
        return migrate_sft_config(old_config_path)
    elif "src_rl" in old_config_path or mode == "grpo":
        return migrate_grpo_config(old_config_path)
```

---

## 6. Benefits of Unified Architecture

### Reduced Complexity
- **~40% fewer files**: Eliminate duplicated components
- **Single configuration system**: No more parallel config implementations
- **Unified logging/checkpointing**: Share TensorBoard and checkpoint logic
- **Common validation**: Single source for tensor alignment checks

### Enhanced Maintainability
- **Mode switching**: Easy to switch between SFT and GRPO
- **Shared bug fixes**: Fix once, benefit both modes
- **Consistent behavior**: Same model loading and data processing
- **Single testing strategy**: Test core components once

### Improved Scalability
- **Easy to add new modes**: Framework extensible to other training paradigms
- **Modular design**: Components can be swapped independently
- **Configuration flexibility**: Mode-specific sections without duplication

### Development Efficiency
- **Single codebase**: Developers work on one unified framework
- **Shared tooling**: Same scripts, debugging, and monitoring
- **Faster iteration**: Changes benefit both training modes

---

## 7. Migration Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Core Infrastructure | Unified config, data, model loading |
| 3 | Mode Adapters | SFT and GRPO trainer adaptations |
| 4 | Testing & Migration | Parity tests, migration scripts |
| 5 | Cleanup & Docs | Remove old modules, update documentation |

**Total Estimated Effort**: 5 weeks with 1-2 developers

---

## 8. Risk Mitigation

### Technical Risks
- **Regression in training behavior**: Comprehensive parity testing
- **Configuration migration issues**: Automated migration with validation
- **Performance impact**: Benchmark unified vs original implementations

### Process Risks
- **Development disruption**: Maintain old modules during transition
- **User confusion**: Clear migration guide and backward compatibility
- **Integration challenges**: Incremental rollout with rollback plan

---

## 9. Success Criteria

### Functional Requirements
- ✅ Both SFT and GRPO training work identically to original implementations
- ✅ Single configuration system supports both modes
- ✅ Shared checkpointing and logging systems
- ✅ Unified model loading and data processing

### Non-Functional Requirements
- ✅ <50% of original combined lines of code
- ✅ <30 total files in unified framework
- ✅ Complete test coverage for core components
- ✅ Performance parity with original implementations

---

This unified framework will transform the current dual-codebase situation into a single, elegant, and maintainable training system for Qwen2.5-VL fine-tuning that serves both SFT and GRPO use cases efficiently.
