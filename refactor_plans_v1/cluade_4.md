# Qwen2.5-VL Unified Training Architecture - Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to merge `src_rl/` and `src_new/` into a unified, modular architecture that supports both **Supervised Fine-tuning (SFT)** and **GRPO post-training** with minimal code duplication and maximum reusability.

---

## Current State Analysis

### Code Duplication Issues
- **Data Loading**: Both codebases load JSONL, process images, and build conversations
- **Model Loading**: Identical Qwen2.5-VL model initialization and validation logic
- **Configuration**: Separate config systems with overlapping concerns
- **Checkpointing**: Duplicate checkpoint saving and management logic
- **Logging**: Separate tensorboard and metrics logging systems

### Key Differences
- **Training Loop**: SFT uses HF Trainer, GRPO uses TRL's GRPOTrainer
- **Optimization Target**: SFT optimizes CE loss, GRPO optimizes rewards
- **Generation**: GRPO needs online generation during training, SFT doesn't

---

## Unified Architecture Design

### Core Principles
1. **Single Source of Truth**: One implementation for shared functionality
2. **Training Mode Agnostic**: Core components work for both SFT and GRPO
3. **Minimal API Surface**: Clean, simple interfaces between components
4. **Fail-Fast Validation**: Comprehensive input validation with clear error messages
5. **HF-First Alignment**: Maintain compatibility with HuggingFace ecosystem

### Target Directory Structure
```
src_unified/
├── config/                    # Unified configuration system
│   ├── schema.py             # Dataclass definitions for all training modes
│   ├── loader.py             # YAML loading and validation
│   └── validators.py         # Cross-field validation logic
│
├── data/                     # Unified data pipeline
│   ├── dataset.py           # Base dataset with SFT/RL modes
│   ├── collator.py          # Unified data collation
│   ├── augmentation.py      # Image/geometry augmentation
│   └── teacher_pool.py      # Teacher sample management
│
├── processing/               # Conversation and tokenization
│   ├── conversation/        # Conversation building (unified)
│   ├── templates.py         # Prompt templates
│   ├── tokenization.py      # Tokenization utilities
│   └── geometry.py          # Coordinate processing
│
├── models/                   # Model loading and wrapping
│   ├── loader.py            # Unified model loading
│   ├── wrapper.py           # Detection model wrapper
│   └── validation.py        # Input validation
│
├── training/                 # Training orchestration
│   ├── base/                # Base training components
│   │   ├── trainer_base.py  # Abstract base trainer
│   │   ├── callbacks.py     # Shared callbacks
│   │   └── state_manager.py # Training state management
│   │
│   ├── sft/                 # SFT-specific components
│   │   ├── trainer.py       # SFT trainer (extends base)
│   │   ├── loss_manager.py  # Loss computation
│   │   └── metrics.py       # SFT metrics
│   │
│   └── rl/                  # GRPO-specific components
│       ├── trainer.py       # GRPO trainer (extends base)
│       ├── rewards/         # Reward functions
│       └── metrics.py       # RL metrics
│
├── utils/                    # Shared utilities
│   ├── checkpointing.py     # Unified checkpoint management
│   ├── logging.py           # Unified logging system
│   ├── freezing.py          # Parameter freezing
│   └── validation.py        # Input validation utilities
│
├── inference/               # Unified inference engine
│   ├── engine.py           # Main inference engine
│   └── parsers.py          # Response parsing
│
└── cli/                     # Command-line interfaces
    ├── train.py            # Unified training entry point
    ├── inference.py        # Inference CLI
    └── eval.py             # Evaluation CLI
```

---

## Core Components Design

### 1. Unified Configuration System

**File**: `src_unified/config/schema.py`

```python
@dataclass
class TrainingConfig:
    """Unified configuration for both SFT and GRPO training."""
    
    # Training mode selection
    training_mode: Literal["sft", "grpo"] = "sft"
    
    # Model configuration (shared)
    model: ModelConfig
    
    # Data configuration (shared)
    data: DataConfig
    
    # Training configuration (mode-specific)
    training: Union[SFTTrainingConfig, GRPOTrainingConfig]
    
    # Features (shared)
    features: FeaturesConfig
    
    # Output/logging (shared)
    output: OutputConfig

@dataclass
class SFTTrainingConfig:
    """SFT-specific training configuration."""
    loss_weights: LossWeights
    teacher_pairing: TeacherPairingConfig
    # ... SFT specific fields

@dataclass  
class GRPOTrainingConfig:
    """GRPO-specific training configuration."""
    reward_weights: Dict[str, float]
    generation_config: GenerationConfig
    # ... GRPO specific fields
```

### 2. Unified Data Pipeline

**File**: `src_unified/data/dataset.py`

```python
class UnifiedDataset:
    """Unified dataset supporting both SFT and GRPO modes."""
    
    def __init__(
        self, 
        config: DataConfig,
        training_mode: Literal["sft", "grpo"],
        conversation_builder: ConversationBuilder,
        is_training: bool = True
    ):
        self.training_mode = training_mode
        self.conversation_builder = conversation_builder
        # ... initialization
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._load_sample(idx)
        
        if self.training_mode == "sft":
            return self._prepare_sft_sample(sample)
        else:  # grpo
            return self._prepare_grpo_sample(sample)
    
    def _prepare_sft_sample(self, sample: Dict) -> Dict[str, Any]:
        """Prepare sample for SFT training with labels."""
        # Use conversation builder to create training sample
        # Include labels, spans, etc.
        
    def _prepare_grpo_sample(self, sample: Dict) -> Dict[str, Any]:
        """Prepare sample for GRPO training (generation inputs only)."""
        # Use conversation builder for generation-ready inputs
        # No labels needed
```

### 3. Unified Training Orchestration

**File**: `src_unified/training/base/trainer_base.py`

```python
class UnifiedTrainerBase(ABC):
    """Abstract base class for both SFT and GRPO trainers."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.checkpoint_manager = None
        self.state_manager = None
        
    def setup(self):
        """Initialize all components."""
        self._load_model_components()
        self._setup_data()
        self._setup_training_components()
        self._setup_callbacks()
    
    @abstractmethod
    def _create_trainer_instance(self) -> Any:
        """Create the actual trainer (HF Trainer or TRL GRPOTrainer)."""
        pass
    
    def train(self):
        """Run training with unified orchestration."""
        trainer = self._create_trainer_instance()
        # Unified training loop management
```

**File**: `src_unified/training/sft/trainer.py`

```python
class SFTTrainer(UnifiedTrainerBase):
    """SFT trainer using HuggingFace Trainer."""
    
    def _create_trainer_instance(self) -> BBUTrainer:
        return BBUTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            training_args=self._build_hf_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
        )
```

**File**: `src_unified/training/rl/trainer.py`

```python
class GRPOTrainer(UnifiedTrainerBase):
    """GRPO trainer using TRL's GRPOTrainer."""
    
    def _create_trainer_instance(self) -> VisionGRPOTrainer:
        return VisionGRPOTrainer(
            model=self.model.base_model,
            reward_funcs=self._build_reward_functions(),
            args=self._build_grpo_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )
```

### 4. Unified CLI Interface

**File**: `src_unified/cli/train.py`

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["sft", "grpo"], required=True)
    args = parser.parse_args()
    
    # Load unified configuration
    config = load_config(args.config)
    config.training_mode = args.mode
    
    # Create appropriate trainer based on mode
    if config.training_mode == "sft":
        trainer = SFTTrainer(config)
    else:
        trainer = GRPOTrainer(config)
    
    # Run training
    trainer.setup()
    trainer.train()
```

---

## Migration Strategy

### Phase 1: Foundation (Week 1-2)
1. **Create unified config system**
   - Extract common config fields from both codebases
   - Implement mode-specific config branches
   - Add comprehensive validation

2. **Unify data pipeline**
   - Extract conversation building to shared module
   - Create mode-aware dataset class
   - Consolidate image processing and augmentation

3. **Unify model loading**
   - Extract model loading logic to shared module
   - Consolidate validation and patching logic

### Phase 2: Core Training (Week 3-4)
1. **Create base trainer abstraction**
   - Extract common training orchestration
   - Implement shared state management
   - Unify callback system

2. **Implement mode-specific trainers**
   - SFT trainer wrapping BBUTrainer
   - GRPO trainer wrapping VisionGRPOTrainer
   - Ensure both use shared components

### Phase 3: Utilities & Polish (Week 5)
1. **Unify checkpoint and logging systems**
   - Single checkpoint saver implementation
   - Unified tensorboard logging
   - Shared metrics collection

2. **Create unified CLI**
   - Single entry point for both modes
   - Backward compatibility with existing scripts

3. **Testing and validation**
   - Ensure feature parity with existing systems
   - Performance regression testing
   - Documentation updates

### Phase 4: Cleanup (Week 6)
1. **Remove deprecated codebases**
   - Archive `src_rl/` and `src_new/`
   - Update all references to use `src_unified/`
   - Clean up configuration files

---

## Key Benefits

### Code Reduction
- **Estimated 40-50% reduction** in total codebase size
- **Single implementation** for shared functionality
- **Unified testing** and maintenance

### Improved Maintainability
- **Single source of truth** for common operations
- **Consistent interfaces** across training modes
- **Centralized validation** and error handling

### Enhanced Developer Experience
- **Unified CLI** with consistent interface
- **Shared configuration system** with cross-mode validation
- **Common debugging and logging** infrastructure

### Scalability
- **Easy to add new training modes** (e.g., DPO, PPO)
- **Modular reward system** for RL experimentation
- **Extensible configuration** system

---

## Implementation Checklist

### Configuration System
- [ ] Design unified config schema with mode branches
- [ ] Implement YAML loading with layered inheritance
- [ ] Add comprehensive validation with clear error messages
- [ ] Create config migration tools for existing files

### Data Pipeline
- [ ] Extract conversation building to shared module
- [ ] Create unified dataset with mode-specific behavior
- [ ] Consolidate image processing and augmentation
- [ ] Implement shared data collation

### Training System
- [ ] Create abstract base trainer with shared orchestration
- [ ] Implement SFT trainer extending base
- [ ] Implement GRPO trainer extending base
- [ ] Unify callback and state management systems

### Utilities
- [ ] Create unified checkpoint manager
- [ ] Implement shared logging and metrics system
- [ ] Consolidate parameter freezing logic
- [ ] Create shared validation utilities

### Interface
- [ ] Create unified CLI with mode selection
- [ ] Implement backward compatibility layer
- [ ] Add comprehensive help and documentation
- [ ] Create migration guides

### Testing
- [ ] Unit tests for all unified components
- [ ] Integration tests for both training modes
- [ ] Performance regression tests
- [ ] Configuration validation tests

---

## Risk Mitigation

### Backward Compatibility
- **Gradual migration**: Keep existing scripts working during transition
- **Configuration migration tools**: Automatic conversion of existing configs
- **Comprehensive testing**: Ensure no regression in functionality

### Performance
- **Shared component optimization**: Profile and optimize shared paths
- **Mode-specific optimization**: Allow mode-specific optimizations where needed
- **Memory management**: Careful attention to memory usage patterns

### Complexity Management
- **Clear abstractions**: Well-defined interfaces between components
- **Comprehensive documentation**: Document all design decisions
- **Incremental rollout**: Validate each phase before proceeding

---

## Conclusion

This refactoring plan creates a unified, elegant architecture that:
- **Eliminates code duplication** while maintaining full functionality
- **Provides a clean foundation** for future training mode additions
- **Improves maintainability** through shared components and consistent interfaces
- **Enhances developer experience** with unified tooling and documentation

The modular design ensures that domain-specific logic (SFT losses vs GRPO rewards) remains separate while maximizing code reuse for shared functionality (data loading, model management, checkpointing, logging).
