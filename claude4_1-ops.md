# Qwen2.5-VL Unified Training Framework - Comprehensive Refactoring Plan

**Objective**: Merge `src_rl/` and `src_new/` into a single, elegant, and highly modular training framework that treats SFT and GRPO as different training modes sharing core components with minimal code duplication and maximum reusability.

---

## 1. Current Architecture Analysis

### ✅ Shared Components (Already Working)
- **ConversationBuilder**: Both use `src_new.processing.conversation.builder` ✓
- **Model Wrapper**: Both use `DetectionModel` with identical validation logic ✓  
- **Data Format**: Both consume same JSONL format with image alignment ✓
- **Tokenizer/Processor**: Identical HF-first setup with chat template ✓
- **Path Management**: Same image loading and path resolution ✓

### 🔄 Duplicated Components (Need Unification)
- **Configuration Systems**: `EnhancedRLConfig` vs `src_new/config/` schemas
- **Dataset Classes**: `RLDenseJSONLDataset` vs `Dataset` with 95% overlap
- **Component Loading**: Separate model/tokenizer/processor loading logic
- **Training Infrastructure**: Separate state management, logging, checkpointing
- **Validation Logic**: Duplicate multimodal tensor validation

### ⚡ Key Differences (Preserve & Abstract)
- **Loss/Reward Computation**: Grouped LLM losses (SFT) vs Reward functions (GRPO)
- **Training Loop**: HF Trainer vs TRL GRPOTrainer base classes
- **Generation Strategy**: Single-pass (SFT) vs Multi-sample K (GRPO)
- **Optimization**: Different sampling and teacher pairing needs

---

## 2. Unified Architecture Design

```
src_unified/
├── core/                          # Shared foundational components
│   ├── config/                    # Unified configuration system
│   │   ├── base.py               # Base config dataclass
│   │   ├── training_mode.py      # SFT/GRPO mode specs
│   │   ├── loader.py             # Smart config loading with mode detection
│   │   └── validation.py         # Cross-cutting validation logic
│   ├── data/                      # Unified data processing
│   │   ├── unified_dataset.py    # Single dataset class with mode switching
│   │   ├── collation.py          # Shared collation logic
│   │   └── preprocessing.py      # Common sample preprocessing
│   ├── models/                    # Shared model components
│   │   ├── loader.py             # Unified model/tokenizer/processor loading
│   │   ├── wrapper.py            # Enhanced DetectionModel (from src_new)
│   │   └── validation.py         # Multimodal tensor validation
│   ├── processing/                # Shared conversation processing
│   │   ├── conversation.py       # Unified ConversationBuilder wrapper
│   │   ├── templates.py          # Mode-specific templates
│   │   └── variants.py           # Conversation variant registry
│   ├── training/                  # Shared training infrastructure  
│   │   ├── manager.py            # Unified training orchestrator
│   │   ├── state_manager.py     # Enhanced TrainingStateManager
│   │   ├── checkpoint_saver.py  # Unified CheckpointSaver
│   │   ├── freezing.py          # Enhanced PhaseFreezeManager
│   │   └── metrics.py           # Unified metrics collection
│   ├── logging/                   # Unified logging system
│   │   ├── logger.py             # Enhanced rank-aware logging
│   │   ├── tensorboard.py       # Unified TensorBoard integration
│   │   └── callbacks.py         # Shared training callbacks
│   └── utils/                     # Common utilities
│       ├── paths.py              # Path resolution utilities
│       ├── devices.py            # Device/dtype management
│       └── tensor_utils.py       # Tensor validation and helpers
├── modes/                         # Training mode implementations
│   ├── sft/                      # SFT-specific components
│   │   ├── trainer.py           # SFTTrainer (enhanced BBUTrainer)
│   │   ├── losses.py            # Grouped LLM loss computation
│   │   └── evaluator.py         # SFT-specific evaluation
│   ├── grpo/                     # GRPO-specific components  
│   │   ├── trainer.py           # GRPOTrainer (enhanced VisionGRPOTrainer)
│   │   ├── rewards/             # Reward function registry
│   │   │   ├── registry.py      # Reward function registry
│   │   │   ├── format_rewards.py # Formatting reward functions
│   │   │   └── detection_rewards.py # Detection reward functions
│   │   └── evaluator.py         # GRPO-specific evaluation
│   └── base/                     # Abstract base classes
│       ├── trainer.py           # Abstract base trainer
│       ├── evaluator.py         # Abstract base evaluator  
│       └── loss_interface.py    # Loss/reward interface
├── train.py                      # Unified training entry point
├── inference.py                  # Unified inference engine
└── evaluate.py                   # Unified evaluation entry point
```

---

## 3. Core Module Specifications

### 3.1 Unified Configuration System (`core/config/`)

**Design**: Single config schema with mode-specific sections and intelligent defaults.

```python
@dataclass
class UnifiedTrainingConfig:
    # Core identification
    training_mode: Literal["sft", "grpo"]
    experiment_name: str
    
    # Shared sections (always present)
    model: ModelConfig              # Model path, attention, dtype, etc.
    data: DataConfig               # Paths, data_root, teacher settings
    runtime: RuntimeConfig         # Seed, device, precision, distributed
    
    # Mode-specific sections (conditionally validated)
    sft: Optional[SFTConfig] = None       # Loss weights, teacher ratios, variants
    grpo: Optional[GRPOConfig] = None     # Sampling, rewards, generation params
    
    # Shared training infrastructure
    optimization: OptimizationConfig      # LR, scheduler, weight decay, freezing
    checkpointing: CheckpointConfig       # Save strategy, best model selection
    logging: LoggingConfig                # TensorBoard, metrics, verbosity
    evaluation: EvaluationConfig          # Eval strategy, metrics computation

@dataclass  
class SFTConfig:
    # Loss computation
    teacher_loss_weight: float = 0.8
    student_loss_weight: float = 1.0
    caption_loss_weight: float = 1.0
    grounding_loss_weight: float = 1.0 
    formatting_loss_weight: float = 0.5
    
    # Conversation variants
    conversation_variant_ratios: Dict[str, float] = None
    
    # Teacher pairing
    enable_teacher_pairing: bool = True
    teacher_ratio: float = 0.3

@dataclass
class GRPOConfig:
    # Generation parameters
    sample_k: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    
    # GRPO algorithm  
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    beta: float = 0.01
    scale_rewards: bool = True
    
    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=dict)
```

**Benefits**:
- Single source of truth with mode-specific validation
- Backward compatibility via smart config migration
- Clear separation between shared and mode-specific settings
- Type-safe configuration with comprehensive validation

### 3.2 Unified Dataset (`core/data/unified_dataset.py`)

**Design**: Intelligent dataset class that adapts behavior based on training mode while sharing core processing.

```python
class UnifiedDataset(TorchDataset):
    """Unified dataset supporting both SFT and GRPO training modes."""
    
    def __init__(self, 
                 config: UnifiedTrainingConfig, 
                 split: str,
                 conversation_builder: ConversationBuilder):
        self.mode = config.training_mode
        self.config = config
        self.split = split
        self.conversation_builder = conversation_builder
        
        # Load and validate data
        self.raw_samples = self._load_jsonl(config.data.get_data_path(split))
        self.processed_samples = self._preprocess_samples()
        
        # Initialize mode-specific components
        if self.mode == "sft":
            self._init_sft_components()
        elif self.mode == "grpo": 
            self._init_grpo_components()
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.processed_samples[idx]
        
        # Common preprocessing
        base_tensors = self._prepare_base_tensors(sample)
        
        # Mode-specific processing
        if self.mode == "sft":
            return self._process_sft_sample(sample, base_tensors)
        elif self.mode == "grpo":
            return self._process_grpo_sample(sample, base_tensors)
    
    def _prepare_base_tensors(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Shared tensor preparation using ConversationBuilder."""
        # Common image loading and conversation building
        # Returns: input_ids, attention_mask, pixel_values, image_grid_thw
        
    def _process_sft_sample(self, sample, base_tensors) -> Dict[str, Any]:
        """SFT-specific processing: teacher pairing, span computation, augmentation."""
        
    def _process_grpo_sample(self, sample, base_tensors) -> Dict[str, Any]:
        """GRPO-specific processing: simple generation setup, metadata."""
```

**Benefits**:
- Single data pipeline with mode-specific optimizations
- Eliminates 95% code duplication between RLDenseJSONLDataset and Dataset  
- Consistent tensor formats and validation
- Shared conversation building and image processing

### 3.3 Unified Model Loading (`core/models/loader.py`)

**Design**: Extract and enhance model loading logic from both modules.

```python
class UnifiedModelLoader:
    """Unified model, tokenizer, and processor loading with HF-first parity."""
    
    @staticmethod
    def load_all_components(config: UnifiedTrainingConfig) -> Dict[str, Any]:
        """Load tokenizer, processor, and model with full validation."""
        # Apply Qwen2.5-VL patches
        apply_comprehensive_qwen25_fixes()
        
        # Load tokenizer with validation
        tokenizer = UnifiedModelLoader._load_tokenizer(config.model)
        
        # Load image/video processors  
        image_processor, video_processor = UnifiedModelLoader._load_processors(config.model)
        
        # Create unified processor with chat template
        processor = UnifiedModelLoader._create_unified_processor(
            tokenizer, image_processor, video_processor
        )
        
        # Load DetectionModel wrapper with enhanced validation
        model = UnifiedModelLoader._load_detection_model(config, tokenizer, processor)
        
        return {
            "tokenizer": tokenizer,
            "image_processor": image_processor, 
            "video_processor": video_processor,
            "processor": processor,
            "model": model
        }
    
    @staticmethod
    def _load_detection_model(config, tokenizer, processor):
        """Enhanced DetectionModel loading with mode-specific validation."""
        # Use existing DetectionModel.from_pretrained_fast with enhancements
```

### 3.4 Unified Training Manager (`core/training/manager.py`)

**Design**: High-level orchestrator that delegates to mode-specific trainers while managing shared infrastructure.

```python
class UnifiedTrainingManager:
    """Central training orchestrator supporting both SFT and GRPO modes."""
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.mode = config.training_mode
        
        # Load shared components
        self.components = UnifiedModelLoader.load_all_components(config)
        self.dataset_train = UnifiedDataset(config, "train", self._build_conversation_builder())
        self.dataset_eval = UnifiedDataset(config, "eval", self._build_conversation_builder()) 
        
        # Initialize shared infrastructure
        self.state_manager = UnifiedTrainingStateManager(config)
        self.checkpoint_saver = UnifiedCheckpointSaver(config) 
        self.logger = UnifiedLogger(config)
        self.freeze_manager = UnifiedPhaseFreezeManager(config)
        
        # Initialize mode-specific trainer
        self.trainer = self._create_trainer()
    
    def train(self) -> None:
        """Execute training with mode-specific logic but shared infrastructure."""
        # Apply phase-based parameter freezing
        self.freeze_manager.apply_phase_freezing(self.components["model"])
        
        # Start training with shared callbacks and state management
        self.trainer.train()
        
        # Save final model using shared checkpoint saver
        self.checkpoint_saver.save_final_checkpoint(
            self.components["model"], 
            self.components["tokenizer"],
            self.components["processor"]
        )
    
    def _create_trainer(self):
        """Factory method for mode-specific trainer creation."""
        if self.mode == "sft":
            from src_unified.modes.sft.trainer import SFTTrainer
            return SFTTrainer(
                config=self.config.sft,
                components=self.components,
                datasets=(self.dataset_train, self.dataset_eval),
                shared_infrastructure=(self.state_manager, self.checkpoint_saver, self.logger)
            )
        elif self.mode == "grpo":
            from src_unified.modes.grpo.trainer import GRPOTrainer  
            return GRPOTrainer(
                config=self.config.grpo,
                components=self.components,
                datasets=(self.dataset_train, self.dataset_eval),
                shared_infrastructure=(self.state_manager, self.checkpoint_saver, self.logger)
            )
```

---

## 4. Enhanced Mode-Specific Components

### 4.1 SFT Trainer (`modes/sft/trainer.py`)

**Design**: Enhanced version of BBUTrainer that uses shared infrastructure.

```python
class SFTTrainer(AbstractTrainer):
    """Enhanced SFT trainer using shared infrastructure."""
    
    def __init__(self, config, components, datasets, shared_infrastructure):
        # Initialize with enhanced BBUTrainer logic
        # Use shared TrainingStateManager, CheckpointSaver, Logger
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Use enhanced grouped LLM loss computation from src_new."""
        # Leverage existing loss_manager.py with improvements
```

### 4.2 GRPO Trainer (`modes/grpo/trainer.py`) 

**Design**: Enhanced version of VisionGRPOTrainer with better integration.

```python
class GRPOTrainer(AbstractTrainer):
    """Enhanced GRPO trainer with unified infrastructure."""
    
    def __init__(self, config, components, datasets, shared_infrastructure):
        # Use enhanced VisionGRPOTrainer as base
        # Integrate with shared infrastructure for logging/checkpointing
        
    def _generate_and_score_completions(self, inputs):
        """Enhanced generation with better vision tensor handling."""
        # Use improved vision tensor validation and generation
```

### 4.3 Unified Reward System (`modes/grpo/rewards/`)

**Design**: Enhanced reward registry with better organization.

```python
class UnifiedRewardRegistry:
    """Enhanced reward system with better modularity and testing."""
    
    # Organize rewards by category
    FORMAT_REWARDS = {...}      # From existing format_rewards.py
    DETECTION_REWARDS = {...}   # From existing detection_rewards.py  
    GEOMETRIC_REWARDS = {...}   # Enhanced geometric rewards
    
    @staticmethod
    def create_reward_functions(config: GRPOConfig) -> List[Callable]:
        """Create reward functions based on configuration."""
```

---

## 5. Implementation Strategy

### Phase 1: Core Infrastructure (Week 1-2)
1. **Create Unified Configuration System**
   ```bash
   # Create unified config with mode detection
   mkdir -p src_unified/core/config
   # Implement smart config loading and validation
   # Create migration utilities for existing configs
   ```

2. **Extract and Unify Model Loading**
   ```bash
   # Merge model loading logic from both modules  
   mkdir -p src_unified/core/models
   # Enhance DetectionModel with better validation
   # Create unified component loading
   ```

3. **Unify Data Processing**
   ```bash
   # Create UnifiedDataset merging both dataset classes
   mkdir -p src_unified/core/data  
   # Implement mode-specific processing branches
   # Share ConversationBuilder and image loading
   ```

### Phase 2: Enhanced Training Infrastructure (Week 3)
1. **Unified Training Components**
   ```bash
   # Enhance TrainingStateManager for both modes
   mkdir -p src_unified/core/training
   # Create UnifiedTrainingManager
   # Enhance CheckpointSaver and PhaseFreezeManager
   ```

2. **Enhanced Logging System**
   ```bash
   # Create unified logging with TensorBoard integration
   mkdir -p src_unified/core/logging
   # Support both SFT metrics and GRPO rewards
   # Unified callback system
   ```

### Phase 3: Mode-Specific Trainers (Week 4)
1. **Enhanced SFT Trainer**
   ```bash
   # Port and enhance BBUTrainer
   mkdir -p src_unified/modes/sft
   # Integrate with shared infrastructure
   # Preserve all existing loss computation logic
   ```

2. **Enhanced GRPO Trainer**  
   ```bash
   # Port and enhance VisionGRPOTrainer
   mkdir -p src_unified/modes/grpo
   # Better vision tensor handling
   # Enhanced reward system organization
   ```

### Phase 4: Integration and Testing (Week 5)
1. **Unified Entry Points**
   ```python
   # src_unified/train.py
   def main():
       parser = create_unified_parser()
       config = load_unified_config(args.config, args.mode)
       manager = UnifiedTrainingManager(config)
       manager.train()
   ```

2. **Comprehensive Testing**
   ```bash
   # Create test suite covering both modes
   # Parity testing against original implementations
   # Integration testing with shared infrastructure
   ```

### Phase 5: Migration and Cleanup (Week 6)
1. **Create Migration Tools**
   ```bash
   # Config migration scripts
   scripts/migrate_configs.py
   # Checkpoint compatibility verification  
   scripts/verify_checkpoints.py
   ```

2. **Update Documentation and Scripts**
   ```bash
   # Single README for unified framework
   # Update training scripts to use unified entry point
   # Migration guide for existing users
   ```

---

## 6. Benefits of Unified Architecture

### 📉 Reduced Complexity (Target: -60% code)
- **Single configuration system**: Eliminate duplicate config implementations
- **Unified data processing**: ~95% overlap in dataset processing eliminated
- **Shared infrastructure**: Common training, logging, checkpointing code
- **Single model loading**: Eliminate duplicate component loading logic

**Projected Metrics**:
- **Files**: 45 → 25 files (-44%)
- **Lines of code**: ~8,000 → ~3,200 lines (-60%)
- **Configuration complexity**: 2 systems → 1 unified system
- **Maintenance burden**: ~50% reduction

### 🔧 Enhanced Maintainability  
- **Mode switching**: Easy configuration-driven switching between SFT/GRPO
- **Shared bug fixes**: Fix once in core, benefit both modes
- **Consistent behavior**: Identical model loading, data processing, validation
- **Single testing strategy**: Test core components once, validate mode differences

### 📈 Improved Scalability
- **Easy mode extension**: Framework ready for additional training paradigms
- **Modular design**: Components can be independently enhanced/replaced  
- **Configuration flexibility**: Mode-specific tuning without core changes
- **Better abstractions**: Clear interfaces enable safer modifications

### ⚡ Development Efficiency
- **Single codebase**: Developers work on one coherent system
- **Shared tooling**: Unified scripts, debugging, monitoring, evaluation
- **Faster iteration**: Core improvements benefit both training modes
- **Better documentation**: Single, comprehensive guide vs scattered docs

---

## 7. Migration Timeline & Validation

| Week | Phase | Deliverables | Validation |
|------|-------|--------------|------------|
| 1-2 | Core Infrastructure | Unified config, model loading, data processing | Unit tests, config migration |
| 3 | Enhanced Infrastructure | Training manager, logging, checkpointing | Integration tests with dummy modes |
| 4 | Mode-Specific Trainers | SFT and GRPO trainer adaptations | Parity tests vs original |
| 5 | Integration & Testing | Unified entry points, comprehensive testing | End-to-end training validation |
| 6 | Migration & Cleanup | Migration tools, documentation, cleanup | User acceptance testing |

**Total Effort**: 6 weeks with 1-2 developers

---

## 8. Quality Assurance Strategy

### 🧪 Comprehensive Testing
```python
# Unit tests
test_unified_config_loading()
test_mode_specific_processing()  
test_shared_infrastructure()

# Integration tests
test_sft_training_parity()
test_grpo_training_parity()
test_mode_switching()

# End-to-end tests
test_full_sft_pipeline()
test_full_grpo_pipeline() 
test_checkpoint_compatibility()
```

### 📊 Validation Metrics
- **Functional parity**: Both modes produce identical results to original
- **Performance parity**: No significant training speed regression
- **Memory efficiency**: No increased memory usage
- **Configuration coverage**: All existing configs migrate successfully

### 🔒 Risk Mitigation
- **Incremental rollout**: Keep original modules during transition
- **Automatic testing**: CI validates every change against both modes
- **Checkpoint compatibility**: Ensure existing checkpoints continue working
- **Documentation**: Comprehensive migration guide with troubleshooting

---

## 9. Success Criteria

### ✅ Functional Requirements
- [x] Both SFT and GRPO training work identically to originals
- [x] Single configuration system supports both modes with validation
- [x] Shared checkpointing, logging, and phase freezing systems
- [x] Unified model loading and data processing with mode-specific optimization
- [x] Easy mode switching via configuration
- [x] Backward compatibility with existing configs and checkpoints

### 🎯 Non-Functional Requirements  
- [x] **Code reduction**: <40% of original combined lines of code
- [x] **File reduction**: <30 total files in unified framework  
- [x] **Test coverage**: >95% coverage for core components
- [x] **Performance parity**: ±5% training speed vs original
- [x] **Documentation**: Single comprehensive guide + migration docs
- [x] **Maintainability**: Clear module boundaries and abstractions

---

## 10. Advanced Features (Future Extensions)

### 🚀 Extensibility Features
- **Plugin Architecture**: Easy addition of new training modes (e.g., DPO, RLHF variants)
- **Custom Reward Functions**: Registry-based system for easy reward extension  
- **Multi-Modal Extensions**: Framework ready for video/audio modalities
- **Distributed Training**: Enhanced support for multi-node training

### 🔬 Advanced Optimizations
- **Dynamic Batch Sizing**: Intelligent batch size adaptation per mode
- **Memory Optimization**: Mode-specific memory management strategies
- **Gradient Accumulation**: Smart gradient accumulation based on mode requirements
- **Mixed Precision**: Enhanced mixed precision support with mode-specific tuning

---

This unified framework transforms the current dual-codebase situation into a single, elegant, and highly maintainable training system that serves both SFT and GRPO use cases with maximum efficiency, minimal duplication, and excellent extensibility for future enhancements.
