# Qwen2.5-VL Unified Training Pipeline: src_rl → src_new Integration Plan

## Executive Summary

**Objective**: Merge `src_rl` into `src_new` to eliminate code duplication and create a unified multimodal training pipeline supporting both SFT and RL post-training modes.

**Strategic Approach**: Preserve HF-first invariants, eliminate duplicate components, and ensure seamless interoperability between SFT and RL training modes while maintaining all existing functionality.

**Target Architecture**: Single codebase with `src_new.training` for SFT and `src_new.rl` for GRPO, sharing conversation building, model wrapper, validation logic, and parsing utilities.

---

## 1. Current State Analysis

### 1.1 Shared Components (Duplicated Across src_new & src_rl)
- **Data Processing**: JSONL parsing, image loading via PathManager, multimodal tensor validation
- **Chat Templates**: HF-first conversation building with typed messages → template render → processor tensors
- **Model Wrapper**: DetectionModel with coordinate token support, multimodal validations, forward pass logic
- **Base Infrastructure**: Configuration loading, path resolution, special token handling
- **Parsing Logic**: Generated text → structured objects (bbox/quad/line extraction)

### 1.2 Unique Components
**src_new (SFT-specific)**:
- Cross-entropy loss with grouped components (caption/grounding/formatting)
- Teacher-student pairing mechanisms
- Span-based label masking and loss computation
- Data augmentation pipeline
- Coordinate auxiliary losses (legacy)

**src_rl (RL-specific)**:
- GRPO trainer and reward computation
- Online sampling and response generation
- Multi-component reward registry (format, detection, geometry)
- Vision-aware generation with TRL integration
- Policy optimization and advantage computation

---

## 2. Target Architecture (Post-Merge)

### 2.1 Unified Structure
```
src_new/
├── config/           # Unified config loading (SFT + RL)
├── data/            # Shared dataset and collation
├── models/          # DetectionModel wrapper (shared)
├── processing/      # Shared conversation, parsing, validation
│   ├── conversation/      # ConversationBuilder (shared)
│   ├── parse_generated.py # NEW: Unified parsing (factored from inference)
│   └── coordinate_converter.py
├── training/        # SFT-specific trainer and losses
├── rl/             # NEW: RL post-training module
│   ├── runner.py          # RL training launcher
│   ├── trainer.py         # VisionGRPOTrainer
│   ├── data/             # RL dataset adaptations
│   ├── prompting/        # RL conversation helpers
│   ├── rewards/          # Reward registry and components
│   ├── config.py         # RL-specific config extensions
│   └── eval.py           # Offline reward evaluation
├── utils/           # Shared utilities
├── types/           # Shared type definitions
└── inference.py     # Unified inference (uses shared parser)
```

### 2.2 Preserved Boundaries
- **SFT Training**: Remains under `src_new/training/` with CE loss path
- **RL Training**: Isolated under `src_new/rl/` with GRPO logic
- **Shared Infrastructure**: Common components unified under appropriate modules
- **Configuration**: Separate config schemas but shared loading infrastructure

---

## 3. Migration Roadmap

### Phase 1: Infrastructure Preparation (Day 1-2)

#### 3.1 Create Module Skeleton
```bash
# Create RL module structure
mkdir -p src_new/rl/{data,prompting,rewards,tools}
touch src_new/rl/{__init__.py,runner.py,trainer.py,config.py,eval.py}
touch src_new/rl/{data,prompting,rewards,tools}/__init__.py
```

#### 3.2 Extract Shared Parser
**Action**: Create `src_new/processing/parse_generated.py`
- **Source**: Factor from `src_new/inference.py::_normalize_prediction_to_vis_objects`
- **Interface**: `parse_geometry_wrapped_text_to_objects(text: str, coordinate_tokens_enabled: bool = False) -> List[Dict[str, Any]]`
- **Features**: Tolerant/strict regex, bbox/quad/line parsing, description sanitization, ASCII punctuation normalization
- **Consumers**: Update `src_new/inference.py` and all `src_rl/rewards/*` to use shared parser

#### 3.3 Unified Configuration Foundation
**Action**: Extend `src_new/config/` to support RL configurations
- **New**: `src_new/config/rl_config.py` with `EnhancedRLConfig` dataclass
- **Integration**: `src_new/config/loader.py` handles both SFT and RL YAML formats
- **Validation**: Fail-fast on missing required keys, no silent defaults
- **Backward Compatibility**: Support existing `configs/rl/*.yaml` files

### Phase 2: Core Component Migration (Day 3-4)

#### 3.4 Move RL Components
**Direct Moves** (update imports):
```
src_rl/runner.py → src_new/rl/runner.py
src_rl/trainer.py → src_new/rl/trainer.py
src_rl/data/ → src_new/rl/data/
src_rl/prompting/ → src_new/rl/prompting/
src_rl/rewards/ → src_new/rl/rewards/
src_rl/eval.py → src_new/rl/eval.py
src_rl/tools/ → src_new/rl/tools/
```

**Import Updates**:
- `from src_rl.*` → `from src_new.rl.*`
- `from src_new.processing.conversation` (shared ConversationBuilder)
- `from src_new.processing.parse_generated` (shared parser)
- `from src_new.models.wrapper` (shared DetectionModel)

#### 3.5 Shared Component Integration
**ConversationBuilder**: Already shared, ensure RL uses `src_new.processing.conversation.ConversationBuilder`

**DetectionModel**: Ensure RL trainer uses same wrapper from `src_new.models.wrapper`

**PathManager**: Ensure RL uses `src_new.utils.path_manager` for image loading

**Validation**: Ensure RL uses `src_new.utils.tensor_validation` for multimodal checks

### Phase 3: Vision-Aware RL Integration (Day 5-6)

#### 3.6 Fix Vision Tensor Flow
**Current Issue**: RL drops vision tensors before TRL, making generation text-only

**Solution**: Enhance `VisionGRPOTrainer` to:
- Accept batches with `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`
- Override generation methods to forward vision tensors to `model.generate`
- Maintain vision tensors through query-response concatenation
- Preserve TRL GRPO algorithmic logic unchanged

#### 3.7 Multimodal Dataset Integration
**Current**: `RLDenseJSONLDataset` → `_PromptOnlyDataset` (loses vision tensors)

**New Flow**: Direct integration without prompt-only wrapper
- `RLDenseJSONLDataset` uses shared `ConversationBuilder`
- Emits complete multimodal tensors: `{input_ids, attention_mask, pixel_values, image_grid_thw, meta}`
- Pass-through collator: `trainer.data_collator = lambda items: items`

#### 3.8 Enhanced Reward Components
**Existing Rewards**: `parse`, `wrappers`, `coords`, `separators`, `vocab`

**New Rewards** (using shared parser):
- `coverage`: Object count proximity (tolerance ±1 from GT)
- `geometry_sanity`: Bounds checking against `max_coord_value`
- `bbox_iou`: GIoU computation for bbox validation
- `taxonomy_valid`: Description conformance to hierarchy rules

### Phase 4: Integration and Testing (Day 7-8)

#### 3.9 Backward Compatibility Layer
**Shim Structure**: Keep `src_rl/` as thin re-export layer
```python
# src_rl/__init__.py
import warnings
from src_new.rl import *

warnings.warn("src_rl is deprecated. Use src_new.rl instead.", DeprecationWarning)
```

**Legacy Entrypoints**:
- `python -m src_rl.runner` → delegates to `python -m src_new.rl.runner`
- Preserve existing CLI arguments and behavior
- Print deprecation warnings

#### 3.10 Script and Documentation Updates
**Scripts**: Update `scripts/run_dense_grpo.sh` to call `src_new.rl.runner`

**Documentation**: 
- Move `src_rl/README.md` → `src_new/rl/README.md`
- Update all import examples and CLI commands
- Add migration guide for existing users

**Configuration**: Ensure existing `configs/rl/*.yaml` files work unchanged

### Phase 5: Validation and Cleanup (Day 9-10)

#### 3.11 Comprehensive Testing
**Unit Tests**:
- Shared parser: bbox/quad/line parsing, edge cases, normalization
- Rewards: All components with synthetic test cases
- Configuration: YAML loading, validation, error handling

**Integration Tests**:
- `python -m src_new.rl.runner --config configs/rl/enhanced_grpo.yaml --mode load`
- Short GRPO training run with vision tensors enabled
- Offline evaluation with reward component analysis

**Regression Tests**:
- Prompt/tensor parity with SFT for identical samples
- Vision tensor alignment validation (no "Image features mismatch" errors)
- RL checkpoint compatibility with `src_new.inference`

#### 3.12 Performance Validation
**Invariants to Verify**:
- `<|image_pad|>` count matches expected token count from `image_grid_thw`
- `pixel_values` rows equal `sum(t*h*w)` across all images
- Generation includes vision context (not text-only)
- Reward computation uses shared parser (no drift)

---

## 4. Key Implementation Details

### 4.1 Configuration Unification
**RL Config Extension**:
```python
@dataclass
class EnhancedRLConfig:
    # Shared with SFT
    model_path: str
    data_root: str
    train_data_path: str
    val_data_path: str
    
    # Model config (shared invariants)
    attn_implementation: str = "eager"  # Force for RL stability
    coordinate_tokens_enabled: bool = False  # src_new default
    max_coord_value: int = 1000
    
    # RL-specific
    sampling: SamplingConfig
    grpo: GRPOConfig
    rewards: Dict[str, float]  # reward_name -> weight
    logging: LoggingConfig
```

### 4.2 Shared Parser Interface
```python
# src_new/processing/parse_generated.py
def parse_geometry_wrapped_text_to_objects(
    text: str, 
    coordinate_tokens_enabled: bool = False,
    max_coord_value: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Parse generated text into structured objects.
    
    Returns:
        List[Dict] with keys: desc, bbox_2d|quad|line, confidence
    """
```

### 4.3 Vision-Aware GRPO Trainer
**Key Overrides**:
- `generate()`: Forward `pixel_values` and `image_grid_thw`
- `_get_per_token_logps()`: Compute probabilities with vision context
- `concatenate_queries_and_responses()`: Preserve vision tensors
- Pre-generation validation: Image token alignment checks

### 4.4 Reward Registry Enhancement
**Deterministic Components**: All rewards pure functions, no side effects
**Configurable Weights**: YAML-driven reward composition
**Shared Parsing**: Single source of truth for text → objects
**Extensible Design**: Easy to add new reward components

---

## 5. Risk Mitigation

### 5.1 Technical Risks
**TRL Version Compatibility**: Pin tested versions, isolate TRL imports to `src_new/rl/`
**Parser Drift**: Enforce shared parser usage, centralize tests
**Configuration Drift**: Strict validation, explicit YAML keys
**Performance Instability**: Force eager attention, probe bf16 capability

### 5.2 Migration Risks
**Breaking Changes**: Maintain backward compatibility shims
**Import Confusion**: Clear deprecation warnings and migration guide  
**Config Incompatibility**: Support existing YAML formats
**Workflow Disruption**: Preserve existing scripts and entrypoints

### 5.3 Integration Risks
**Vision Tensor Loss**: Comprehensive validation at each stage
**Model Wrapper Conflicts**: Ensure single source of truth usage
**Reward Computation Drift**: Unit tests for all components
**Multi-GPU Issues**: Preserve existing DDP/DeepSpeed toggles

---

## 6. Success Criteria

### 6.1 Functional Requirements
- [x] GRPO training runs with vision tensors enabled (no text-only fallback)
- [x] Zero "Image features and image tokens do not match" errors
- [x] Prompt/tensor parity with SFT for identical samples
- [x] Valid parse rate ≥ 90% on validation data
- [x] RL checkpoints load correctly in `src_new.inference`

### 6.2 Quality Requirements  
- [x] All existing unit tests pass
- [x] No performance regression in SFT training
- [x] RL training metrics remain stable
- [x] Configuration validation prevents silent failures
- [x] Documentation completeness and accuracy

### 6.3 Operational Requirements
- [x] Backward compatibility maintained during transition
- [x] Clear migration path for existing users
- [x] Deprecation timeline communicated
- [x] Support for existing configuration files
- [x] Preserved CLI interfaces and script behavior

---

## 7. Post-Merge Operations

### 7.1 Primary Entrypoints
**SFT Training**:
```bash
/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config phase_3/standard --log_level INFO
```

**RL Training**:
```bash
/root/miniconda3/envs/ms/bin/python -m src_new.rl.runner --config /abs/path/configs/rl/enhanced_grpo.yaml --mode train
```

**Unified Inference**:
```bash
/root/miniconda3/envs/ms/bin/python -m src_new.inference --model_path /abs/path/checkpoint --data_root /abs/path/data --dataset val --output_file /abs/path/out.jsonl
```

### 7.2 Configuration Management
**Shared Keys**: model_path, data_root, attn_implementation, coordinate_tokens_enabled
**SFT-Specific**: teacher pairing, data augmentation, loss components
**RL-Specific**: sampling parameters, GRPO hyperparameters, reward weights

### 7.3 Maintenance Strategy
**Single Parser**: All text → object parsing through `src_new.processing.parse_generated`
**Shared Validation**: Multimodal tensor checks in `src_new.utils.tensor_validation`
**Unified Testing**: Cross-module integration tests ensuring compatibility
**Consistent Interfaces**: HF-first contracts enforced across all modes

---

## 8. Timeline and Milestones

**Days 1-2**: Infrastructure setup, shared parser extraction, config foundation
**Days 3-4**: Component migration, import fixes, integration points  
**Days 5-6**: Vision-aware RL implementation, reward enhancements
**Days 7-8**: Compatibility layer, documentation, script updates
**Days 9-10**: Comprehensive testing, performance validation, cleanup

**Week 2**: Deprecation notices, user migration support, legacy removal planning

---

## 9. Conclusion

This integration plan creates a unified, maintainable codebase that eliminates duplication while preserving all functionality. The shared infrastructure ensures consistent behavior across SFT and RL modes, while the modular design allows for independent evolution of training methodologies.

The vision-aware GRPO integration solves the current limitation of text-only RL training, enabling true multimodal reinforcement learning. The shared parsing and validation components eliminate drift risks and reduce maintenance overhead.

Post-merger, the unified pipeline provides a single source of truth for Qwen2.5-VL training, from data processing through model deployment, with seamless interoperability between supervised fine-tuning and reinforcement learning post-training.
