# Differential Learning Rates for Coordinate Tokens

## Overview

This document explains the implementation of differential learning rates for coordinate tokens in the Qwen2.5-VL training framework. The system allows different components of the model to be trained with different learning rates, which is critical for proper spatial relationship learning in visual tasks.

## Problem Statement

The warning message `"⚠️ Coordinate tokens enabled but coordinate_lr is 0"` indicated that coordinate tokens were active but not being trained with their own learning rate. This implementation addresses that issue and provides a comprehensive solution for differential learning rates.

## Architecture

### Parameter Groups

The training system categorizes model parameters into the following groups:

1. **Vision Parameters** (`vision_lr`): Visual encoder components
2. **Merger Parameters** (`merger_lr`): Vision-language fusion components  
3. **LLM Parameters** (`llm_lr`): Language model components
4. **Coordinate Parameters** (`coordinate_lr`): Coordinate token embeddings and projections
5. **Adapter Parameters** (`adapter_lr`): LoRA or other adapter components

### Coordinate Token Detection

The system uses a sophisticated detection mechanism to identify coordinate token parameters:

#### 1. Explicit Module Detection
```python
# Direct coordinate token modules
patterns = [
    "extended_embeddings",
    "extended_lm_head", 
    "coordinate_tokens",
    "coord_tokens",
    "coordinate_head"
]
```

#### 2. Extended Vocabulary Detection
```python
# For models with coordinate tokens enabled
if model.coordinate_tokens_enabled:
    # Check if embed_tokens or lm_head have extended vocabulary
    if "embed_tokens.weight" in param_name or "lm_head.weight" in param_name:
        # Verify actual vocabulary size matches extended size
        return actual_vocab_size == extended_vocab_size > original_vocab_size
```

#### 3. Pattern-Based Detection
```python
# Coordinate-specific patterns
patterns = ["coordinate", "coord_", "bbox", "detection_head"]
```

## Configuration

### Required Settings

```yaml
# Learning rates (all must be explicitly set)
learning_rate: 5e-6        # Base learning rate
vision_lr: 5e-7            # Vision encoder (conservative)
merger_lr: 5e-5            # Vision-language fusion (higher)
llm_lr: 5e-6               # Language model (base rate)
coordinate_lr: 2.5e-5      # Coordinate tokens (5x llm_lr for spatial learning)
adapter_lr: 0              # Adapter layers (0 = frozen)

# Coordinate token system
coordinate_config_enable_coordinate_tokens: true
coordinate_config_max_coord_value: 2048
```

### Learning Rate Ratios

The recommended learning rate ratios are:

- **Coordinate LR**: 2-5x the LLM learning rate
- **Merger LR**: 5-10x the LLM learning rate  
- **Vision LR**: 0.1-0.5x the LLM learning rate (conservative)
- **LLM LR**: Base reference rate

## Implementation Details

### Parameter Manager Integration

The `ParameterGroupManager` handles parameter categorization:

```python
class ParameterGroupManager:
    def _categorize_parameter(self, param_name: str) -> str:
        # 1. Check for explicit coordinate token modules
        if self._is_coordinate_token_parameter(param_name):
            return "coordinate"
        
        # 2. Check for extended vocabulary parameters
        if self._is_extended_vocabulary_parameter(param_name):
            return "coordinate"
        
        # 3. Continue with other parameter types...
```

### Trainer Integration

The `BBUTrainer` creates parameter groups for the optimizer:

```python
def create_optimizer(self) -> Union[Optimizer, DummyOptim]:
    if self.config.use_differential_lr:
        # Create parameter groups with different learning rates
        lr_map = {
            "vision": self.config.vision_lr,
            "merger": self.config.merger_lr, 
            "llm": self.config.llm_lr,
            "coordinate": self.config.coordinate_lr,
        }
        
        optimizer_grouped_parameters = []
        for group_name, params in self._param_groups.items():
            if params:
                optimizer_grouped_parameters.append({
                    "params": params,
                    "lr": lr_map[group_name],
                })
```

### Validation and Warnings

The system provides comprehensive validation:

```python
def validate_configuration(self) -> List[str]:
    warnings = []
    
    # Check coordinate token configuration
    if (self.config.coordinate_config_enable_coordinate_tokens 
        and self.config.coordinate_lr <= 0):
        warnings.append(
            "Coordinate tokens enabled but coordinate_lr is 0 - "
            "coordinate tokens will not be trained"
        )
    
    return warnings
```

## Usage Examples

### Basic Configuration

```yaml
# configs/coordinate_training.yaml
learning_rate: 5e-6
coordinate_lr: 2.5e-5  # 5x base rate for coordinate learning
vision_lr: 5e-7        # Conservative for pretrained vision
merger_lr: 5e-5        # Higher for fusion learning
llm_lr: 5e-6           # Base rate for language model

coordinate_config_enable_coordinate_tokens: true
coordinate_config_max_coord_value: 2048
use_differential_lr: true
```

### Training Script Integration

```python
from src.training.trainer_factory import create_trainer_with_coordinator

# Create trainer with differential learning rates
trainer = create_trainer_with_coordinator(training_args)

# The trainer automatically:
# 1. Detects coordinate token parameters
# 2. Creates parameter groups with different learning rates
# 3. Validates configuration
# 4. Logs parameter group statistics
```

### Monitoring and Debugging

```python
# Check parameter group statistics
coordinator = TrainingCoordinator(model, tokenizer, config)
setup_info = coordinator.setup_training()

print(f"Parameter groups: {len(setup_info['optimizer_groups'])}")
for group in setup_info['optimizer_groups']:
    print(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']:.2e}")
```

## Benefits

### 1. Improved Spatial Learning
- Coordinate tokens get higher learning rates for better spatial relationship learning
- Prevents coordinate tokens from being undertrained relative to language components

### 2. Stable Training
- Vision encoder uses conservative learning rates to preserve pretrained features
- Language model maintains stable learning with base rates

### 3. Efficient Fine-tuning
- Merger components get higher rates for effective vision-language fusion
- Adapter components can be frozen or trained with specific rates

### 4. Comprehensive Validation
- Automatic detection of configuration issues
- Clear warnings for common mistakes
- Parameter group statistics for monitoring

## Troubleshooting

### Common Issues

1. **"Coordinate tokens enabled but coordinate_lr is 0"**
   - Solution: Set `coordinate_lr` to a non-zero value (recommended: 2-5x `llm_lr`)

2. **"No coordinate parameters found"**
   - Check that `coordinate_config_enable_coordinate_tokens: true`
   - Verify model has extended vocabulary size
   - Check parameter detection logic

3. **"Uncategorized parameters found"**
   - Review parameter names in error message
   - Update parameter detection patterns if needed
   - Ensure all trainable parameters are categorized

### Debugging Commands

```bash
# Test parameter detection
python test_parameter_detection.py

# Validate configuration
python -c "
from src.config import init_config
from src.training.training_coordinator import TrainingCoordinator
config = init_config('configs/bbu_v2.yaml')
coordinator = TrainingCoordinator(model, tokenizer, config)
warnings = coordinator.validate_configuration()
for warning in warnings:
    print(f'Warning: {warning}')
"
```

## Future Enhancements

1. **Adaptive Learning Rates**: Automatically adjust ratios based on training progress
2. **Component-Specific Schedules**: Different learning rate schedules for different components
3. **Dynamic Parameter Groups**: Add/remove parameter groups during training
4. **Advanced Validation**: More sophisticated parameter detection and validation

## Related Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Configuration Guide](docs/core/configuration.md)
- [Training System](docs/components/training-system.md)
- [Performance Optimization](src/utils/performance_optimizer.py)
