# Soft Expectation Coordinate Regression

This document describes the implementation of soft expectation coordinate regression for Qwen2.5-VL, which replaces traditional DETR-style detection with a unified token-based approach.

## Overview

### Key Innovation
Instead of using separate detection heads with cross-entropy loss, we extend the vocabulary with coordinate tokens (0-2048) and use soft expectation regression to predict bbox coordinates naturally within the LLM framework.

### Format
```
Input: "There is a screw at <|box_start|><coord_100><coord_200><coord_1500><coord_1800><|box_end|>"
Output: Bbox coordinates [0.048, 0.097, 0.732, 0.878] (normalized)
```

## Implementation

### 1. Non-Destructive Extension
- **Preserves all pretrained weights**: Original embeddings and LM head weights are frozen
- **Extends vocabulary**: Adds 2048 coordinate tokens (reuses official <|box_start|> & <|box_end|>)
- **Zero impact when disabled**: Standard Qwen2.5-VL behavior when coordinate tokens are off

### 2. Soft Expectation Loss
```python
# Traditional cross-entropy: argmax(softmax(logits))
# Soft expectation: sum(softmax(logits) * coordinate_range)

soft_weights = F.softmax(coord_logits / temperature, dim=-1)
expected_coords = torch.sum(soft_weights * coord_range, dim=-1)
loss = F.l1_loss(expected_coords, target_coords) + focal_loss
```

### 3. Hybrid Loss Computation
- **Regular tokens**: Standard cross-entropy loss (unchanged)
- **Coordinate tokens**: Soft expectation + focal loss for sharpness
- **Automatic detection**: Uses token ID ranges to identify coordinate tokens

## Usage

### Basic Setup
```python
from src.models.wrapper import Qwen25VLWithDetection, CoordinateConfig
from transformers import AutoTokenizer

# Configure coordinate tokens
coord_config = CoordinateConfig(
    enable_coordinate_tokens=True,    # Enable the feature
    max_coord_value=2048,            # Resolution (0-2047)
    coordinate_loss_weight=1.0,       # Weight for coordinate loss
    regular_loss_weight=1.0,          # Weight for regular tokens
    soft_expectation_temperature=1.0, # Softmax temperature
)

# Load model with coordinate support
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = Qwen25VLWithDetection(
    base_model_path=model_path,
    num_queries=100,
    max_caption_length=50,
    tokenizer=tokenizer,
    coordinate_config=coord_config  # Pass coordinate config
)

# The model automatically handles:
# 1. Tokenizer extension (adds coordinate tokens)
# 2. Embedding extension (preserves pretrained weights)
# 3. LM head extension (preserves pretrained weights)
# 4. Hybrid loss computation during training
```

### Data Processing
```python
# Get coordinate utilities
utils = model.get_coordinate_tokenizer_utils()

# Convert bbox to tokens
bbox = [0.1, 0.2, 0.8, 0.9]  # normalized coordinates
token_ids = utils['convert_bbox_to_tokens'](bbox)
# Returns: [151648, 151938, 151956, 152345, 152700, 151649]
#          [<|box_start|>, <coord_0>, <coord_204>, <coord_1638>, <coord_1843>, <|box_end|>]

# Convert tokens back to bbox
recovered_bbox = utils['convert_tokens_to_bbox'](token_ids)
# Returns: [0.1, 0.2, 0.8, 0.9] (with quantization)
```

### Training Integration
```python
# Works seamlessly with existing training pipeline
# Model automatically detects coordinate tokens and applies appropriate loss

# Standard training input
inputs = tokenizer("There is a screw at <|box_start|><coord_100><coord_200><coord_1500><coord_1800><|box_end|>", 
                  return_tensors="pt")

# Forward pass with automatic loss computation
outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])

# outputs.loss contains:
# - Regular token cross-entropy loss
# - Coordinate token soft expectation loss
# - Properly weighted and combined
```

## Testing

Run the phased test suite to verify implementation:

```bash
cd temporal
python test_coordinate_tokens.py
```

The test suite verifies:
1. **Weight Preservation**: All pretrained weights are preserved exactly
2. **Tokenizer Extension**: Coordinate tokens are added correctly
3. **Forward Pass**: Model processes coordinate tokens properly
4. **Loss Computation**: Hybrid loss works as expected

## Configuration Options

```python
@dataclass
class CoordinateConfig:
    max_coord_value: int = 2048              # Coordinate resolution
    coord_token_init_std: float = 0.01       # Initialization std for new tokens
    coordinate_loss_weight: float = 1.0      # Weight for coordinate loss
    regular_loss_weight: float = 1.0         # Weight for regular tokens
    soft_expectation_temperature: float = 1.0 # Softmax temperature
    focal_loss_alpha: float = 0.25           # Focal loss alpha
    focal_loss_gamma: float = 2.0            # Focal loss gamma
    enable_coordinate_tokens: bool = False   # Feature flag
```

## Performance Characteristics

### Memory Impact
- **Embedding increase**: +2048 tokens × hidden_size (e.g., +4.2M params for 3B model)
- **LM head increase**: +2048 tokens × hidden_size (e.g., +4.2M params for 3B model)
- **Total increase**: ~8.4M parameters (~0.3% of 3B model)

### Training Benefits
- **Unified gradients**: Coordinate learning benefits from full LLM gradient flow
- **Continuous representation**: Soft expectation provides richer coordinate representation
- **Natural integration**: No separate matching algorithms or detection pipelines

### Compatibility
- **Backward compatible**: Disable coordinate tokens for standard Qwen2.5-VL behavior
- **DeepSpeed compatible**: Extended vocab_size properly reported to optimizer
- **Generation compatible**: Coordinate tokens disabled during inference by default

## Migration from DETR

To migrate from your current DETR-style detection:

1. **Enable coordinate tokens** in your model config
2. **Update data processing** to use `<|box_start|>` / `<|box_end|>` coordinate format
3. **Remove detection head configuration** (optional, can coexist)
4. **Run comparative training** to evaluate performance

The implementation allows both approaches to coexist, so you can compare performance directly.

## Technical Details

### Token ID Mapping
```
Original vocab:     0 to 151935
<|box_start|>:     151648 (official token)
<|box_end|>:       151649 (official token)
<coord_0>:         151936
<coord_1>:         151937
...
<coord_2047>:      153983
Total vocab size:  153984
```

### Loss Computation Flow
1. **Token classification**: Identify coordinate vs regular tokens
2. **Separate processing**: Different loss functions for each type
3. **Gradient combination**: Unified backpropagation through extended LM head
4. **Weight preservation**: Original token weights remain frozen

This approach represents a fundamental shift from traditional object detection toward unified language-based coordinate regression, leveraging the full power of pretrained language models for spatial reasoning.