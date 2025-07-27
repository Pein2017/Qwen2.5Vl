# Design Document

## Overview

The detection loss visibility issue stems from a disconnect between where coordinate token losses are computed (in the model wrapper) and where they are extracted and logged (in the loss manager and trainer). The coordinate token losses are being computed correctly but are not being properly propagated through the training pipeline for logging.

## Architecture

The current flow is:
1. **Model Wrapper** (`src/models/wrapper.py`) computes coordinate losses and attaches them to model outputs
2. **Loss Manager** (`src/training/loss_manager.py`) extracts losses from model outputs 
3. **Trainer** (`src/training/trainer.py`) logs the extracted losses

The issue is that the loss extraction and logging logic has gaps that prevent coordinate losses from appearing in logs.

## Components and Interfaces

### Model Wrapper Enhancement
- **Location**: `src/models/wrapper.py`
- **Function**: `_forward_with_coordinate_tokens()`
- **Issue**: Loss components are being attached to outputs but may not be accessible
- **Fix**: Ensure all coordinate loss components are properly attached to model outputs

### Loss Manager Enhancement  
- **Location**: `src/training/loss_manager.py`
- **Function**: `_extract_coordinate_losses()`
- **Issue**: May not be extracting all coordinate loss components
- **Fix**: Ensure robust extraction of all coordinate loss attributes from model outputs

### Trainer Logging Enhancement
- **Location**: `src/training/trainer.py` 
- **Function**: `log()` method and loss accumulation
- **Issue**: Coordinate losses may not be properly accumulated and logged
- **Fix**: Ensure coordinate losses are always logged, even when zero

## Data Models

### Coordinate Loss Components
```python
@dataclass
class CoordinateLossComponents:
    coordinate_loss: float = 0.0    # Main coordinate regression loss
    focal_loss: float = 0.0         # Focal loss for distribution sharpening
    regular_loss: float = 0.0       # Regular token cross-entropy loss
    l1_loss: float = 0.0           # L1 component of coordinate loss
    giou_loss: float = 0.0         # GIoU component of coordinate loss
```

### Model Output Extensions
The model wrapper should attach these attributes to outputs:
- `outputs._coordinate_loss`
- `outputs._focal_loss` 
- `outputs._regular_loss`
- `outputs._l1_loss`
- `outputs._giou_loss`

## Error Handling

### Graceful Degradation
- When coordinate tokens are disabled, all coordinate losses should default to 0.0
- Missing coordinate loss attributes should not cause crashes
- Loss extraction should handle both presence and absence of coordinate loss components

### Validation
- Verify that coordinate token configuration is properly loaded
- Ensure model wrapper is using coordinate tokens when enabled
- Validate that loss components are numeric and finite

## Testing Strategy

### Unit Tests
1. **Model Wrapper Tests**: Verify coordinate losses are computed and attached to outputs
2. **Loss Manager Tests**: Verify coordinate losses are extracted from model outputs
3. **Trainer Tests**: Verify coordinate losses appear in training logs

### Integration Tests  
1. **End-to-End Training**: Run training with coordinate tokens enabled and verify losses appear in logs
2. **Configuration Tests**: Test with coordinate tokens enabled/disabled
3. **Loss Accumulation Tests**: Verify coordinate losses are properly accumulated across gradient accumulation steps

### Validation Scenarios
1. **Coordinate Tokens Enabled**: All coordinate losses should appear in logs
2. **Coordinate Tokens Disabled**: Coordinate losses should be 0.0 in logs  
3. **Mixed Batches**: Batches with and without coordinate tokens should be handled correctly
4. **Gradient Accumulation**: Coordinate losses should accumulate and average correctly