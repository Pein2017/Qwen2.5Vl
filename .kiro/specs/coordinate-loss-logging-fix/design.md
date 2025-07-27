# Design Document

## Overview

The coordinate loss logging issue stems from a disconnect between where coordinate losses are computed (in the model wrapper) and where they are extracted and logged (in the loss manager and trainer). The coordinate losses are being computed correctly but are not being properly propagated through the training pipeline for logging.

## Architecture

The current flow is:
1. **Model Wrapper** (`src/models/wrapper.py`) computes coordinate losses and stores them in `_last_*` attributes
2. **Model Wrapper** attaches these losses to model outputs as `_coordinate_loss`, `_focal_loss`, etc.
3. **Loss Manager** (`src/training/loss_manager.py`) extracts losses from model outputs in `_extract_coordinate_losses()`
4. **Trainer** (`src/training/trainer.py`) accumulates and logs the extracted losses

The issue is that the loss extraction and accumulation logic has gaps that prevent coordinate losses from appearing in final logs.

## Root Cause Analysis

Based on the debug logs and code analysis, the problem occurs in these areas:

1. **Model Wrapper**: Loss components are computed and stored in `_last_*` attributes, but may not be properly attached to outputs
2. **Loss Manager**: The `_extract_coordinate_losses()` method may not be extracting all components correctly
3. **Trainer**: Coordinate loss accumulators may not be properly initialized or accumulated

## Components and Interfaces

### Model Wrapper Enhancement
- **Location**: `src/models/wrapper.py`
- **Function**: `_forward_with_coordinate_tokens()`
- **Issue**: Loss components are stored in `_last_*` attributes but may not be consistently attached to outputs
- **Fix**: Ensure all coordinate loss components are always attached to model outputs, even when zero

### Loss Manager Enhancement  
- **Location**: `src/training/loss_manager.py`
- **Function**: `_extract_coordinate_losses()`
- **Issue**: May not be extracting all coordinate loss components or updating current loss tracking
- **Fix**: Ensure robust extraction and proper updating of current loss components

### Trainer Logging Enhancement
- **Location**: `src/training/trainer.py` 
- **Issue**: Coordinate loss accumulators may not be properly initialized or accumulated
- **Fix**: Ensure coordinate losses are always accumulated and logged, even when zero

## Data Models

### Coordinate Loss Components
The model wrapper should attach these attributes to outputs:
- `outputs._coordinate_loss` - Main coordinate regression loss (detection_loss)
- `outputs._focal_loss` - Focal loss for distribution sharpening
- `outputs._regular_loss` - Regular token cross-entropy loss
- `outputs._l1_loss` - L1 component of coordinate loss
- `outputs._giou_loss` - GIoU component of coordinate loss

### Loss Flow
```
Model Wrapper:
  _last_coordinate_loss = detection_loss.item()
  _last_focal_loss = focal_loss.item()
  _last_l1_loss = l1_loss.item()
  _last_giou_loss = giou_loss.item()
  
  outputs._coordinate_loss = _last_coordinate_loss
  outputs._focal_loss = _last_focal_loss
  outputs._l1_loss = _last_l1_loss
  outputs._giou_loss = _last_giou_loss

Loss Manager:
  coordinate_components["coordinate_loss"] = float(model_outputs._coordinate_loss)
  coordinate_components["focal_loss"] = float(model_outputs._focal_loss)
  coordinate_components["l1_loss"] = float(model_outputs._l1_loss)
  coordinate_components["giou_loss"] = float(model_outputs._giou_loss)

Trainer:
  _accumulated_coordinate_loss += _current_coordinate_loss
  _accumulated_focal_loss += _current_focal_loss
  _accumulated_l1_loss += _current_l1_loss
  _accumulated_giou_loss += _current_giou_loss
```

## Error Handling

### Graceful Degradation
- When coordinate tokens are disabled, all coordinate losses should default to 0.0
- Missing coordinate loss attributes should not cause crashes
- Loss extraction should handle both presence and absence of coordinate loss components

### Validation
- Verify that coordinate token configuration is properly loaded
- Ensure model wrapper is computing coordinate losses when enabled
- Validate that loss components are numeric and finite
- Add debug logging to track loss flow through the pipeline

## Implementation Strategy

### Phase 1: Model Wrapper Fix
1. Ensure all coordinate loss components are always attached to model outputs
2. Add validation that loss attributes are properly set
3. Add debug logging to verify loss attachment

### Phase 2: Loss Manager Fix
1. Fix the `_extract_coordinate_losses()` method to extract all components
2. Ensure proper updating of current loss tracking variables
3. Add validation for extracted loss values

### Phase 3: Trainer Fix
1. Ensure coordinate loss accumulators are properly initialized
2. Fix accumulation logic to include all coordinate loss components
3. Ensure coordinate losses appear in final logged output

## Testing Strategy

### Unit Tests
1. **Model Wrapper Tests**: Verify coordinate losses are computed and attached to outputs
2. **Loss Manager Tests**: Verify coordinate losses are extracted from model outputs
3. **Trainer Tests**: Verify coordinate losses appear in training logs

### Integration Tests  
1. **End-to-End Training**: Run training with coordinate tokens enabled and verify losses appear in logs
2. **Loss Value Validation**: Verify logged loss values match debug output values
3. **Gradient Accumulation Tests**: Verify coordinate losses are properly accumulated across micro-batches

### Validation Scenarios
1. **Coordinate Tokens Enabled**: All coordinate losses should appear in logs with correct values
2. **Coordinate Tokens Disabled**: Coordinate losses should be 0.0 in logs  
3. **Mixed Batches**: Batches with and without coordinate tokens should be handled correctly
4. **Debug vs Final Logs**: Debug loss values should match final logged values