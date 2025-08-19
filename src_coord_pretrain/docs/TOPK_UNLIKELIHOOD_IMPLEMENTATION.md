# Top-K Unlikelihood Training Implementation

## Overview

This document describes the implementation of advanced Top-K Unlikelihood training for coordinate token learning in Qwen2.5-VL. The system provides sophisticated negative sampling and conflict resolution for mixed forward/reverse mapping tasks.

## Key Features

### 1. **Top-K Negative Sampling**
- **Coordinate targets** (forward mapping): Select top-K highest-probability non-coordinate tokens as negatives
- **Text targets** (reverse mapping): Select top-K highest-probability coordinate tokens as negatives
- Configurable K values for fine-tuned control

### 2. **Token-Level Conflict Resolution**
- `coord_target_mask`: Positions where labels are coordinate tokens in assistant spans
- `text_target_mask`: Positions where labels are regular text tokens in assistant spans
- Mutually exclusive masks with fail-fast validation

### 3. **Enhanced Configuration**
- Backward compatible with existing simple unlikelihood
- New Top-K parameters with sensible defaults
- Debug-friendly smaller values for development

## Configuration Parameters

### Production Settings (`coord_bootstrap.yaml`)
```yaml
# UNLIKELIHOOD TRAINING (Bootstrap Stage)
unlikelihood_enabled: true
unlikelihood_lambda_digits: 1.0  # Weight for digit token suppression
unlikelihood_lambda_coords: 1.0  # Weight for coordinate window suppression
unlikelihood_coord_window: 8     # Suppress neighboring coord tokens within ±window

# TOP-K UNLIKELIHOOD (Advanced Features)
ul_topk_noncoord: 100  # Top-K non-coordinate tokens for coordinate targets
ul_topk_coord: 100     # Top-K coordinate tokens for text targets
ul_neighbor_window: 8  # Coordinate neighbor suppression window (±W around gold)
```

### Debug Settings (`coord_bootstrap_debug.yaml`)
```yaml
# TOP-K UNLIKELIHOOD (Advanced Features) - DEBUG SETTINGS
ul_topk_noncoord: 20   # Smaller K for debugging
ul_topk_coord: 20      # Smaller K for debugging
ul_neighbor_window: 4  # Smaller window for debugging
```

## Implementation Architecture

### 1. **Method Selection Logic**
```python
def _compute_unlikelihood_loss(self, outputs, inputs):
    """Compute advanced Top-K Unlikelihood loss with conflict resolution."""
    # Use Top-K approach if configured, otherwise fall back to legacy
    if hasattr(self, 'ul_topk_noncoord') and self.ul_topk_noncoord > 0:
        return self._compute_topk_unlikelihood_loss(logits, labels, assistant_mask)
    else:
        return self._compute_legacy_unlikelihood_loss(logits, labels, assistant_mask)
```

### 2. **Mask Computation with Conflict Resolution**
```python
# Compute target masks with conflict resolution
coord_target_mask = torch.isin(labels, coord_ids_tensor) & assistant_mask
text_target_mask = (~torch.isin(labels, coord_ids_tensor)) & (labels != -100) & assistant_mask

# Verify masks are mutually exclusive (fail-fast validation)
if (coord_target_mask & text_target_mask).any():
    raise RuntimeError("Target masks are not mutually exclusive - this indicates a bug")
```

### 3. **Top-K Selection for Coordinate Targets**
```python
def _compute_coordinate_target_loss(self, logits, labels, coord_target_mask, coord_ids_tensor):
    """Compute Top-K non-coordinate token suppression for coordinate targets."""
    
    for batch_idx, seq_idx in zip(coord_positions[0], coord_positions[1]):
        # Get probabilities
        probs = torch.softmax(position_logits, dim=-1)
        
        # Create mask for valid negative tokens (exclude gold, coordinates, special tokens)
        exclude_ids = set(coord_ids_tensor.tolist()) | {gold_token_id, 151643}  # 151643 is <|im_end|>
        valid_mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        for exclude_id in exclude_ids:
            if exclude_id < logits.shape[-1]:
                valid_mask[exclude_id] = False
        
        # Select top-K from valid non-coordinate tokens
        if valid_mask.sum() > 0:
            valid_probs = probs.clone()
            valid_probs[~valid_mask] = -float('inf')
            
            k = min(self.ul_topk_noncoord, valid_mask.sum().item())
            if k > 0:
                topk_probs, topk_indices = torch.topk(valid_probs, k=k)
                
                # Apply unlikelihood to top-K negatives
                clamped_complement = torch.clamp(1.0 - topk_probs, min=eps)
                position_loss = -torch.log(clamped_complement).sum()
                total_loss = total_loss + position_loss
```

### 4. **Top-K Selection for Text Targets**
```python
def _compute_text_target_loss(self, logits, labels, text_target_mask, coord_ids_tensor):
    """Compute Top-K coordinate token suppression for text targets."""
    
    for batch_idx, seq_idx in zip(text_positions[0], text_positions[1]):
        # Create mask for coordinate tokens only (exclude gold if it's accidentally a coord token)
        coord_mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        for coord_id in coord_ids_tensor:
            if coord_id < logits.shape[-1] and coord_id != gold_token_id:
                coord_mask[coord_id] = True
        
        # Select top-K from coordinate tokens
        if coord_mask.sum() > 0:
            coord_probs = probs.clone()
            coord_probs[~coord_mask] = -float('inf')
            
            k = min(self.ul_topk_coord, coord_mask.sum().item())
            if k > 0:
                topk_probs, topk_indices = torch.topk(coord_probs, k=k)
                
                # Apply unlikelihood to top-K coordinate negatives
                clamped_complement = torch.clamp(1.0 - topk_probs, min=eps)
                position_loss = -torch.log(clamped_complement).sum()
                total_loss = total_loss + position_loss
```

## Numerical Stability

### 1. **Probability Clamping**
```python
eps = 1e-5  # Numerical stability
clamped_complement = torch.clamp(1.0 - topk_probs, min=eps)
ul_loss = -torch.log(clamped_complement)
```

### 2. **Finite Loss Verification**
- All loss computations include `torch.isfinite()` checks in tests
- Graceful handling of edge cases (empty masks, no valid negatives)
- Position-wise averaging to prevent loss explosion

## Testing Framework

### 1. **Unit Tests** (`test_topk_unlikelihood.py`)
- Mixed forward/reverse batch handling
- Mutually exclusive mask computation
- Coordinate target negative selection
- Text target negative selection
- Numerical stability guards

### 2. **Integration Tests** (`test_topk_simple.py`)
- Configuration parameter loading
- Mask computation and conflict resolution
- Top-K loss computation methods
- Numerical stability with real data

### 3. **Verification Checklist**
```python
# Mask verification
assert not (coord_target_mask & text_target_mask).any(), "Masks should be mutually exclusive"

# Negative set verification
assert gold_token_id not in topk_indices, "Gold token should not be in negatives"
assert 151643 not in topk_indices, "<|im_end|> should not be in negatives"

# Numerical verification
assert torch.isfinite(ul_loss).all(), "Unlikelihood loss should be finite"
assert (clamped_complement >= eps).all(), "Clamped probabilities should be ≥ eps"
```

## Usage Examples

### 1. **Training with Top-K Enabled**
```bash
# Use production config with Top-K enabled
python -m src_coord_pretrain.training.trainer \
  --config src_coord_pretrain/config/coord_bootstrap.yaml
```

### 2. **Debug Training with Smaller K**
```bash
# Use debug config with smaller K values
python -m src_coord_pretrain.training.trainer \
  --config src_coord_pretrain/config/coord_bootstrap_debug.yaml
```

### 3. **Legacy Mode (Backward Compatibility)**
```yaml
# Disable Top-K to use legacy implementation
ul_topk_noncoord: 0
ul_topk_coord: 0
```

## Performance Characteristics

### 1. **Computational Complexity**
- **Top-K selection**: O(V log K) per position, where V is vocab size
- **Mask computation**: O(B × S) where B is batch size, S is sequence length
- **Memory overhead**: Minimal additional memory for masks and Top-K indices

### 2. **Training Stability**
- Numerical stability through probability clamping
- Fail-fast validation prevents silent bugs
- Graceful degradation when no valid negatives exist

### 3. **Hyperparameter Sensitivity**
- **K values**: Higher K = more aggressive suppression, slower training
- **Lambda weights**: Balance between LLM loss and unlikelihood loss
- **Window size**: Controls coordinate neighbor suppression range

## Migration Guide

### From Simple to Top-K Unlikelihood
1. **Update configuration**: Add Top-K parameters to YAML config
2. **Test compatibility**: Run existing training with Top-K enabled
3. **Tune hyperparameters**: Adjust K values based on training dynamics
4. **Monitor metrics**: Watch for improved coordinate token precision

### Backward Compatibility
- Set `ul_topk_noncoord: 0` to disable Top-K and use legacy implementation
- All existing configurations continue to work without modification
- Gradual migration path available through debug configurations

## Future Enhancements

### 1. **Adaptive K Selection**
- Dynamic K based on training progress
- Per-layer or per-head K values
- Curriculum learning for K values

### 2. **Advanced Negative Sampling**
- Semantic similarity-based negative selection
- Hard negative mining strategies
- Multi-scale negative sampling

### 3. **Distributed Training Optimizations**
- Efficient Top-K computation across GPUs
- Gradient synchronization optimizations
- Memory-efficient negative sampling
