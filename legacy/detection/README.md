# Legacy Detection Module

## Status: MOVED TO LEGACY

This directory contains the original DETR-style object detection implementation that was used before migrating to the **coordinate token soft expectation approach**.

## Contents

- `detection_head.py` - DETR-style detection head with cross-attention decoder
- `detection_loss.py` - Hungarian matching loss with bbox regression, objectness, and caption components
- `detection_adapter.py` - Vision and language adapters for detection features
- `__init__.py` - Package initialization

## Why Moved to Legacy

With the implementation of **soft expectation coordinate regression**, the DETR-style detection approach is no longer the primary method. The new coordinate token approach provides:

- ✅ **Unified Training Objective**: Single loss function instead of separate detection head
- ✅ **Superior Gradient Flow**: Rich gradients through 2048-dimensional coordinate space
- ✅ **No Hungarian Matching**: Autoregressive token prediction (natural for LLMs)
- ✅ **Better Integration**: Seamless integration with VLM architecture

## Historical Value

This implementation is preserved for:
- **Research comparison**: Comparing DETR vs coordinate token approaches
- **Reference implementation**: Useful patterns for future detection work  
- **Fallback option**: If coordinate tokens need debugging
- **Educational value**: Understanding evolution of the detection approach

## Usage (If Needed)

To re-enable the legacy detection approach:

1. Move files back to `src/detection/`
2. Update imports in `src/models/wrapper.py`
3. Set `detection_enabled=True` in configuration
4. Set `coordinate_tokens.enable_coordinate_tokens=False`

## Migration Date

Moved to legacy: $(date)
Replaced by: Coordinate token soft expectation regression
Location: `src/models/wrapper.py` (CoordinateConfig)