# Coordinate Token Training Fixes - Complete Implementation

## Issues Fixed

### 1. Coordinate Token Parameter Registration
- **Problem**: Coordinate tokens existed in vocabulary but had no trainable parameters
- **Solution**: Extended embeddings/LM head properly created and registered in model parameter tree
- **Files**: `src/models/wrapper.py`

### 2. Loss Computation Architecture
- **Problem**: Manual CE loss computation + incorrect student loss scaling
- **Solution**: Use Qwen2.5-VL's built-in CE loss + separate LLM/coordinate losses
- **Files**: `src/models/wrapper.py`, `src/training/loss_manager.py`

### 3. Loss Scaling Balance
- **Problem**: Coordinate losses 10-100x higher than LLM losses
- **Solution**: Reduced coordinate loss weight from 1.0 to 0.05
- **Files**: `configs/bbu_v2.yaml`

### 4. Parameter Group Detection
- **Problem**: Parameter manager couldn't find coordinate parameters
- **Solution**: Updated categorization to detect extended embeddings/LM head by name patterns
- **Files**: `src/training/parameter_manager.py`

## Current Status
- Coordinate tokens: 2048 coordinate values + 4 special tokens
- Extended vocab: 151669 (original) + 2052 (coordinate) = 153721 total
- Proper weight preservation and initialization
- Clean loss reporting with 4 essential losses only