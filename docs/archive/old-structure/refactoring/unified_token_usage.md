# Unified Token Management Usage Guide

## Overview

The new unified token management system dramatically simplifies coordinate token handling by:

1. **Automatic Token Detection**: Reuses existing tokens (box, object_ref) automatically
2. **Automatic Token Addition**: Adds new tokens (line, square, coordinates) as needed  
3. **No Manual Configuration**: Token IDs are handled automatically
4. **Simple API**: Just specify `max_coord_value` and everything else is automatic

## Basic Usage

### 1. Initialize Token Manager

```python
from transformers import AutoTokenizer, AutoModel
from src.utils.tokens import create_unified_token_manager

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Create unified token manager (automatically handles everything)
token_manager = create_unified_token_manager(
    tokenizer=tokenizer,
    model=model, 
    max_coord_value=2048  # Creates coordinate tokens [0, 2047]
)
```

### 2. Format Objects with Coordinates

```python
# Format a bounding box object
bbox_obj = {
    "bbox_2d": [100, 200, 300, 400],
    "desc": "A red car"
}

formatted = token_manager.format_object(bbox_obj)
# Result: "<|box_start|><coord_100><coord_200><coord_300><coord_400><|box_end|><|object_ref_start|>A red car<|object_ref_end|>"

# Format a line object  
line_obj = {
    "line": [50, 100, 150, 200],
    "desc": "Power line"
}

formatted = token_manager.format_object(line_obj)
# Result: "<|line_start|><coord_50><coord_100><coord_150><coord_200><|line_end|><|object_ref_start|>Power line<|object_ref_end|>"
```

### 3. Get Token IDs

```python
# Get geometry token IDs (automatically assigned)
box_start_id = token_manager.get_token_id("box_start")
line_start_id = token_manager.get_token_id("line_start")

# Get coordinate token IDs
coord_100_id = token_manager.get_coordinate_token_id(100)
coord_200_id = token_manager.get_coordinate_token_id(200)
```

## Configuration Changes

### Old Complex Configuration (REMOVED)
```yaml
# OLD - No longer needed!
coordinate_config_box_start_id: 151648
coordinate_config_box_end_id: 151649
coordinate_config_square_start_id: 151650
coordinate_config_square_end_id: 151651
coordinate_config_line_start_id: 151652
coordinate_config_line_end_id: 151653
coordinate_config_enable_validation: true
coordinate_config_enable_caching: true
coordinate_config_batch_processing: true
# ... 15+ more complex fields
```

### New Simple Configuration
```yaml
# NEW - Simple and clean!
coordinate_tokens_enabled: true
max_coord_value: 2048
coordinate_loss_weight: 1.0
regular_loss_weight: 1.0
```

## Integration with Model Wrapper

### Before (Complex)
```python
# OLD - Complex manual configuration
coordinate_config_dict = {
    "box_start_id": getattr(self.coordinate_config, "box_start_id", 151648),
    "box_end_id": getattr(self.coordinate_config, "box_end_id", 151649),
    "square_start_id": getattr(self.coordinate_config, "square_start_id", 151650),
    # ... 15+ more getattr() calls with fallbacks
}
```

### After (Simple)
```python
# NEW - Automatic and clean
if config.coordinate_tokens_enabled:
    self.token_manager = create_unified_token_manager(
        tokenizer=self.tokenizer,
        model=self.base_model,
        max_coord_value=config.max_coord_value
    )
```

## Token System Details

### Automatic Token Detection
The system automatically detects and reuses existing tokens:
- `<|object_ref_start|>` / `<|object_ref_end|>` (usually IDs 151646/151647)
- `<|box_start|>` / `<|box_end|>` (usually IDs 151648/151649)

### Automatic Token Addition
New tokens are added automatically as needed:
- `<|line_start|>` / `<|line_end|>` (auto-assigned IDs)
- `<|square_start|>` / `<|square_end|>` (auto-assigned IDs)
- `<coord_0>` through `<coord_{max_coord_value-1}>` (auto-assigned IDs)

### Token ID Management
- **No manual ID assignment**: All token IDs are assigned automatically
- **No hardcoded values**: Token IDs adapt to the tokenizer's vocabulary
- **Collision-free**: New tokens get the next available IDs
- **Consistent**: Same tokens always get same IDs within a session

## Benefits

### Code Reduction
- **Eliminated 200+ lines** of complex token configuration
- **Removed 37+ getattr() fallback patterns** from model wrapper
- **Simplified configuration** from 20+ fields to 4 fields

### Improved Reliability  
- **No manual token ID conflicts**: All IDs assigned automatically
- **No missing token errors**: System ensures all required tokens exist
- **Fail-fast validation**: Clear errors if setup fails

### Better Maintainability
- **Single source of truth**: All token logic in one place
- **Automatic adaptation**: Works with any tokenizer vocabulary
- **Clear API**: Simple methods for common operations

## Migration Guide

### Step 1: Update Configuration
Replace complex coordinate config with simple version:
```yaml
# Remove all coordinate_config_* fields
# Add these 4 simple fields:
coordinate_tokens_enabled: true
max_coord_value: 2048
coordinate_loss_weight: 1.0
regular_loss_weight: 1.0
```

### Step 2: Update Model Initialization
```python
# Replace complex coordinate manager setup with:
if config.coordinate_tokens_enabled:
    self.token_manager = create_unified_token_manager(
        tokenizer=self.tokenizer,
        model=self.base_model,
        max_coord_value=config.max_coord_value
    )
```

### Step 3: Update Token Usage
```python
# Replace manual token ID lookups with:
box_start_id = self.token_manager.get_token_id("box_start")
coord_token_id = self.token_manager.get_coordinate_token_id(coord_value)

# Replace manual formatting with:
formatted_obj = self.token_manager.format_object(obj_dict)
```

## Error Handling

The unified system provides clear error messages:

```python
# Missing token
token_id = token_manager.get_token_id("nonexistent")
# ValueError: Token 'nonexistent' not found

# Invalid coordinate
coord_id = token_manager.get_coordinate_token_id(3000)  # max_coord_value=2048
# ValueError: Coordinate value 3000 out of range [0, 2048)

# Invalid object format
formatted = token_manager.format_object({"invalid": "object"})
# ValueError: Object missing geometry: {'invalid': 'object'}
```

## Performance

- **Faster initialization**: No complex configuration parsing
- **Efficient token lookup**: Direct dictionary access
- **Minimal memory overhead**: Only stores necessary token mappings
- **Batch token addition**: Coordinate tokens added in efficient batches
