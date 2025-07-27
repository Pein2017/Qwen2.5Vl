# Token Validation System

## Overview

The token validation system ensures that all training samples contain the required special tokens for object descriptions and geometry information. This helps catch data quality issues early in the training process and prevents silent failures where samples are missing critical annotation components.

## Required Tokens

All training samples must contain the following types of tokens:

### 1. Object Reference Tokens (Required for Descriptions)
- `<|object_ref_start|>` (ID: 151646)
- `<|object_ref_end|>` (ID: 151647)

These tokens wrap object descriptions and are essential for the model to understand where object descriptions begin and end.

### 2. Geometry Tokens (At Least One Type Required)
- **Bounding Box**: `<|box_start|>` (ID: 151648) and `<|box_end|>` (ID: 151649)
- **Square**: `<|square_start|>` (ID: 151667) and `<|square_end|>` (ID: 151668)
- **Line**: `<|line_start|>` (ID: 151665) and `<|line_end|>` (ID: 151666)

Each sample must have at least one geometry type with proper start/end token pairs.

### 3. Coordinate Values
- Multiple approaches supported:
  - **Simple tokens**: Comma-separated coordinate values within geometry tokens
  - **Coordinate tokens**: Special `<coord_N>` tokens for precise value encoding

## Simple Token Approach

The system now uses a simple token approach inspired by the ms-swift methodology:

```python
# Token Philosophy
simple_tokens_enabled: True      # Enable simple token system (default)
coordinate_tokens_enabled: False # Legacy coordinate token system disabled

# Token Addition
tokenizer.add_special_tokens()   # Standard HuggingFace method
model.resize_token_embeddings()  # Expand embedding matrix

# Expected Training Format
"<|box_start|>100, 200, 300, 400<|box_end|> <|object_ref_start|>BBU设备<|object_ref_end|>"
"<|square_start|>150, 10, 211, 35, 218, 16, 166, 0<|square_end|> <|object_ref_start|>标签<|object_ref_end|>"
"<|line_start|>579, 1385, 679, 1451, 764, 1444<|line_end|> <|object_ref_start|>光纤<|object_ref_end|>"
```

### Coordinate Token Integration

The system also supports coordinate tokens for precise value encoding:

```python
# Coordinate tokens for precise value encoding
coord_tokens = []
for coord in coords:
    # Normalize to [0, 2047] range
    coord_int = max(0, min(int(coord), 2047))
    coord_tokens.append(f"<coord_{coord_int}>")

coord_str = "".join(coord_tokens)  # No spaces between coordinate tokens
return f"{start_token}{coord_str}{end_token}"
```

## Validation Levels

### 1. Chat Processor Validation
Validates tokens immediately after sample processing:

```python
# Enable in chat processor
chat_processor = ChatProcessor(
    tokenizer=tokenizer,
    image_processor=processor,
    enable_token_validation=True  # Enable validation
)
```

### 2. Training-Time Validation
Validates tokens during training before model forward pass:

```python
# Automatically enabled when model has validate_sample_tokens method
# Validates each sample in the batch during training
```

### 3. Model Wrapper Validation
Validates tokens at the model level:

```python
# Available through model wrapper
model.validate_sample_tokens(input_ids, sample_info)
```

## Configuration

### YAML Configuration
Add to your training configuration file:

```yaml
# Enable token validation in chat processor
chat_processor_enable_token_validation: true
# Use simple token system (default)
simple_tokens_enabled: true
# Disable legacy coordinate tokens
coordinate_tokens_enabled: false
```

### Programmatic Configuration
```python
# Enable validation when creating chat processor
chat_processor = ChatProcessor(
    # ... other parameters ...
    enable_token_validation=True,
    enable_simple_tokens=True
)
```

## Error Messages

The validation system provides detailed error messages:

### Missing Object Reference Tokens
```
❌ SPECIAL_TOKEN_VIOLATION: Sample test_sample missing object reference tokens.
All samples must have descriptions wrapped with <|object_ref_start|> and <|object_ref_end|>.
Found object_ref_start: False, Found object_ref_end: False
```

### Missing Geometry Tokens
```
❌ SPECIAL_TOKEN_VIOLATION: Sample test_sample missing geometry tokens.
All samples must have at least one geometry type (bbox, square, or line) with proper start/end tokens.
```

### Mismatched Geometry Token Pairs
```
❌ SPECIAL_TOKEN_VIOLATION: Sample test_sample has mismatched bbox tokens.
Found 2 start tokens and 1 end tokens. Each geometry must have matching start/end token pairs.
```

## Simple Token Manager Integration

The simple token manager is automatically initialized by the unified model loader:

```python
# Initialization (automatic in training)
from src.utils.simple_token_manager import create_simple_token_manager
token_manager = create_simple_token_manager(tokenizer, model)

# Token Wrapping
manager.wrap_coordinates([150, 10, 211, 35], "bbox_2d")
# → "<|box_start|>150.0, 10.0, 211.0, 35.0<|box_end|>"

manager.wrap_description("BBU设备/华为,显示完整")  
# → "<|object_ref_start|>BBU设备/华为,显示完整<|object_ref_end|>"
```

## STRICT Validation

The SimpleTokenManager performs strict validation on all objects:

```python
# STRICT VALIDATION: Ensure exactly one geometry type exists
available_geom_types = [
    geom for geom in ["bbox_2d", "square", "line"] if geom in obj_dict
]

if len(available_geom_types) == 0:
    raise ValueError(
        f"❌ SPECIAL_TOKEN_VIOLATION: Object must have exactly one geometry type"
    )
elif len(available_geom_types) > 1:
    raise ValueError(
        f"❌ SPECIAL_TOKEN_VIOLATION: Object must have exactly one geometry type, found multiple"
    )

# STRICT VALIDATION: Ensure description exists
if "desc" not in obj_dict:
    raise ValueError(
        f"❌ SPECIAL_TOKEN_VIOLATION: Object must have 'desc' field"
    )
```

## Performance Considerations

### When to Enable
- **Development/Debugging**: Always enable to catch data issues
- **Production Training**: Enable for first few epochs, then disable for performance
- **Data Validation**: Enable when processing new datasets

### Performance Impact
- Minimal overhead (~1-2% training time increase)
- Most validation work happens during data loading
- Can be disabled after data quality is confirmed

## Troubleshooting

### Common Issues

1. **Empty Descriptions**: Ensure all objects have non-empty descriptions
2. **Missing Geometry**: Verify all objects have bbox_2d, square, or line coordinates
3. **Malformed Coordinates**: Check coordinate arrays have correct lengths
4. **Token ID Mismatches**: Verify tokenizer has been properly extended with special tokens

### Debug Commands
```python
# Check tokenizer vocabulary
vocab = tokenizer.get_vocab()
print(f"object_ref_start: {vocab.get('<|object_ref_start|>', 'MISSING')}")
print(f"box_start: {vocab.get('<|box_start|>', 'MISSING')}")

# Validate specific sample
model.validate_sample_tokens(input_ids, {"index": "debug_sample"})
```

## Best Practices

1. **Enable During Development**: Always use validation when developing or debugging
2. **Validate New Data**: Run validation on any new datasets before training
3. **Monitor Logs**: Check for validation warnings during training
4. **Test Edge Cases**: Validate samples with different geometry types
5. **Performance Tuning**: Disable validation for long production runs after data quality is confirmed

## Related Documentation

- [Training Architecture](architecture-overview.md)
- [Multi-Geometry Support](V2_MIGRATION_COMPLETE.md)
- [Training Configuration](configuration.md)
- [Troubleshooting Guide](quick-reference/problem-solution-lookup.md)
