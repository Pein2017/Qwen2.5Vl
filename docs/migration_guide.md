# Migration Guide: From Fallback to Explicit Configuration

## Overview

This guide helps you migrate from the old fallback-based configuration system to the new explicit configuration system. The migration eliminates 200+ fallback patterns and simplifies token management dramatically.

## Pre-Migration Checklist

- [ ] Backup your current configuration files
- [ ] Review your current training setup
- [ ] Identify any custom modifications to token management
- [ ] Test the migration on a small dataset first

## Step-by-Step Migration

### Step 1: Update Configuration Files

#### Old Configuration (REMOVE)
```yaml
# OLD - Complex coordinate configuration (20+ fields)
coordinate_config_enable_coordinate_tokens: true
coordinate_config_max_coord_value: 2048
coordinate_config_coord_token_init_std: 0.02
coordinate_config_coordinate_loss_weight: 1.0
coordinate_config_regular_loss_weight: 1.0
coordinate_config_soft_expectation_temperature: 1.0
coordinate_config_use_official_box_tokens: true
coordinate_config_enable_multi_geometry: false
coordinate_config_max_line_coordinates: 50
coordinate_config_box_start_id: 151648
coordinate_config_box_end_id: 151649
coordinate_config_square_start_id: 151650
coordinate_config_square_end_id: 151651
coordinate_config_line_start_id: 151652
coordinate_config_line_end_id: 151653
coordinate_config_enable_validation: true
coordinate_config_enable_caching: true
coordinate_config_batch_processing: true
coordinate_config_strict_coordinate_validation: true
coordinate_config_log_raw_text_on_error: true
coordinate_config_required_loss_components: ["coordinate_l1_loss"]

# OLD - Chat processor settings
chat_processor_max_coord_value: 2048
chat_processor_enable_token_validation: true

# OLD - Stability settings
stability_max_consecutive_nan: 3
stability_max_consecutive_zero: 5
stability_nan_monitoring_window: 20
stability_max_nan_ratio: 0.3
stability_nan_recovery_enabled: true
stability_learning_rate_reduction_factor: 0.5
stability_gradient_clip_reduction_factor: 0.5
```

#### New Configuration (ADD)
```yaml
# NEW - Simplified coordinate configuration (4 fields only!)
coordinate_tokens_enabled: true  # Enable coordinate token system
max_coord_value: 2048  # Maximum coordinate value (creates tokens [0, 2047])
coordinate_loss_weight: 1.0  # Weight for coordinate loss
regular_loss_weight: 1.0  # Weight for regular LLM loss

# NEW - Add any missing required fields
training_prompt_style: true
use_consistent_prompts: true
detection_freeze_epochs: 0
```

### Step 2: Update Code Imports

#### Old Imports (REPLACE)
```python
# OLD - Complex token management
from src.utils.coordinate_token_manager import CoordinateTokenManager
from src.utils.simple_token_manager import SimpleTokenManager

# OLD - Complex configuration
from src.config.global_config import DirectConfig
```

#### New Imports (USE)
```python
# NEW - Unified token management
from src.utils.tokens import create_unified_token_manager

# NEW - Explicit configuration
from src.config.explicit_config import load_explicit_config
```

### Step 3: Update Token Management Code

#### Old Token Management (REMOVE)
```python
# OLD - Complex manual setup
coordinate_config_dict = {
    "enable_coordinate_tokens": True,
    "max_coord_value": getattr(self.coordinate_config, "max_coord_value", 2048),
    "box_start_id": getattr(self.coordinate_config, "box_start_id", 151648),
    "box_end_id": getattr(self.coordinate_config, "box_end_id", 151649),
    # ... 15+ more getattr() calls with fallbacks
}

# OLD - Manual coordinate manager creation
self.coordinate_manager = CoordinateTokenManager(
    tokenizer=self.tokenizer,
    config=coordinate_config_dict,
    logger=self.logger
)
```

#### New Token Management (USE)
```python
# NEW - Simple automatic setup
if config.coordinate_tokens_enabled:
    self.token_manager = create_unified_token_manager(
        tokenizer=self.tokenizer,
        model=self.base_model,
        max_coord_value=config.max_coord_value
    )
```

### Step 4: Update Configuration Loading

#### Old Configuration Loading (REPLACE)
```python
# OLD - With fallbacks
from src.config import config as global_config

use_consistent_prompts = getattr(config, "use_consistent_prompts", True)
training_prompt_style = getattr(config, "training_prompt_style", True)
max_coord_value = getattr(config, "coordinate_config_max_coord_value", 2048)
teacher_ratio = getattr(config, "teacher_ratio", 0.0)
```

#### New Configuration Loading (USE)
```python
# NEW - Explicit validation
config = load_explicit_config("configs/training.yaml")

# All fields guaranteed to exist - no fallbacks needed
use_consistent_prompts = config.use_consistent_prompts
training_prompt_style = config.training_prompt_style
max_coord_value = config.max_coord_value
teacher_ratio = config.teacher_ratio
```

### Step 5: Update Object Formatting

#### Old Object Formatting (REPLACE)
```python
# OLD - Manual token handling
def format_object_with_tokens(obj):
    bbox = obj.get("bbox_2d", [])  # Fallback to empty list
    desc = obj.get("desc", "")     # Fallback to empty string
    
    # Manual token ID lookup
    box_start_id = getattr(self.coordinate_config, "box_start_id", 151648)
    # ... complex manual formatting
```

#### New Object Formatting (USE)
```python
# NEW - Automatic formatting
def format_object_with_tokens(obj):
    # Automatic validation and formatting
    formatted = self.token_manager.format_object(obj)
    return formatted
```

## Migration Validation

### Step 1: Configuration Validation
```bash
# Run configuration validation
python tests/run_validation_tests.py
```

### Step 2: Test Token Management
```python
# Test token manager creation
from src.utils.tokens import create_unified_token_manager

token_manager = create_unified_token_manager(
    tokenizer=tokenizer,
    model=model,
    max_coord_value=2048
)

# Test object formatting
test_obj = {"bbox_2d": [100, 200, 300, 400], "desc": "test object"}
formatted = token_manager.format_object(test_obj)
print(f"Formatted: {formatted}")
```

### Step 3: Training Validation
```python
# Test training setup with new configuration
config = load_explicit_config("configs/your_config.yaml")

# Verify all required fields are present
assert hasattr(config, "coordinate_tokens_enabled")
assert hasattr(config, "max_coord_value")
assert hasattr(config, "coordinate_loss_weight")
```

## Common Migration Issues

### Issue 1: Missing Configuration Fields
**Error**: `ValueError: Missing required configuration fields`

**Solution**: Add all required fields to your YAML configuration. Use `configs/explicit_template.yaml` as reference.

### Issue 2: Invalid Token IDs
**Error**: Token ID conflicts or missing tokens

**Solution**: Remove all manual token ID configuration. The new system handles this automatically.

### Issue 3: Coordinate Range Errors
**Error**: `ValueError: Coordinate value out of range`

**Solution**: Ensure all coordinates in your data are within `[0, max_coord_value)` range.

### Issue 4: Import Errors
**Error**: `ImportError: cannot import name 'CoordinateTokenManager'`

**Solution**: Update imports to use the new unified token management system.

## Rollback Plan

If you need to rollback the migration:

1. **Restore Configuration**: Use your backed-up configuration files
2. **Restore Code**: Revert to the previous commit before migration
3. **Clear Cache**: Remove any cached model files that may have new tokens

## Benefits After Migration

### Immediate Benefits
- **Faster Startup**: No complex token configuration parsing
- **Clear Errors**: Immediate feedback on configuration issues
- **Simplified Code**: Much less code to maintain

### Long-term Benefits
- **Easier Debugging**: No hidden defaults to confuse troubleshooting
- **Better Reliability**: Fail-fast behavior prevents silent failures
- **Improved Performance**: Optimized token management

## Migration Checklist

- [ ] **Backup**: Save current configuration and code
- [ ] **Update Config**: Convert to explicit configuration format
- [ ] **Update Imports**: Use new unified token management
- [ ] **Update Code**: Remove fallback patterns
- [ ] **Test Configuration**: Validate configuration loading
- [ ] **Test Token Management**: Verify token operations work
- [ ] **Test Training**: Run a short training test
- [ ] **Validate Results**: Ensure training works as expected
- [ ] **Clean Up**: Remove old configuration files
- [ ] **Document**: Update any custom documentation

## Getting Help

### Validation Tools
- `tests/run_validation_tests.py` - Comprehensive validation
- `src.config.explicit_config.validate_config_completeness()` - Config validation
- `configs/explicit_template.yaml` - Complete configuration template

### Common Commands
```bash
# Validate configuration
python -c "from src.config.explicit_config import load_explicit_config; load_explicit_config('your_config.yaml')"

# Check for fallback patterns
grep -r "getattr.*," src/ --include="*.py"
grep -r "\.get.*," src/ --include="*.py"

# Run validation tests
python tests/run_validation_tests.py
```

### Support Resources
- **Configuration Guide**: `docs/explicit_configuration_guide.md`
- **Token Management Guide**: `docs/refactoring/unified_token_usage.md`
- **Validation Tests**: `tests/test_explicit_config_validation.py`
