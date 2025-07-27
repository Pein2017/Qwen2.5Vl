# Fallback Patterns Inventory - /src/ Directory

## Executive Summary

**Total Fallback Patterns Found: 200+**
- **getattr() with defaults**: 85+ instances
- **hasattr() checks**: 100+ instances  
- **dict.get() with defaults**: 15+ instances
- **Dataclass defaults**: 20+ fields
- **Try/except fallbacks**: 5+ instances

## Critical Files Requiring Refactoring

### 1. src/models/wrapper.py (HIGHEST PRIORITY)
**37+ fallback patterns** - Most problematic file

#### getattr() Patterns (25+ instances):
```python
# Line 284: Coordinate config fallback
max_coord_value = getattr(self.coordinate_config, "max_coord_value", 0)

# Lines 321-351: Multiple coordinate config fallbacks
"box_start_id": getattr(self.coordinate_config, "box_start_id", 151648),
"box_end_id": getattr(self.coordinate_config, "box_end_id", 151649),
"square_start_id": getattr(self.coordinate_config, "square_start_id", 151650),
"square_end_id": getattr(self.coordinate_config, "square_end_id", 151651),
"line_start_id": getattr(self.coordinate_config, "line_start_id", 151652),
"line_end_id": getattr(self.coordinate_config, "line_end_id", 151653),
"enable_validation": getattr(self.coordinate_config, "enable_validation", True),
"enable_caching": getattr(self.coordinate_config, "enable_caching", True),
"batch_processing": getattr(self.coordinate_config, "batch_processing", True),

# Lines 470, 475, 564, 569: Model config fallbacks
embedding_dim = getattr(self.base_model.config, "hidden_size", 0)
max_coord_value = getattr(self.coordinate_config, "max_coord_value", 0)

# Lines 382, 413, 2120: Tokenizer fallbacks
existing_tokens = getattr(self.tokenizer, "additional_special_tokens", [])
```

#### hasattr() Patterns (12+ instances):
```python
# Lines 273, 302, 461, 555: Coordinate config existence checks
if not hasattr(self, "coordinate_config") or self.coordinate_config is None:

# Lines 1144, 1324: Manager existence checks  
if not hasattr(self, "coordinate_manager") or self.coordinate_manager is None:

# Lines 1564-1572: Loss tracking fallbacks
if not hasattr(self, "_last_llm_loss"):
if not hasattr(self, "_last_coordinate_l1_loss"):
```

### 2. src/training/trainer.py (HIGH PRIORITY)
**50+ fallback patterns**

#### getattr() Patterns (30+ instances):
```python
# Lines 2255-2282: Configuration fallbacks
use_consistent_prompts = getattr(config, "use_consistent_prompts", True)
training_prompt_style = getattr(config, "training_prompt_style", True)
coordinate_tokens_enabled = getattr(config, "coordinate_config_enable_coordinate_tokens", False)
max_coord_value = getattr(config, "coordinate_config_max_coord_value", 2048)
merge_size = getattr(config, "merge_size", 4)
max_length = getattr(config, "max_total_length", 8192)
teacher_ratio = getattr(config, "teacher_ratio", 0.0)
data_path = getattr(config, "train_data_path", "data/train.jsonl")
collator_type = getattr(config, "collator_type", "standard")
```

#### hasattr() Patterns (20+ instances):
```python
# Lines 620-621: Model attribute checks
if hasattr(self.model, "original_vocab_size") and hasattr(self.model, "extended_vocab_size"):

# Lines 1152, 1216, 1288: Training coordinator checks
hasattr(self.config, "coordinate_config_enable_coordinate_tokens")
hasattr(self, "training_coordinator")
```

### 3. src/training/stability.py (MEDIUM PRIORITY)
**15+ fallback patterns**

#### getattr() Patterns:
```python
# Lines 93-96: Stability config fallbacks
max_consecutive_nan = getattr(config, "max_consecutive_nan", 3)
max_consecutive_zero = getattr(config, "max_consecutive_zero", 5)
nan_monitoring_window = getattr(config, "nan_monitoring_window", 20)
max_nan_ratio = getattr(config, "max_nan_ratio", 0.3)
```

### 4. src/training/loss_manager.py (MEDIUM PRIORITY)
**10+ fallback patterns**

#### hasattr() + getattr() Patterns:
```python
# Lines 56-58: LLM loss extraction fallback
if hasattr(model_outputs, "_llm_loss"):
    return self._safe_item(model_outputs._llm_loss)
return self._safe_item(model_outputs.loss)

# Lines 63-70: Generic loss extraction with fallback
if hasattr(outputs, "get") and key in outputs:
    return self._safe_item(outputs[key])
elif hasattr(outputs, key):
    return self._safe_item(getattr(outputs, key))
else:
    self.logger.debug(f"Loss key '{key}' not found, returning 0.0")
    return 0.0
```

### 5. src/utils/utils.py (LOW PRIORITY)
**5 dict.get() patterns**

```python
# Lines 45-46: Object formatting fallbacks
bbox_2d = obj.get("bbox_2d", [])
desc = obj.get("desc", "")

# Lines 58, 76, 79: Data processing fallbacks
objects = data.get("objects", [])
teachers = data.get("teachers", data.get("examples", []))
teacher_objects = teacher.get("objects", [])
```

## Configuration System Issues

### src/config/global_config.py
**Dataclass fields with defaults (20+ instances):**

```python
# Lines 52-66: Optional fields with defaults
dataset_name: Optional[str] = None
language: str = "chinese"
response_types: List[str] = field(default_factory=list)
log_level: str = "INFO"
fail_fast: bool = True
geometry_diversity_weight: float = 4.0

# Lines 224-226: Error handling defaults
coordinate_config_strict_coordinate_validation: bool = True
coordinate_config_log_raw_text_on_error: bool = True
```

## Refactoring Priority Matrix

| File | Fallback Count | Complexity | Priority | Risk Level |
|------|----------------|------------|----------|------------|
| models/wrapper.py | 37+ | Very High | 1 | High |
| training/trainer.py | 50+ | High | 2 | High |
| training/stability.py | 15+ | Medium | 3 | Medium |
| training/loss_manager.py | 10+ | Medium | 4 | Medium |
| config/global_config.py | 20+ | Low | 5 | Low |
| utils/utils.py | 5 | Low | 6 | Low |

## Next Steps

1. **Phase 1**: Refactor `models/wrapper.py` - eliminate all 37+ getattr/hasattr patterns
2. **Phase 2**: Update configuration schema to require all parameters explicitly
3. **Phase 3**: Refactor training components to use explicit configuration
4. **Phase 4**: Update all YAML files with explicit values
5. **Phase 5**: Implement fail-fast validation testing

## Impact Assessment

**Benefits:**
- Eliminate 200+ implicit fallback patterns
- Enforce explicit configuration for all parameters
- Improve error messages and debugging
- Reduce defensive programming complexity

**Risks:**
- Existing configurations may break without explicit values
- Need comprehensive testing to ensure no regressions
- Requires updating all YAML configuration files
