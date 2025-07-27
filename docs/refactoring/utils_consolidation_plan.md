# Utils Directory Consolidation Plan

## Current State Analysis

### Identified Redundancies

#### 1. Token Management Duplication
- **`coordinate_token_manager.py`** (1,126 lines) - Complex, production-ready coordinate token system
- **`simple_token_manager.py`** (309 lines) - Simplified ms-swift approach
- **Overlap**: Both handle coordinate tokens, geometry tokens, and tokenizer extension
- **Status**: Documentation indicates `coordinate_token_manager.py` is "legacy/deprecated"

#### 2. Utility Function Overlap
- **`utils.py`** - General utilities, conversation formatting, tensor operations
- **`response_parser.py`** - Response parsing with some utility functions
- **Overlap**: JSON handling, data validation, formatting functions

#### 3. Configuration Handling Patterns
- Multiple files use `dict.get()` with fallbacks
- Inconsistent error handling approaches
- Redundant validation logic

#### 4. Parsing and Validation Logic
- Response parsing in `response_parser.py`
- Object validation in multiple files
- Coordinate validation scattered across modules

## Consolidation Strategy

### Phase 1: Token Management Unification

**Action**: Eliminate `coordinate_token_manager.py` (deprecated) and enhance `simple_token_manager.py`

**Rationale**:
- Documentation explicitly marks `coordinate_token_manager.py` as "legacy/deprecated"
- `simple_token_manager.py` follows modern ms-swift approach
- Reduces codebase by 1,126 lines
- Eliminates maintenance burden of two token systems

**Implementation**:
1. Audit dependencies on `coordinate_token_manager.py`
2. Migrate any missing functionality to `simple_token_manager.py`
3. Update imports throughout codebase
4. Remove deprecated file

### Phase 2: Utility Function Consolidation

**Create New Structure**:
```
src/utils/
├── __init__.py                 # Consolidated exports
├── core/                       # Core utilities
│   ├── __init__.py
│   ├── data_utils.py          # JSONL, tensor ops, data handling
│   ├── format_utils.py        # Conversation formatting, JSON formatting
│   └── validation_utils.py    # Input validation, type checking
├── tokens/                     # Token management
│   ├── __init__.py
│   ├── token_manager.py       # Renamed from simple_token_manager.py
│   └── special_tokens.py      # Token definitions
├── parsing/                    # Response parsing
│   ├── __init__.py
│   └── response_parser.py     # Enhanced response parser
├── performance/                # Performance utilities
│   ├── __init__.py
│   └── optimizer.py           # Renamed from performance_optimizer.py
└── legacy/                     # Deprecated files (for reference)
    └── coordinate_token_manager.py
```

### Phase 3: Function Consolidation

#### 3.1 Data Utilities (`core/data_utils.py`)
**Consolidate from**: `utils.py`, parts of `response_parser.py`
```python
# JSONL operations
def load_jsonl(file_path: str) -> List[Dict[str, Any]]
def save_jsonl(data: List[Dict], file_path: str) -> None

# Tensor operations  
def debug_input_shapes(inputs: Dict[str, Any]) -> None
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]) -> None

# Data filtering
def filter_inputs_for_model(inputs: Dict[str, Any]) -> Dict[str, Any]
def filter_inputs_for_generation(inputs: Dict[str, Any]) -> Dict[str, Any]
```

#### 3.2 Format Utilities (`core/format_utils.py`)
**Consolidate from**: `utils.py`, `prompt.py`
```python
# Conversation formatting
def format_object_description(obj: Dict[str, Any]) -> str
def format_single_round_conversation(data: Dict[str, Any]) -> List[Dict[str, str]]
def format_multi_round_conversation(data: Dict[str, Any]) -> str
def format_conversation(data: Dict[str, Any], multi_round: bool = False) -> Union[List[Dict[str, str]], str]

# Prompt utilities
def get_system_prompt(language: str = "chinese") -> str
def get_optimized_prompt_for_context(context_type: str) -> str
```

#### 3.3 Validation Utilities (`core/validation_utils.py`)
**Consolidate from**: Multiple files
```python
# Input validation
def validate_required_fields(data: Dict, required_fields: List[str]) -> None
def validate_coordinate_format(coords: List[float]) -> None
def validate_object_structure(obj: Dict) -> bool

# Type validation
def ensure_tensor_type(value: Any, expected_type: type) -> Any
def validate_config_completeness(config: Dict, required_keys: List[str]) -> List[str]
```

## Implementation Plan

### Step 1: Create New Directory Structure
```bash
mkdir -p src/utils/core src/utils/tokens src/utils/parsing src/utils/performance src/utils/legacy
```

### Step 2: Migrate Token Management
1. Move `simple_token_manager.py` → `tokens/token_manager.py`
2. Move `coordinate_token_manager.py` → `legacy/coordinate_token_manager.py`
3. Update all imports from `simple_token_manager` to `tokens.token_manager`
4. Enhance token manager with any missing functionality

### Step 3: Consolidate Core Utilities
1. Extract data operations from `utils.py` → `core/data_utils.py`
2. Extract formatting functions → `core/format_utils.py`
3. Create validation utilities → `core/validation_utils.py`
4. Update `utils.py` to re-export consolidated functions

### Step 4: Update Response Parser
1. Move `response_parser.py` → `parsing/response_parser.py`
2. Remove redundant utility functions (now in core/)
3. Focus on parsing-specific functionality

### Step 5: Update Imports and Exports
1. Update `src/utils/__init__.py` to export all consolidated functions
2. Maintain backward compatibility for existing imports
3. Update all files that import from utils

## Benefits

### Code Reduction
- **Eliminate 1,126 lines** from deprecated `coordinate_token_manager.py`
- **Reduce duplication** by ~200-300 lines across utility functions
- **Total reduction**: ~1,400+ lines

### Improved Organization
- **Clear separation** of concerns (data, formatting, validation, tokens, parsing)
- **Easier maintenance** with focused modules
- **Better discoverability** of utility functions

### Enhanced Reliability
- **Eliminate fallback patterns** in consolidated utilities
- **Consistent error handling** across all utility functions
- **Centralized validation** logic

### Performance Benefits
- **Reduced import overhead** with focused modules
- **Better caching** of utility functions
- **Cleaner dependency graph**

## Migration Checklist

- [ ] Create new directory structure
- [ ] Move and rename token management files
- [ ] Extract and consolidate data utilities
- [ ] Extract and consolidate format utilities
- [ ] Create validation utilities module
- [ ] Update response parser location
- [ ] Update all imports throughout codebase
- [ ] Update `__init__.py` exports
- [ ] Test all functionality works after consolidation
- [ ] Remove deprecated files
- [ ] Update documentation

## Risk Mitigation

1. **Backward Compatibility**: Maintain imports in `__init__.py` during transition
2. **Incremental Migration**: Move one module at a time
3. **Comprehensive Testing**: Validate each step before proceeding
4. **Rollback Plan**: Keep original files until migration is complete and tested
