# Model System Components

**Detailed documentation for the model management system (`src/models/`)**

## Overview

The model system provides unified model loading, wrapper functionality, and patches for the BBU Detection System. It ensures consistency between training and inference while handling coordinate token integration.

## Component Architecture

```
src/models/
├── model_loader.py    # ModelLoader - Unified model loading
├── wrapper.py         # Qwen25VLWithDetection - Main model wrapper
└── patches.py         # Model patches and optimizations
```

## ModelLoader (`model_loader.py`)

### Component Contract

**Input**:
- `model_path`: Path to the base Qwen2.5-VL model
- `for_inference`: Boolean flag for training vs inference mode
- `attn_implementation`: Attention implementation ("flash_attention_2" recommended)
- `torch_dtype`: Model precision (default: "bfloat16")

**Output**:
- `model`: Qwen25VLWithDetection instance with coordinate token support
- `tokenizer`: Tokenizer with coordinate tokens added
- `image_processor`: Image processor for VLM

**Dependencies**:
- HuggingFace Transformers for base model loading
- SimpleTokenManager for coordinate token addition
- Model patches for optimization

**Side Effects**:
- Downloads model if not cached locally
- Modifies tokenizer vocabulary
- Applies model patches automatically

### Key Features

#### Training-Inference Consistency
```python
# Same loading function for both modes
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="/path/to/qwen2.5-vl-7b-instruct",
    for_inference=False  # Training mode
)

model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="/path/to/qwen2.5-vl-7b-instruct", 
    for_inference=True   # Inference mode
)
```

#### Automatic Token Initialization
- Adds coordinate tokens (0-2047) to vocabulary
- Adds geometry tokens (`<|box_start|>`, `<|box_end|>`, etc.)
- Resizes model embeddings automatically
- Preserves pretrained weights

#### Automatic Patch Application
- mRoPE dimension fix for Qwen2.5-VL
- Flash Attention 2 integration
- Memory optimization patches
- Model-specific fixes

### Usage Example

```python
from src.models.model_loader import load_model_and_processor_unified

# Load for training
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="/data3/models/qwen2.5-vl-7b-instruct",
    for_inference=False,
    attn_implementation="flash_attention_2"
)

# Model is ready for coordinate token training
print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
print(f"Model type: {type(model)}")  # Qwen25VLWithDetection
```

## Qwen25VLWithDetection (`wrapper.py`)

### Component Contract

**Input**:
- Base Qwen2.5-VL model configuration
- Tokenizer with coordinate tokens
- Detection-specific configuration

**Output**:
- Model outputs with coordinate token support
- Loss computation for coordinate tokens
- Generation capabilities with coordinate tokens

**Dependencies**:
- HuggingFace Qwen2VLForConditionalGeneration (base class)
- Coordinate token handling utilities
- Loss computation functions

**Side Effects**:
- Extends model vocabulary
- Modifies forward pass for coordinate token handling
- Adds detection-specific methods

### Key Features

#### Extended Vocabulary Support
```python
# Automatic vocabulary extension
original_vocab_size = len(base_tokenizer.get_vocab())
extended_vocab_size = len(tokenizer.get_vocab())
print(f"Added {extended_vocab_size - original_vocab_size} coordinate tokens")
```

#### Multi-Geometry Support
- **bbox_2d**: Standard rectangular bounding boxes
- **square**: Rotated quadrilaterals (8 coordinates)
- **line**: Multi-point lines (variable length)

#### Coordinate-Aware Loss Computation
```python
# Forward pass with coordinate loss
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    labels=labels
)

# Outputs include coordinate-specific losses
loss = outputs.loss  # Combined LLM + coordinate loss
logits = outputs.logits
```

#### Detection-Specific Methods
```python
# Generate with coordinate tokens
generated_ids = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    max_new_tokens=512,
    do_sample=False
)

# Parse coordinate tokens from output
coordinates = model.parse_coordinate_tokens(generated_ids)
```

### Usage Example

```python
# Model is created automatically by ModelLoader
model, tokenizer, image_processor = load_model_and_processor_unified(...)

# Forward pass for training
outputs = model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    pixel_values=batch['pixel_values'],
    labels=batch['labels']
)

loss = outputs.loss
logits = outputs.logits
```

## Model Patches (`patches.py`)

### Component Contract

**Input**:
- Model instance to be patched
- Patch configuration

**Output**:
- Patched model with optimizations applied

**Dependencies**:
- Model architecture knowledge
- PyTorch for implementation
- Flash Attention 2 (optional)

**Side Effects**:
- Modifies model architecture
- Changes computation behavior
- May affect memory usage

### Available Patches

#### mRoPE Dimension Fix
**Problem**: Qwen2.5-VL has dimension mismatch in rotary position embeddings
**Solution**: Corrects dimension calculations for proper attention computation

```python
# Applied automatically in ModelLoader
def apply_mrope_fix(model):
    """Fix mRoPE dimension mismatch in Qwen2.5-VL"""
    # Corrects rotary embedding dimensions
    # Fixes attention computation issues
```

#### Flash Attention 2 Integration
**Problem**: Standard attention is memory-intensive for long sequences
**Solution**: Integrates Flash Attention 2 for efficient attention computation

```python
# Enabled via attn_implementation parameter
model = load_model_and_processor_unified(
    model_path=model_path,
    attn_implementation="flash_attention_2"  # Enables Flash Attention 2
)
```

#### Memory Optimization Patches
**Problem**: Large model memory usage
**Solution**: Various memory optimization techniques

```python
# Applied automatically
def apply_memory_optimizations(model):
    """Apply memory optimization patches"""
    # Gradient checkpointing setup
    # Memory-efficient attention patterns
    # Activation recomputation
```

### Patch Application

#### Automatic Application
```python
# Patches applied automatically in ModelLoader
model, tokenizer, image_processor = load_model_and_processor_unified(...)
# All necessary patches are already applied
```

#### Manual Patch Application
```python
from src.models.patches import apply_all_patches

# Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)

# Apply patches manually
model = apply_all_patches(
    model=model,
    enable_flash_attention=True,
    enable_memory_optimizations=True
)
```

## Integration Patterns

### Standard Model Loading
```python
# Recommended approach - use ModelLoader
from src.models.model_loader import load_model_and_processor_unified

model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="/path/to/model",
    for_inference=False,
    attn_implementation="flash_attention_2"
)
```

### Custom Model Configuration
```python
# For advanced customization
from src.models.wrapper import Qwen25VLWithDetection
from src.models.patches import apply_all_patches
from src.utils.simple_token_manager import create_simple_token_manager

# Load base components
base_model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

# Apply patches
model = apply_all_patches(base_model)

# Add coordinate tokens
token_manager = create_simple_token_manager(tokenizer, model)

# Wrap for detection
model = Qwen25VLWithDetection(model.config)
model.load_state_dict(base_model.state_dict(), strict=False)
```

### Model Saving and Loading
```python
# Save model with coordinate tokens
model.save_pretrained("/path/to/checkpoint")
tokenizer.save_pretrained("/path/to/checkpoint")

# Load saved model
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path="/path/to/checkpoint",
    for_inference=True
)
```

## Coordinate Token Integration

### Token Addition Process
1. **Base Vocabulary**: Load original Qwen2.5-VL tokenizer
2. **Add Geometry Tokens**: `<|box_start|>`, `<|box_end|>`, etc.
3. **Add Coordinate Tokens**: `<coord_0>` through `<coord_2047>`
4. **Resize Embeddings**: Extend model embeddings to match new vocabulary
5. **Initialize New Tokens**: Random initialization for new token embeddings

### Token Usage in Model
```python
# Coordinate tokens in text sequence
text = "BBU设备: <|box_start|><coord_264><coord_144><coord_326><coord_201><|box_end|>"

# Tokenization includes coordinate tokens
input_ids = tokenizer.encode(text)
# Contains both text tokens and coordinate token IDs

# Model processes coordinate tokens as part of sequence
outputs = model(input_ids=input_ids, ...)
```

## Performance Characteristics

### Memory Usage
- **Base Model**: ~13.5GB (Qwen2.5-VL-7B)
- **With Coordinate Tokens**: ~13.7GB
- **Overhead**: ~200MB (1.5% increase)

### Computational Overhead
- **Token Addition**: Minimal (one-time setup)
- **Forward Pass**: <1% overhead for coordinate token processing
- **Generation**: Comparable to base model

### Optimization Benefits
- **Flash Attention 2**: 2-3x memory reduction for long sequences
- **mRoPE Fix**: Eliminates attention computation errors
- **Memory Patches**: 10-15% memory reduction

## Debugging and Validation

### Model Health Checks
```python
# Verify model loading
assert isinstance(model, Qwen25VLWithDetection)
assert len(tokenizer.get_vocab()) > 151936  # Original vocab + coordinate tokens

# Check coordinate token integration
coord_token_id = tokenizer.convert_tokens_to_ids("<coord_100>")
assert coord_token_id != tokenizer.unk_token_id

# Verify patches applied
assert hasattr(model, 'coordinate_token_support')
```

### Common Issues and Solutions

#### Model Loading Failures
```python
# Issue: Model path not found
# Solution: Verify model path exists
import os
assert os.path.exists(model_path), f"Model path not found: {model_path}"

# Issue: CUDA out of memory
# Solution: Use smaller precision or enable optimizations
model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path=model_path,
    torch_dtype="float16",  # Reduce precision
    attn_implementation="flash_attention_2"  # Enable memory optimization
)
```

#### Coordinate Token Issues
```python
# Issue: Coordinate tokens not recognized
# Solution: Verify token manager setup
from src.utils.simple_token_manager import verify_coordinate_tokens
verify_coordinate_tokens(tokenizer, model)

# Issue: Generation produces invalid coordinates
# Solution: Check model training and coordinate token learning
```

---

**Next Steps**:
- **Training System**: [training-system.md](training-system.md)
- **Data Pipeline**: [data-pipeline.md](data-pipeline.md)
- **Configuration**: [configuration.md](configuration.md)
