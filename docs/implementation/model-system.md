# Model System Module Documentation

## Overview

The `src/models/` module provides the core model architecture for BBU equipment detection training, featuring a specialized Qwen2.5-VL wrapper with coordinate token support, advanced loss computation, and comprehensive model management capabilities.

## Core Architecture

### Primary Components

#### 1. Qwen25VLWithDetection (`wrapper.py`)
The main model wrapper that extends Qwen2.5-VL with detection capabilities.

**Key Features:**
- **Coordinate Token System**: Soft expectation regression for bbox prediction
- **Extended Vocabulary**: Additional tokens for coordinate representation
- **Multi-Geometry Support**: Handles bbox, line, and square annotations
- **Advanced Loss Computation**: Coordinate-aware loss with L1 and GIoU components
- **HuggingFace Integration**: Compatible with Trainer and save/load mechanisms

**Architecture Components:**
```python
class Qwen25VLWithDetection(nn.Module):
    def __init__(self, base_model_path: str, tokenizer: PreTrainedTokenizerBase,
                 coordinate_config: CoordinateConfig):
        """Initialize with coordinate token support"""
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass with coordinate-aware loss computation"""
        
    def _forward_with_coordinate_tokens(self, inputs, labels, **kwargs):
        """Enhanced forward pass with coordinate token processing"""
```

**Coordinate Token Integration:**
- **Token Range Management**: `x0`-`x1023`, `y0`-`y1023` for coordinate representation
- **Soft Expectation Loss**: Probabilistic coordinate prediction
- **Geometry Token Support**: Special tokens for different annotation types
- **Extended Embeddings**: Seamless integration with base model vocabulary

#### 2. CoordinateConfig (`wrapper.py`)
Configuration class for coordinate token system.

**Configuration Options:**
```python
@dataclass
class CoordinateConfig:
    coordinate_tokens_enabled: bool = True
    coordinate_loss_weight: float = 1.0
    coordinate_token_num: int = 1024  # x0-x1023, y0-y1023
    geometry_tokens_enabled: bool = True
    giou_loss_weight: float = 0.5
    focal_loss_weight: float = 1.0
    coordinate_vocab_size: int = 2048  # Total coordinate tokens
```

### Model Loading and Management

#### 3. Model Loader (`model_loader.py`)
Unified model loading with validation and compatibility checks.

**Key Functions:**
```python
def load_model_and_processor_unified(
    model_path: str,
    coordinate_config: Optional[CoordinateConfig] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None
) -> Tuple[Qwen25VLWithDetection, Any]:
    """Load model and processor with coordinate token support"""

def validate_training_inference_consistency(
    model: Qwen25VLWithDetection,
    tokenizer: PreTrainedTokenizerBase
) -> bool:
    """Validate model-tokenizer compatibility"""
```

**Loading Features:**
- **Automatic Device Management**: GPU memory optimization
- **Dtype Configuration**: Mixed precision support (float16, bfloat16)
- **Checkpoint Validation**: Ensures model integrity
- **Extension Loading**: Coordinate token extensions from checkpoints

#### 4. Model Patches (`patches.py`)
Critical fixes for Qwen2.5-VL integration and stability.

**Applied Patches:**
```python
def apply_comprehensive_qwen25_fixes():
    """Apply all necessary patches for stable training"""
    
def patch_torch_library_wrap_triton():
    """Fix triton compilation issues"""
    
def safe_visual_forward(original_forward):
    """Add safety checks to visual processing"""
    
def official_apply_multimodal_rotary_pos_emb():
    """Fix rotary position embedding for multimodal inputs"""
```

**Patch Categories:**
- **Triton Compilation**: Fixes GPU kernel compilation issues
- **Visual Processing**: Stabilizes image feature extraction
- **Position Embeddings**: Corrects multimodal position encoding
- **Memory Management**: Prevents CUDA memory leaks

## Advanced Features

### Coordinate Token System

#### Token Architecture
```python
# Coordinate representation
x_tokens = ["x0", "x1", ..., "x1023"]  # 1024 x-coordinate tokens
y_tokens = ["y0", "y1", ..., "y1023"]  # 1024 y-coordinate tokens

# Geometry markers
geometry_tokens = ["<bbox>", "</bbox>", "<line>", "</line>", "<square>", "</square>"]

# Total extension: 2048 + 6 = 2054 new tokens
```

#### Soft Expectation Loss
```python
def _compute_soft_expectation_loss_with_components(
    self, logits: torch.Tensor, target_coords: torch.Tensor, 
    coord_mask: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute probabilistic coordinate loss with L1 and GIoU components
    
    Returns:
        total_loss: Combined coordinate loss
        loss_components: Dictionary with individual loss components
    """
```

**Loss Components:**
- **L1 Loss**: Direct coordinate regression
- **GIoU Loss**: Geometric IoU for bbox relationships
- **Focal Loss**: Class-aware coordinate prediction
- **Consistency Loss**: Multi-geometry consistency enforcement

### Model Persistence

#### Save/Load Mechanisms
```python
def save_pretrained(self, save_directory: str, **kwargs):
    """Save model with coordinate extensions"""
    # Save base model components
    # Save coordinate token extensions
    # Save configuration metadata
    
def _load_coordinate_extensions(self, model_path: str):
    """Load coordinate token extensions from checkpoint"""
    # Load extended embeddings
    # Load extended language model head
    # Restore coordinate token mappings
```

**Checkpoint Structure:**
```
checkpoint/
├── model.safetensors              # Base model weights
├── coordinate_extensions.pt       # Extended embeddings & LM head
├── coordinate_config.json         # Coordinate configuration
├── tokenizer.json                # Extended tokenizer
└── generation_config.json        # Generation parameters
```

### Training Integration

#### Parameter Management
```python
def get_input_embeddings(self) -> nn.Module:
    """Return extended embeddings for training"""
    
def get_output_embeddings(self) -> nn.Module:  
    """Return extended LM head for generation"""
    
def resize_token_embeddings(self, new_num_tokens: int):
    """Handle dynamic vocabulary resizing"""
```

#### Loss Computation Pipeline
```python
def _compute_coordinate_aware_loss(
    self, logits: torch.Tensor, labels: torch.Tensor,
    coordinate_spans: List[Tuple[int, int]]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss with coordinate token awareness
    
    Pipeline:
    1. Separate coordinate vs regular tokens
    2. Apply appropriate loss functions
    3. Combine with weighted averaging
    4. Track loss components for monitoring
    """
```

## Usage Examples

### Basic Model Loading

```python
from src.models.model_loader import load_model_and_processor_unified
from src.models.wrapper import CoordinateConfig

# Configure coordinate tokens
coord_config = CoordinateConfig(
    coordinate_tokens_enabled=True,
    coordinate_loss_weight=1.0,
    coordinate_token_num=1024,
    geometry_tokens_enabled=True
)

# Load model
model, processor = load_model_and_processor_unified(
    model_path="/path/to/qwen2.5-vl",
    coordinate_config=coord_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Coordinate Token Conversion

```python
# Convert bbox to coordinate tokens
bbox = [100, 150, 200, 250]  # [x1, y1, x2, y2]
image_width, image_height = 1792, 1008

coord_tokens = model._convert_bbox_to_tokens(bbox, image_width, image_height)
# Returns: ["x57", "y152", "x114", "y253"] (normalized to token range)

# Convert back to bbox
recovered_bbox = model._convert_tokens_to_bbox(coord_tokens, image_width, image_height)
# Returns: [100, 150, 200, 250] (approximately, due to quantization)
```

### Training Integration

```python
from transformers import TrainingArguments
from src.training.trainer import BBUTrainer

# Model is already coordinate-aware
trainer = BBUTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args
)

# Training automatically uses coordinate-aware loss
trainer.train()
```

### Advanced Loss Monitoring

```python
# During training, access detailed loss components
loss_components = model.get_last_coordinate_losses()
print(f"LLM Loss: {loss_components['_llm_loss']:.4f}")
print(f"Coordinate L1: {loss_components['_coordinate_l1_loss']:.4f}")
print(f"GIoU Loss: {loss_components['_geometry_bbox_giou_loss']:.4f}")
print(f"Focal Loss: {loss_components['_geometry_focal_loss']:.4f}")
```

## Performance Optimizations

### Memory Management
- **Gradient Checkpointing**: Reduces memory usage during training
- **Extended Vocabulary Caching**: Efficient coordinate token storage
- **Dynamic Batch Processing**: Handles variable sequence lengths

### Computation Efficiency
- **Selective Loss Computation**: Only compute coordinate loss for relevant tokens
- **Vectorized Operations**: Efficient coordinate transformations
- **Mixed Precision**: Automatic support for float16/bfloat16

### GPU Optimization
- **Device Mapping**: Automatic multi-GPU distribution
- **Memory Pooling**: Efficient CUDA memory management
- **Kernel Fusion**: Optimized coordinate loss kernels

## Architecture Decisions

### Design Principles
1. **Minimal Base Model Changes**: Preserve Qwen2.5-VL architecture
2. **Extension-Based Approach**: Add coordinate capabilities without modification
3. **Backward Compatibility**: Support both coordinate and standard modes
4. **Robust Error Handling**: Fail-fast with clear error messages

### Integration Strategy
- **Composition over Inheritance**: Wrap rather than modify base model
- **Configuration-Driven**: All features controllable via config
- **Progressive Enhancement**: Coordinate tokens are optional
- **Validation-First**: Extensive checks for model consistency

### Performance Considerations
- **Lazy Loading**: Load extensions only when needed
- **Caching Strategy**: Cache frequently accessed components
- **Memory Profiling**: Built-in memory usage monitoring
- **Batch Optimization**: Efficient handling of mixed batch types

## Integration Points

### Data Pipeline Integration
- Seamless connection with data conversion module
- Automatic coordinate token generation from geometric annotations
- Support for multi-geometry training data

### Training System Integration
- Compatible with BBUTrainer enhanced training loops
- Supports teacher-student learning paradigms
- Integrates with loss management and monitoring systems

### Inference Integration
- Standard HuggingFace generation interface
- Coordinate token decoding to bounding boxes
- Multi-geometry prediction capabilities

This model system provides a robust, scalable foundation for vision-language training with specialized coordinate prediction capabilities while maintaining full compatibility with the HuggingFace ecosystem.