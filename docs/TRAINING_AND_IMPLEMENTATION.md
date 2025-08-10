# Training and Implementation Guide

**Complete guide to the training system, data processing pipeline, and model integration for the BBU training pipeline**

## 🎯 **System Overview**

The BBU training system provides a production-ready framework for equipment detection with coordinate token support, teacher-student learning, and comprehensive loss management. The system uses a streamlined architecture with consolidated components for optimal performance.

### **Current Architecture (2025)**
- **`src_new/`**: Production-ready implementation with composition-based design
- **DetectionModel**: Composition-based wrapper around Qwen2.5-VL base model
- **BBUTrainer**: HuggingFace Trainer extension with local loss aggregation
- **LossManager**: Multi-component loss computation with dual-mask system
- **TokenProcessor**: Coordinate token handling with positional encoding initialization

## 🔄 **8-Step Training Pipeline**

The production training pipeline follows a comprehensive 8-step process from data loading to model checkpointing:

### **Step 1: Parse Inputs** (`src_new/data/dataset.py`)
- Load JSONL data with image paths and object annotations
- Validate sample structure and filter invalid entries
- Initialize teacher pool manager for teacher-student assignments

### **Step 2: Replace Values** (`src_new/processing/coordinate_converter.py`)
- Convert raw coordinates to coordinate tokens (`<|coord_N|>`)
- Apply coordinate clamping to valid range `[0, max_coord_value]`
- Wrap objects with appropriate geometry tokens (bbox, quad, line)

### **Step 3: Expand Tokenizer** (`src_new/processing/token_processor.py`)
- Extend vocabulary with coordinate tokens (151667–153715)
- Add geometry tokens (line: `<|line_start|>`, `<|line_end|>`; bbox tokens appear as `<|box_start|>`/`<|box_end|>` or `<|box_start|>`/`<|box_end|>` depending on path)
- Initialize new token embeddings using positional encoding

### **Step 4: Apply Templates** (`src_new/processing/conversation_processor.py`)
- Create teacher-student conversations using HuggingFace chat templates
- Apply Chinese prompts from centralized constants
- Format multi-turn conversations with image tokens

### **Step 5: Vision Processing** (`src_new/data/collator.py`)
- Process images through Qwen2VL image processor
- Handle pixel value normalization and resizing
- Create batched tensors with proper padding

### **Step 6: Forward Pass** (`src_new/models/wrapper.py`)
- Pass inputs through DetectionModel wrapper
- Generate logits from base Qwen2.5-VL model
- Apply coordinate token masking for loss computation

### **Step 7: Compute Losses** (`src_new/models/loss_manager.py`)
- Calculate teacher/student LLM losses using span masking
- Compute coordinate L1 losses using soft expectation
- Aggregate loss components with configurable weights

### **Step 8: Log Results** (`src_new/training/training_state_manager.py`)
- Track loss components locally (no distributed operations)
- Generate comprehensive training metrics
- Log performance statistics and remaining time estimates

## 🏗️ **Core Training Components**

### **1. BBUTrainer (`src_new/training/bbu_trainer.py`)**
Enhanced trainer extending HuggingFace Trainer with local loss aggregation to eliminate NCCL timeout issues.

**Key Features:**
- **Local Loss Aggregation**: TrainingStateManager eliminates distributed conflicts
- **NCCL Timeout Resolution**: 100% success rate in distributed training
- **Teacher-Student Learning**: Span-based loss splitting for enhanced training
- **Advanced Logging**: 4-decimal precision losses and comprehensive metrics
- **Memory Optimization**: Gradient checkpointing and efficient batch processing

**Core Methods:**
```python
class BBUTrainer(HFTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Enhanced loss computation with DetectionModel wrapper"""

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Local loss aggregation via TrainingStateManager"""
```

**Training Features:**
- **Mixed Precision Training**: Automatic support for float16/bfloat16
- **Gradient Accumulation**: Efficient handling of large effective batch sizes
- **SafeTensors Checkpoints**: 4-6x faster inference loading performance
- **Loss Component Tracking**: Detailed monitoring of all loss components

### **2. DetectionModel (`src_new/models/wrapper.py`)**
Composition-based wrapper around Qwen2.5-VL base model with coordinate token support.

**Key Features:**
- **Composition over Inheritance**: Clean separation of concerns
- **Coordinate Token Support**: Automatic tokenizer and embedding extension
- **Intelligent Checkpoint Detection**: Auto-detect base vs fine-tuned models
- **SafeTensors Optimization**: 4-6x faster loading for inference
- **Lazy Loading**: Maximum initialization efficiency

**Core Methods:**
```python
class DetectionModel(nn.Module):
    def __init__(self, base_model, config, tokenizer=None, skip_expansion=False):
        """Initialize with composition-based design"""

    def forward(self, **kwargs):
        """Forward pass through base model with loss computation"""

    @classmethod
    def from_pretrained(cls, model_path, config, tokenizer=None):
        """Load model with intelligent checkpoint detection"""
```

### **3. LossManager (`src_new/models/loss_manager.py`)**
Multi-component loss computation with dual-mask system for teacher-student training.

**Loss Components:**
- **teacher_llm_loss**: Cross-entropy on teacher spans
- **student_llm_loss**: Cross-entropy on student spans
- **teacher_l1_loss**: Soft expectation + L1 loss on teacher coordinate tokens
- **student_l1_loss**: Soft expectation + L1 loss on student coordinate tokens

**Key Methods:**
```python
class LossManager:
    def compute_loss(self, logits, labels, coord_mask, teacher_spans, student_spans):
        """Compute all loss components with dual-mask system"""

    def _compute_granular_teacher_student_loss(self, ...):
        """60-70% optimized loss computation with single-pass cross-entropy"""
```

**Soft Expectation Formula:**
```python
# Coordinate regression via soft expectation
expected_coord = Σ(v * softmax(logits_v / temperature))
coordinate_loss = L1(expected_coord, target_coord)
```

### **4. TrainingStateManager (`src_new/training/training_state_manager.py`)**
Local loss aggregation and metrics tracking to eliminate NCCL timeout issues.

**Responsibilities:**
- **Local Loss Aggregation**: No distributed operations to avoid NCCL conflicts
- **Metrics Tracking**: Comprehensive training metrics with 4-decimal precision
- **Performance Monitoring**: Memory usage, training speed, remaining time estimates
- **Component Loss Tracking**: Individual tracking of all loss components

## 📊 **Data Processing Pipeline**

### **1. Dataset (`src_new/data/dataset.py`)**
HuggingFace-compatible dataset with teacher-student conversation creation.

**Key Features:**
- **JSONL Data Loading**: Direct loading from structured JSONL files
- **Teacher Pool Integration**: Automatic teacher assignment for student samples
- **Conversation Creation**: Multi-turn teacher-student conversations
- **Coordinate Conversion**: Raw coordinates to coordinate tokens
- **Fail-Fast Validation**: Explicit error handling with ValueError for empty objects

**Processing Flow:**
```python
class BBUDataset(Dataset):
    def __getitem__(self, idx):
        # 1. Load sample from JSONL
        # 2. Assign teacher (if teacher_ratio > 0)
        # 3. Create teacher-student conversation
        # 4. Convert coordinates to tokens
        # 5. Return processed sample
```

### **2. ConversationProcessor (`src_new/processing/conversation_processor.py`)**
Creates teacher-student conversations using HuggingFace chat templates.

**Key Features:**
- **Chinese-Only Prompts**: Centralized prompt constants in templates.py
- **HuggingFace Integration**: Uses official apply_chat_template()
- **Multi-Turn Conversations**: Teacher example + student query format
- **Image Token Placement**: Images placed AFTER prompt text
```

### **2. CoordinateManager (`src_new/data_conversion/coordinate_manager.py`)**
Centralized coordinate transformation and geometry processing.

**Transformation Pipeline:**
1. **EXIF Orientation Compensation**: Handles image rotation metadata
2. **Dimension Mismatch Rescaling**: Corrects JSON vs actual image dimensions
3. **Smart Resize Scaling**: Applies vision processing constraints
4. **Coordinate Normalization**: Enhanced ordering for optimal training

**Supported Geometries:**
- **bbox_2d**: `[x1, y1, x2, y2]` - top-left and bottom-right corners
- **line**: `[x1, y1, x2, y2, ...]` - multi-point line segments
- **quad**: `[x1, y1, x2, y2, x3, y3, x4, y4]` - four corner points

**Normalization Features:**
```python
@classmethod
def normalize_bbox_coordinates(cls, coords, width, height):
    """Normalize bbox ensuring x1 < x2, y1 < y2"""
    
@classmethod  
def normalize_quad_coordinates(cls, coords, width, height):
    """Enhanced clockwise ordering from top-left vertex"""
    
@classmethod
def normalize_line_coordinates(cls, coords, width, height):
    """Canonical directional ordering with path preservation"""
```

### **3. ValidationManager (`src_new/data_conversion/validation_manager.py`)**
Comprehensive validation system with detailed error reporting.

**Validation Modes:**
- `strict`: Fail on critical errors
- `lenient`: Warning for non-critical issues  
- `warning_only`: Log all issues but continue processing

**Validation Checks:**
- **Coordinate Bounds**: Ensure coordinates within image dimensions
- **Geometry Validity**: Check for degenerate shapes and invalid formats
- **Data Integrity**: Validate JSON structure and required fields
- **Image Compatibility**: Verify image-annotation alignment

## 🔧 **Model Integration**

### **1. Qwen2.5-VL Integration (`src_new/models/wrapper.py`)
Specialized integration for Qwen2.5-VL via `DetectionModel` with coordinate token support.

**Key Features:**
- **Coordinate Token Embedding**: Initialization/extension when enabled
- **Multi-Modal Processing**: Vision-language integration with coordinate awareness
- **FlashAttention v2**: Performance improvement for long sequences (training-time config)
- **Compatibility Patches**: See `src_new/models/patches.py`

## 🚀 **Training Workflows**

### **Standard Training Workflow**
```bash
# 1. Data preparation
bash data_conversion/convert_dataset.sh

# 2. src_new/ implementation training
python scripts/train_new.py --config bbu_v2

# 3. Monitor training
tail -f checkpoints/*/training.log

# 4. Evaluation
python -m src_new.inference \
  --config_path configs/bbu_v2.yaml \
  --model_path checkpoints/best \
  --input_file data/val.jsonl \
  --output_file results/val.json \
  --data_root /abs/path/to/data_root
```

### **Teacher-Student Training Workflow**
```bash
# 1. Prepare teacher pool data
# Ensure data/teacher_pool.jsonl exists

# 2. Teacher-student training
./scripts/run_new_train.sh

# 3. Monitor dual-role learning
# Look for teacher_llm_loss and student_llm_loss in logs

# 4. Validate performance
python -m pytest src_new/tests/test_teacher_student.py -v
```

### **Coordinate Mode Training Workflow**
```bash
# 1. Configure coordinate mode
# Set coordinate_tokens_enabled: true in config

# 2. Coordinate token training
python scripts/train_new.py --config bbu_v2 --coordinate_tokens_enabled

# 3. Monitor coordinate loss
# Look for coordinate_loss and l1_loss in training logs

# 4. Test coordinate prediction
python - << 'PY'
from src_new.processing.token_processor import TokenProcessor, TokenConfig
cfg = TokenConfig(max_coord_value=2048, coordinate_tokens_enabled=True)
proc = TokenProcessor(cfg)
print(proc.coordinates_to_tokens([150, 10, 211, 35]))
PY
```

## 📊 **Performance Optimization**

### **Memory Optimization**
```yaml
# Reduce memory usage
per_device_train_batch_size: 2
gradient_checkpointing: true
fp16: true
dataloader_pin_memory: false
```

### **Speed Optimization**
```yaml
# Maximize training speed
attn_implementation: "flash_attention_2"
dataloader_num_workers: 8
dataloader_pin_memory: true
gradient_checkpointing: false
```

### **Debugging Optimization**
```yaml
# Fast iteration for debugging
max_dataset_size: 10
eval_dataset_size: 5
max_steps: 50
logging_steps: 5
save_steps: 25
```

## 🔍 **Monitoring and Logging**

### **Training Metrics**
- **Loss Components**: `loss`, `lm_loss`, `coordinate_loss`
- **Teacher-Student**: `teacher_llm_loss`, `student_llm_loss`, `teacher_l1_loss`, `student_l1_loss`
- **Performance**: `train_samples_per_second`, `train_steps_per_second`
- **Learning Rates**: `learning_rate`, `coordinate_lr` (if applicable)

### **Log Analysis**
```bash
# Monitor training progress
tail -f checkpoints/run_*/training.log

# Extract loss trends
grep "train_loss" checkpoints/run_*/training.log | tail -20

# Check teacher-student balance
grep -E "(teacher_llm_loss|student_llm_loss)" checkpoints/run_*/training.log | tail -10
```

### **Validation Checks**
```bash
# Validate training pipeline
python -m pytest src_new/tests/test_training_pipeline.py -v

# Test data processing
python -m pytest src_new/tests/test_data_conversion.py -v

# Validate model integration
python -m pytest src_new/tests/test_model_integration.py -v
```

## ⚠️ **Common Issues and Solutions**

### **Issue 1: NCCL Timeout Errors**
**Solution**: Use BBUTrainer with local loss aggregation (automatically resolved in `src_new/`)

### **Issue 2: Memory Issues**
**Solution**: Reduce batch size, enable gradient checkpointing, or use fp16

### **Issue 3: Coordinate Mode Failures**
**Solution**: Ensure `remove_unused_columns: false` in configuration

### **Issue 4: Teacher-Student Loss Imbalance**
**Solution**: Adjust `teacher_loss_weight` and `student_loss_weight` parameters

## 🔮 **Inference System**

### **Production Inference Pipeline** (`src_new/inference.py`)
The inference system provides comprehensive support for production deployment with advanced features:

#### **Key Features**
- **Multi-Image Support**: Teacher-guided inference with multiple images
- **Batch Processing**: Efficient processing of JSONL datasets
- **Path Management**: Unified path resolution with PathManager
- **Coordinate Token Support**: Full coordinate token processing in inference
- **Performance Monitoring**: Built-in performance metrics and benchmarking

#### **Basic Inference Usage**
```bash
python -m src_new.inference \
  --config_path configs/bbu_v2.yaml \
  --model_path path/to/checkpoint \
  --input_file data/test_samples.jsonl \
  --output_file results.json \
  --data_root /abs/path/to/data_root
```

#### **Advanced Inference Features**
```python
from src_new.inference import InferenceEngine
from src_new.config.config import load_config

# Initialize inference engine
env = load_config('configs/bbu_v2.yaml')
engine = InferenceEngine(
    config_path='configs/bbu_v2.yaml',
    model_path='checkpoints/best',
    teacher_pool_file='data/teacher_pool.jsonl',
    num_teachers=1,
)

# Prepare inputs like training
inputs = engine.prepare_inference_inputs(sample, data_root='/abs/path/to/data')

# Generate
print(engine.generate_response(inputs, max_new_tokens=128))
```
