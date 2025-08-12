# Qwen2.5-VL BBU Detection Training Pipeline - Unified Documentation

**Complete Reference for Vision-Language Model Fine-tuning with Coordinate Token System**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Training Pipeline Flow](#training-pipeline-flow)
3. [Input Data Format](#input-data-format)
4. [Token Conversion Pipeline](#token-conversion-pipeline)
5. [Conversation Structure & Templates](#conversation-structure--templates)
6. [Span Detection & Loss Masking](#span-detection--loss-masking)
7. [Training Architecture](#training-architecture)
8. [Loss Computation System](#loss-computation-system)
9. [Distributed Training & Checkpoints](#distributed-training--checkpoints)
10. [Configuration & Setup](#configuration--setup)
11. [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

The Qwen2.5-VL training pipeline implements **teacher-student learning** for BBU equipment detection with coordinate token regression. The system converts raw object coordinates into special tokens and trains the model to generate accurate coordinate predictions through multi-task learning.

### Key Features
- **Vision-Language Integration**: End-to-end dense detection → Chinese captions
- **Coordinate Token System**: Automatic bbox ↔ token conversion via soft regression
- **Multi-Task Training**: Teacher–student span-based loss splits
- **Modular Architecture**: Clean component boundaries, plug-and-play design
- **Loss Management**: Mode-aware switching between coordinate and LLM loss
- **NCCL Timeout Resolution**: Local loss aggregation eliminates distributed conflicts
- **SafeTensors Optimization**: 4-6x faster checkpoint loading for inference

### Architecture Components
```
Raw Data (JSONL) → Coordinate Conversion → Conversation Templates →
Tokenization → Span Detection → Training → Model Checkpoints
```

---

## 🔄 Training Pipeline Flow

### Complete 8-Step Training Flow

The training pipeline follows a comprehensive 8-step process from data loading to model checkpointing:

#### **Step 1: Parse Inputs** (`src_new/data/dataset.py`)
- Load JSONL data with image paths and object annotations
- Validate sample structure and filter invalid entries
- Initialize teacher pool manager for teacher-student assignments

#### **Step 2: Replace Values** (`src_new/processing/coordinate_converter.py`)
- Convert raw coordinates to coordinate tokens (`<|coord_N|>`)
- Apply coordinate clamping to valid range `[0, max_coord_value]`
- Wrap objects with appropriate geometry tokens (bbox, quad, line)

#### **Step 3: Expand Tokenizer** (`src_new/processing/token_processor.py`)
- Extend vocabulary with coordinate tokens (151667-153715)
- Add geometry tokens (`<|line_start|>`, `<|line_end|>`)
- Initialize new token embeddings using positional encoding

#### **Step 4: Apply Templates** (`src_new/processing/conversation_processor.py`)
- Create teacher-student conversations using HuggingFace chat templates
- Apply Chinese prompts from centralized constants
- Format multi-turn conversations with image tokens

#### **Step 5: Vision Processing** (`src_new/data/collator.py`)
- Process images through Qwen2VL image processor
- Handle pixel value normalization and resizing
- Create batched tensors with proper padding

#### **Step 6: Forward Pass** (`src_new/models/wrapper.py`)
- Pass inputs through DetectionModel wrapper
- Generate logits from base Qwen2.5-VL model
- Apply coordinate token masking for loss computation

#### **Step 7: Compute Losses** (`src_new/models/loss_manager.py`)
- Calculate teacher/student LLM losses using span masking
- Compute coordinate L1 losses using soft expectation
- Aggregate loss components with configurable weights

#### **Step 8: Log Results** (`src_new/training/training_state_manager.py`)
- Track loss components locally (no distributed operations)
- Generate comprehensive training metrics
- Log performance statistics and remaining time estimates

### Component Interactions

#### **Data Flow Architecture**
```
JSONL Input → Dataset → ConversationProcessor → Collator → DetectionModel → LossManager → BBUTrainer
     ↓            ↓            ↓                    ↓            ↓              ↓            ↓
Raw Objects → Structured → Conversations → Batched → Model → Loss → Training
              Samples      + Images        Tensors   Outputs  Components  Loop
```

#### **Module Dependencies**
- **`Dataset`** → **`ConversationProcessor`**: Creates teacher-student conversations
- **`ConversationProcessor`** → **`CoordinateTokenConverter`**: Converts coordinates to tokens
- **`TokenProcessor`** → **`DetectionModel`**: Extends tokenizer and model embeddings
- **`DetectionModel`** → **`LossManager`**: Computes multi-component losses
- **`BBUTrainer`** → **`TrainingStateManager`**: Aggregates metrics locally
- **`BBUTrainer`** → **`UnifiedCheckpointManager`**: Direct folder copy for best checkpoints

#### **Data Transformations**

**Raw JSONL Sample**:
```json
{
  "images": ["path/to/image.jpg"],
  "objects": [{"bbox_2d": [100, 200, 300, 400], "desc": "BBU设备"}],
  "width": 532, "height": 728
}
```

**After Coordinate Conversion**:
```
<|object_ref_start|>BBU设备<|object_ref_end|><|box_start|>[<|coord_100|>, <|coord_200|>, <|coord_300|>, <|coord_400|>]<|box_end|>  # Inference accepts `<|obj_ref_*|>` synonymously
```

**After Conversation Creation**:
```
<|im_start|>system\n你是通信机房设备检测AI助手...<|im_end|>
<|im_start|>user\n现在请你回答，请检测图像中的设备和部件:<image><|im_end|>
<|im_start|>assistant\n<|object_ref_start|>BBU设备<|object_ref_end|>...<|im_end|>
```

**After Tokenization & Span Detection**:
```python
input_ids: [151644, 8948, 198, 你是通信..., 151645, 151644, 872, 198, 现在请你..., <image_tokens>, 151645, 151644, 77091, 198, 151652, BBU设备, 151653, 151659, 91, 151767, 151867, 151967, 152067, 93, 151660, 151645]
labels:    [-100, -100, -100, -100, ..., -100, -100, -100, -100, 151652, BBU设备, 151653, 151659, 91, 151767, 151867, 151967, 152067, 93, 151660, 151645]
teacher_spans: []  # No teacher in this example
student_spans: [(span_start, span_end)]  # Assistant content span
```

---

## 📊 Input Data Format

### JSONL Schema

Each training sample follows this exact structure:

```json
{
  "images": ["path/to/image.jpeg"],
  "objects": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "desc": "设备描述文本"
    },
    {
      "quad": [x1, y1, x2, y2, x3, y3, x4, y4],
      "desc": "四边形设备描述"
    },
    {
      "line": [x1, y1, x2, y2, ..., xn, yn],
      "desc": "线性设备描述"
    }
  ],
  "width": 532,
  "height": 728
}
```

### Supported Geometry Types

| Type | Format | Description | Example Objects |
|------|--------|-------------|-----------------|
| `bbox_2d` | `[x1, y1, x2, y2]` | Rectangular bounding box | BBU设备, 标签贴纸 |
| `quad` | `[x1, y1, x2, y2, x3, y3, x4, y4]` | Four-point polygon | 倾斜的BBU设备 |

| `line` | `[x1, y1, x2, y2, ..., xn, yn]` | Multi-point line | 光纤, 电线 |

### Object Types

The system supports 6 main object categories:
- **`bbu`**: BBU设备 (Base Band Unit equipment)
- **`bbu_shield`**: 挡风板 (Wind shields)
- **`connect_point`**: 螺丝、光纤插头 (Connection points)
- **`label`**: 标签贴纸 (Labels and stickers)
- **`fiber`**: 光纤 (Optical fibers) - line geometry only
- **`wire`**: 电线 (Electrical wires) - line geometry only

### Example Complete Sample

```json
{
  "images": ["ds_output/QC-20230312-0000764_74877.jpeg"],
  "objects": [
    {
      "bbox_2d": [293.40, 350.26, 412.27, 409.30],
      "desc": "标签贴纸"
    }
  ],
  "width": 532,
  "height": 728
}
```

---

## 🔄 Token Conversion Pipeline

### Coordinate Token System (`src_new/processing/token_processor.py`)

The system extends the Qwen2.5-VL vocabulary with coordinate tokens for precise spatial understanding:

#### **Vocabulary Extension**
- **Original Vocabulary**: 151,665 tokens (standard Qwen2.5-VL)
- **Extended Vocabulary**: 151,665 + (max_coord + 3) tokens
- **Total Extension**: 2,051 new tokens (2 geometry + 2,049 coordinate tokens)

#### **Token ID Ranges**

| Token Type | ID Range | Count | Purpose |
|------------|----------|-------|---------|
| Original Tokens | 0 - 151,664 | 151,665 | Standard Qwen2.5-VL vocabulary |
| Line Tokens | 151,665 - 151,666 | 2 | `<|line_start|>`, `<|line_end|>` |
| Coordinate Tokens | 151,667 - 153,715 | 2,049 | `<|coord_0|>` to `<|coord_2048|>` |

#### **Token Initialization Strategy**

**Geometry Tokens**: Initialized from existing similar tokens
```python
# Line tokens initialized from quad tokens
line_start_embedding = quad_start_embedding.clone()
line_end_embedding = quad_end_embedding.clone()
```

**Coordinate Tokens**: Deterministic sinusoidal initialization
```python
# Sinusoidal (positional-encoding-like) initialization for coordinate tokens
# Generates sin/cos features over scaled coordinate values and matches base std
coord_ids = [tokenizer.get_vocab()[f"<|coord_{i}|>"] for i in range(max_coord_value + 1)]
pos = torch.arange(0, max_coord_value + 1, dtype=torch.float32).unsqueeze(1)
half = embedding_dim // 2
inv_freq = torch.exp(torch.arange(0, half) * (-(math.log(10000.0) / max(1, half))))
angles = (pos / max_coord_value * 10000.0) * inv_freq
sin = torch.sin(angles); cos = torch.cos(angles)
pe = torch.zeros(len(coord_ids), embedding_dim)
pe[:, :half] = sin; pe[:, half:half*2] = cos
pe = pe * base_std  # base_std from pretrained embedding slice
input_embeddings.weight[coord_ids] = pe.to(input_embeddings.weight.dtype)
```

#### **Coordinate Token Processing**

**Coordinate Clamping**:
```python
# All coordinates clamped to valid range [0, max_coord_value]
clamped_coord = max(0, min(int(coord), max_coord_value))
coord_token = f"<|coord_{clamped_coord}|>"
```

**Coordinate Mask Creation**:
```python
# Create boolean mask for coordinate token positions
def create_coordinate_mask(input_ids, tokenizer):
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    # Mark positions where coordinate tokens appear
    coord_positions = (input_ids >= 151667) & (input_ids <= 152691)
    mask[coord_positions] = True
    return mask
```

- **Strict Inference Parsing (src_new/inference.py)**:
  - Requires training-format blocks with `<|object_ref_start|>...<|object_ref_end|>` (synonyms `<|obj_ref_*|>` accepted), geometry tokens, and `<|coord_N|>`; otherwise raises.
  - Geometry lengths must be exact: bbox 4, quad 8, line even ≥ 4; overlapping spans are rejected.

### Geometry Token Mapping

```python
GEOMETRY_TOKENS = {
    "bbox_2d": ("<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>"),
    "quad": ("<|object_ref_start|>", "<|object_ref_end|>", "<|quad_start|>", "<|quad_end|>"),
    "line": ("<|object_ref_start|>", "<|object_ref_end|>", "<|line_start|>", "<|line_end|>")
}
# Note: `<|obj_ref_*|>` and `<|object_ref_*|>` are treated synonymously at inference; training conversion uses `<|object_ref_*|>`.
```

### Conversion Process

**Input Object**:
```json
{
  "bbox_2d": [100, 200, 300, 400],
  "desc": "BBU设备"
}
```

**Output Token String**:
```
<|object_ref_start|>BBU设备<|object_ref_end|><|box_start|>[<|coord_100|>, <|coord_200|>, <|coord_300|>, <|coord_400|>]<|box_end|>
```

### Coordinate Clamping

All coordinates are clamped to the valid range: `[0, max_coord_value]` (default: 2048)

```python
clamped_coord = max(0, min(int(coord), max_coord_value))
coord_token = f"<|coord_{clamped_coord}|>"
```

---

## 💬 Conversation Structure & Templates

### Teacher-Student Training System

#### **Teacher Pool Management** (`src_new/data/teacher_pool.py`)

The system uses a teacher pool for providing reference examples during training:

**Teacher Assignment Strategy**:
- **Teacher Ratio**: Configurable percentage of samples that receive teachers (default: 50%)
- **Random Selection**: Teachers randomly selected from pool for diversity
- **Image-Specific**: Optional image-specific teacher assignment for targeted learning
- **Fallback Handling**: Graceful degradation when teacher pool is insufficient

**Teacher Pool Structure**:
```json
{
  "images": ["teacher_image.jpg"],
  "objects": [{"bbox_2d": [x1, y1, x2, y2], "desc": "reference_description"}],
  "width": 532, "height": 728
}
```

#### **Conversation Creation** (`src_new/processing/conversation_processor.py`)

**Teacher-Student Conversation Format**:

The system creates multi-turn conversations with teacher examples followed by student queries:

```
<|im_start|>system
你是通信机房设备检测AI助手，专门识别和定位BBU设备、光纤、电线等通信设备。
请准确检测图像中的设备位置并提供中文描述。
<|im_end|>

<|im_start|>user
这是示例，请检测图像中的设备和部件:<image>
<|im_end|>

<|im_start|>assistant
<|object_ref_start|>BBU设备<|object_ref_end|><|box_start|>[<|coord_100|>, <|coord_200|>, <|coord_300|>, <|coord_400|>]<|box_end|>
<|im_end|>

<|im_start|>user
现在请你回答，请检测图像中的设备和部件:<image>
<|im_end|>

<|im_start|>assistant
<|object_ref_start|>标签贴纸<|object_ref_end|><|box_start|>[<|coord_150|>, <|coord_250|>, <|coord_350|>, <|coord_450|>]<|box_end|>
<|im_end|>
```

### Template Constants

```python
SYSTEM_PROMPT = "你是通信机房设备检测AI助手，专门识别和定位BBU设备、光纤、电线等通信设备。请准确检测图像中的设备位置并提供中文描述。"

TEACHER_USER_PROMPT = "这是示例，请检测图像中的设备和部件:"
STUDENT_USER_PROMPT = "现在请你回答，请检测图像中的设备和部件:"
```

### HuggingFace Integration

All conversation formatting uses the official HuggingFace processor:

```python
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False  # Complete conversations include responses
)
```

---

## 🎯 Span Detection & Loss Masking

### Offset Mapping Approach

The system uses tokenizer's `offset_mapping` for accurate token-level span detection:

```python
# Get full conversation text
full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

# Re-tokenize with offset mapping
tokenized_with_offsets = tokenizer(
    full_text,
    return_offsets_mapping=True,
    add_special_tokens=False,
    return_tensors="pt"
)

offset_mapping = tokenized_with_offsets['offset_mapping'][0]
```

### Span Detection Logic

1. **Text Pattern Matching**: Find assistant content using regex
   ```python
   pattern = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>'
   ```

2. **Character-to-Token Conversion**: Use offset mapping for precise alignment
   ```python
   start_token = char_to_token_position(content_start_char, offset_mapping)
   end_token = char_to_token_position(content_end_char, offset_mapping)
   ```

3. **Teacher/Student Classification**:
   - First assistant response = Teacher (when `has_teachers=True`)
   - Subsequent assistant responses = Students

### Loss Masking Strategy

**Masked Tokens** (set to -100):
- System prompts
- User prompts  
- Image pad tokens
- Assistant prefixes (`<|im_start|>assistant\n`)

**Unmasked Tokens** (original token IDs):
- Assistant response content (coordinate tokens + descriptions)
- End tokens (`<|im_end|>`) — included as the final token of each assistant span so the model learns when to stop

### Span Output Format

```python
teacher_spans = [(start_token, end_token), ...]  # Teacher assistant content (end_token is exclusive)
student_spans = [(start_token, end_token), ...]  # Student assistant content (end_token is exclusive)
```

Notes:
- Spans are extended to include the immediate `<|im_end|>` token if present; i.e., `end_token` points one past `<|im_end|>` so that the labels unmask it and loss is computed on it.

---

## 🏗️ Training Architecture

### Model Wrapper System

The `DetectionModel` wraps the base Qwen2.5-VL model with coordinate token support:

```python
class DetectionModel(nn.Module):
    def __init__(self, base_model, config, coordinate_processor):
        self.base_model = base_model
        self.config = config
        self.coordinate_processor = coordinate_processor
        self.loss_manager = LossManager(config)
```

### Dual-Loss Architecture

The system computes separate losses for different learning objectives:

| Loss Component | Purpose | Computation |
|----------------|---------|-------------|
| `teacher_llm_loss` | Teacher example learning | Cross-entropy on teacher spans |
| `student_llm_loss` | Student response learning | Cross-entropy on student spans |
| `teacher_l1_loss` | Teacher coordinate regression | Soft expectation + L1 loss |
| `student_l1_loss` | Student coordinate regression | Soft expectation + L1 loss |

Note (2025-08): We use joint training with span-aligned masks and next-token shifting.
- CE covers all assistant tokens (text + coordinate tokens) within teacher/student spans for comprehensive language learning.
- L1 covers only coordinate-token targets within assistant spans for precise coordinate regression.
- Soft-expectation temperature is configurable via `coordinate_temperature` (preferred; legacy: `coordinate_loss_temperature`).

### Soft Expectation Coordinate Loss

Instead of cross-entropy, coordinate tokens use soft expectation for better regression:

```python
# Soft expectation formula
expected_coord = Σ(v * softmax(logits_v / temperature))

# L1 loss between expected and target coordinates
l1_loss = |expected_coord - target_coord|
```

---

## 🧠 Loss Computation System

### Dual-Loss Architecture (`src_new/models/loss_manager.py`)

The system implements a sophisticated dual-loss architecture combining language modeling and coordinate regression:

#### **Loss Components**

| Component | Purpose | Computation Method | Weight |
|-----------|---------|-------------------|---------|
| `teacher_llm_loss` | Teacher example learning | Cross-entropy on teacher spans | `teacher_loss_weight` |
| `student_llm_loss` | Student response learning | Cross-entropy on student spans | `student_loss_weight` |
| `teacher_l1_loss` | Teacher coordinate regression | Soft expectation + L1 loss | `coordinate_loss_weight` |
| `student_l1_loss` | Student coordinate regression | Soft expectation + L1 loss | `coordinate_loss_weight` |

#### **Soft Expectation Coordinate Loss**

Instead of cross-entropy, coordinate tokens use soft expectation for better regression:

```python
# Soft expectation formula (src_new/models/coordinate_loss.py)
P(coord_value = v) = softmax(logits_v / temperature)
expected_coord = Σ(v * P(coord_value = v))  # v ∈ [0, MAX_COORD]
coordinate_loss = L1(expected_coord, ground_truth_coord)
```

#### **Optimized Loss Computation**

**Solution 1 Optimization** (60-70% performance improvement):
- Single-pass cross-entropy computation for all tokens
- Reuse per-token results for teacher/student loss calculation
- Maintains mathematical equivalence with significant speedup

#### **Span-Based Loss Masking**

The system uses precise span detection for loss separation:

1. **Teacher Spans**: First assistant response in teacher-student conversations
2. **Student Spans**: Subsequent assistant responses
3. **Masked Tokens**: System prompts, user prompts, image pad tokens (set to -100)
4. **Unmasked Tokens**: Assistant content including coordinate tokens

---

## 🌐 Distributed Training & Checkpoints

### NCCL Timeout Resolution (`src_new/training/bbu_trainer.py`)

**Problem Solved**: 100% failure rate in distributed training due to NCCL timeouts

**Solution**: BBUTrainer with local loss aggregation:
- **Local Loss Computation**: All loss calculations done locally
- **No Custom Distributed Operations**: Eliminates NCCL conflicts
- **TrainingStateManager**: Local metrics aggregation without distributed sync
- **Standard HuggingFace Integration**: Uses built-in distributed mechanisms only

### Checkpoint Optimization

#### **SafeTensors Format** (4-6x faster loading)
```python
# Automatic SafeTensors usage in checkpoint saving
model.save_pretrained(
    checkpoint_dir,
    safe_serialization=True,  # Use SafeTensors format
    max_shard_size="5GB",     # Optimize shard size
)
```

#### **BestCheckpointCallback** (`src_new/training/callbacks.py`)
- **Rotation-Safe**: Creates independent copies that survive checkpoint rotation
- **Descriptive Naming**: `best-{step}-{eval_loss}` format
- **Automatic Updates**: Creates new copy when better model is found
- **Cleanup**: Removes old best copies when updating

#### **Distributed Coordination**
- **Rank 0 Only**: Checkpoint saving only on rank 0 to avoid redundancy
- **Pre-Distributed Expansion**: All tokenizer/model expansion before distributed training
- **Lazy Loading**: Maximum efficiency with deferred initialization

### Model State Synchronization

#### **Pre-Distributed Expansion Pattern**
```python
# All expansion operations BEFORE distributed training (scripts/train_new.py)
def perform_pre_distributed_expansion(base_model, tokenizer, config):
    """Perform ALL tokenizer and model expansion operations BEFORE distributed training"""
    # 1. Extend tokenizer vocabulary
    expanded_tokenizer = token_processor.extend_tokenizer_vocabulary(tokenizer)
    # 2. Extend model embeddings
    expanded_model = token_processor.extend_model_embeddings(base_model, expanded_tokenizer)
    return expanded_tokenizer, expanded_model
```

#### **Checkpoint Components Saved**
- **Model Weights**: SafeTensors format with optimized sharding
- **Extended Tokenizer**: With coordinate tokens (vocab > 151665)
- **Processor Configuration**: Chat templates and image processing settings
- **Coordinate Config**: Token ranges and coordinate system parameters
- **Training Arguments**: Complete training configuration for resumption

---

## ⚙️ Configuration & Setup

### Configuration System (`src_new/config/config.py`)

The configuration system uses a unified dataclass approach with comprehensive validation:

#### **Configuration Loading**
```python
from src_new.config.config import load_config

# Load and validate configuration
config = load_config("configs/bbu_v2.yaml")

# Automatic validation with detailed error messages
# - File existence checks
# - Type validation
# - Range validation
# - Fail-fast error handling
```

#### **Key Configuration Categories**

**Model Settings**:
```yaml
model_path: "/path/to/Qwen2.5-VL-3B-Instruct"
model_size: "3B"
model_max_length: 32000
attn_implementation: "flash_attention_2"
torch_dtype: "bfloat16"
use_cache: false
model_hidden_size: 2048
```

**Training Parameters**:
```yaml
num_train_epochs: 20
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5e-6
vision_lr: 5e-7      # Vision encoder learning rate
merger_lr: 1e-5      # Vision-language merger learning rate
llm_lr: 5e-6         # Language model learning rate
warmup_ratio: 0.1
weight_decay: 0.0001
lr_scheduler_type: "cosine"
gradient_checkpointing: true
bf16: true
```

**Coordinate Token System**:
```yaml
coordinate_tokens_enabled: true
max_coord_value: 1024
coordinate_loss_weight: 0.05
regular_loss_weight: 1.0
# Preferred key (legacy alias supported):
coordinate_temperature: 1.0
```

**Teacher-Student Training**:
```yaml
teacher_ratio: 0.5           # 50% of samples get teachers
num_teacher_samples: 1       # Number of teachers per student
teacher_loss_weight: 0.3     # Weight for teacher loss component
student_loss_weight: 1.0     # Weight for student loss component
```

**Data Processing**:
```yaml
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
teacher_pool_file: "data/teacher_pool.jsonl"
max_total_length: 12000
collator_type: "packed"      # or "standard"
language: "chinese"
max_pixels: 401408           # Image processor pixel limit
```

**Performance Optimization**:
```yaml
use_flash_attention: true
mixed_precision: "bf16"
dataloader_num_workers: 4
pin_memory: true
prefetch_factor: 2
save_safetensors: true       # Use SafeTensors format
```

#### **Training Entry Point** (`scripts/train_new.py`)

The training script orchestrates the complete pipeline:

```python
def main():
    # 1. Load and validate configuration
    config = load_config(f"configs/{args.config}.yaml")

    # 2. Create training arguments with DeepSpeed support
    training_args = create_training_arguments_with_deepspeed(config)

    # 3. Create trainer with new architecture
    trainer = create_trainer_with_new_architecture(training_args, config)

    # 4. Start training with automatic checkpointing
    trainer.train()
```

#### **Setup Process**

1. **Configuration Loading**: Load and validate YAML configuration with fail-fast validation
2. **Model Loading**: Load base Qwen2.5-VL model with optimized loading parameters
3. **Pre-Distributed Expansion**: Extend tokenizer and model embeddings BEFORE distributed training
4. **Dataset Creation**: Initialize HuggingFace-first datasets with teacher pool management
5. **Trainer Initialization**: Configure BBUTrainer with local loss aggregation
6. **Processor Setup**: Configure extended tokenizer and processor for checkpoint saving

#### **Critical Setup Details**

**Tokenizer Extension** (`src_new/processing/token_processor.py`):
```python
# Add coordinate tokens: <|coord_0|> to <|coord_MAX_COORD|>
# Add geometry tokens: <|line_start|>, <|line_end|>
# Initialize embeddings using positional encoding
expanded_tokenizer = token_processor.extend_tokenizer_vocabulary(tokenizer)
expanded_model = token_processor.extend_model_embeddings(base_model, expanded_tokenizer)
```

**Processing Class Setup**:
```python
# CRITICAL: Use tokenizer (not processor) for checkpoint compatibility
trainer.processing_class = tokenizer  # Has get_vocab() method
trainer.set_processor(processor)      # Saved separately for inference
```

**Model Wrapper**:
```python
# DetectionModel wraps base model with coordinate token support
model = DetectionModel(
    base_model=expanded_model,
    config=config,
    tokenizer=expanded_tokenizer,
    skip_expansion=True,  # Already expanded
)
```

---

## 🔧 Troubleshooting

### Critical Issues Resolved

#### **1. NCCL Timeout Failures** ✅ RESOLVED
**Problem**: 100% failure rate in distributed training
**Root Cause**: Custom distributed operations conflicting with HuggingFace Trainer
**Solution**: BBUTrainer with local loss aggregation eliminates all custom distributed operations

#### **2. Checkpoint Saving Errors** ✅ RESOLVED
**Problem**: `AttributeError: 'Qwen2VLProcessor' object has no attribute 'get_vocab'`
**Root Cause**: Setting `processing_class = processor` instead of tokenizer
**Solution**: Use `trainer.processing_class = tokenizer` for checkpoint compatibility

#### **3. Span Detection Misalignment** ✅ RESOLVED
**Problem**: Incorrect teacher/student span boundaries causing loss computation errors
**Root Cause**: Character-based vs token-based position mismatch
**Solution**: Offset mapping for precise token-level alignment in `_create_masked_labels_with_spans`

#### **4. Missing Student Responses** ✅ RESOLVED
**Problem**: Student response tokens = 0, causing NaN losses
**Root Cause**: Incomplete conversation generation in teacher-student mode
**Solution**: Ensure `create_teacher_student_conversation` includes complete student responses

### Inference Alignment & Generation Notes (src_new)

- **Use generation builders**: Build inputs with `ConversationProcessor.create_teacher_student_conversation_for_generation(...)` or `create_simple_conversation_for_generation(...)`. Avoid tokenizer-only re-tokenization after any truncation.
- **Template placeholder check**: Before passing through the HF processor, ensure the number of `<|image_pad|>` placeholders in template text equals the number of images via `validate_image_token_consistency(...)`.
- **Decode without skipping specials**: Use `skip_special_tokens=False` to preserve extended geometry tokens (e.g., `<|coord_xxx|>`).
- **Strict parsing**: Inference strictly parses `<|object_ref_start|>...<|object_ref_end|>` (or `<|obj_ref_*|>`) with geometry tokens and `<|coord_N|>`. Any deviation raises immediately; no fallbacks.
- **Operational tips**: Prefer `attn_implementation="eager"` for inference stability in this environment; keep batch size = 1 for teacher-guided inference.
- **Canonical reference**: See `docs/INFERENCE_ROOT_CAUSE_AND_FIXES.md` for the root causes and full fix details.

### Performance Optimization Guidelines

#### **Model Configuration**
- **Model Size**: 3B model recommended for development, 7B for production
- **Sequence Length**: `max_total_length: 12000` optimal for most conversations
- **Attention**: Use `flash_attention_2` for 5x performance improvement
- **Precision**: `bf16: true` for optimal memory/performance balance

#### **Training Parameters**
- **Batch Size**: Start with `per_device_train_batch_size: 1` for 7B model
- **Gradient Accumulation**: Use `gradient_accumulation_steps: 8` for effective batch size
- **Learning Rates**: Differential LRs (vision: 5e-7, merger: 1e-5, llm: 5e-6)
- **Checkpointing**: `gradient_checkpointing: true` to reduce memory usage

#### **Data Processing**
- **Collator**: Use `collator_type: "packed"` for efficient sequence packing
- **Workers**: `dataloader_num_workers: 4` with `pin_memory: true`
- **Teacher Ratio**: `teacher_ratio: 0.5` for balanced teacher-student training

#### **Checkpoint Optimization**
- **Format**: Always use SafeTensors (`save_safetensors: true`) for 4-6x faster loading
- **Sharding**: `max_shard_size: "5GB"` for optimal loading performance
- **Rotation**: Use BestCheckpointCallback for rotation-safe best model preservation

### Validation Checks

#### **Training Health Indicators**
- **Unmasked Token Ratio**: Should be 30-60% for healthy training
- **Loss Components**: All components (teacher_llm, student_llm, teacher_l1, student_l1) should have finite values
- **Span Coverage**: Both teacher and student spans should be detected in teacher-student mode
- **Coordinate Tokens**: Should be within expected ID range (151667-153715)

#### **Performance Monitoring**
- **Training Speed**: ~2-3 samples/second on A100 for 7B model
- **Memory Usage**: ~24GB VRAM for batch_size=1 with 7B model
- **Convergence**: Expect convergence within 1000-2000 steps for fine-tuning

#### **Data Quality Checks**
- **Empty Objects**: Fail-fast on empty object lists (indicates preprocessing issues)
- **Coordinate Range**: All coordinates should be within [0, max_coord_value]
- **Image Loading**: Verify all image paths are accessible and valid
- **Teacher Pool**: Ensure teacher pool contains sufficient diversity

### Debugging Tools

#### **Debug Logging**
```python
# Enable comprehensive debug logging
python scripts/train_new.py --config bbu_v2 --log_level DEBUG
```

#### **Sample Inspection**
- **Conversation Text**: Full conversation logging (no truncation)
- **Token Analysis**: Coordinate token detection and validation
- **Span Visualization**: Teacher/student span boundaries with offset mapping
- **Loss Breakdown**: Component-wise loss tracking and validation

---

## 🔬 Implementation Details: Teacher/Student Flow, Spans, Masks, Losses

This section captures the exact algorithmic steps and code locations for teacher assignment, span detection, masking, and loss computation.

### 1) Teacher assignment and conversation creation

- Configuration used by dataset:
  - `teacher_ratio`, `num_teacher_samples` read at dataset init
  - Random teacher assignment when `random()<teacher_ratio`
  - Teachers are sampled from `TeacherPoolManager.teacher_pool`

```300:346:/data3/Qwen2.5-VL-main/src_new/data/dataset.py
# ... inside __getitem__ → _process_sample_unified
labels, teacher_spans, student_spans = (
    self._create_masked_labels_with_spans(
        inputs["input_ids"],
        self.tokenizer,
        has_teachers=(len(teacher_samples) > 0),
    )
)
inputs["labels"] = labels
inputs["teacher_assistant_spans"] = teacher_spans
inputs["student_assistant_spans"] = student_spans
```

```116:129:/data3/Qwen2.5-VL-main/src_new/data/dataset.py
self.teacher_ratio = getattr(self.config, "teacher_ratio", 0.5)
self.num_teacher_samples = getattr(self.config, "num_teacher_samples", 1)
```

```257:275:/data3/Qwen2.5-VL-main/src_new/data/dataset.py
if (
    self.teacher_pool_manager
    and random.random() < self.teacher_ratio
    and len(self.teacher_pool_manager.teacher_pool) > 0
):
    num_teachers = min(
        random.randint(1, self.num_teacher_samples),
        len(self.teacher_pool_manager.teacher_pool),
    )
    selected_teachers = random.sample(
        self.teacher_pool_manager.teacher_pool, num_teachers
    )
    structured_sample["teacher_samples"] = selected_teachers
```

- Conversation building:
```312:333:/data3/Qwen2.5-VL-main/src_new/data/dataset.py
inputs = (
    self.conversation_processor.create_teacher_student_conversation(
        student_sample=structured_sample,
        teacher_samples=teacher_samples,
        student_images=student_images,
        teacher_images_list=teacher_images_list,
    )
)
# or, when no teachers
inputs = self.conversation_processor.create_simple_conversation(
    sample=structured_sample, images=images
)
```

- Teacher pool manager (loading, indexing, random sampling) is implemented here:
```100:129:/data3/Qwen2.5-VL-main/src_new/data/teacher_pool.py
self.teacher_pool = self._load_teacher_pool()
self.image_to_teachers = self._build_image_index()
```
```244:274:/data3/Qwen2.5-VL-main/src_new/data/teacher_pool.py
def get_random_teachers(self, num_samples: int = 1) -> List[Dict[str, Any]]:
    # ... random sampling from self.teacher_pool with teacher_id assignment
```

### 2) Span detection and masked labels (offset mapping)

- Exact algorithm in `_create_masked_labels_with_spans`:
  1. Decode full conversation text
  2. Re-tokenize with `return_offsets_mapping=True`
  3. Find assistant content spans via regex and offset mapping
  4. Initialize labels to `-100`, then unmask assistant spans by copying `input_ids`
  5. Mask `<|image_pad|>` tokens to `-100`

```354:409:/data3/Qwen2.5-VL-main/src_new/data/dataset.py
def _create_masked_labels_with_spans(...):
    full_text = tokenizer.decode(input_ids_1d, skip_special_tokens=False)
    tokenized_with_offsets = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    offset_mapping = tokenized_with_offsets["offset_mapping"][0]
    assistant_spans = self._find_assistant_spans_with_offsets(
        full_text, offset_mapping, has_teachers
    )
    labels.fill_(-100)
    for start_token, end_token, is_teacher in assistant_spans:
        if 0 <= start_token < end_token <= len(labels):
            labels[start_token:end_token] = input_ids_1d[start_token:end_token]
            span = (start_token, end_token)
            if is_teacher:
                teacher_spans.append(span)
            else:
                student_spans.append(span)
```

```423:428:/data3/Qwen2.5-VL-main/src_new/data/dataset.py
image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
if image_pad_id is not None:
    image_pad_mask = input_ids_1d == image_pad_id
    labels[image_pad_mask] = -100
```

- Assistant span discovery (regex + offset mapping):
```465:493:/data3/Qwen2.5-VL-main/src_new/data/dataset.py
pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
for match in re.finditer(pattern, full_text, re.DOTALL):
    content_start_char = match.start(1)
    content_end_char = match.end(1)
    start_token = self._char_to_token_position(content_start_char, offset_mapping)
    end_token = self._char_to_token_position(content_end_char, offset_mapping)
    is_teacher = has_teachers and assistant_count == 0
    assistant_spans.append((start_token, end_token, is_teacher))
```

- Trainer cross-checks spans from labels (grouping unmasked regions):
```1542:1560:/data3/Qwen2.5-VL-main/src_new/training/bbu_trainer.py
unmasked_positions = (labels != -100).nonzero(as_tuple=True)[0]
# group into consecutive spans (start, end+1)
```
```1564:1588:/data3/Qwen2.5-VL-main/src_new/training/bbu_trainer.py
has_teachers = chat_text.count("<|im_start|>assistant") > 1
if has_teachers:
    # teacher spans first, last span is student
    teacher_spans = spans[:-1] if len(spans) > 1 else spans
    student_spans = spans[-1:] if len(spans) > 1 else []
else:
    teacher_spans = []
    student_spans = spans
```

### 3) Loss calculation and masks

- Forward pass supplies spans to loss manager:
```596:608:/data3/Qwen2.5-VL-main/src_new/models/wrapper.py
loss_components = self.loss_manager.compute_loss_components(
    logits=base_outputs.logits,
    labels=labels,
    coord_mask=None,
    teacher_spans=teacher_spans,
    student_spans=student_spans,
)
```

- In coordinate mode, a coordinate mask is also passed:
```702:713:/data3/Qwen2.5-VL-main/src_new/models/wrapper.py
coord_mask = self.coordinate_processor.get_coordinate_mask(input_ids)
loss_components = self.loss_manager.compute_loss_components(
    logits=masked_logits,
    labels=labels,
    coord_mask=coord_mask,
    teacher_spans=teacher_spans,
    student_spans=student_spans,
)
```

- Weighted loss components (final sum used for backprop):
```206:268:/data3/Qwen2.5-VL-main/src_new/models/loss_manager.py
# teacher_llm: teacher_loss_weight * regular_loss_weight
# teacher_l1:  teacher_loss_weight * coordinate_loss_weight
# student_llm: student_loss_weight * regular_loss_weight
# student_l1:  student_loss_weight * coordinate_loss_weight
# total_loss = sum of non-None weighted components
```

- Single-pass cross-entropy (+ mask reuse) for teacher/student LLM losses:
```485:509:/data3/Qwen2.5-VL-main/src_new/models/loss_manager.py
per_token_loss = self._compute_per_token_cross_entropy(logits, labels)
if need_teacher_llm:
    teacher_llm_loss = self._apply_mask_to_per_token_loss(per_token_loss, teacher_mask)
if need_student_llm:
    student_llm_loss = self._apply_mask_to_per_token_loss(per_token_loss, student_mask)
```

- Coordinate L1 losses are computed on masked coordinate positions only:
```514:528:/data3/Qwen2.5-VL-main/src_new/models/loss_manager.py
teacher_coord_mask = teacher_mask & coord_mask
student_coord_mask = student_mask & coord_mask
if teacher_coord_mask.any():
    teacher_l1_loss = self._compute_coordinate_loss(logits, labels, teacher_coord_mask)
if student_coord_mask.any():
    student_l1_loss = self._compute_coordinate_loss(logits, labels, student_coord_mask)
```

### 4) Coordinate loss specifics (soft expectation + L1)

- Extract logits for coordinate token vocab range and positions from `coord_mask`:
```188:205:/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py
coord_logits_full = logits[:, :, self.coord_start_id : self.coord_end_id]
coord_positions = torch.where(coord_mask)
coord_logits = coord_logits_full[coord_positions]
```

- Use labels to filter valid positions (ignore -100), then convert token IDs to coordinate values:
```220:246:/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py
target_coord_ids = labels[coord_positions]
valid_mask = target_coord_ids != -100
valid_coord_logits = coord_logits[valid_mask]
valid_target_coord_ids = target_coord_ids[valid_mask]
target_coords = valid_target_coord_ids - self.coord_start_id
# clamp out-of-range and continue
```

- Numerical stability, temperature scaling, soft expectation, and L1:
```130:152:/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py
scaled_logits = torch.clamp(scaled_logits, min=-50.0, max=50.0)
coord_probs = F.softmax(scaled_logits, dim=-1)
coord_probs = coord_probs + 1e-8
coord_probs = coord_probs / coord_probs.sum(dim=-1, keepdim=True)
expected_coords = torch.sum(coord_probs * coord_values, dim=-1)
```
```260:271:/data3/Qwen2.5-VL-main/src_new/models/coordinate_loss.py
expected_coords = self.compute_soft_expectation(valid_coord_logits, temperature)
coordinate_loss = F.l1_loss(expected_coords, target_coords.float())
```

### 5) Shapes, masks, and conventions

- `labels`: `[batch, seq_len]` with `-100` for masked tokens; assistant content unmasked
- `teacher_spans` / `student_spans`: lists of `(start, end)` indices per batch item (end-exclusive)
- `teacher_mask` / `student_mask`: boolean masks built from spans; applied to per-token loss and to intersect with `coord_mask`
- `coord_mask`: boolean `[batch, seq_len]` true where coordinate tokens appear; generated from token IDs

These details reflect the exact implementation used during training/inference. Adjust weights via config keys: `teacher_loss_weight`, `student_loss_weight`, `regular_loss_weight`, `coordinate_loss_weight`, and set `teacher_ratio`, `num_teacher_samples` for teacher sampling.

---

## 🔄 Data Conversion (Preprocessing Canonicalization)

- `data_conversion/unified_processor.py` and `data_conversion/coordinate_manager.py` canonicalize geometry before training output:
  - **Quad**: `_canonical_quad_ordering` produces clockwise order starting from top-left; values are clamped to image bounds and cast to int.
  - **Line**: `_canonical_line_ordering` lexicographically orders 2‑point lines; multi‑point lines preserve path with canonical start; degenerate horizontal/vertical handled with minimal padding.
  - **BBox**: normalized to `x1<x2, y1<y2`.

This ensures model input uses a single deterministic coordinate ordering, eliminating randomness in vertex/direction permutations.

---

*This documentation reflects the production-ready state of the Qwen2.5-VL training pipeline with all critical issues resolved and performance optimizations applied.*
