# Qwen2.5-VL BBU Detection Training Pipeline - Unified Documentation

**Complete Reference for Vision-Language Model Fine-tuning with Coordinate Token System**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [Training Pipeline Flow](#training-pipeline-flow)
4. [Input Data Format](#input-data-format)
5. [Token Conversion Pipeline](#token-conversion-pipeline)
6. [Conversation Structure & Templates](#conversation-structure--templates)
7. [Span Detection & Loss Masking](#span-detection--loss-masking)
8. [Training Architecture](#training-architecture)
9. [Loss Computation System](#loss-computation-system)
10. [Data Augmentation](#data-augmentation)
11. [Distributed Training & Checkpoints](#distributed-training--checkpoints)
12. [Configuration & Setup](#configuration--setup)
13. [Testing](#testing)
14. [Troubleshooting](#troubleshooting)

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

## 🚀 Quick Start

### Training
```bash
cd /data3/Qwen2.5-VL-main
source ~/.bashrc && conda activate ms
bash scripts/run_new_train.sh
```

### Inference
```bash
source ~/.bashrc && conda activate ms
python -m src_new.inference --config /abs/path/to/config.yaml --checkpoint /abs/path/to/checkpoint --image /abs/path/to/image.jpg
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
- Extend vocabulary with geometry + coordinate tokens (derived at runtime; no hard-coded ID ranges)
- Add geometry tokens (`<|line_start|>`, `<|line_end|>`)
- Initialize new token embeddings using positional encoding

#### **Step 4: Apply Templates** (`src_new/processing/conversation/builder.py`)
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
- Compute coordinate losses using auxiliary losses (Kernelized-KL + Unlikelihood)
- Aggregate loss components with configurable weights

#### **Step 8: Log Results** (`src_new/training/training_state_manager.py`)
- Track loss components locally (no distributed operations)
- Generate comprehensive training metrics
- Log performance statistics and remaining time estimates

### Component Interactions

#### **Data Flow Architecture**
```
JSONL Input → Dataset → ConversationBuilder → Collator → DetectionModel → LossManager → BBUTrainer
     ↓            ↓            ↓                    ↓            ↓              ↓            ↓
Raw Objects → Structured → Conversations → Batched → Model → Loss → Training
              Samples      + Images        Tensors   Outputs  Components  Loop
```

#### **Module Dependencies**
- **`Dataset`** → **`ConversationBuilder`**: Creates teacher-student conversations
- **`ConversationBuilder`** → **`CoordinateTokenConverter`**: Converts coordinates to tokens
- **`TokenProcessor`** → **`DetectionModel`**: Extends tokenizer and model embeddings
- **`DetectionModel`** → **`LossManager`**: Computes multi-component losses
- **`BBUTrainer`** → **`TrainingStateManager`**: Aggregates metrics locally
- **`BBUTrainer`** → **`CheckpointSaver`** + **`BestCheckpointManager`**: Inference-ready SafeTensors checkpoints; best checkpoint via direct folder copy and rotation

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
input_ids: [...]
labels:    [...]
teacher_spans: []
student_spans: [(span_start, span_end)]
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
- Tokens are added dynamically at runtime:
  - 2 geometry tokens: `<|line_start|>`, `<|line_end|>`
  - `(max_coord_value + 1)` coordinate tokens: `<|coord_0|>` … `<|coord_{max_coord_value}|>`
- Final vocabulary size = base_vocab_size + 2 + (max_coord_value + 1). IDs are derived from the tokenizer.

#### **Token ID Ranges**

Token IDs are not hard-coded. Coordinate and geometry token IDs are derived from the tokenizer at runtime:

```python
from src_new.processing.special_tokens import get_coord_token_range, validate_geometry_tokens
validate_geometry_tokens(tokenizer)
rng = get_coord_token_range(tokenizer)  # rng.start_id, rng.end_exclusive (0,0 if none present)
```
- Geometry tokens (`<|line_start|>`, `<|line_end|>`) are ensured by vocabulary extension when needed.
- Coordinate tokens are added for bins `0..max_coord_value` during pre-distributed extension; range is re-derived from the tokenizer (no hard-coded IDs).

#### **Token Initialization Strategy**

**Geometry Tokens**: Initialized from existing similar tokens
```python
# Line tokens initialized from quad tokens
line_start_embedding = quad_start_embedding.clone()
line_end_embedding = quad_end_embedding.clone()
```

**Coordinate Tokens**: Configurable deterministic initialization
- `ms_mean`: Neutral mean-resizing with smart init only for new rows; preserves pretrained rows and pads to multiples of 128. Matches `TokenProcessor._smart_initialize_new_embeddings(...)` when `coordinate_init_mode: ms_mean`.
- `fourier_ramp`: Deterministic Fourier ramp over coordinate bins; produces smooth sin/cos features. Use `coordinate_init_mode: fourier_ramp`.

Configuration key: `coordinate_init_mode` in YAML; implementation lives in `src_new/processing/token_processor.py`.

#### **Coordinate Token Processing**

**Coordinate Clamping**:
```python
# All coordinates clamped to valid range [0, max_coord_value]
clamped_coord = max(0, min(int(coord), max_coord_value))
coord_token = f"<|coord_{clamped_coord}|>"
```

**Coordinate Mask Creation**:
```python
# Derive coordinate ID range from tokenizer; avoid hard-coded ranges
from src_new.processing.special_tokens import get_coord_token_range

rng = get_coord_token_range(tokenizer)  # start_id, end_exclusive
coord_positions = (input_ids >= rng.start_id) & (input_ids < rng.end_exclusive)
```

- **Strict Inference Parsing (src_new/inference.py)**:
  - In coordinate-token mode, requires training-format blocks with `<|object_ref_start|>...<|object_ref_end|>` (synonyms `<|obj_ref_*|>` accepted), geometry tokens, and `<|coord_N|>`; otherwise falls back to best-effort parsing or returns empty.
  - Geometry lengths must be exact: bbox 4, quad 8, line even ≥ 4; parsing filters out invalid objects rather than raising globally.

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

All coordinates are clamped to the valid range: `[0, max_coord_value]` (as configured in YAML)

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
- **Teacher Ratio**: Configurable percentage of samples that receive teachers (set explicitly via YAML; no in-code defaults)
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

#### **Conversation Creation** (`src_new/processing/conversation/builder.py`)

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
| `teacher_kce_loss` | Teacher coordinate regression | Kernelized-KL loss |
| `teacher_unlike_loss` | Teacher coordinate regression | Unlikelihood loss |
| `student_kce_loss` | Student coordinate regression | Kernelized-KL loss |
| `student_unlike_loss` | Student coordinate regression | Unlikelihood loss |

Note (2025-08): We use joint training with span-aligned masks and next-token shifting.
- CE covers all assistant tokens (text + coordinate tokens) within teacher/student spans for comprehensive language learning.
- Auxiliary losses cover only coordinate-token targets within assistant spans for precise coordinate regression.

### Auxiliary Coordinate Losses

When enabled, coordinate tokens add auxiliary losses (in addition to CE training) for better regression:

- **Kernelized-KL**: Sparse window around ground truth coordinate bin
- **Unlikelihood**: Penalizes non-coordinate tokens at coordinate positions

---

## 🧠 Loss Computation System

### Dual-Loss Architecture (`src_new/models/loss_manager.py`)

The system implements a sophisticated dual-loss architecture combining language modeling and coordinate regression:

| Component | Purpose | Computation Method | Weight |
|-----------|---------|-------------------|---------|
| `teacher_llm_loss` | Teacher example learning | Cross-entropy on teacher spans | `teacher_loss_weight` |
| `student_llm_loss` | Student response learning | Cross-entropy on student spans | `student_loss_weight` |
| `teacher_kce_loss` | Teacher coordinate regression | Kernelized-KL loss | `coord_aux_lambda_kce` |
| `teacher_unlike_loss` | Teacher coordinate regression | Unlikelihood loss | `coord_aux_lambda_unlike` |
| `student_kce_loss` | Student coordinate regression | Kernelized-KL loss | `coord_aux_lambda_kce` |
| `student_unlike_loss` | Student coordinate regression | Unlikelihood loss | `coord_aux_lambda_unlike` |

### Auxiliary Coordinate Losses (Optional)

When `coord_aux_enabled: true` in YAML, auxiliary losses are computed on coordinate-token targets:

- **Kernelized-KL**: Sparse Gaussian kernel around the correct coordinate bin
- **Unlikelihood**: Top-k penalty on non-coordinate tokens at coordinate positions

If disabled, these components are zero while the CE path remains unchanged.

### Optional Auxiliary Coordinate Losses (Kernelized‑KL + Unlikelihood)

When enabled via YAML, `LossManager` switches the coordinate path at shifted positions whose labels are coordinate tokens and computes separate components:

- Kernelized‑KL (sparse window) around the correct bin (temperature-scaled)
- Unlikelihood on non‑coordinate tokens at coordinate positions (top‑K)

Behavior and wiring:
- CE path is unchanged and continues to train language tokens (including coordinate targets) under span masks.
- Separate components are exposed for logging and total loss summation:
  - `teacher_kce_loss`, `teacher_unlike_loss`, `student_kce_loss`, `student_unlike_loss`
- Legacy L1 has been retired in the standard path; coordinate regression now flows through auxiliary losses.
- Laplacian regularizer on the coordinate embedding slice has been removed.

YAML configuration:
```yaml
coord_aux_enabled: true
coord_aux_tau: 1.2
coord_aux_sigma_bins: 8
coord_aux_window_bins: 32
coord_aux_topk: 100
coord_aux_lambda_kce: 1
coord_aux_lambda_unlike: 1
```

Implementation highlights:
- `src_new/losses/coord_aux.py` provides:
  - `build_kernel_indices_and_q`, `kernelized_kl_sparse`, `unlikelihood_topk_text`
- `src_new/models/loss_manager.py` computes auxiliary losses at shifted coordinate positions (teacher/student separately) and returns separate components for logging and weighting.
- Laplacian code and metrics were removed entirely to simplify the system.

### Optimized Loss Computation

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

## 🧠 Loss Computation System


### Coordinate Diagnostics (New)

To monitor and guide coordinate learning, we log the following diagnostics per group (teacher/student) during auxiliary loss computation:

- `teacher_window_mass`, `student_window_mass`: Sum of probability mass inside the Gaussian window around the GT bin, averaged across positions. Target: should increase over time.
- `teacher_coord_slice_mass`, `student_coord_slice_mass`: Fraction of full-vocabulary mass assigned to the coordinate slice (softmax over coord bins vs full vocab). Helps detect leakage into non-coordinate tokens.
- `teacher_gt_prob`, `student_gt_prob`: Probability on the exact GT bin within the coordinate slice. Target: should increase.
- `teacher_expected_mae_bins`, `student_expected_mae_bins`: Expected absolute error in bin units using the coordinate-slice distribution (distance-aware quality). Target: should decrease.
- `teacher_top1_acc`, `student_top1_acc`: Argmax bin equals GT (within coordinate slice). Target: should increase.
- `teacher_top5_acc`, `student_top5_acc`: GT within top-5 bins (within coordinate slice). Target: should increase.
- `teacher_outside_window_mass`, `student_outside_window_mass`: 1 - window_mass; should decrease as learning concentrates mass.
- `teacher_noncoord_topk_mass`, `student_noncoord_topk_mass`: Sum of top‑K non‑coordinate probabilities at coord positions; should decrease. Correlates with unlikelihood.
- `teacher_window_entropy`, `student_window_entropy`: Entropy within the window (normalized); should decrease as the model sharpens near GT.
- `teacher_margin_top1_top2`, `student_margin_top1_top2`: Mean margin between top‑1 and top‑2 probs on coord slice; should increase.
- `teacher_mean_bin_offset`, `student_mean_bin_offset`: Signed offset of expected bin relative to GT (bias indicator); aim near 0.
- `teacher_coord_pos_count`, `student_coord_pos_count`: Number of coord positions encountered (sanity check for masking/spans).

These metrics are aggregated locally (no distributed ops) by `TrainingStateManager` and appear in the regular logs alongside loss components and learning rates. Use them to adjust:
- Window/σ (locality), τ (sharpness), and λ weights (relative strength) without widening the window excessively.

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
# Centralized SafeTensors usage via CheckpointSaver
from src_new.training.checkpoint_saver import CheckpointSaver, BestCheckpointManager
saver = CheckpointSaver(args=training_args, checkpoint_manager=BestCheckpointManager(metric_name=metric, greater_is_better=flag))
saver.save_checkpoint(model=model, processing_class=tokenizer, processor=processor, step=global_step,
                      current_metrics=current_eval_metrics, is_deepspeed_enabled=is_deepspeed, training_start_time=start_time)
```

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

The configuration uses a unified dataclass with explicit, required fields. Coordinate auxiliary losses are controlled via the following keys:

```yaml
# Coordinate auxiliary losses
coord_aux_enabled: true
coord_aux_tau: 1.2
coord_aux_sigma_bins: 8
coord_aux_window_bins: 32
coord_aux_topk: 100
coord_aux_lambda_kce: 0.5
coord_aux_lambda_unlike: 0.05
```

- Debug config: `configs/bbu_v2/debug.yaml` demonstrates enabling and tuning auxiliary losses.
- Baseline training config: `configs/bbu_v2/standard.yaml` keeps `coord_aux_enabled: false` by default.

### Phase-based Freezing for Separate Runs (New)

We removed the legacy progressive-unfreeze callback and epoch-knob settings. Instead, use explicit phases per run:

- `phase_name: off | phase_1 | phase_2 | phase_3`
- Implemented by `src_new/training/phase_freeze_manager.py` and applied in `scripts/train_new.py` before optimizer creation.

Behavior per phase:
- `phase_1`: Train `visual.merger` and, when coordinate tokens exist, apply coord-slice grad masks to the coordinate-token rows of `embed_tokens.weight` and `lm_head.weight`. Freeze all LLM decoder layers and the vision backbone.
- `phase_2`: Phase 1 + unfreeze last-K LLM decoder layers. If not provided, the manager uses an internal default of `K=6` that works well for both 3B and 7B. Vision backbone remains frozen; `visual.patch_embed` remains frozen.
- `phase_3`: Unfreeze all parameters by default. Keep `visual.patch_embed` frozen by default for stability, and optionally restrict to the last-K vision blocks if configured. The aligner (`visual.merger`) stays trainable in all phases.

Run pattern:
- Run Phase 1 → save checkpoint → Phase 2 resume → save checkpoint → Phase 3 resume.
- Each phase uses its own cosine schedule.

Minimal YAML examples:
```yaml
# base.yaml
phase_name: off

# standard.yaml (phase 1)
phase_name: phase_1

# coord_aux.yaml (phase 1)
phase_name: phase_1
```

Internal per‑phase defaults and semantics (PhaseFreezeManager):
- Components (Qwen2.5‑VL):
  - Vision backbone: `model.visual.patch_embed`, `model.visual.blocks.*`
  - Aligner (patch‑merger MLP): `model.visual.merger` (always trainable)
  - Language model: `model.language_model.embed_tokens`, `model.language_model.layers.*`, `lm_head`
- Keys (resolved internally when only `phase_name` is provided):
  - `top_k_layers`: number of last LLM decoder blocks to unfreeze (0 keeps all LLM blocks frozen)
  - `vision_top_k_blocks` (phase_3 only): unfreeze only the last K vision blocks (0 disables restriction)
  - `coord_slice_only`: enable coord‑slice grad masks on embeddings/LM head in phase_1/2 when coord tokens exist
  - `freeze_patch_embed` (phase_3): keep `visual.patch_embed` frozen for stability
- Default values:
  - phase_1 → `top_k_layers: 0`, `vision_top_k_blocks: 0`, `coord_slice_only: true`, `freeze_patch_embed: true`
  - phase_2 → `top_k_layers: 6`, `vision_top_k_blocks: 0`, `coord_slice_only: true`, `freeze_patch_embed: true`
  - phase_3 → `top_k_layers: 0`, `vision_top_k_blocks: 0`, `coord_slice_only: true`, `freeze_patch_embed: true`

Entry point applies the phase:
```python
# scripts/train_new.py
from src_new.training.phase_freeze_manager import PhaseFreezeManager
pfm = PhaseFreezeManager()
phase = config.phase_name  # off|phase_1|phase_2|phase_3
if phase != "off":
    pfm.apply_phase(model, tokenizer, phase=phase)  # uses internal per-phase defaults
```

Optimizer learning‑rate groups (aligned with phases)
- The trainer groups parameters by component and uses your YAML LR keys:
  - `vision` group (vision backbone) → `vision_lr`
  - `merger` group (aligner `visual.merger`) → `merger_lr`
  - `top_layers` group (when only a suffix subset of LLM layers is trainable) → `lr_top_layers` (fallback to `llm_lr`)
  - `coord_slice` group (`embed_tokens.weight`, `lm_head.weight` when trainable) → `lr_coord_slice` (fallback to `llm_lr`)
  - `llm` group (remaining LLM params) → `llm_lr` (or `lr_full_model` if explicitly set)
- Grouping derives from `requires_grad`, which PhaseFreezeManager sets per phase. LLM layer detection supports Qwen2.5‑VL naming (`model.language_model.layers.<idx>.*`).

Optional per-group LRs (validated by `config.py`):
```yaml
lr_merger: 3.0e-5
lr_coord_slice: 5.0e-6
lr_top_layers: 1.0e-5
lr_full_model: 1.0e-5
```

---

## 🧪 Testing

```bash
cd /data3/Qwen2.5-VL-main/src_new/tests
python run_comprehensive_tests.py
```

Augmentation visualization (optional):
```bash
python /data3/Qwen2.5-VL-main/src_new/tests/vis_aug_angle_rotate_real.py
```

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

- **Use generation builders**: Build inputs with `ConversationBuilder.create_teacher_student_conversation_for_generation(...)` or `create_simple_conversation_for_generation(...)`. Avoid tokenizer-only re-tokenization after any truncation.
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
// Rotation is handled by the unified checkpoint saver integrated in the training loop.

### Validation Checks

#### **Training Health Indicators**
- **Unmasked Token Ratio**: Should be 30-60% for healthy training
- **Loss Components**: All components (teacher_llm, student_llm, teacher_l1, student_l1) should have finite values
- **Span Coverage**: Both teacher and student spans should be detected in teacher-student mode
- **Coordinate Tokens**: Verify presence via runtime‑derived range from `get_coord_token_range(tokenizer)` (no hard‑coded ranges)

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

### 4) Coordinate loss specifics (auxiliary losses)

- Extract coordinate logits and build sparse kernel windows around ground truth:
```python
coord_logits_full = shifted_logits[..., coord_start:coord_end_exclusive]
# Build Gaussian kernel around ground truth coordinate bin
idxs, q_vals = build_kernel_indices_and_q(y=y, K=K, sigma=sigma_bins, window=window_bins)
```

- Compute Kernelized-KL loss with temperature scaling:
```python
kce = kernelized_kl_sparse(coord_logits=coord_logits, idxs=idxs, q_vals=q_vals, tau=tau)
```

- Compute Unlikelihood loss on non-coordinate tokens:
```python
unlike = unlikelihood_topk_text(logits_all=shifted_logits, coord_mask=group_mask,
                               noncoord_vocab_mask=noncoord_mask, topk=topk)
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

## 📦 Module-by-Module Overview (Refactored `src_new`)

- **`config/`**: Unified, frozen dataclass config with strict validation and path normalization.
  - `config/config.py`: `Config` schema, `load_config` (auto-resolve dataset paths via `DataResolver`), fail-fast checks, `phase_name` control and LR-group validation.
- **`data/`**: Data loading and HuggingFace-first conversation assembly.
  - `data/dataset.py`: Reads JSONL, validates samples, loads images via `utils.path_manager`, builds conversations via `processing/conversation/builder.py`, creates masked labels and teacher/student spans using offset mapping, masks `<|image_pad|>`.
  - `data/teacher_pool.py`: Loads teacher pool JSONL, builds image index, random sampling APIs.
  - `data/collator_{packed,standard}.py`: Batching strategies; choose via `config.collator_type`.
- **`processing/`**: Prompting, token/embedding extension, and coordinate conversion.
  - `processing/conversation_processor.py`: HuggingFace-first builder; validates structure, interleaves images, applies chat templates, enforces image token consistency, offers robust/multi-teacher/truncation/inference builders.
  - `processing/conversation/builder.py`: Thin wrapper providing the public ConversationBuilder API used by `data/dataset.py` and `inference.py`.
  - `processing/coordinate_converter.py`: Converts objects to geometry + `<|coord_N|>` strings with clamping; only custom logic preserved from legacy.
  - `processing/token_processor.py`: Extends tokenizer and model embeddings; adds geometry tokens if missing; adds coord tokens `0..max_coord_value`; initialization mode: `ms_mean` (neutral mean-resize + smart init) or `fourier_ramp` (deterministic Fourier features); pads embeddings to multiples of 128; validates alignment.
  - `processing/special_tokens.py`: Canonical tokens, assistant span pattern, and `get_coord_token_range(tokenizer)`; no hard-coded ID ranges.
  - `processing/templates.py`: Centralized Chinese prompts and constants.
- **`models/`**: Model wrapper and loss system.
  - `models/wrapper.py`: `DetectionModel` composition around Qwen2.5‑VL; validates image tensors and image token counts vs grids; SOLUTION‑1 single‑pass CE path; coordinates via `LossManager`; exposes `model.config` for HF integrations; saves tokenizer/coordinate config.
  - `models/loss_manager.py`: Teacher/student CE masks from spans; optional auxiliary coordinate losses (Kernelized‑KL + Unlikelihood) with diagnostics; strict decomposition and weighting.
  - `models/coord_metrics.py`: Metrics like window mass, gt prob, expected MAE (bins), top‑k acc, non‑coord top‑k mass, etc.
  - `models/patches.py`: Qwen2.5‑VL safety/compatibility patches.
- **`losses/`**: Auxiliary loss implementations.
  - `losses/coord_aux.py`: `build_kernel_indices_and_q`, `kernelized_kl_sparse`, `unlikelihood_topk_text`, `build_noncoord_vocab_mask`.
- **`training/`**: Trainer and checkpointing.
  - `training/phase_freeze_manager.py`: Phase-based freezing for separate runs.
  - `training/bbu_trainer.py`: Local aggregation; proper LR logging; eval/save cadence.
  - `training/training_state_manager.py`: Local metrics aggregation.
  - `training/checkpoint_saver.py`: SafeTensors + best-checkpoint manager.
- **`inference.py`**: Inference engine.
- **`utils/`**: Rank-aware logging, path/data resolution, validation, debug logging.

## 🧭 Refactored Roadmap & Training Procedure

1) **Configuration**
   - Write a YAML with paths (`data_root`, `train_data_path`, `val_data_path`, `teacher_pool_file`, `output_dir`, `tb_dir`). Relative paths and `@src_new/` alias are accepted; they will be normalized relative to the config file.
   - Load with `/root/miniconda3/envs/ms/bin/python -m src_new.config.config` utilities (`load_config`), which validates strictly.
2) **Model & Tokenizer (pre‑distributed expansion)**
   - Load base model/tokenizer; extend vocabulary and embeddings with `processing.token_processor.TokenProcessor`.
   - Geometry tokens are ensured; coord tokens added for `0..max_coord_value` with `coordinate_init_mode: {ms_mean|fourier_ramp}`; embeddings padded to 128‑multiple, original rows preserved.
3) **Dataset & Conversations**
   - Initialize `data.Dataset` with tokenizer and config; set HF `Qwen2VLProcessor` via `dataset.set_processor(processor)`.
   - Build conversations with `processing.conversation.ConversationBuilder` (robust HF templates, strict image token checks).
   - Create labels with offset‑mapping spans; include `<|im_end|>` in assistant spans; mask `<|image_pad|>`.
4) **Collation**
   - Choose `collator_packed` or `collator_standard` via `config.collator_type`; ensure shapes/types match Qwen2.5‑VL.
5) **Forward & Losses**
   - `models/wrapper.DetectionModel` validates multimodal tensors and image token counts; bypasses HF loss when spans provided.
   - `models/loss_manager.LossManager` computes single‑pass CE for teacher/student; optionally adds coordinate aux losses with diagnostics.
6) **Training Loop**
   - Use `training/BBUTrainer` with local loss aggregation (`TrainingStateManager`); clean logging only; no custom distributed ops.
- Checkpoints saved with SafeTensors via `CheckpointSaver`; best checkpoints managed by `BestCheckpointManager` with rotation.
7) **Inference**
   - Use `inference.InferenceEngine` or builders `ConversationBuilder.*for_generation`; strict parsing; teacher‑guided mode supported; dynamic coord range.

## 🔁 Key Differences vs `src_new_bak/`
- **HuggingFace‑first conversations** replace custom chat builders; image and token tensors come from the official processor.
- **Strict config & fail‑fast** validation; normalized paths accepted; progressive unfreeze as explicit knobs.
- **Dynamic coordinate IDs** via `get_coord_token_range(tokenizer)`; no hard‑coded ranges.
- **Loss system**: single‑pass CE; optional auxiliary coordinate losses with diagnostics; legacy L1 path retired; Laplacian regularizers removed.
- **Trainer**: no custom distributed ops; local aggregation only; unified best‑checkpoint handling.
- **Embedding extension**: padded to 128; original pretrained rows preserved; deterministic coordinate init (`ms_mean`/`fourier_ramp`).

## 🎛️ Data Augmentation

Integrated, training‑ready data augmentation for detection‑focused VL fine‑tuning. The module runs inside `src_new/data/dataset.py` before conversation assembly/tokenization, keeping downstream tokenization and span logic unchanged.

### Quick start (apply in 60 seconds)

Add under your training YAML:
```yaml
augmentation:
  preset: moderate   # off|conservative|moderate|aggressive
  rng_seed: 12345
```
Then launch training (ensure `ms` conda env):
```bash
cd /data3/Qwen2.5-VL-main
source ~/.bashrc && conda activate ms
bash scripts/run_new_train.sh
```

### Preset‑based configuration

- off: augmentation disabled (rotation fixed at 0°)
- conservative: light photometric + small geometry
- moderate: balanced default for most runs
- aggressive: strong perturbations, higher CPU cost

Advanced users can still provide the explicit block below for fine‑grained control. Teachers are not augmented by default (`apply_to_teachers: false`).

### Integration & execution order

1) Load images
2) Image‑level AngleRotate (image + coordinates)
3) Albumentations RandAug (image‑only photometric)
4) Per‑object Local Affine (pixels + coordinates) if `apply_pixels: true`
5) Object Copy‑Paste (with feathered alpha) if enabled
6) Object Blur (masked blur) if enabled
7) Optional RandAug pool over object‑level ops
8) Conversation building and tokenization

Implementation: `src_new/augmentation/base.py::AugmentationPipeline`, called from `src_new/data/dataset.py`.

### Human‑readable object transform wrappers

Switch behaviors via a single key:
- `off`: no object‑level movement
- `cut_paste`: move objects (erase source and paste at new location) — uses `object_local_affine` with `apply_pixels: true`
- `copy_paste`: duplicate objects (keep source, add a new object) — uses `object_copy_paste`
- `both`: enable both behaviors (order: cut/move → copy/duplicate)

Examples:
```yaml
# Cut‑paste (move objects, no duplicates)
augmentation:
  object_transform: cut_paste
  object_local_affine:
    enabled: true
    apply_pixels: true
    per_object_prob: 0.8
    max_rotation_deg: 10
    translate_px: 8
    avoid_overlap: true
    iou_thresh: 0.05
    max_resample: 10
```

```yaml
# Copy‑paste (duplicate objects)
augmentation:
  object_transform: copy_paste
  object_copy_paste:
    enabled: true
    per_object_prob: 0.6
    num_copies_per_object: 1
    translate_px: 48
    rotation_jitter_deg: 8.0
    scale_jitter_min: 0.95
    scale_jitter_max: 1.05
    occ_grid_downscale: 8
    occ_margin_px: 6
    max_occ_fraction: 0.15
    max_iou_with_existing: 0.25
    attempts: 20
    alpha_feather_px: 1.5
```

### Full configuration (YAML)

```yaml
augmentation:
  enabled: true
  preset: moderate           # off|conservative|moderate|aggressive (optional; overrides detailed knobs)
  rng_seed: 12345
  apply_to_teachers: false
  lines_policy: transform    # identity|drop_objects|error|transform
  debug_visualization: false
  debug_output_dir: null

  # Image-level rotation (backbone)
  op:
    sample_mode: uniform_range   # fixed|uniform_range|set
    angle_min_deg: -30
    angle_max_deg: 30
    angles_set_deg: null
    expand: true
    interpolation: bilinear      # nearest|bilinear|bicubic
    fill_color: [0, 0, 0]

  # Photometric RandAug (image-only)
  albumentations_rand:
    enabled: true
    apply_prob: 0.8
    num_ops: 2
    magnitude: 0.6               # 0..1, coarse strength scale
    safe_ops_only: true

  # Per-object local affine (v1)
  object_local_affine:
    enabled: true
    apply_pixels: true           # REQUIRED for v1; no coords-only jitter
    per_object_prob: 0.8
    max_rotation_deg: 10.0
    translate_px: 8
    avoid_overlap: true
    iou_thresh: 0.05
    max_resample: 10

  # Object copy-paste (optional)
  object_copy_paste:
    enabled: true
    per_object_prob: 0.6
    num_copies_per_object: 1
    translate_px: 48
    rotation_jitter_deg: 8.0
    scale_jitter_min: 0.95
    scale_jitter_max: 1.05
    occ_grid_downscale: 8
    occ_margin_px: 6
    max_occ_fraction: 0.15
    max_iou_with_existing: 0.25
    attempts: 20
    allowed_types: null
    alpha_feather_px: 1.5        # soften pasted edges to avoid halos

  # Object blur (optional)
  object_blur:
    enabled: true
    per_object_prob: 0.5
    blur_type: gaussian          # gaussian|box
    radius_min: 0.5
    radius_max: 1.2

  # Object-level RandAug pool (optional)
  rand_pool:
    enabled: true
    apply_prob: 1.0
    num_ops: 2
    include_object_affine: true
    include_object_copy_paste: true
    include_object_blur: true
```

### Canonicalization & guarantees

- Quad ordering: top‑left → clockwise
- Bounds: every vertex `0 ≤ x < width`, `0 ≤ y < height`
- Non‑degenerate: polygon area must be positive (invalid candidates rejected)
- Pixel–coord consistency: per‑object affine runs only when `apply_pixels: true`; otherwise skipped entirely (no coordinates‑only jitter)
- Lines: receive image‑level rotation; skipped in v1 for object‑level transforms

### Debugging & visualization

- Real‑data visualization script: `src_new/tests/vis_aug_angle_rotate_real.py`
  - Outputs in `outputs/aug_vis/aug_vis_*.jpg`

### Dependencies

- Albumentations (tested: 1.3.x)
- `albucore<0.1` and `opencv-python-headless~4.10` for 1.3.x

---
