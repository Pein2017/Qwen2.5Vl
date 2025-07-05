# 🧩 Prompt & Training Overview — Qwen-BBU-VL (Updated)

*Last updated: 2025-01-18 – updated with current teacher-student implementation and data schema*

---
## 0. TL;DR
• Fine-tune **Qwen-2.5-VL-3B** for dense object detection + captioning on BBU equipment images  
• **Teacher-Student Learning**: 70% samples use teacher demonstrations, 30% single-shot  
• **Current Schema**: Explicit `teachers`/`student` structure in JSONL  
• **Prompts**: Managed in `src/prompt.py` with Chinese primary, English fallback  
• **Teacher Pool**: Curated diverse demonstrations via `TeacherPoolManager`  
• **Packed Collator**: Zero-padding with Flash Attention 2 for efficiency  
• **Loss Splitting**: Automatic teacher/student loss separation for analysis

---
## 1. Current Data & Conversation Flow

### 1.1 Data Schema (Teacher-Student)
```jsonc
{
  "teachers": [
    {
      "images": ["ds_rescaled/teacher1.jpeg"],
      "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}]
    }
  ],
  "student": {
    "images": ["ds_rescaled/student.jpeg"],
    "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"}]
  }
}
```

### 1.2 Conversation Templates

**Teacher-Student Format (70% of training)**:
```
System: <detection_system_prompt>
User: <teacher_image_1> + <detection_query>
Assistant: <teacher_response_1>              # → Teacher LM Loss
User: <teacher_image_2> + <detection_query>  
Assistant: <teacher_response_2>              # → Teacher LM Loss
User: <student_image> + <detection_query>
Assistant: <student_response>                # → Student LM Loss + Detection Loss
```

**Single-Shot Format (30% training + 100% validation)**:
```
System: <detection_system_prompt>
User: <student_image> + <detection_query>
Assistant: <student_response>                # → Student LM Loss + Detection Loss
```

**Key Points**:
- Both teacher and student assistant tokens contribute to LM loss
- Detection loss only applied to final (student) image
- Conversation spans automatically extracted for loss splitting
- Training ratio controlled by `teacher_ratio` config parameter

---
## 2. Prompt System (src/prompt.py)

### Current Prompt Structure:
```python
def get_detection_system_prompt(language="chinese", use_training_prompt=True):
    if language == "chinese":
        return """你是专业的BBU基站设备检测专家。请仔细分析图像中的设备状态，
        识别螺丝连接点、线缆、标签贴纸等关键部件，并描述它们的安装状态和连接情况。
        
        对于每个检测到的目标，请提供：
        1. 精确的位置描述
        2. 部件类型和状态
        3. 是否存在问题或异常"""
    else:  # English fallback
        return """You are a professional BBU base station equipment inspector..."""
```

### Prompt Categories:
- **Training Prompts**: Detailed instructions for learning phase
- **Evaluation Prompts**: Concise instructions for inference
- **Language Support**: Chinese (primary), English (fallback)
- **Task Specificity**: Customized for BBU equipment detection

### Dynamic Prompt Selection:
- `use_training_prompt: true/false` switches between detailed/concise versions
- `language: "chinese"/"english"` controls primary language
- Automatic fallback to English if Chinese prompts unavailable

---
## 3. Teacher Pool Management (Enhanced)

### Teacher Pool Creation Algorithm:
```python
# From create_teacher_pool.py - Selection Criteria:
class TeacherPoolSelector:
    def select_diverse_teachers(self, samples):
        # 1. Semantic Coverage - every label represented
        # 2. Scene Complexity - sparse/medium/dense object counts
        # 3. Spatial Distribution - 3×3 grid coverage of bbox centers  
        # 4. Object Size Diversity - small/medium/large objects
        # 5. Image Quality - prefer clear, well-lit samples
```

### Runtime Teacher Sampling:
```python
# From src/data.py - TeacherPoolManager
class TeacherPoolManager:
    def sample_teachers(self, student_sample, num_teachers=1):
        # Smart sampling based on student image characteristics
        # Preferred: similar scene complexity, complementary labels
        # Fallback: random selection from pool
```

### Configuration:
- **Pool Size**: Automatically determined by dataset diversity
- **Sampling Strategy**: Semantic similarity + random diversity
- **Teacher Count**: `num_teacher_samples` (typically 1-2)
- **Training Ratio**: `teacher_ratio: 0.7` (70% use teachers)

---
## 4. Training Schedule & Configuration

### Current Training Parameters:
```yaml
# Key training settings from configs/base_flat.yaml
teacher_ratio: 0.7              # 70% teacher-student, 30% single-shot
collator_type: "packed"         # Zero-padding efficiency
use_training_prompt: true       # Detailed vs concise prompts
num_teacher_samples: 1          # Teachers per sample

# Learning rates (differential)
vision_lr: 5e-7                # Vision encoder
merger_lr: 1e-5                # Vision-language merger  
llm_lr: 5e-6                   # Language model
detection_lr: 1e-5             # Detection head
adapter_lr: 5e-3               # Adapter modules

# Detection configuration  
detection_enabled: false        # Can be toggled
detection_freeze_epochs: 0      # Staged training delay
```

### Training Phases:
| Phase | Epochs | Active Components | Strategy |
|-------|--------|------------------|----------|
| **Warm-up** | 0-1 | Vision + Merger + Adapters | Stabilize cross-modal connections |
| **Joint** | 1-N | All components active | Full multi-task learning |
| **Fine-tune** | Last 25% | Detection-focused | Reduce LLM LR, maintain detection |

---
## 5. Token Packing & Efficiency

### Packed Collator Benefits:
```python
# Memory efficiency comparison:
# Standard: (4, 2048) = 8192 tokens (with padding waste)
# Packed:   (1, 6134) = 6134 tokens (actual content only)
# Savings:  ~25% memory reduction typical
```

### Implementation Details:
- **Concatenation**: All sequences → single row `(1, ΣL_i)`
- **Attention Masking**: Flash Attention 2 compatible masks
- **RoPE Handling**: 3D positional embeddings via `get_rope_index_25()`
- **Span Adjustment**: Teacher-student spans adjusted for concatenation offsets

### Configuration Requirements:
- `collator_type: "packed"` 
- `per_device_train_batch_size: 1` (typically)
- `gradient_accumulation_steps: ≥2` (to maintain effective batch size)

---
## 6. Loss Splitting & Analysis

### Automatic Span Extraction:
```python
# From src/chat_processor.py
def _extract_assistant_token_spans(self, conversation, formatted_text):
    teacher_spans = []  # [(start_idx, end_idx), ...]
    student_spans = []  # [(start_idx, end_idx), ...]
    
    # Extract spans from conversation structure
    # All assistant responses except last = teachers
    # Last assistant response = student
```

### Loss Computation:
```python
# From src/training/trainer.py - _compute_teacher_student_losses()
def _compute_teacher_student_losses(self, logits, labels, inputs):
    teacher_loss = compute_span_loss(logits, labels, teacher_spans)
    student_loss = compute_span_loss(logits, labels, student_spans)
    
    # Both contribute to total LM loss for backprop
    total_lm_loss = teacher_loss + student_loss
    return teacher_loss, student_loss
```

### Monitoring Benefits:
- **Ablation Analysis**: Quantify teacher contribution to learning
- **Training Diagnostics**: Separate learning curves for teacher/student
- **Loss Balancing**: Adjust `teacher_ratio` based on loss trends
- **Research Insights**: Teacher-student effectiveness analysis

---
## 7. Validation & Inference Modes

### Validation Strategy:
- **Format**: Always single-shot (no teachers)
- **Prompt**: Concise evaluation prompts (`use_training_prompt: false`)
- **Metrics**: Detection mAP + caption quality metrics
- **Purpose**: Match deployment/inference conditions

### Inference Configuration:
```python
# Inference always uses single-shot format
inference_sample = {
    "student": {
        "images": ["test_image.jpeg"],
        "objects": []  # Empty for inference
    }
    # No "teachers" field
}
```

### Production Deployment:
- **Teacher Augmentation**: Optionally prepend teacher examples for rare cases
- **Prompt Selection**: Concise prompts for efficiency
- **Batch Processing**: Standard collator for consistent latencies

---
## 8. Current Implementation Status

### Fully Implemented:
✅ **Teacher-Student Data Schema**: Complete with validation  
✅ **Conversation Templates**: Automatic span extraction  
✅ **Loss Splitting**: Separate teacher/student loss tracking  
✅ **Packed Collator**: Zero-padding efficiency  
✅ **Teacher Pool**: Curated demonstration selection  
✅ **Prompt System**: Multi-language, configurable complexity

### Configuration Integration:
✅ **YAML Parameters**: All settings in `configs/base_flat.yaml`  
✅ **DirectConfig**: No hardcoded defaults, fail-fast validation  
✅ **Runtime Controls**: Dynamic teacher ratio, prompt selection

---
## 9. File Reference Map

| Component | Implementation File | Configuration |
|-----------|-------------------|---------------|
| **Prompt Templates** | `src/prompt.py` | `language`, `use_training_prompt` |
| **Teacher Pool** | `src/data.py` → `TeacherPoolManager` | `teacher_pool_file`, `num_teacher_samples` |
| **Conversation Building** | `src/chat_processor.py` | `teacher_ratio`, `max_examples` |
| **Loss Splitting** | `src/training/trainer.py` | Automatic span extraction |
| **Data Collation** | `src/data.py` → Collators | `collator_type` |
| **Teacher Selection** | `scripts/create_teacher_pool.py` | Generated offline |

---
## 10. Best Practices

### Training Tips:
1. **Start Conservative**: `teacher_ratio: 0.7`, `num_teacher_samples: 1`
2. **Monitor Split Losses**: Watch teacher vs student loss trends
3. **Packed Collator**: Use for efficiency, but validate spans carefully  
4. **Staged Training**: Consider `detection_freeze_epochs > 0` for stability
5. **Prompt Consistency**: Keep training/eval prompt styles consistent

### Debugging Guidelines:
1. **Span Validation**: Log first few samples to verify teacher/student spans
2. **Loss Balance**: Ensure neither teacher nor student loss dominates
3. **Memory Monitoring**: Watch for packed collator memory spikes
4. **Teacher Quality**: Validate teacher pool diversity and quality

---
**The training system now provides comprehensive teacher-student learning with full observability and control.** 🚀 