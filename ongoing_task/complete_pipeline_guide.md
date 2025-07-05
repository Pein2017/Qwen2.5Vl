# 🚀 Complete Qwen2.5-VL Training & Inference Pipeline Guide

*Last updated: 2025-01-18 – Comprehensive end-to-end pipeline with resolved inference issues*

---

## 📋 Pipeline Overview

This guide covers the complete end-to-end pipeline for Qwen-BBU-VL, from data preparation through training, inference, and evaluation.

```
Data Preparation → Training → Inference → Evaluation
       ↓             ↓          ↓          ↓
   Clean JSONL → BBUTrainer → Teacher-    → COCO
   Teacher Pool              Guided      Metrics
                            Inference
```

**Key Features:**
- **Unified Inference Pipeline**: Standard and teacher-guided modes using identical ChatProcessor
- **COCO-Style Evaluation**: mAP, AP@0.5, per-category analysis  
- **Resolved Issues**: Chat template alignment, batch processing fixes
- **Teacher Guidance**: Implemented and tested teacher sample guidance

---

## 1. Data Preparation

### 1.1 Data Structure
```bash
# Expected structure:
data/
├── chinese-train.jsonl     # Teacher-student format
├── chinese-val.jsonl       # Validation set
└── teacher_pool.jsonl      # Curated teacher demonstrations

ds_rescaled/                # Preprocessed images
├── image_001.jpeg
└── ...
```

### 1.2 Data Conversion
```bash
# Full conversion pipeline
bash data_conversion/convert_dataset.sh \
    --input_folder raw_annotations/ \
    --image_folder images/ \
    --output_root data/
```

### 1.3 Data Validation
```bash
# Validate converted data
python data_conversion/simple_validate.py \
    --jsonl_path data/chinese-train.jsonl \
    --image_root ds_rescaled/
```

---

## 2. Training Configuration & Execution

### 2.1 Configuration Setup
```yaml
# configs/my_experiment.yaml

# --- Experiment identification ---
run_name: "my_experiment_001"
output_dir: "output_exp001"

# --- Data paths ---
train_data_path: "data/chinese-train.jsonl"
val_data_path: "data/chinese-val.jsonl"
teacher_pool_file: "data/teacher_pool.jsonl"

# --- Model settings ---
model_path: "/path/to/Qwen2.5-VL-3B-Instruct"
model_max_length: 12000

# --- Training parameters ---
num_train_epochs: 20
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# --- Learning rates (differential) ---
vision_lr: 1e-6      # Vision encoder
merger_lr: 5e-6      # Vision-language merger
llm_lr: 2e-6         # Language model
detection_lr: 1e-4   # Detection head
adapter_lr: 1e-3     # Adapter modules

# --- Teacher-student configuration ---
teacher_ratio: 0.7             # 70% use teachers
num_teacher_samples: 1         # Teachers per sample
collator_type: "packed"        # For efficiency

# --- Detection head ---
detection_enabled: true
detection_num_queries: 100
detection_bbox_weight: 10.0
detection_caption_weight: 1.0

# --- Performance optimizations ---
bf16: true
gradient_checkpointing: true
use_flash_attention: true
```

### 2.2 Training Execution
```bash
# Single GPU training
python scripts/train.py \
    --config my_experiment \
    --log_level INFO \
    --log_verbose true

# Multi-GPU with DeepSpeed ZeRO-2
export BBU_DEEPSPEED_ENABLED=true
export BBU_DEEPSPEED_CONFIG=scripts/zero2.json

python scripts/train.py \
    --config my_experiment \
    --log_level INFO \
    --log_verbose true
```

### 2.3 Training Monitoring
```bash
# TensorBoard monitoring
tensorboard --logdir output_exp001/my_experiment_001/tb --port 6006

# Key metrics to monitor:
# - loss: Overall training loss
# - lm_loss: Language modeling component
# - teacher_lm_loss: Teacher demonstration loss
# - student_lm_loss: Student learning loss
# - bbox_l1_loss: Bounding box regression
# - objectness_loss: Object presence prediction
# - caption_loss: Object captioning
```

---

## 3. Inference Pipeline (✅ Issues Resolved)

### 3.1 Critical Issue Resolved
**Problem**: Standard inference was NOT using the same prompt structure as training.
- ❌ Standard inference used hardcoded system prompt
- ❌ Training used ChatProcessor with candidate phrases and specific formatting
- ✅ **Solution**: Modified `InferenceEngine` to ALWAYS use `ChatProcessor` for both modes

### 3.2 Unified Inference Modes

#### Standard Inference (now uses ChatProcessor)
```bash
python src/inference.py \
  --model_path output-626/626-random_teacher-packed-04mini/checkpoint-180 \
  --input_jsonl data/chinese-val.jsonl \
  --output_file infer_result/standard.json \
  --data_root ./ \
  --batch_size 1
```

#### Teacher-Guided Inference
```bash
python src/inference.py \
  --model_path output-626/626-random_teacher-packed-04mini/checkpoint-180 \
  --input_jsonl data/chinese-val.jsonl \
  --output_file infer_result/teacher_guided.json \
  --data_root ./ \
  --batch_size 1 \
  --teacher_pool_file data/teacher_pool.jsonl \
  --num_teachers 2
```

#### Unified Script
```bash
# Use eval/infer_dataset.sh for both modes
# Standard mode: NUM_TEACHERS=0
# Teacher-guided: NUM_TEACHERS=2
./eval/infer_dataset.sh
```

### 3.3 Key Implementation Details

#### ChatProcessor Integration (Fixed)
```python
# Always create ChatProcessor for consistent prompts
self.chat_processor = ChatProcessor(
    tokenizer=self.processor.tokenizer,
    image_processor=self.processor.image_processor,
    use_training_prompts=True,
    language=self.language,
)
```

#### Batch Processing Fix
```python
# Process each sample individually due to processor issues
responses = []
for i, (text, images) in enumerate(zip(texts, images_list)):
    response = self.generate_response(
        images=images,
        text=text,
        max_new_tokens=max_new_tokens,
        # ... other params
    )
    responses.append(response)
```

---

## 4. Evaluation Pipeline

### 4.1 COCO-Style Evaluation
```bash
# Basic evaluation
python eval/eval_dataset.py \
    --responses_file eval_results_chinese/chinese-train_responses.json \
    --output_file eval_results_chinese/chinese-train_metrics.json \
    --iou_threshold 0.5 \
    --semantic_threshold 0.7

# Batch evaluation
./eval/run_evaluation.sh
```

### 4.2 Evaluation Metrics

| Metric | Description | IoU Range |
|--------|-------------|-----------|
| `mAP` | Mean Average Precision | 0.5:0.05:0.95 |
| `AP@0.5` | Average Precision at IoU=0.5 | 0.5 |
| `AP@0.75` | Average Precision at IoU=0.75 | 0.75 |
| `mAR` | Mean Average Recall | 0.5:0.05:0.95 |

### 4.3 Per-Category Analysis
Automatic detection of equipment categories:
- **BBU**: Huawei BBU, ZTE BBU, Ericsson BBU
- **Cable**: Fiber cable, non-fiber cable
- **Connection**: CPRI connection, ODF connection
- **Screw**: Install screw, floor screw
- **Shield**: BBU shield
- **Cabinet**: Cabinet
- **Label**: Label matches
- **Grounding**: Grounding

### 4.4 Output Format
```json
{
  "evaluation_info": {
    "responses_file": "path/to/responses.json",
    "total_samples": 100,
    "total_gt_objects": 250,
    "total_pred_objects": 240,
    "iou_thresholds": [0.5, 0.55, ..., 0.95],
    "semantic_threshold": 0.7
  },
  "overall_metrics": {
    "mAP": 0.8245,
    "AP@0.5": 0.9123,
    "AP@0.75": 0.7854,
    "mAR": 0.8567
  },
  "category_metrics": {
    "bbu": {"mAP": 0.8901, "AP@0.5": 0.9456, ...},
    "cable": {"mAP": 0.7234, "AP@0.5": 0.8234, ...}
  }
}
```

---

## 5. Performance Comparison Results

### 5.1 Standard vs Teacher-Guided Inference

**Ground Truth (4 objects)**:
- 机柜空间: [6, 4, 416, 835]
- 挡风板: [0, 28, 399, 510]  
- 螺丝连接点/BBU安装螺丝: [325, 249, 401, 327]
- 螺丝连接点/BBU安装螺丝: [43, 585, 106, 638]

**Standard Inference (6 objects - over-detection)**:
- Detected extra BBU unit (华为) not in ground truth
- Detected more screw connection points
- Shows tendency to over-detect objects

**Teacher-Guided Inference (4 objects - same count as GT)**:
- Detected BBU unit (中兴) instead of 机柜空间
- More conservative in number of detections
- Shows influence from teacher examples in object type predictions

### 5.2 Key Insights

1. **Prompt Consistency is Critical**: The mismatch between training and inference prompts was a major contributor to poor performance.

2. **Teacher Guidance Trade-offs**: 
   - ✅ Helps constrain the number of predictions (reduces over-detection)
   - ⚠️ May bias predictions towards object types seen in teacher examples
   - 🔧 Requires careful selection of teacher examples

---

## 6. Complete Workflow Example

### Step 1: Prepare Data
```bash
bash data_conversion/convert_dataset.sh \
    --input_folder raw_annotations/ \
    --image_folder images/ \
    --output_root data/
```

### Step 2: Train Model
```bash
python scripts/train.py \
    --config my_experiment \
    --log_level INFO \
    --log_verbose true
```

### Step 3: Run Inference
```bash
# Standard inference
python src/inference.py \
  --model_path output_exp001/my_experiment_001/checkpoint-best \
  --input_jsonl data/chinese-val.jsonl \
  --output_file infer_result/standard.json \
  --data_root ./

# Teacher-guided inference
python src/inference.py \
  --model_path output_exp001/my_experiment_001/checkpoint-best \
  --input_jsonl data/chinese-val.jsonl \
  --output_file infer_result/teacher_guided.json \
  --data_root ./ \
  --teacher_pool_file data/teacher_pool.jsonl \
  --num_teachers 2
```

### Step 4: Evaluate Results
```bash
# Evaluate both modes
python eval/eval_dataset.py \
    --responses_file infer_result/standard.json \
    --output_file eval_results/standard_metrics.json

python eval/eval_dataset.py \
    --responses_file infer_result/teacher_guided.json \
    --output_file eval_results/teacher_guided_metrics.json
```

---

## 7. Troubleshooting & Best Practices

### 7.1 Common Issues & Solutions

**Issue**: Loss not decreasing
```bash
# Solution: Reduce learning rates by 10x
# Edit config: llm_lr: 2e-7, detection_lr: 1e-5
```

**Issue**: GPU OOM
```bash
# Solution: Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

**Issue**: Poor inference results
```bash
# Ensure ChatProcessor consistency (already fixed)
# Try teacher-guided inference with 2-3 teachers
```

### 7.2 Best Practices

#### ✅ Do:
- Start with small learning rates and conservative settings
- Monitor all loss components, not just total loss
- Use packed collator for efficiency with variable-length data
- Always use ChatProcessor for inference (already implemented)
- Validate configuration before starting long training runs
- Compare standard and teacher-guided inference results

#### ❌ Don't:
- Set learning rates too high initially
- Ignore teacher vs student loss balance
- Skip data validation steps
- Train without monitoring
- Use different prompt structures for training vs inference
- Mix different data formats without validation

---

## 8. Future Enhancements

### 8.1 Planned Features
- **Confidence-based metrics**: Precision-recall curves
- **Dynamic teacher selection**: Based on image similarity
- **Batch processing optimization**: Fix Qwen2.5-VL batch issues
- **Interactive visualization**: Detection result browser

### 8.2 Research Directions
- **Teacher pool optimization**: Which examples lead to better performance
- **Prompt engineering**: Fine-tune system prompts for accuracy
- **Multi-teacher strategies**: Optimal number and selection of teachers

---

**This comprehensive pipeline provides a complete, battle-tested approach to training and deploying Qwen-BBU-VL models with resolved inference issues and comprehensive evaluation capabilities.** 🚀 