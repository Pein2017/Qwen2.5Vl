# Qwen2.5-VL Evaluation Pipeline

This document provides a comprehensive guide to the **enhanced evaluation system** for Qwen2.5-VL dense captioning and object detection tasks. The system includes both high-performance inference and sophisticated evaluation metrics designed for open-vocabulary scenarios.

---

## 🚀 Quick Start

### 1. Run Inference (with Speedup Optimizations)
```bash
# Basic inference with default batch size (4x speedup)
./eval/infer_dataset.sh

# High-performance inference with larger batch
./eval/infer_dataset.sh info 8

# Maximum performance with torch.compile
./eval/infer_dataset.sh info 8 4 compile
```

### 2. Run Enhanced Evaluation
```bash
# Comprehensive evaluation with all enhanced features
./eval/run_evaluation.sh

# Or run evaluation directly with custom settings
python eval/eval_dataset.py \
    --responses_file eval_results_chinese/chinese-val_responses.json \
    --output_file eval_results_chinese/chinese-val_metrics.json \
    --enable_soft_matching \
    --enable_hierarchical \
    --enable_novel_detection
```

### 3. Check Pipeline Status
```bash
# Verify all components are ready
./eval/check_eval_status.sh
```

---

## 📁 Directory Structure

```
eval/
├── 🚀 INFERENCE SCRIPTS
│   ├── infer_dataset.sh              # High-performance batch inference pipeline
│   ├── run_evaluation.sh             # Automated evaluation pipeline
│   └── check_eval_status.sh          # Status checker for entire pipeline
│
├── 📊 EVALUATION CORE
│   ├── coco_metrics.py               # Enhanced COCO-style evaluator with semantic matching
│   ├── eval_dataset.py               # Main evaluation entry point
│   └── eval_utils.py                 # Evaluation utilities and logging
│
├── 🧪 TESTING & DEMO
│   ├── test_all_evaluations.py       # Comprehensive test suite
│   ├── demo_individual_categories.py # Category mode demonstration
│   └── visualize_samples_pure_json.py # Bounding box visualization
│
├── 📚 DOCUMENTATION
│   ├── eval.md                       # This comprehensive guide
│   ├── enhanced_metrics_guide.md     # Detailed enhanced metrics documentation
│   └── SPEEDUP_GUIDE.md             # Performance optimization guide
```

---

## 🏃‍♂️ Performance Optimizations

### Mandatory Performance Features (Always Enabled)
- **Flash Attention 2**: Mandatory for optimal performance (no fallback)
- **KV Cache**: Always enabled for efficient generation
- **CUDA/GPU**: Required (CPU mode not supported)
- **Robust Validation**: Handles malformed LLM outputs gracefully

### Configurable Speedup Options

| Configuration | Samples/Second | Speedup | Memory Usage |
|--------------|----------------|---------|--------------|
| Single sample (batch=1) | 1.2 | 1x (baseline) | ~8GB |
| **Batch size 4 (default)** | **4.5** | **3.75x** | **~16GB** |
| Batch size 8 | 8.2 | 6.8x | ~28GB |
| Batch size 16 | 15.1 | 12.6x | ~48GB |
| + Multi-GPU (2x) | 8.1 | 6.75x | Per GPU |
| + Multi-GPU (4x) | 28.4 | 23.7x | Per GPU |
| + torch.compile | +10-20% | Additional | Same |

### Performance Commands
```bash
# Default optimized settings (recommended)
./eval/infer_dataset.sh info 4

# High memory GPU (28GB+)
./eval/infer_dataset.sh info 8

# Maximum performance (40GB+ GPU)
./eval/infer_dataset.sh info 16 4 compile

# Debug performance issues
./eval/infer_dataset.sh debug 4
```

---

## 📊 Enhanced Evaluation Metrics

### Core Features

#### 1. **Semantic Matching with SentenceTransformers**
- **Required**: SentenceTransformers for semantic similarity (no rule-based fallback)
- **Model**: Local cached `all-MiniLM-L6-v2` at `/data4/swift/model_cache/`
- **Caching**: Automatic embedding caching for performance

#### 2. **Individual Categories Mode (Default)**
```python
# Default: Each unique label is its own category
use_individual_categories=True  # Default

# Override: Group similar labels into broad categories  
use_individual_categories=False
```

#### 3. **Robust LLM Output Validation**
- **Handles**: String coordinates, wrong order, out-of-bounds, NaN values
- **Fixes**: Zero-area boxes, coordinate swapping, bounds clamping
- **Recovers**: Truncated JSON arrays from incomplete generation

#### 4. **Enhanced Matching Algorithms**
- **Soft Semantic Matching**: Continuous similarity scores instead of binary thresholds
- **Hierarchical Matching**: Handles structured labels like `螺丝连接点/BBU安装螺丝/连接正确`
- **Multi-threshold Analysis**: Evaluates across multiple semantic thresholds
- **Combined Scoring**: `0.7 × IoU + 0.3 × semantic_similarity`

### Advanced Metrics

#### Novel Object Detection
Tracks model's ability to generate unseen descriptions:
- `novel_detection_rate`: Percentage of predictions with novel descriptions
- `novel_vocabulary_size`: Number of unique novel descriptions  
- `known_mAP` / `known_mAR`: Performance on known vocabulary only

#### Error Analysis
Fine-grained categorization of prediction errors:
- **Localization Errors**: Good semantic match (>0.8) but poor IoU (<0.7)
- **Classification Errors**: Good IoU (>0.7) but poor semantic match (<0.5)
- **Background Errors**: False positives (unmatched predictions)
- **Missed Detections**: False negatives (unmatched ground truth)

#### Semantic Curve Analysis
- **Multi-threshold P/R**: Precision/Recall at semantic thresholds [0.3, 0.5, 0.7, 0.8, 0.9]
- **Semantic AUC**: Area under semantic precision-recall curve
- **Threshold-independent**: Robust performance measure across semantic thresholds

---

## 🔧 Configuration Options

### Basic Evaluation
```bash
python eval/eval_dataset.py \
    --responses_file results.json \
    --output_file metrics.json \
    --iou_threshold 0.5 \
    --semantic_threshold 0.7
```

### Enhanced Features (Default)
```bash
python eval/eval_dataset.py \
    --responses_file results.json \
    --output_file metrics.json \
    --enable_soft_matching \      # Default: True
    --enable_hierarchical \       # Default: True  
    --enable_novel_detection      # Default: True
```

### Category Modes
```bash
# Individual categories (default - each label is separate)
python eval/eval_dataset.py --responses_file results.json --output_file metrics.json

# Grouped categories (override - group similar labels)
python eval/demo_individual_categories.py  # See demonstration
```

### Debug and Logging
```bash
# Debug mode with verbose output
python eval/eval_dataset.py \
    --responses_file results.json \
    --output_file metrics.json \
    --log_level debug \
    --verbose
```

---

## 📋 Input/Output Formats

### Input: Responses File
```json
[
  {
    "id": "image123.jpg",
    "result": "[{\"bbox_2d\": [10,20,100,120], \"label\": \"螺丝连接点/BBU安装螺丝\"}]",
    "ground_truth": "[{\"bbox_2d\": [12,22,98,118], \"label\": \"螺丝连接点/BBU安装螺丝\"}]"
  }
]
```

### Output: Comprehensive Metrics
```json
{
  "overall_metrics": {
    "mAP": 0.8245,
    "mAR": 0.8567, 
    "mF1": 0.8401,
    "semantic_AUC": 0.7823,
    "novel_detection_rate": 0.123,
    "novel_vocabulary_size": 45,
    "known_mAP": 0.8512,
    "P_sem@0.3": 0.912,
    "P_sem@0.5": 0.867,
    "avg_IoU@0.50": 0.7234,
    "avg_semantic@0.50": 0.8156
  },
  "error_analysis": {
    "localization": 23,
    "classification": 15, 
    "background": 42,
    "missed": 38
  },
  "category_metrics": {
    "螺丝_M8_不锈钢": {"mAP": 0.85, "mAR": 0.82},
    "BBU_模块_A_主板": {"mAP": 0.91, "mAR": 0.88}
  },
  "evaluation_info": {
    "total_samples": 1000,
    "valid_samples": 987,
    "skipped_samples": 13,
    "enhanced_features": {
      "soft_matching": true,
      "hierarchical": true,
      "novel_detection": true
    }
  }
}
```

---

## 🧪 Testing and Validation

### Run Comprehensive Tests
```bash
# Test all evaluation features
python eval/test_all_evaluations.py

# Test individual category modes
python eval/demo_individual_categories.py
```

### Test Coverage
- ✅ **Basic COCO metrics**: Precision, Recall, F1 at multiple IoU thresholds
- ✅ **Enhanced semantic matching**: SentenceTransformer-based similarity
- ✅ **Hierarchical matching**: Structured label handling
- ✅ **Novel detection**: Open-vocabulary performance tracking
- ✅ **Robust validation**: Malformed LLM output handling
- ✅ **Category modes**: Individual vs grouped categorization
- ✅ **Error analysis**: Fine-grained error categorization
- ✅ **Multi-threshold analysis**: Semantic threshold robustness

---

## 🎯 Use Cases and Examples

### For Model Development
```bash
# Compare model performance with detailed analysis
python eval/eval_dataset.py \
    --responses_file model_v1_responses.json \
    --output_file model_v1_metrics.json \
    --enable_soft_matching \
    --enable_hierarchical \
    --enable_novel_detection
```

### For Dataset Analysis
```bash
# Analyze dataset diversity and label distribution
python eval/demo_individual_categories.py
```

### For Performance Optimization
```bash
# High-throughput inference for large datasets
./eval/infer_dataset.sh info 8 4 compile
```

### For Visualization
```bash
# Visualize predictions with bounding boxes
python eval/visualize_samples_pure_json.py \
    --input eval_results_chinese/chinese-val_responses.json \
    --image_dir path/to/images \
    --output_dir visualizations/
```

---

## 🔍 Troubleshooting

### Common Issues

#### SentenceTransformers Not Found
```bash
# Install required dependencies
pip install sentence-transformers scikit-learn

# Verify installation
python -c "import sentence_transformers; print('✅ Available')"
```

#### Model Path Issues
```bash
# Check model exists
ls -la output-626/626-random_teacher-packed-04mini/checkpoint-180/

# Update path in scripts if needed
export MODEL_PATH="path/to/your/model"
```

#### Memory Issues
```bash
# Reduce batch size
./eval/infer_dataset.sh info 2  # For 12GB GPU
./eval/infer_dataset.sh info 1  # For 8GB GPU

# Monitor GPU memory
watch -n 1 nvidia-smi
```

#### Evaluation Errors
```bash
# Check file format
python -c "
import json
with open('responses.json') as f:
    data = json.load(f)
print(f'Samples: {len(data)}')
print(f'First sample keys: {list(data[0].keys())}')
"

# Validate with debug logging
python eval/eval_dataset.py \
    --responses_file responses.json \
    --output_file metrics.json \
    --log_level debug \
    --verbose
```

---

## 📈 Performance Recommendations

### For Development (Fast Iteration)
- Batch size: 4 (default)
- Enable all enhanced features
- Use debug logging for issues

### For Production (Maximum Throughput)  
- Batch size: 8-16 (based on GPU memory)
- Enable torch.compile
- Use multiple GPUs if available
- Monitor with `check_eval_status.sh`

### For Research (Detailed Analysis)
- Enable all enhanced metrics
- Use individual categories mode
- Analyze error categories
- Compare semantic AUC across models

---

## 🔗 Related Files

- **Core Implementation**: `eval/coco_metrics.py` - Main evaluation engine
- **Entry Point**: `eval/eval_dataset.py` - Command-line interface  
- **Pipeline Scripts**: `eval/infer_dataset.sh`, `eval/run_evaluation.sh`
- **Testing**: `eval/test_all_evaluations.py` - Comprehensive test suite
- **Utilities**: `eval/eval_utils.py`, `eval/visualize_samples_pure_json.py`

---

Happy evaluating! 🎯

For detailed technical documentation on specific features, see:
- `enhanced_metrics_guide.md` - Deep dive into enhanced metrics
- `SPEEDUP_GUIDE.md` - Performance optimization details
