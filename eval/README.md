# Qwen2.5-VL Evaluation System

## Overview

The modular evaluation system provides comprehensive assessment of fine-tuned Qwen2.5-VL models with clean separation of concerns. Each component has a single responsibility and can be used independently.

## Architecture

### Modular Pipeline Design
```
eval/
├── infer_dataset.py     # Pure inference (no evaluation logic)
├── metrics.py           # Response parsing & metrics calculation  
├── eval_dataset.py      # Pipeline orchestration
└── eval_utils.py        # Evaluation-specific utilities
```

### Clean Separation from Training
- **`src/`** - Training components only (BBUTrainer, Config, datasets)
- **`eval/`** - Evaluation components only (inference, parsing, metrics)
- **No cross-dependencies** - each module is self-contained

## Usage

### 1. Pure Inference (Raw Responses)
```bash
python eval/infer_dataset.py \
  --model_path output/checkpoint-XXX \
  --validation_jsonl 521_qwen_val.jsonl \
  --output_file raw_responses.json \
  --max_new_tokens 8192
```

**Output**: JSON file with raw model responses and metadata

### 2. Metrics Calculation (Parse & Evaluate)
```bash
python eval/metrics.py \
  --responses_file raw_responses.json \
  --output_file evaluation_results.json \
  --iou_threshold 0.5
```

**Output**: Evaluation metrics with detailed per-sample results

### 3. Complete Pipeline (Inference → Metrics)
```bash
python eval/eval_dataset.py \
  --model_path output/checkpoint-XXX \
  --validation_jsonl 521_qwen_val.jsonl \
  --output_dir eval_results \
  --iou_threshold 0.5
```

**Output**: Complete evaluation with organized directory structure

### 4. Skip Inference (Use Existing Responses)
```bash
python eval/eval_dataset.py \
  --skip_inference \
  --responses_file existing_responses.json \
  --validation_jsonl 521_qwen_val.jsonl \
  --output_dir eval_results
```

## Component Details

### `infer_dataset.py` - Pure Inference Engine
- **Purpose**: Generate raw model responses only
- **No evaluation logic**: Just saves responses with metadata
- **Reusable**: Can be used for any inference task
- **Output format**:
```json
{
  "sample_id": 0,
  "image_path": "ds_rescaled/IMG.jpeg",
  "system_prompt": "You are a helpful assistant",
  "user_prompt": "<image>\nPlease first output bbox coordinates...",
  "ground_truth": "[{\"bbox_2d\":[x1,y1,x2,y2],\"description\":\"...\"}]",
  "prediction": "[{\"bbox_2d\":[x1,y1,x2,y2],\"description\":\"...\"}]",
  "metadata": {"input_tokens": 1024, "output_tokens": 512},
  "timestamp": "2025-01-26T12:00:00"
}
```

### `metrics.py` - Response Parsing & Metrics
- **Purpose**: Parse responses and calculate evaluation metrics
- **Features**:
  - Robust JSON parsing with multiple fallback strategies
  - Handles truncated responses and malformed JSON
  - Post-processing to remove duplicates and invalid objects
  - IoU-based object detection metrics
- **Metrics calculated**:
  - Precision, Recall, F1 Score
  - Overall and per-sample averages
  - Object counts and match statistics

### `eval_dataset.py` - Pipeline Orchestration
- **Purpose**: Coordinate the complete evaluation process
- **Features**:
  - Subprocess-based module execution
  - Error handling and progress reporting
  - Unified output directory structure
  - Support for skipping inference step

### `eval_utils.py` - Evaluation Utilities
- **Purpose**: Shared utilities for evaluation tasks
- **Components**:
  - `EvaluationLogger`: Simple logging for evaluation
  - `SimpleModelLoader`: Model loading with eager attention
  - `SimpleDataPreprocessor`: Image preprocessing for inference
  - `SimpleOutputManager`: Output file management

## Command Line Options

### Common Options
| Option               | Description                               | Default     |
| -------------------- | ----------------------------------------- | ----------- |
| `--model_path`       | Path to fine-tuned model directory        | Required    |
| `--validation_jsonl` | Path to validation JSONL file             | Required    |
| `--device`           | Device to load model on                   | `auto`      |
| `--max_new_tokens`   | Maximum tokens to generate                | `8192`      |
| `--max_samples`      | Maximum samples to evaluate (for testing) | All samples |
| `--iou_threshold`    | IoU threshold for detection matching      | `0.5`       |

### Pipeline-Specific Options
| Option             | Module          | Description                  |
| ------------------ | --------------- | ---------------------------- |
| `--output_file`    | `infer_dataset` | Path for raw responses JSON  |
| `--responses_file` | `metrics`       | Path to raw responses JSON   |
| `--output_dir`     | `eval_dataset`  | Directory for all outputs    |
| `--skip_inference` | `eval_dataset`  | Skip inference, use existing |

## Output Structure

### Complete Pipeline Output
```
eval_results/
├── raw_responses_20250126_120000.json     # Raw inference responses
├── evaluation_results_20250126_120000.json # Detailed evaluation metrics
└── logs/                                   # Future: detailed logs
```

### Response File Format
Each raw response contains:
- Sample metadata (ID, image path, timestamps)
- Input prompts (system, user)
- Model prediction (raw text)
- Ground truth (from validation data)
- Generation metadata (token counts, model settings)

### Metrics File Format
Evaluation results include:
- Overall metrics (precision, recall, F1)
- Per-sample detailed results
- Dataset statistics (object counts, matches)
- Evaluation configuration (IoU threshold, model path)

## Benefits of Modular Design

### 1. Single Responsibility
- Each module has one clear purpose
- Easy to understand and maintain
- Focused testing and debugging

### 2. Reusable Components
- Inference engine can be used for other tasks
- Metrics calculation works with any response format
- Utilities can be shared across projects

### 3. Better Debugging
- Issues isolated to specific pipeline stages
- Can test each component independently
- Clear error boundaries and reporting

### 4. Flexible Workflows
- Run inference once, calculate metrics multiple times
- Compare different models using same responses
- Skip expensive inference when iterating on metrics

### 5. Maintainable Architecture
- Training and evaluation evolve independently
- Clean imports and dependencies
- Easy to add new evaluation modes

## Integration Examples

### Batch Evaluation
```bash
# Evaluate multiple checkpoints
for checkpoint in output/checkpoint-*; do
  echo "Evaluating $checkpoint"
  python eval/eval_dataset.py \
    --model_path "$checkpoint" \
    --validation_jsonl 521_qwen_val.jsonl \
    --output_dir "eval_$(basename $checkpoint)"
done
```

### Development Workflow
```bash
# 1. Generate responses once (expensive)
python eval/infer_dataset.py \
  --model_path output/checkpoint-XXX \
  --validation_jsonl 521_qwen_val.jsonl \
  --output_file responses.json

# 2. Iterate on metrics (fast)
python eval/metrics.py --responses_file responses.json --output_file results_iou05.json --iou_threshold 0.5
python eval/metrics.py --responses_file responses.json --output_file results_iou07.json --iou_threshold 0.7
```

### Model Comparison
```bash
# Compare different models on same data
python eval/infer_dataset.py --model_path model_A --validation_jsonl data.jsonl --output_file responses_A.json
python eval/infer_dataset.py --model_path model_B --validation_jsonl data.jsonl --output_file responses_B.json

python eval/metrics.py --responses_file responses_A.json --output_file metrics_A.json
python eval/metrics.py --responses_file responses_B.json --output_file metrics_B.json
```

## Troubleshooting

### Common Issues
| Issue             | Module          | Solution                               |
| ----------------- | --------------- | -------------------------------------- |
| Empty predictions | `infer_dataset` | Check model training, adjust prompts   |
| Parse errors      | `metrics`       | Check response format, improve parsing |
| Memory errors     | `infer_dataset` | Reduce batch size, use smaller model   |
| Missing files     | `eval_dataset`  | Check file paths and permissions       |

### Debug Commands
```bash
# Test single sample
python eval/infer_dataset.py --max_samples 1 --model_path output/checkpoint-XXX --validation_jsonl 521_qwen_val.jsonl --output_file test.json

# Validate response format
python -c "import json; print(json.load(open('responses.json'))[0].keys())"

# Check metrics calculation
python eval/metrics.py --responses_file test.json --output_file test_metrics.json
```

This modular architecture provides a clean, maintainable, and flexible evaluation system that scales from development to production use cases. 