# Evaluation Pipeline

Clean, simple evaluation scripts with configuration at the top of each file.

## Quick Start

1. **Configure and run inference**:
   ```bash
   # Edit configuration at top of file
   vim eval/infer_dataset.sh
   # Run inference
   ./eval/infer_dataset.sh
   ```

2. **Configure and run evaluation**:
   ```bash
   # Edit configuration at top of file  
   vim eval/run_evaluation.sh
   # Run evaluation
   ./eval/run_evaluation.sh
   ```

3. **Compare experiments**:
   ```bash
   # Edit configuration at top of file
   vim eval/compare_experiments.py
   # Run comparison
   python eval/compare_experiments.py
   ```

## Clean Output Structure

```
experiments/
└── baseline_eval_20241205_143022/
    ├── config.json                    # Auto-generated config
    ├── inference/
    │   ├── train_tokens_1024_no_teacher_model_qwen2_5_vl.json
    │   └── val_tokens_1024_no_teacher_model_qwen2_5_vl.json
    └── evaluation/
        ├── train_tokens_1024_no_teacher_model_qwen2_5_vl_metrics.json
        ├── val_tokens_1024_no_teacher_model_qwen2_5_vl_metrics.json
        └── evaluation_summary.json    # Aggregated results
```

## Configuration

### Inference Script (`eval/infer_dataset.sh`)
Edit these parameters at the top:
- `EXPERIMENT_NAME`: Experiment identifier
- `MODEL_PATH`: Path to model checkpoint
- `NUM_TEACHERS`: Teacher guidance (0=disabled, 1+=enabled)
- `MAX_NEW_TOKENS`: Generation length
- `DATASETS`: List of dataset files to process

### Evaluation Script (`eval/run_evaluation.sh`)
Edit these parameters at the top:
- `EXPERIMENT_NAME`: Must match inference experiment
- `IOU_THRESHOLD`: IoU threshold for evaluation
- `SEMANTIC_THRESHOLD`: Semantic similarity threshold
- `ENABLE_*`: Feature toggles

### Comparison Script (`eval/compare_experiments.py`)
Edit these parameters at the top:
- `EXPERIMENTS_DIR`: Directory containing experiments
- `METRICS_TO_COMPARE`: Metrics to show in comparison
- `GENERATE_PLOTS`: Whether to create visualization plots

## Features

- **Clean output organization** with timestamped experiment directories
- **Consistent file naming** with descriptive suffixes
- **Automatic configuration tracking** in JSON format
- **Comprehensive evaluation summaries** with aggregated metrics
- **Simple comparison tools** for analyzing multiple experiments
- **No CLI complexity** - just edit config at top of files

## Example Workflow

```bash
# 1. Configure inference for baseline experiment
vim eval/infer_dataset.sh
# Set: EXPERIMENT_NAME="baseline_eval", NUM_TEACHERS=0

# 2. Run inference
./eval/infer_dataset.sh

# 3. Configure evaluation to match experiment
vim eval/run_evaluation.sh  
# Set: EXPERIMENT_NAME="baseline_eval"

# 4. Run evaluation
./eval/run_evaluation.sh

# 5. Configure and run teacher-guided experiment
vim eval/infer_dataset.sh
# Set: EXPERIMENT_NAME="teacher_guided_eval", NUM_TEACHERS=3
./eval/infer_dataset.sh
./eval/run_evaluation.sh

# 6. Compare experiments
python eval/compare_experiments.py
```

This produces a clean, organized evaluation pipeline with all results properly tracked and easy to compare.