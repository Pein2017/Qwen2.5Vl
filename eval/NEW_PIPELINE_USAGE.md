# Clean Evaluation Pipeline Usage Guide

## New Directory Structure

The refactored pipeline creates a clean, organized structure:

```
experiments/
├── 1_teacher/                    # Experiment name (no timestamps!)
│   ├── config.json              # Experiment configuration
│   ├── train/                   # Training dataset results
│   │   ├── inference/
│   │   │   ├── predictions.json # Inference results
│   │   │   └── inference.log   # Inference logs
│   │   └── evaluation/
│   │       ├── metrics.json    # Evaluation metrics
│   │       └── evaluation.log  # Evaluation logs
│   └── val/                     # Validation dataset results
│       ├── inference/
│       │   ├── predictions.json
│       │   └── inference.log
│       └── evaluation/
│           ├── metrics.json
│           └── evaluation.log
└── no_teacher/                  # Another experiment
    ├── config.json
    ├── train/
    └── val/
```

## Key Improvements

1. **No timestamps**: Clean experiment names only
2. **Dataset separation**: Train/val results are clearly separated  
3. **Organized structure**: Each dataset has its own inference/evaluation folders
4. **Single source of truth**: One unique result per experiment+dataset combination
5. **Easy comparison**: Direct access to results without searching for "latest"

## Usage

### Method 1: Individual Scripts

#### 1. Run Inference
```bash
# Configure the script first
EXP_NAME="1_teacher" DATASET="val" bash eval/infer_dataset_new.sh

# Or for training dataset
EXP_NAME="1_teacher" DATASET="train" bash eval/infer_dataset_new.sh
```

#### 2. Run Evaluation 
```bash
# After inference completes
EXP_NAME="1_teacher" DATASET="val" bash eval/run_evaluation_new.sh

# Or for training dataset  
EXP_NAME="1_teacher" DATASET="train" bash eval/run_evaluation_new.sh
```

### Method 2: Complete Pipeline (Recommended)

```bash
# Run inference + evaluation for specific dataset
bash eval/run_experiment.sh 1_teacher val
bash eval/run_experiment.sh no_teacher train

# Run for all datasets
bash eval/run_experiment.sh 1_teacher all
```

### Method 3: Compare Results

```bash
# Compare all experiments
python eval/compare_experiments_new.py

# Compare specific experiments
python eval/compare_experiments_new.py --experiments 1_teacher no_teacher

# Compare only validation results
python eval/compare_experiments_new.py --dataset val

# Show detailed category breakdown
python eval/compare_experiments_new.py --detailed
```

## Configuration

### Inference Configuration
Edit the top section of `eval/infer_dataset_new.sh`:

```bash
# Experiment name (set manually)
EXP_NAME="1_teacher"           # Your experiment name

# Dataset to process (single dataset per run)  
DATASET="val"                  # "train" or "val"

# Teacher configuration
NUM_TEACHERS=1                 # 0 for no teacher
TEACHER_POOL_FILE="data/teacher.jsonl"

# Model configuration
MODEL_PATH="output-74/checkpoint-280"
MODEL_NAME="qwen2_5_vl"

# Generation parameters
MAX_NEW_TOKENS=2048
BATCH_SIZE=64
```

### Evaluation Configuration
Edit the top section of `eval/run_evaluation_new.sh`:

```bash
# Experiment name (must match inference)
EXP_NAME="1_teacher"          

# Dataset to evaluate  
DATASET="val"                 # "train" or "val"

# Evaluation parameters
IOU_THRESHOLD=0.3
SEMANTIC_THRESHOLD=0.7
ENABLE_SOFT_MATCHING=true
```

## Migration from Old Pipeline

### 1. Backup existing results (optional)
```bash
mv experiments experiments_old_$(date +%Y%m%d)
```

### 2. Use new pipeline
```bash
# Run your experiments with new structure
bash eval/run_experiment.sh 1_teacher all
bash eval/run_experiment.sh no_teacher all
```

### 3. Compare results
```bash
python eval/compare_experiments_new.py
```

## Advantages

- **Clean naming**: `experiments/1_teacher/val/` instead of `experiments/1_teacher_20250706_130402/`
- **No duplicates**: Each experiment+dataset combination has exactly one result
- **Easy access**: Direct paths instead of searching for "latest"
- **Clear organization**: Train/val results are separated
- **Better comparison**: Easy to compare same datasets across experiments
- **Simpler debugging**: Logs and results are clearly organized

## File Locations

- **Inference results**: `experiments/{exp_name}/{dataset}/inference/predictions.json`
- **Evaluation metrics**: `experiments/{exp_name}/{dataset}/evaluation/metrics.json`
- **Configuration**: `experiments/{exp_name}/config.json`
- **Logs**: `experiments/{exp_name}/{dataset}/{inference|evaluation}/*.log`

## Troubleshooting

1. **Experiment already exists**: Results will be overwritten (by design)
2. **Missing datasets**: Make sure `data/train.jsonl` and `data/val.jsonl` exist
3. **Permission errors**: Check write permissions on `experiments/` directory
4. **Evaluation fails**: Ensure inference completed successfully first