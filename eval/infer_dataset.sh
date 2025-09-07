#!/bin/bash
# Clean inference pipeline with improved directory structure
set -e

###############################################################################
# EXPERIMENT CONFIGURATION - EDIT THIS SECTION
###############################################################################

# Experiment name (set manually)
MODEL_PATH="outputs/8-30-standard_phase_3/8-30-standard_phase_3-ep50-merger_5e-4-top_lr-5e-6-vison_1e-7/checkpoint-1100"  
EXP_NAME="8-30-standard_phase_3"           
CONFIG_PATH="configs/phase_3/standard.yaml"

# Dataset to process (single dataset per run)
DATASET="train"                  # "train" or "val"
DATA_ROOT="data/ds_v2_bbu_bbu_shield"       # Root directory - centralized data resolver will auto-discover all files
OUTPUT_BASE="infer_results" 


# Teacher configuration (set manually)
NUM_TEACHERS=1

# Model configuration

MODEL_NAME="qwen2_5_vl"
MAX_NEW_TOKENS=1024
BATCH_SIZE=1          # Use batch_size=1 for coordinate token models
NUM_WORKERS=4         # Use 0 workers to avoid memory issues
ENABLE_TORCH_COMPILE=false
# Force eager attention to avoid Flash Attention triton issues
FORCE_EAGER_ATTENTION=true

MAX_SAMPLES=40      

# Logging level (debug shows validation details)
LOG_LEVEL="debug"                       # "debug" for detailed validation info, "info" for normal

# Set PROJECT_ROOT dynamically
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH=.:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=7

# Normalize to absolute paths per repository rules
ABS_REPO_ROOT="."
ABS_CONFIG_PATH=$(readlink -f "$ABS_REPO_ROOT/$CONFIG_PATH")
ABS_MODEL_PATH=$(readlink -f "$ABS_REPO_ROOT/$MODEL_PATH")
ABS_DATA_ROOT=$(readlink -f "$ABS_REPO_ROOT/$DATA_ROOT")

# Validate dataset parameter
if [[ "$DATASET" != "train" && "$DATASET" != "val" ]]; then
    echo "❌ DATASET must be 'train' or 'val', got: $DATASET"
    exit 1
fi

# Create clean experiment structure
EXPERIMENT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
DATASET_DIR="${EXPERIMENT_DIR}/${DATASET}"
INFERENCE_DIR="${DATASET_DIR}/inference"

echo "🚀 Starting inference for experiment: ${EXP_NAME}"
echo "📊 Dataset split: ${DATASET}"
echo "📁 Output directory: ${INFERENCE_DIR}"

# Create directories
mkdir -p "${INFERENCE_DIR}"

# Generate output file names
OUTPUT_SUFFIX="predictions"
OUTPUT_FILE="${INFERENCE_DIR}/${OUTPUT_SUFFIX}.jsonl"
LOG_FILE="${INFERENCE_DIR}/inference.log"

# Save experiment configuration (for bookkeeping only)
CONFIG_FILE="${EXPERIMENT_DIR}/config.json"
cat > "$CONFIG_FILE" << EOF
{
  "exp_name": "$EXP_NAME",
  "model": {
    "name": "$MODEL_NAME",
    "path": "$ABS_MODEL_PATH"
  },
  "generation": {
    "max_new_tokens": $MAX_NEW_TOKENS,
    "batch_size": $BATCH_SIZE,
    "num_workers": $NUM_WORKERS,
    "enable_torch_compile": $ENABLE_TORCH_COMPILE
  },
  "teacher": {
    "num_teachers": $NUM_TEACHERS
  },
  "debug": {
    "max_samples": ${MAX_SAMPLES:-null}
  },
  "log_level": "$LOG_LEVEL"
}
EOF

echo "💾 Saved experiment config: $CONFIG_FILE"

# Determine teacher arguments
if [ "$NUM_TEACHERS" -gt 0 ]; then
    TEACHER_ARGS="--num_teachers $NUM_TEACHERS"
    echo "👨‍🏫 Using $NUM_TEACHERS teacher(s). Teacher pool will be auto-resolved from data_root."
else
    TEACHER_ARGS=""
    echo "🚫 No teacher mode"
fi

# Validate config file exists
if [ ! -f "$ABS_CONFIG_PATH" ]; then
    echo "❌ Configuration file not found: $ABS_CONFIG_PATH"
    echo "Available config files:"
    ls -1 $ABS_REPO_ROOT/configs/*.yaml 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Build inference command (use ms env's python directly)
PY_BIN="/root/miniconda3/envs/ms/bin/python"
INFERENCE_CMD="$PY_BIN $ABS_REPO_ROOT/src_new/inference.py \
    --config_path \"$ABS_CONFIG_PATH\" \
    --model_path \"$ABS_MODEL_PATH\" \
    --dataset \"$DATASET\" \
    --output_file \"$OUTPUT_FILE\" \
    --data_root \"${ABS_DATA_ROOT}\" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --log_level \"$LOG_LEVEL\" \
    $TEACHER_ARGS"

# Add max_samples parameter if set
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "None" ] && [ "$MAX_SAMPLES" != "null" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --max_samples $MAX_SAMPLES"
fi

if [ "$ENABLE_TORCH_COMPILE" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --use_torch_compile"
fi

if [ "$FORCE_EAGER_ATTENTION" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --force_eager_attention"
fi

# Resolve dataset file using centralized data resolver for logging and stats
DERIVED_DATASET_FILE=$($PY_BIN - << PY
from src_new.utils.data_resolver import DataResolver
print(DataResolver.resolve_dataset_paths("$ABS_DATA_ROOT").train_data_path if "$DATASET"=="train" else DataResolver.resolve_dataset_paths("$ABS_DATA_ROOT").val_data_path)
PY
)

echo ""
echo "🔧 Configuration:"
echo "   Config: $ABS_CONFIG_PATH"
echo "   Model: $ABS_MODEL_PATH"
echo "   Dataset file (derived): $DERIVED_DATASET_FILE"
echo "   Data root: ${ABS_DATA_ROOT}"
echo "   Output: $OUTPUT_FILE"
echo "   Teachers: $NUM_TEACHERS"
echo "   Max tokens: $MAX_NEW_TOKENS"
echo "   Batch size: $BATCH_SIZE"
echo "   Max samples: ${MAX_SAMPLES:-'all'}"
echo "   Multi-geometry parsing: Auto-detected from config"
echo ""

# Run inference
echo "🚀 Starting inference..."
echo "Command: $INFERENCE_CMD"
echo ""

if eval "$INFERENCE_CMD" 2>&1 | tee "$LOG_FILE"; then
    if [ -f "$OUTPUT_FILE" ]; then
        # Get result statistics
        if [ -f "$DERIVED_DATASET_FILE" ]; then
            TOTAL_SAMPLES=$(wc -l < "$DERIVED_DATASET_FILE")
        else
            TOTAL_SAMPLES=0
        fi
        OUTPUT_SAMPLES=$($PY_BIN -c "
import json
try:
    with open('$OUTPUT_FILE') as f:
        data = [line for line in f if line.strip()]
    print(len(data))
except Exception:
    print('0')
" 2>/dev/null || echo "0")
        
        echo ""
        echo "✅ Inference completed successfully!"
        echo "📊 Processed: $OUTPUT_SAMPLES/$TOTAL_SAMPLES samples"
        echo "📁 Results: $OUTPUT_FILE"
        echo "📋 Log: $LOG_FILE"
        echo ""
        echo "🎯 Next step: Run evaluation"
        echo "   bash eval/run_evaluation.sh"
    else
        echo "❌ Inference failed: Output file not created"
        exit 1
    fi
else
    echo "❌ Inference failed"
    exit 1
fi