#!/bin/bash
# Clean inference pipeline with improved directory structure
set -e

###############################################################################
# EXPERIMENT CONFIGURATION - EDIT THIS SECTION
###############################################################################

# Experiment name (set manually)
EXP_NAME="det_coordinates"           # e.g., "1_teacher", "no_teacher", "baseline"

# Dataset to process (single dataset per run)
DATASET="val"                  # "train" or "val"

# Teacher configuration (set manually)
NUM_TEACHERS=1                          # Set number of teachers manually (0 for no teacher)
TEACHER_POOL_FILE="data/teacher.jsonl"

# Model configuration  
MODEL_PATH="output-714/coordinate-det/checkpoint-150"
MODEL_NAME="qwen2_5_vl"
CONFIG_PATH="configs/base_flat_det.yaml"  # EXPLICIT configuration file path - no fallbacks

# Generation parameters (optimized for coordinate token models)
MAX_NEW_TOKENS=2048
BATCH_SIZE=1          # Use batch_size=1 for coordinate token models
NUM_WORKERS=0         # Use 0 workers to avoid memory issues
ENABLE_TORCH_COMPILE=false

# Force eager attention to avoid Flash Attention triton issues
FORCE_EAGER_ATTENTION=true

# Logging level (debug shows validation details)  
LOG_LEVEL="info"                       # "debug" for detailed validation info, "info" for normal

###############################################################################
# PIPELINE EXECUTION - DO NOT EDIT BELOW
###############################################################################

# Environment
export PYTHONPATH=/data3/Qwen2.5-VL-main:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=5

# Validate dataset parameter
if [[ "$DATASET" != "train" && "$DATASET" != "val" ]]; then
    echo "❌ DATASET must be 'train' or 'val', got: $DATASET"
    exit 1
fi

# Create clean experiment structure
OUTPUT_BASE="exp_det_coordinates"
EXPERIMENT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
DATASET_DIR="${EXPERIMENT_DIR}/${DATASET}"
INFERENCE_DIR="${DATASET_DIR}/inference"

echo "🚀 Starting inference for experiment: ${EXP_NAME}"
echo "📊 Dataset: ${DATASET}"
echo "📁 Output directory: ${INFERENCE_DIR}"

# Create directories
mkdir -p "${INFERENCE_DIR}"

# Generate output file names
OUTPUT_SUFFIX="predictions"
DATASET_FILE="data/${DATASET}.jsonl"
OUTPUT_FILE="${INFERENCE_DIR}/${OUTPUT_SUFFIX}.json"
LOG_FILE="${INFERENCE_DIR}/inference.log"

# Validate input dataset exists
if [ ! -f "$DATASET_FILE" ]; then
    echo "❌ Dataset file not found: $DATASET_FILE"
    echo "Available dataset files:"
    ls -1 data/*.jsonl 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Save experiment configuration
CONFIG_FILE="${EXPERIMENT_DIR}/config.json"
cat > "$CONFIG_FILE" << EOF
{
  "exp_name": "$EXP_NAME",
  "model": {
    "name": "$MODEL_NAME",
    "path": "$MODEL_PATH"
  },
  "generation": {
    "max_new_tokens": $MAX_NEW_TOKENS,
    "batch_size": $BATCH_SIZE,
    "num_workers": $NUM_WORKERS,
    "enable_torch_compile": $ENABLE_TORCH_COMPILE
  },
  "teacher": {
    "num_teachers": $NUM_TEACHERS,
    "teacher_pool_file": "$TEACHER_POOL_FILE"
  },
  "log_level": "$LOG_LEVEL"
}
EOF

echo "💾 Saved experiment config: $CONFIG_FILE"

# Determine teacher arguments
if [ "$NUM_TEACHERS" -gt 0 ]; then
    TEACHER_ARGS="--num_teachers $NUM_TEACHERS --teacher_pool_file $TEACHER_POOL_FILE"
    echo "👨‍🏫 Using $NUM_TEACHERS teacher(s) from $TEACHER_POOL_FILE"
else
    TEACHER_ARGS=""
    echo "🚫 No teacher mode"
fi

# Validate config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Configuration file not found: $CONFIG_PATH"
    echo "Available config files:"
    ls -1 configs/*.yaml 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Build inference command
INFERENCE_CMD="python src/inference.py \
    --config_path \"$CONFIG_PATH\" \
    --model_path \"$MODEL_PATH\" \
    --input_file \"$DATASET_FILE\" \
    --output_file \"$OUTPUT_FILE\" \
    --data_root \".\" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --log_level \"$LOG_LEVEL\" \
    $TEACHER_ARGS"

if [ "$ENABLE_TORCH_COMPILE" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --use_torch_compile"
fi

echo ""
echo "🔧 Configuration:"
echo "   Config: $CONFIG_PATH"
echo "   Model: $MODEL_PATH"
echo "   Dataset: $DATASET_FILE"
echo "   Output: $OUTPUT_FILE"
echo "   Teachers: $NUM_TEACHERS"
echo "   Max tokens: $MAX_NEW_TOKENS"
echo "   Batch size: $BATCH_SIZE"
echo ""

# Run inference
echo "🚀 Starting inference..."
echo "Command: $INFERENCE_CMD"
echo ""

if eval "$INFERENCE_CMD" 2>&1 | tee "$LOG_FILE"; then
    if [ -f "$OUTPUT_FILE" ]; then
        # Get result statistics
        TOTAL_SAMPLES=$(wc -l < "$DATASET_FILE")
        if [ -f "$OUTPUT_FILE" ]; then
            OUTPUT_SAMPLES=$(python -c "
import json
try:
    with open('$OUTPUT_FILE') as f:
        data = json.load(f)
    if isinstance(data, list):
        print(len(data))
    else:
        print('1')
except:
    print('0')
" 2>/dev/null || echo "0")
        else
            OUTPUT_SAMPLES=0
        fi
        
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