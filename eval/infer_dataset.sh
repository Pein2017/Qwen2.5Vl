#!/bin/bash
# Simple inference pipeline - just set EXP_NAME and run
set -e

###############################################################################
# EXPERIMENT CONFIGURATION - ONLY EDIT THIS SECTION
###############################################################################

# Experiment name (this determines everything else)
EXP_NAME="baseline_no_teacher"           # e.g., "baseline_no_teacher", "test_3_teachers", "checkpoint_500_eval"

# Model configuration  
MODEL_PATH="output-74/7-4-teacher_student_loss_weight_roundFloats/checkpoint-280"
MODEL_NAME="qwen2_5_vl"

# Generation parameters
MAX_NEW_TOKENS=1024
BATCH_SIZE=64
NUM_WORKERS=8
ENABLE_TORCH_COMPILE=false

# Teacher guidance (auto-detected from EXP_NAME or set manually)
# NUM_TEACHERS will be auto-detected if EXP_NAME contains "teacher" pattern
# Or set manually: NUM_TEACHERS=3
NUM_TEACHERS="auto"                     # "auto" or specific number
TEACHER_POOL_FILE="data/teacher.jsonl"

# Datasets to process
DATASETS=(
    "data/train.jsonl"
    "data/val.jsonl"
)

# Logging level (debug shows validation details)
LOG_LEVEL="debug"                       # "debug" for detailed validation info, "info" for normal

###############################################################################
# AUTO-CONFIGURATION - DO NOT EDIT
###############################################################################

# Auto-detect teacher count from experiment name
if [ "$NUM_TEACHERS" = "auto" ]; then
    if [[ "$EXP_NAME" =~ no_teacher|0_teacher ]]; then
        NUM_TEACHERS=0
    elif [[ "$EXP_NAME" =~ ([0-9]+)_teacher ]]; then
        NUM_TEACHERS=${BASH_REMATCH[1]}
    elif [[ "$EXP_NAME" =~ teacher ]]; then
        NUM_TEACHERS=1  # Default if "teacher" mentioned but no number
    else
        NUM_TEACHERS=0  # Default to no teacher
    fi
fi

# Environment
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4

# Create experiment structure
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_ID="${EXP_NAME}_${TIMESTAMP}"
OUTPUT_BASE="experiments"
EXPERIMENT_DIR="${OUTPUT_BASE}/${EXPERIMENT_ID}"
INFERENCE_DIR="${EXPERIMENT_DIR}/inference"
mkdir -p "${INFERENCE_DIR}"

# Generate consistent naming
TEACHER_SUFFIX=$([ "$NUM_TEACHERS" -gt 0 ] && echo "teacher_${NUM_TEACHERS}" || echo "no_teacher")
OUTPUT_SUFFIX="tokens_${MAX_NEW_TOKENS}_${TEACHER_SUFFIX}_model_${MODEL_NAME}"

echo "🚀 Starting inference for experiment: ${EXP_NAME}"
echo "📁 Experiment ID: ${EXPERIMENT_ID}"
echo "📊 Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "👥 Teachers: ${NUM_TEACHERS} (auto-detected from name)"
echo "📋 Datasets: ${#DATASETS[@]}"
echo "📁 Output: ${INFERENCE_DIR}"
echo "🔧 Output suffix: ${OUTPUT_SUFFIX}"
echo ""

# Save experiment configuration
cat > "${EXPERIMENT_DIR}/config.json" << EOF
{
  "exp_name": "${EXP_NAME}",
  "experiment_id": "${EXPERIMENT_ID}", 
  "timestamp": "${TIMESTAMP}",
  "model": {
    "name": "${MODEL_NAME}",
    "path": "${MODEL_PATH}"
  },
  "generation": {
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "batch_size": ${BATCH_SIZE},
    "num_workers": ${NUM_WORKERS},
    "enable_torch_compile": ${ENABLE_TORCH_COMPILE}
  },
  "teacher": {
    "num_teachers": ${NUM_TEACHERS},
    "teacher_pool_file": "${TEACHER_POOL_FILE}"
  },
  "datasets": $(printf '%s\n' "${DATASETS[@]}" | jq -R . | jq -s .),
  "output_suffix": "${OUTPUT_SUFFIX}",
  "log_level": "${LOG_LEVEL}"
}
EOF

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Validate teacher pool if needed
if [ "$NUM_TEACHERS" -gt 0 ] && [ ! -f "$TEACHER_POOL_FILE" ]; then
    echo "❌ Teacher pool file not found: $TEACHER_POOL_FILE"
    exit 1
fi

# Process each dataset
successful_count=0
failed_datasets=()

for dataset_path in "${DATASETS[@]}"; do
    dataset_name=$(basename "$dataset_path" .jsonl)
    output_file="${INFERENCE_DIR}/${dataset_name}_${OUTPUT_SUFFIX}.json"
    log_file="${INFERENCE_DIR}/${dataset_name}_inference.log"
    
    echo "=== Processing: ${dataset_name} ==="
    echo "Input: ${dataset_path}"
    echo "Output: ${output_file}"
    
    # Check if dataset exists
    if [ ! -f "$dataset_path" ]; then
        echo "❌ Dataset not found: $dataset_path"
        failed_datasets+=("${dataset_name}")
        continue
    fi
    
    # Build inference command
    CMD="python src/inference.py \
        --model_path \"${MODEL_PATH}\" \
        --input_jsonl \"${dataset_path}\" \
        --output_file \"${output_file}\" \
        --data_root \".\" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --log_level \"${LOG_LEVEL}\" \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS}"
    
    # Add teacher guidance if enabled
    if [ "$NUM_TEACHERS" -gt 0 ]; then
        CMD="${CMD} --teacher_pool_file \"${TEACHER_POOL_FILE}\" --num_teachers ${NUM_TEACHERS}"
    fi
    
    # Add torch compile if enabled
    if [ "$ENABLE_TORCH_COMPILE" = true ]; then
        CMD="${CMD} --use_torch_compile"
    fi
    
    echo "🚀 Running inference..."
    
    if bash -c "$CMD" 2>&1 | tee "$log_file"; then
        if [ -f "$output_file" ]; then
            result_count=$(python -c "
import json
try:
    with open('$output_file') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
" 2>/dev/null)
            echo "✅ Generated ${result_count} predictions for ${dataset_name}"
            ((successful_count++))
        else
            echo "❌ Output file not created for ${dataset_name}"
            failed_datasets+=("${dataset_name}")
        fi
    else
        echo "❌ Inference failed for ${dataset_name}"
        failed_datasets+=("${dataset_name}")
    fi
    echo ""
done

# Final summary
echo "🏁 Inference completed for experiment: ${EXP_NAME}"
echo "✅ Successful: ${successful_count}/${#DATASETS[@]} datasets"
if [ ${#failed_datasets[@]} -gt 0 ]; then
    echo "❌ Failed: ${failed_datasets[*]}"
fi
echo "📁 Results: ${INFERENCE_DIR}"
echo "📋 Config: ${EXPERIMENT_DIR}/config.json"
echo ""
echo "🎯 Next step: Run evaluation with the same EXP_NAME"
echo "   Edit eval/run_evaluation.sh and set EXP_NAME=\"${EXP_NAME}\""