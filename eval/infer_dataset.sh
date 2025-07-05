#!/bin/bash
# Inference pipeline for multiple datasets with teacher guidance support
# Usage: ./eval/infer_dataset.sh
# Configure NUM_TEACHERS directly in the script to tune teacher guidance

set -e  # Exit on any error

# Set Python path and offline mode
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4


# Model and data configuration
MODEL_PATH="output-74/7-4-teacher_student_loss_weight_roundFloats/checkpoint-280"
DATA_ROOT="./"
MAX_NEW_TOKENS=1024

# Teacher guidance configuration (MAIN TUNING PARAMETER)
NUM_TEACHERS=0                                                      # Number of teachers per sample (0 = no teacher guidance)
TEACHER_POOL_FILE="data/teacher.jsonl"                             # Teacher pool file location

# Performance configuration
BATCH_SIZE=64                                                   
NUM_WORKERS=8                                                   

LOG_LEVEL="info"                                                    
ENABLE_TORCH_COMPILE=false                                         # Enable torch.compile optimization

# Output directory
RESULTS_DIR="infer_result"
mkdir -p "$RESULTS_DIR"


# Generate output suffix based on teacher configuration
if [ "$NUM_TEACHERS" -gt 0 ]; then
    TEACHER_SUFFIX="teacher_${NUM_TEACHERS}"
else
    TEACHER_SUFFIX="no_teacher"
fi

# Configurable list of datasets to process (output names include teacher info)
DATASETS=(
    "data/train.jsonl:train-${MAX_NEW_TOKENS}-74-${TEACHER_SUFFIX}"
    # "data/chinese-val.jsonl:val-${MAX_NEW_TOKENS}-${TEACHER_SUFFIX}"
)

# Allow single-dataset mode for compatibility with run_inference.sh
if [ -n "$INPUT_JSONL" ]; then
    # If INPUT_JSONL is set, override DATASETS to only use this file
    DATASETS=("$INPUT_JSONL:chinese-val-${TEACHER_SUFFIX}")
fi


# Internal flag for torch compile
USE_TORCH_COMPILE=""
if [ "$ENABLE_TORCH_COMPILE" = true ]; then
    USE_TORCH_COMPILE="--use_torch_compile"
fi

# Determine inference mode for display
if [ "$NUM_TEACHERS" -gt 0 ]; then
    INFERENCE_MODE="Teacher-Guided Inference (${NUM_TEACHERS} teachers per sample)"
else
    INFERENCE_MODE="Standard Inference (no teacher guidance)"
fi

# Display configuration
echo "🚀 Unified Dataset Inference Runner"
echo "=================================================="
echo "📁 Model: $MODEL_PATH"
echo "📊 Output directory: $RESULTS_DIR"
echo "🎯 Mode: $INFERENCE_MODE"
echo "📋 Datasets: ${#DATASETS[@]} to process"
echo ""
echo "⚙️  Generation Settings:"
echo "   Max new tokens: $MAX_NEW_TOKENS"
echo "   Max samples per dataset: $([ "$MAX_SAMPLES" -eq -1 ] && echo "All" || echo "$MAX_SAMPLES")"
echo "   Log level: $LOG_LEVEL"
echo ""
echo "🔧 Performance Settings:"
echo "   Batch size: $BATCH_SIZE $([ "$NUM_TEACHERS" -gt 0 ] && echo "(will be forced to 1 for teacher guidance)" || echo "")"
echo "   Num workers: $NUM_WORKERS"
echo "   Torch compile: $([ "$ENABLE_TORCH_COMPILE" = true ] && echo "ENABLED" || echo "DISABLED")"
if [ "$NUM_TEACHERS" -gt 0 ]; then
    echo "   Teacher pool: $TEACHER_POOL_FILE"
fi
echo ""

# Print optimization tips
echo "💡 Performance Features:"
echo "   ⚡ Flash Attention 2: MANDATORY (always enabled)"
echo "   💾 KV Cache: MANDATORY (always enabled)"
echo "   🚀 Batch Processing: Configurable (default: 32)"
echo "   🎯 Single GPU: Optimized for single GPU inference"
echo "   🔥 Torch Compile: Optional (set ENABLE_TORCH_COMPILE=true)"
echo "   👥 Teacher Guidance: Configurable (NUM_TEACHERS parameter)"
echo ""

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model path does not exist: $MODEL_PATH"
    echo "Please check the model path and try again."
    exit 1
fi

# Validate teacher pool if teacher guidance is enabled
if [ "$NUM_TEACHERS" -gt 0 ]; then
    if [ ! -f "$TEACHER_POOL_FILE" ]; then
        echo "❌ Teacher pool file not found: $TEACHER_POOL_FILE"
        echo "Please ensure the teacher pool file exists before running inference."
        exit 1
    else
        teacher_count=$(wc -l < "$TEACHER_POOL_FILE")
        echo "✅ Found $teacher_count teacher samples in pool"
        if [ "$teacher_count" -lt "$NUM_TEACHERS" ]; then
            echo "⚠️  Warning: Only $teacher_count teachers available, but $NUM_TEACHERS requested"
            echo "   Will use all available teachers"
        fi
    fi
fi

# Validate input datasets
echo "📋 Validating input datasets..."
for dataset_entry in "${DATASETS[@]}"; do
    IFS=':' read -r input_jsonl dataset_name <<< "$dataset_entry"
    
    if [ ! -f "$input_jsonl" ]; then
        echo "❌ Dataset file not found: $input_jsonl"
        echo "Please ensure all dataset files exist before running inference."
        exit 1
    else
        # Count lines in the dataset
        line_count=$(wc -l < "$input_jsonl")
        echo "✅ $dataset_name: $line_count samples in $input_jsonl"
    fi
done
echo ""

###############################################################################
# INFERENCE EXECUTION                                                          #
###############################################################################

# Function to run inference on a dataset
run_inference() {
    local input_jsonl=$1
    local dataset_name=$2
    local output_file="$RESULTS_DIR/${dataset_name}.json"
    
    echo "🚀 Running inference on $dataset_name..."
    echo "   Input: $input_jsonl"
    echo "   Output: $output_file"
    if [ "$NUM_TEACHERS" -gt 0 ]; then
        echo "   Using teacher guidance: $NUM_TEACHERS teachers per sample"
    fi
    
    # Check if output already exists
    if [ -f "$output_file" ]; then
        echo "⚠️  Output file already exists: $output_file"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping $dataset_name (output file exists)"
            return 0
        fi
    fi
    
    # Build command with teacher guidance if enabled
    CMD="python src/inference.py \
        --model_path \"$MODEL_PATH\" \
        --input_jsonl \"$input_jsonl\" \
        --output_file \"$output_file\" \
        --data_root \"$DATA_ROOT\" \
        --max_new_tokens $MAX_NEW_TOKENS \
        --log_level \"$LOG_LEVEL\" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS"
    
    # Add max samples limit if specified
    if [ "$MAX_SAMPLES" -gt 0 ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
    
    # Add teacher guidance parameters if enabled
    if [ "$NUM_TEACHERS" -gt 0 ]; then
        CMD="$CMD --teacher_pool_file \"$TEACHER_POOL_FILE\" --num_teachers $NUM_TEACHERS"
    fi
    
    # Add torch compile if enabled
    if [ -n "$USE_TORCH_COMPILE" ]; then
        CMD="$CMD $USE_TORCH_COMPILE"
    fi
    
    bash -c "$CMD" < /dev/null 2>&1 | tee "$RESULTS_DIR/${dataset_name}_inference.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        if [ -f "$output_file" ]; then
            # Validate output file using separate script
            echo "✅ Inference completed for $dataset_name"
            python eval/validate_results.py "$output_file" --no-preview
            return 0
        else
            echo "❌ Inference completed but output file not found: $output_file"
            return 1
        fi
    elif [ $exit_code -eq 124 ]; then
        echo "❌ Inference timed out for $dataset_name (1 hour limit)"
        return 1
    else
        echo "❌ Inference failed for $dataset_name (exit code: $exit_code)"
        echo "Check log file: $RESULTS_DIR/${dataset_name}_inference.log"
        return 1
    fi
}

# Process all datasets
successful_count=0
failed_count=0
failed_datasets=()

for dataset_entry in "${DATASETS[@]}"; do
    # Split entry into path and name
    IFS=':' read -r input_jsonl dataset_name <<< "$dataset_entry"
    
    echo "=== PROCESSING: $dataset_name ==="
    
    if run_inference "$input_jsonl" "$dataset_name"; then
        ((successful_count++))
    else
        ((failed_count++))
        failed_datasets+=("$dataset_name")
    fi
    
    echo ""
done

###############################################################################
# FINAL SUMMARY                                                               #
###############################################################################

echo "🏁 All inference completed!"
echo "📊 Summary:"
echo "   ✅ Successful: $successful_count datasets"
echo "   ❌ Failed: $failed_count datasets"

if [ $failed_count -gt 0 ]; then
    echo "   Failed datasets: ${failed_datasets[*]}"
fi

echo "📁 Results directory: $RESULTS_DIR"

# Show detailed results for single dataset runs
if [ $failed_count -eq 0 ] && [ ${#DATASETS[@]} -eq 1 ]; then
    dataset_entry="${DATASETS[0]}"
    IFS=':' read -r input_jsonl dataset_name <<< "$dataset_entry"
    output_file="$RESULTS_DIR/${dataset_name}.json"
    
    if [ -f "$output_file" ]; then
        result_count=$(python -c "
import json
try:
    with open('$output_file') as f:
        data = json.load(f)
    successful = len([r for r in data if 'error' not in r])
    failed = len([r for r in data if 'error' in r])
    print(f'{len(data)}:{successful}:{failed}')
except Exception as e:
    print('0:0:0')
" 2>/dev/null)
        IFS=':' read -r total successful failed <<< "$result_count"
        echo "✅ Inference completed successfully!"
        echo "📊 Results Summary:"
        echo "   Total samples: $total"
        echo "   Successful: $successful"
        echo "   Failed: $failed"
        echo "   Output file: $output_file"
        echo ""
        if [ "$successful" -gt 0 ]; then
            echo "📋 Sample Results:"
            python -c "
import json
with open('$output_file') as f:
    results = json.load(f)
    for i, r in enumerate([r for r in results if 'error' not in r][:2]):
        print(f'\\n  Sample {i+1}:')
        print(f'    Image: {r.get(\"image\", \"unknown\")}')
        pred = r.get(\"pred_result\", \"\")
        if pred:
            print(f'    Prediction: {pred[:100]}...')
        else:
            print(f'    No prediction generated')
" 2>/dev/null || echo "   Could not parse results for preview"
        fi
    fi
fi

echo ""
echo "🎯 Usage:"
echo "   Edit NUM_TEACHERS in the script to tune teacher guidance"
echo "   Standard inference:     NUM_TEACHERS=0"
echo "   Teacher-guided:         NUM_TEACHERS=1, 2, 3, etc."
echo "   Then run: ./eval/infer_dataset.sh"
