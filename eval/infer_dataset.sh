#!/bin/bash
# Inference pipeline for multiple datasets
# Usage: ./eval/infer_dataset.sh [log_level] [batch_size] [num_workers] [use_torch_compile]
# Examples:
#   ./eval/infer_dataset.sh                    # Default: info log, batch_size=4
#   ./eval/infer_dataset.sh info 8             # Batch size 8
#   ./eval/infer_dataset.sh info 8 4           # Batch size 8 with 4 workers
#   ./eval/infer_dataset.sh info 8 4 compile   # With torch.compile optimization

set -e  # Exit on any error

# Set Python path and offline mode
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

# Model configuration
MODEL_PATH="output-630/630-filtered_sample-updated_prompt/checkpoint-150"
DATA_ROOT="./"
MAX_NEW_TOKENS=1024

# Configurable list of datasets to process
DATASETS=(
    "data/chinese-train.jsonl:train-$MAX_NEW_TOKENS-teacher-1-updated_prompt"
    # "data/chinese-val.jsonl:val-$MAX_NEW_TOKENS-updated_prompt"
)

# Allow single-dataset mode for compatibility with run_inference.sh
if [ -n "$INPUT_JSONL" ]; then
    # If INPUT_JSONL is set, override DATASETS to only use this file
    if [ "$NUM_TEACHERS" -gt 0 ]; then
        DATASETS=("$INPUT_JSONL:chinese-val-teacher-guided-${NUM_TEACHERS}teachers")
    else
        DATASETS=("$INPUT_JSONL:chinese-val-standard")
    fi
fi

###############################################################################
# USER-CONFIGURABLE PARAMETERS (edit here, then simply run the script)         #
###############################################################################

# Verbosity for Python logger (debug | info | warning | error)
LOG_LEVEL="info"

# Inference batch size
BATCH_SIZE=64
# Data-loader workers
NUM_WORKERS=8

# Enable torch.compile (true/false)
ENABLE_TORCH_COMPILE=false

# Teacher guidance configuration (defaults)
DEFAULT_TEACHER_POOL_FILE="data/teacher_pool.jsonl"  # Default teacher pool location
TEACHER_POOL_FILE=""  # Path to teacher pool JSONL file (empty = no teacher guidance)
NUM_TEACHERS=1        # Number of teachers per sample (0 = no teacher guidance)

# Override with environment variables for easy mode switching
# Examples:
#   NUM_TEACHERS=2 ./eval/infer_dataset.sh     # Enable teacher guidance
#   MAX_SAMPLES=5 ./eval/infer_dataset.sh      # Test on limited samples
NUM_TEACHERS=${NUM_TEACHERS:-0}                                    # Default: standard inference

# Auto-set teacher pool file if teacher guidance is requested but no pool file specified
if [ "$NUM_TEACHERS" -gt 0 ] && [ -z "$TEACHER_POOL_FILE" ]; then
    TEACHER_POOL_FILE="$DEFAULT_TEACHER_POOL_FILE"
fi
BATCH_SIZE=${BATCH_SIZE:-32}                                       # Allow override (default: 32)
MAX_SAMPLES=${MAX_SAMPLES:--1}                                     # -1 = all samples

# Internal flag passed to Python when enabled
USE_TORCH_COMPILE=""
if [ "$ENABLE_TORCH_COMPILE" = true ]; then
    USE_TORCH_COMPILE="--use_torch_compile"
fi

# Output directory
RESULTS_DIR="infer_result"
mkdir -p "$RESULTS_DIR"

# Determine inference mode for display
if [ -n "$TEACHER_POOL_FILE" ] && [ "$NUM_TEACHERS" -gt 0 ]; then
    INFERENCE_MODE="Teacher-Guided Inference (${NUM_TEACHERS} teachers per sample)"
else
    INFERENCE_MODE="Standard Inference (no teacher guidance)"
fi

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
if [ -n "$TEACHER_POOL_FILE" ] && [ "$NUM_TEACHERS" -gt 0 ]; then
    echo "   Teacher pool: $TEACHER_POOL_FILE"
fi
echo ""

# Print optimization tips
echo "💡 Performance Features:"
echo "   ⚡ Flash Attention 2: MANDATORY (always enabled)"
echo "   💾 KV Cache: MANDATORY (always enabled)"
echo "   🚀 Batch Processing: Configurable (default: 4)"
echo "   🎯 Single GPU: Optimized for single GPU inference"
echo "   🔥 Torch Compile: Optional (use 'compile' flag)"
echo "   👥 Teacher Guidance: Optional (matches training pipeline)"
echo ""
echo "📋 Optimization Tips:"
echo "   - Batch size 4: ~3.75x speedup (16GB GPU memory)"
echo "   - Batch size 8: ~6.8x speedup (28GB GPU memory)"
echo "   - Single GPU: Optimized for single GPU inference"
echo "   - Torch compile: Additional 10-20% speedup on newer GPUs"
echo "   - Teacher guidance: May improve accuracy by providing context"
echo ""

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model path does not exist: $MODEL_PATH"
    echo "Please check the model path and try again."
    exit 1
fi

# Validate teacher pool if specified
if [ -n "$TEACHER_POOL_FILE" ] && [ "$NUM_TEACHERS" -gt 0 ]; then
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

# Function to run inference on a dataset
run_inference() {
    local input_jsonl=$1
    local dataset_name=$2
    local output_file="$RESULTS_DIR/${dataset_name}.json"
    
    echo "🚀 Running inference on $dataset_name..."
    echo "   Input: $input_jsonl"
    echo "   Output: $output_file"
    if [ -n "$TEACHER_POOL_FILE" ] && [ "$NUM_TEACHERS" -gt 0 ]; then
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
    if [ -n "$TEACHER_POOL_FILE" ] && [ "$NUM_TEACHERS" -gt 0 ]; then
        CMD="$CMD --teacher_pool_file \"$TEACHER_POOL_FILE\" --num_teachers $NUM_TEACHERS"
    fi
    
    # Add torch compile if enabled
    if [ -n "$USE_TORCH_COMPILE" ]; then
        CMD="$CMD $USE_TORCH_COMPILE"
    fi
    
    # Run inference with timeout protection
    timeout 3600 bash -c "$CMD" < /dev/null 2>&1 | tee "$RESULTS_DIR/${dataset_name}_inference.log"
    
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

# Final summary
echo "🏁 All inference completed!"
echo "📊 Summary:"
echo "   ✅ Successful: $successful_count datasets"
echo "   ❌ Failed: $failed_count datasets"

if [ $failed_count -gt 0 ]; then
    echo "   Failed datasets: ${failed_datasets[*]}"
fi

echo "📁 Results directory: $RESULTS_DIR"

if [ $failed_count -eq 0 ] && [ ${#DATASETS[@]} -eq 1 ]; then
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

echo "🎯 Quick Mode Switching:"
echo "   Standard inference:      NUM_TEACHERS=0 ./eval/infer_dataset.sh"
echo "   Teacher-guided (2):      NUM_TEACHERS=2 ./eval/infer_dataset.sh"
echo "   Quick test (5 samples):  MAX_SAMPLES=5 ./eval/infer_dataset.sh"
echo "   Both modes for comparison:"
echo "     NUM_TEACHERS=0 ./eval/infer_dataset.sh && NUM_TEACHERS=2 ./eval/infer_dataset.sh"
