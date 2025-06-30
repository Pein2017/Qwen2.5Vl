#!/bin/bash
# Enhanced evaluation script for Qwen2.5-VL dense captioning/grounding inference results
# Usage: ./eval/run_evaluation.sh [debug|info] [--check-only] [--enhanced] [--semantic-threshold] [--iou-threshold]
# Examples:
#   ./eval/run_evaluation.sh                                    # Standard evaluation with all enhancements
#   ./eval/run_evaluation.sh debug                              # Debug mode with detailed logging
#   ./eval/run_evaluation.sh info --check-only                 # Validate files only
#   ./eval/run_evaluation.sh info --enhanced --semantic-threshold 0.8  # Custom semantic threshold
#   ./eval/run_evaluation.sh debug --iou-threshold 0.7         # Custom IoU threshold

set -e  # Exit on any error

# Set Python path and offline mode
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Configurable list of result files to evaluate
RESULT_FILES=(
    "infer_result/chinese-train.json:chinese-train"
    # "infer_result/chinese-val.json:chinese-val"
)

###############################################################################
# USER-CONFIGURABLE PARAMETERS (edit here, then simply run the script)         #
###############################################################################

# Python logging level (debug | info)
LOG_LEVEL="info"

# Whether to only check files without running evaluation
CHECK_ONLY=false

# Enhanced feature toggles
ENHANCED_MODE=true    # master switch
ENABLE_SOFT_MATCHING=true
ENABLE_HIERARCHICAL=true
ENABLE_NOVEL_DETECTION=true

# Thresholds
SEMANTIC_THRESHOLD=0.7
IOU_THRESHOLD=0.3

# Verbose bash output
VERBOSE=false

# Output directory for metrics/logs
RESULTS_DIR="eval_result"
mkdir -p "$RESULTS_DIR"

echo "🎯 Enhanced Qwen2.5-VL Dense Captioning/Grounding Evaluation Pipeline"
echo "====================================================================="
echo "📊 Processing ${#RESULT_FILES[@]} result files with advanced metrics"
echo "📁 Output directory: $RESULTS_DIR"
echo ""
echo "🚀 Evaluation Configuration:"
echo "   Log level: $LOG_LEVEL"
echo "   IoU threshold: $IOU_THRESHOLD"
echo "   Semantic threshold: $SEMANTIC_THRESHOLD"
echo "   Enhanced mode: $([ "$ENHANCED_MODE" = true ] && echo 'ENABLED' || echo 'DISABLED')"
echo "   Verbose logging: $([ "$VERBOSE" = true ] && echo 'ENABLED' || echo 'DISABLED')"

if [ "$CHECK_ONLY" = true ]; then
    echo "   🔍 Mode: Validation only (no evaluation execution)"
else
    echo "   🔍 Mode: Full evaluation with metrics computation"
fi

echo ""
echo "💡 Enhanced Features Status:"
echo "   🧠 Soft Semantic Matching: $([ "$ENABLE_SOFT_MATCHING" = true ] && echo 'ENABLED' || echo 'DISABLED')"
echo "   🌳 Hierarchical Label Matching: $([ "$ENABLE_HIERARCHICAL" = true ] && echo 'ENABLED' || echo 'DISABLED')"
echo "   🔍 Novel Object Detection: $([ "$ENABLE_NOVEL_DETECTION" = true ] && echo 'ENABLED' || echo 'DISABLED')"
echo "   📂 Individual Categories: ENABLED (default)"
echo "   🎯 Multi-threshold Analysis: ENABLED (0.3, 0.5, 0.7, 0.8, 0.9)"
echo "   📊 Error Categorization: ENABLED (localization, classification, background, missed)"
echo "   🔧 Robust LLM Validation: ENABLED (handles malformed outputs)"
echo ""

# Validate thresholds using bash (requires bc)
if (( $(echo "$IOU_THRESHOLD < 0.0" | bc -l) )) || (( $(echo "$IOU_THRESHOLD > 1.0" | bc -l) )); then
    echo "❌ IoU threshold must be between 0.0 and 1.0, got $IOU_THRESHOLD"
    exit 1
fi

if (( $(echo "$SEMANTIC_THRESHOLD < 0.0" | bc -l) )) || (( $(echo "$SEMANTIC_THRESHOLD > 1.0" | bc -l) )); then
    echo "❌ Semantic threshold must be between 0.0 and 1.0, got $SEMANTIC_THRESHOLD"
    exit 1
fi

echo "✅ Thresholds validated"

# Function to check if file exists and is valid JSON with enhanced validation
check_responses_file() {
    local file_path="$1"
    local dataset_name="$2"
    
    if [ ! -f "$file_path" ]; then
        echo "   ❌ File not found: $file_path"
        return 1
    fi
    echo "   ✅ Found file for $dataset_name"
    return 0
}

# Function to run evaluation for a single dataset with enhanced features
run_single_evaluation() {
    local responses_file="$1"
    local dataset_name="$2"
    
    local output_file="${RESULTS_DIR}/${dataset_name}_metric.json"
    local log_file="${RESULTS_DIR}/${dataset_name}_evaluation.log"
    
    echo "🚀 Running enhanced evaluation on $dataset_name dataset..."
    echo "   📂 Input:  $responses_file"
    echo "   📊 Output: $output_file"
    echo "   📝 Log:    $log_file"
    
    # Prepare evaluation command with all enhanced features
    local eval_cmd="python eval/eval_dataset.py \
        --responses_file \"$responses_file\" \
        --output_file \"$output_file\" \
        --iou_threshold $IOU_THRESHOLD \
        --semantic_threshold $SEMANTIC_THRESHOLD \
        --log_level \"$LOG_LEVEL\""
    
    # Add feature flags
    if [ "$ENABLE_SOFT_MATCHING" = true ]; then
        eval_cmd="$eval_cmd --enable_soft_matching"
    else
        eval_cmd="$eval_cmd --disable_soft_matching"
    fi
    
    if [ "$ENABLE_HIERARCHICAL" = true ]; then
        eval_cmd="$eval_cmd --enable_hierarchical"
    else
        eval_cmd="$eval_cmd --disable_hierarchical"
    fi
    
    if [ "$ENABLE_NOVEL_DETECTION" = true ]; then
        eval_cmd="$eval_cmd --enable_novel_detection"
    else
        eval_cmd="$eval_cmd --disable_novel_detection"
    fi
    
    if [ "$VERBOSE" = true ]; then
        eval_cmd="$eval_cmd --verbose"
    fi
    
    echo "   🔧 Command: $eval_cmd"
    echo ""
    
    # Run evaluation with timing
    local start_time=$(date +%s)
    
    if bash -c "$eval_cmd" 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo "   ✅ Evaluation completed in ${duration}s"
        
        # Optional: print basic summary
        if [ -f "$output_file" ]; then
            echo "   📊 Metrics saved to $output_file"
        fi
        
        return 0
    else
        echo "   ❌ Evaluation failed - check $log_file for details"
        return 1
    fi
}

# Main evaluation loop with enhanced progress tracking
success_count=0
total_count=${#RESULT_FILES[@]}
start_time=$(date +%s)

echo "🏁 Starting enhanced evaluation pipeline..."
echo ""

for dataset_entry in "${RESULT_FILES[@]}"; do
    # Parse dataset entry
    IFS=':' read -r responses_file dataset_name <<< "$dataset_entry"
    
    echo "========================================"
    echo "--- Processing: $dataset_name ---"
    echo "========================================"
    
    # Check if responses file exists and is valid
    if ! check_responses_file "$responses_file" "$dataset_name"; then
        echo "   ⚠️ Skipping $dataset_name due to validation errors"
        continue
    fi
    
    # Skip evaluation if in check-only mode
    if [ "$CHECK_ONLY" = true ]; then
        echo "   ✅ $dataset_name validation passed (check-only mode)"
        ((success_count++))
        continue
    fi
    
    # Run enhanced evaluation
    dataset_start=$(date +%s)
    if run_single_evaluation "$responses_file" "$dataset_name"; then
        ((success_count++))
        dataset_end=$(date +%s)
        dataset_duration=$((dataset_end - dataset_start))
        echo "   ⏱️  Dataset evaluation completed in ${dataset_duration}s"
    else
        echo "   ❌ Dataset evaluation failed"
    fi
    
    echo ""
done

# Final comprehensive summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))
total_minutes=$((total_duration / 60))
total_seconds=$((total_duration % 60))

echo "============================================================"
echo "🏁 Enhanced Evaluation Pipeline Completed!"
echo "============================================================"
echo "⏱️  Total time: ${total_minutes}m ${total_seconds}s"
echo "📊 Evaluation Summary:"
echo "   Total datasets: $total_count"
echo "   Successful: $success_count"
echo "   Failed: $((total_count - success_count))"

if [ "$CHECK_ONLY" = true ]; then
    echo "   Mode: Validation only (no evaluation executed)"
fi

echo ""

if [ $success_count -eq $total_count ]; then
    echo "🎉 All evaluations completed successfully!"
    echo "📁 Results saved in: $RESULTS_DIR/"
    echo ""
    echo "📋 Generated files:"
    for dataset_entry in "${RESULT_FILES[@]}"; do
        IFS=':' read -r responses_file dataset_name <<< "$dataset_entry"
        metrics_file="$RESULTS_DIR/${dataset_name}_metric.json"
        log_file="$RESULTS_DIR/${dataset_name}_evaluation.log"
        
        if [ -f "$metrics_file" ]; then
            file_size=$(du -h "$metrics_file" | cut -f1)
            echo "   📊 $metrics_file ($file_size)"
        fi
        if [ -f "$log_file" ]; then
            file_size=$(du -h "$log_file" | cut -f1)
            echo "   📝 $log_file ($file_size)"
        fi
    done
    echo ""
    echo "📋 Next steps:"
    echo "   1. Check detailed results: cat $RESULTS_DIR/*_metric.json | jq ."
    echo "   2. Visualize samples: python eval/visualize_samples_pure_json.py"
    echo "   3. Run individual categories demo: python eval/demo_individual_categories.py"
    echo "   4. Test enhanced features: python eval/test_all_evaluations.py"
    exit 0
else
    echo "⚠️ Some evaluations failed. Check individual log files for details:"
    for dataset_entry in "${RESULT_FILES[@]}"; do
        IFS=':' read -r responses_file dataset_name <<< "$dataset_entry"
        log_file="$RESULTS_DIR/${dataset_name}_evaluation.log"
        if [ -f "$log_file" ]; then
            echo "   📝 $log_file"
        fi
    done
    exit 1
fi 