#!/bin/bash
# Evaluation script for Qwen2.5-VL model
# Updated to use latest COCO-style metrics with comprehensive evaluation options

# =============================================================================
# Configuration
# =============================================================================

# Model and data configuration
max_new_tokens=1024
validation_jsonl=eval_results/train_raw_responses_512.json
output_dir=eval_results

# Evaluation configuration
iou_threshold=0.5
semantic_threshold=0.7
evaluation_mode=comprehensive  # basic, coco, comprehensive

# File paths
response_file="${output_dir}/raw_responses_${max_new_tokens}.json"
basic_metrics_file="${output_dir}/basic_metrics_${max_new_tokens}.json"
coco_metrics_file="${output_dir}/coco_metrics_${max_new_tokens}.json"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

print_config() {
    echo "📊 Evaluation Configuration:"
    echo "   - Response file: ${response_file}"
    echo "   - Validation JSONL: ${validation_jsonl}"
    echo "   - Output directory: ${output_dir}"
    echo "   - Max new tokens: ${max_new_tokens}"
    echo "   - IoU threshold: ${iou_threshold}"
    echo "   - Semantic threshold: ${semantic_threshold}"
    echo "   - Evaluation mode: ${evaluation_mode}"
    echo ""
}

check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    # Check if response file exists
    if [[ ! -f "${response_file}" ]]; then
        echo "❌ Response file not found: ${response_file}"
        echo "💡 Please run inference first using infer_dataset.py"
        echo "   Example: python eval/infer_dataset.py --model_path output/checkpoint-XXX --validation_jsonl ${validation_jsonl} --output_file ${response_file}"
        exit 1
    fi
    
    # Check if validation file exists
    if [[ ! -f "${validation_jsonl}" ]]; then
        echo "❌ Validation file not found: ${validation_jsonl}"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "${output_dir}"
    
    echo "✅ Dependencies check passed"
    echo ""
}

run_basic_metrics() {
    print_header "Running Basic Metrics Evaluation"
    
    python eval/metrics.py \
        --responses_file "${response_file}" \
        --output_file "${basic_metrics_file}" \
        --iou_threshold "${iou_threshold}"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Basic metrics completed: ${basic_metrics_file}"
    else
        echo "❌ Basic metrics failed"
        return 1
    fi
}

run_coco_metrics() {
    print_header "Running COCO-Style Metrics Evaluation"
    
    # Check if semantic evaluation should be disabled
    if [[ "${DISABLE_SEMANTIC:-false}" == "true" ]]; then
        semantic_flag="--no_semantic"
        echo "⚠️  Semantic evaluation disabled"
    else
        semantic_flag=""
        echo "🧠 Semantic evaluation enabled (threshold: ${semantic_threshold})"
    fi
    
    python eval/coco_metrics.py \
        --responses_file "${response_file}" \
        --output_file "${coco_metrics_file}" \
        --semantic_threshold "${semantic_threshold}" \
        ${semantic_flag}
    
    if [[ $? -eq 0 ]]; then
        echo "✅ COCO-style metrics completed: ${coco_metrics_file}"
    else
        echo "❌ COCO-style metrics failed"
        return 1
    fi
}

display_results() {
    print_header "Evaluation Results Summary"
    
    echo "📁 Generated Files:"
    if [[ -f "${basic_metrics_file}" ]]; then
        echo "   ✅ Basic metrics: ${basic_metrics_file}"
    fi
    if [[ -f "${coco_metrics_file}" ]]; then
        echo "   ✅ COCO metrics: ${coco_metrics_file}"
    fi
    echo ""
    
    # Display basic metrics summary if available
    if [[ -f "${basic_metrics_file}" ]] && command -v jq &> /dev/null; then
        echo "📊 Basic Metrics Summary:"
        jq -r '.overall_metrics | "   - Precision: \(.precision // 0 | . * 100 | floor / 100)\n   - Recall: \(.recall // 0 | . * 100 | floor / 100)\n   - F1 Score: \(.f1 // 0 | . * 100 | floor / 100)"' "${basic_metrics_file}"
        echo ""
    fi
    
    # Display COCO metrics summary if available
    if [[ -f "${coco_metrics_file}" ]] && command -v jq &> /dev/null; then
        echo "🏆 COCO-Style Metrics Summary:"
        jq -r '.overall_metrics | "   - mAP (0.5:0.95): \(.mAP // 0 | . * 100 | floor / 100)\n   - AP@0.5: \(.["AP@0.5"] // 0 | . * 100 | floor / 100)\n   - AP@0.75: \(.["AP@0.75"] // 0 | . * 100 | floor / 100)\n   - mAR: \(.mAR // 0 | . * 100 | floor / 100)"' "${coco_metrics_file}"
        echo ""
    fi
    
    echo "💡 For detailed analysis, examine the JSON files or use visualization tools."
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    print_header "Qwen2.5-VL Model Evaluation Pipeline"
    print_config
    check_dependencies
    
    case "${evaluation_mode}" in
        "basic")
            echo "🎯 Running basic evaluation only..."
            run_basic_metrics
            ;;
        "coco")
            echo "🎯 Running COCO-style evaluation only..."
            run_coco_metrics
            ;;
        "comprehensive"|*)
            echo "🎯 Running comprehensive evaluation (basic + COCO)..."
            run_basic_metrics
            echo ""
            run_coco_metrics
            ;;
    esac
    
    echo ""
    display_results
    
    print_header "Evaluation Pipeline Completed"
    echo "🎉 All evaluations completed successfully!"
    echo "📁 Results saved in: ${output_dir}"
}

# =============================================================================
# Usage Information
# =============================================================================

show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Environment Variables:"
    echo "  MAX_NEW_TOKENS=2048          Maximum tokens for generation (default: 2048)"
    echo "  VALIDATION_JSONL=path        Path to validation JSONL file"
    echo "  OUTPUT_DIR=eval_results      Output directory for results"
    echo "  IOU_THRESHOLD=0.5            IoU threshold for detection matching"
    echo "  SEMANTIC_THRESHOLD=0.7       Semantic similarity threshold"
    echo "  EVALUATION_MODE=comprehensive Evaluation mode: basic|coco|comprehensive"
    echo "  DISABLE_SEMANTIC=false       Set to 'true' to disable semantic evaluation"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  bash eval/eval.sh"
    echo ""
    echo "  # Custom configuration"
    echo "  MAX_NEW_TOKENS=1024 IOU_THRESHOLD=0.7 bash eval/eval.sh"
    echo ""
    echo "  # COCO-style evaluation only"
    echo "  EVALUATION_MODE=coco bash eval/eval.sh"
    echo ""
    echo "  # Disable semantic evaluation"
    echo "  DISABLE_SEMANTIC=true bash eval/eval.sh"
    echo ""
    echo "Prerequisites:"
    echo "  1. Raw responses file must exist (generated by infer_dataset.py)"
    echo "  2. Validation JSONL file must be available"
    echo ""
}

# Handle help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"