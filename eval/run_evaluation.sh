#!/bin/bash
# Clean evaluation pipeline with improved directory structure
set -e

###############################################################################
# EVALUATION CONFIGURATION - EDIT THIS SECTION
###############################################################################

# Experiment name (must match inference experiment)
EXP_NAME="det_coordinates"          # Must match the EXP_NAME from inference

# Dataset to evaluate (single dataset per run)
DATASET="train"

# Evaluation parameters
IOU_THRESHOLD=0.3
SEMANTIC_THRESHOLD=0.7
ENABLE_SOFT_MATCHING=true
ENABLE_HIERARCHICAL=true
ENABLE_NOVEL_DETECTION=true
MINIMAL_METRICS=true

# Detailed analysis parameters
ENABLE_DETAILED_ANALYSIS=true

# Logging level (debug shows detailed validation issues)
LOG_LEVEL="debug"

###############################################################################
# PIPELINE EXECUTION - DO NOT EDIT BELOW
###############################################################################

# Environment
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Validate dataset parameter
if [[ "$DATASET" != "train" && "$DATASET" != "val" ]]; then
    echo "❌ DATASET must be 'train' or 'val', got: $DATASET"
    exit 1
fi

# Build paths
OUTPUT_BASE="experiments"
EXPERIMENT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
DATASET_DIR="${EXPERIMENT_DIR}/${DATASET}"
INFERENCE_DIR="${DATASET_DIR}/inference"
EVALUATION_DIR="${DATASET_DIR}/evaluation"

echo "🎯 Starting evaluation for experiment: ${EXP_NAME}"
echo "📊 Dataset: ${DATASET}"
echo "📁 Experiment directory: ${EXPERIMENT_DIR}"
echo "📊 Inference results: ${INFERENCE_DIR}"
echo "📈 Evaluation output: ${EVALUATION_DIR}"
echo ""

# Validate experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Experiment directory not found: $EXPERIMENT_DIR"
    echo ""
    echo "Available experiments:"
    ls -1 "$OUTPUT_BASE" 2>/dev/null || echo "  (none found)"
    echo ""
    echo "💡 Make sure you:"
    echo "   1. Set the correct EXP_NAME"
    echo "   2. Run inference first: bash eval/infer_dataset_new.sh"
    exit 1
fi

# Validate inference directory exists
if [ ! -d "$INFERENCE_DIR" ]; then
    echo "❌ Inference directory not found: $INFERENCE_DIR"
    echo "   Run inference first for $DATASET dataset:"
    echo "   DATASET=\"$DATASET\" EXP_NAME=\"$EXP_NAME\" bash eval/infer_dataset_new.sh"
    exit 1
fi

# Find inference results file
INFERENCE_FILE="${INFERENCE_DIR}/predictions.json"

if [ ! -f "$INFERENCE_FILE" ]; then
    echo "❌ Inference results not found: $INFERENCE_FILE"
    echo ""
    echo "Available files in ${INFERENCE_DIR}:"
    ls -1 "$INFERENCE_DIR" 2>/dev/null || echo "  (empty)"
    echo ""
    echo "💡 Run inference first:"
    echo "   DATASET=\"$DATASET\" EXP_NAME=\"$EXP_NAME\" bash eval/infer_dataset_new.sh"
    exit 1
fi

# Create evaluation directory
mkdir -p "$EVALUATION_DIR"

# Load experiment configuration
CONFIG_FILE="${EXPERIMENT_DIR}/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Experiment configuration not found: $CONFIG_FILE"
    exit 1
fi

echo "📋 Loading experiment configuration..."

# Set output paths
EVALUATION_FILE="${EVALUATION_DIR}/metrics.json"
LOG_FILE="${EVALUATION_DIR}/evaluation.log"

echo "📊 Evaluating: ${DATASET}"
echo "Input: ${INFERENCE_FILE}"
echo "Output: ${EVALUATION_FILE}"
echo ""

# Build evaluation command
CMD="python eval/eval_dataset.py \
    --responses_file \"${INFERENCE_FILE}\" \
    --output_file \"${EVALUATION_FILE}\" \
    --iou_threshold ${IOU_THRESHOLD} \
    --semantic_threshold ${SEMANTIC_THRESHOLD} \
    --log_level \"${LOG_LEVEL}\""

# Add evaluation flags
if [ "$MINIMAL_METRICS" = true ]; then
    CMD="${CMD} --minimal"
fi

if [ "$ENABLE_SOFT_MATCHING" = true ]; then
    CMD="${CMD} --enable_soft_matching"
else
    CMD="${CMD} --disable_soft_matching"
fi

if [ "$ENABLE_HIERARCHICAL" = true ]; then
    CMD="${CMD} --enable_hierarchical"
else
    CMD="${CMD} --disable_hierarchical"
fi

if [ "$ENABLE_NOVEL_DETECTION" = true ]; then
    CMD="${CMD} --enable_novel_detection"
else
    CMD="${CMD} --disable_novel_detection"
fi

echo "🚀 Running evaluation..."
echo "Command: $CMD"
echo ""

# Run evaluation with detailed logging
if bash -c "$CMD" 2>&1 | tee "$LOG_FILE"; then
    if [ -f "$EVALUATION_FILE" ]; then
        # Extract and display key metrics
        metrics=$(python -c "
import json
try:
    with open('$EVALUATION_FILE') as f:
        data = json.load(f)
    overall = data.get('overall_metrics', {})
    info = data.get('evaluation_info', {})
    
    # Show metrics
    print(f'mAP: {overall.get(\"mAP\", 0):.4f}, mAR: {overall.get(\"mAR\", 0):.4f}, mF1: {overall.get(\"mF1\", 0):.4f}')
    
    # Show validation summary
    total = info.get('total_samples', 0)
    valid = info.get('valid_samples', 0)
    skipped = info.get('skipped_samples', 0)
    if skipped > 0:
        print(f'⚠️  Validation: {valid}/{total} valid samples ({skipped} skipped)')
    else:
        print(f'✅ Validation: {valid}/{total} samples processed')
        
except Exception as e:
    print(f'Unable to parse metrics: {e}')
" 2>/dev/null)
        
        echo "✅ Evaluation completed for ${DATASET}:"
        echo "   ${metrics}"
        
        # Show validation issues if any (from log)
        validation_issues=$(grep -c "Skipped.*invalid" "$LOG_FILE" 2>/dev/null || echo "0")
        if [ "$validation_issues" -gt 0 ]; then
            echo "   ⚠️  Found ${validation_issues} validation warnings in log"
            echo "   📋 Check detailed log: ${LOG_FILE}"
        fi
        
        echo ""
        echo "📁 Results saved to: ${EVALUATION_FILE}"
        echo "📋 Log saved to: ${LOG_FILE}"
        
        # Run detailed analysis if enabled
        if [ "$ENABLE_DETAILED_ANALYSIS" = true ]; then
            echo ""
            echo "🔍 Running detailed performance analysis..."
            
            DETAILED_ANALYSIS_DIR="${EVALUATION_DIR}/detailed_analysis"
            DETAILED_CMD="python eval/detailed_analysis.py \
                --predictions_file \"${INFERENCE_FILE}\" \
                --output_dir \"${DETAILED_ANALYSIS_DIR}\" \
                --label_vocab \"data/label_vocabulary.json\" \
                --log_level \"${LOG_LEVEL}\""
            
            if bash -c "$DETAILED_CMD" 2>&1 | tee -a "$LOG_FILE"; then
                echo "✅ Detailed analysis completed"
                echo "📊 Human-readable report: ${DETAILED_ANALYSIS_DIR}/analysis_summary.txt"
                echo "📈 Visual analysis: ${DETAILED_ANALYSIS_DIR}/visual_analysis.txt"
                echo "📋 Raw data: ${DETAILED_ANALYSIS_DIR}/detailed_analysis.json"
                echo "📊 Confusion matrix: ${DETAILED_ANALYSIS_DIR}/confusion_matrix.csv"
            else
                echo "⚠️  Detailed analysis failed (evaluation results still available)"
            fi
        fi
        
    else
        echo "❌ Evaluation file not created for ${DATASET}"
        exit 1
    fi
else
    echo "❌ Evaluation failed for ${DATASET}"
    exit 1
fi

echo ""
echo "🏁 Evaluation completed for experiment: ${EXP_NAME}, dataset: ${DATASET}"
echo "📁 All results in: ${DATASET_DIR}"
echo ""
echo "🎯 Next steps:"
echo "   1. Review detailed analysis: ${DATASET_DIR}/evaluation/detailed_analysis/analysis_summary.txt"
echo "   2. Compare experiments: python eval/compare_experiments.py"
echo "   3. Evaluate other datasets by changing DATASET parameter"