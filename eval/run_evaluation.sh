#!/bin/bash
# Simple evaluation pipeline - just set EXP_NAME and run
set -e

###############################################################################
# EVALUATION CONFIGURATION - ONLY EDIT THIS SECTION
###############################################################################

# Experiment name (must match inference experiment)
EXP_NAME="baseline_no_teacher"          # Must match the EXP_NAME from inference

# Evaluation parameters
IOU_THRESHOLD=0.3
SEMANTIC_THRESHOLD=0.7
ENABLE_SOFT_MATCHING=true
ENABLE_HIERARCHICAL=true
ENABLE_NOVEL_DETECTION=true
MINIMAL_METRICS=true

# Logging level (debug shows detailed validation issues)
LOG_LEVEL="debug"                       # "debug" for detailed validation info, "info" for normal

###############################################################################
# AUTO-CONFIGURATION - DO NOT EDIT
###############################################################################

# Environment
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Find latest experiment with matching name
OUTPUT_BASE="experiments"
EXPERIMENT_DIR=$(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "${EXP_NAME}_*" | sort | tail -1)

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "❌ No experiments found matching: ${EXP_NAME}_*"
    echo ""
    echo "Available experiments:"
    ls -1 "$OUTPUT_BASE" 2>/dev/null | grep -E "^[^.]*_[0-9]{8}_[0-9]{6}$" || echo "  (none found)"
    echo ""
    echo "💡 Make sure you:"
    echo "   1. Set the correct EXP_NAME (must match inference experiment)"
    echo "   2. Run inference first: ./eval/infer_dataset.sh"
    exit 1
fi

EXPERIMENT_ID=$(basename "$EXPERIMENT_DIR")
INFERENCE_DIR="${EXPERIMENT_DIR}/inference"
EVALUATION_DIR="${EXPERIMENT_DIR}/evaluation"

echo "🎯 Starting evaluation for experiment: ${EXP_NAME}"
echo "📁 Experiment ID: ${EXPERIMENT_ID}"
echo "📊 Inference results: ${INFERENCE_DIR}"
echo "📈 Evaluation output: ${EVALUATION_DIR}"
echo ""

# Validate inference directory exists
if [ ! -d "$INFERENCE_DIR" ]; then
    echo "❌ Inference directory not found: $INFERENCE_DIR"
    echo "   Run inference first: ./eval/infer_dataset.sh"
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
OUTPUT_SUFFIX=$(python -c "
import json
try:
    with open('$CONFIG_FILE') as f:
        config = json.load(f)
    print(config['output_suffix'])
except Exception as e:
    print(f'Error: {e}', file=__import__('sys').stderr)
    exit(1)
" 2>/dev/null)

if [ -z "$OUTPUT_SUFFIX" ]; then
    echo "❌ Could not extract output suffix from config"
    exit 1
fi

echo "🔧 Output suffix: ${OUTPUT_SUFFIX}"

# Find all inference result files
INFERENCE_FILES=($(find "$INFERENCE_DIR" -name "*_${OUTPUT_SUFFIX}.json" -type f))

if [ ${#INFERENCE_FILES[@]} -eq 0 ]; then
    echo "❌ No inference result files found matching: *_${OUTPUT_SUFFIX}.json"
    echo ""
    echo "Available files in ${INFERENCE_DIR}:"
    ls -1 "$INFERENCE_DIR" 2>/dev/null || echo "  (empty)"
    exit 1
fi

echo "📊 Found ${#INFERENCE_FILES[@]} inference result files to evaluate"
echo ""

# Process each inference result file
successful_count=0
failed_datasets=()

for inference_file in "${INFERENCE_FILES[@]}"; do
    # Extract dataset name from filename
    filename=$(basename "$inference_file")
    dataset_name="${filename%_${OUTPUT_SUFFIX}.json}"
    
    evaluation_file="${EVALUATION_DIR}/${dataset_name}_${OUTPUT_SUFFIX}_metrics.json"
    log_file="${EVALUATION_DIR}/${dataset_name}_evaluation.log"
    
    echo "=== Evaluating: ${dataset_name} ==="
    echo "Input: ${inference_file}"
    echo "Output: ${evaluation_file}"
    
    # Build evaluation command
    CMD="python eval/eval_dataset.py \
        --responses_file \"${inference_file}\" \
        --output_file \"${evaluation_file}\" \
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
    
    # Run evaluation with detailed logging
    if bash -c "$CMD" 2>&1 | tee "$log_file"; then
        if [ -f "$evaluation_file" ]; then
            # Extract and display key metrics
            metrics=$(python -c "
import json
try:
    with open('$evaluation_file') as f:
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
            echo "✅ Evaluation completed for ${dataset_name}:"
            echo "   ${metrics}"
            ((successful_count++))
            
            # Show validation issues if any (from log)
            validation_issues=$(grep -c "Skipped.*invalid" "$log_file" 2>/dev/null || echo "0")
            if [ "$validation_issues" -gt 0 ]; then
                echo "   ⚠️  Found ${validation_issues} validation warnings in log"
                echo "   📋 Check detailed log: ${log_file}"
            fi
        else
            echo "❌ Evaluation file not created for ${dataset_name}"
            failed_datasets+=("${dataset_name}")
        fi
    else
        echo "❌ Evaluation failed for ${dataset_name}"
        failed_datasets+=("${dataset_name}")
    fi
    echo ""
done

# Generate summary report
SUMMARY_FILE="${EVALUATION_DIR}/evaluation_summary.json"
echo "📋 Generating evaluation summary..."

python -c "
import json
import os
from glob import glob

evaluation_dir = '$EVALUATION_DIR'
output_suffix = '$OUTPUT_SUFFIX'
exp_name = '$EXP_NAME'

# Collect all metrics
results = {}
validation_summary = {}

for metrics_file in glob(os.path.join(evaluation_dir, f'*_{output_suffix}_metrics.json')):
    dataset_name = os.path.basename(metrics_file).replace(f'_{output_suffix}_metrics.json', '')
    try:
        with open(metrics_file) as f:
            data = json.load(f)
        results[dataset_name] = data.get('overall_metrics', {})
        
        # Collect validation info
        eval_info = data.get('evaluation_info', {})
        validation_summary[dataset_name] = {
            'total_samples': eval_info.get('total_samples', 0),
            'valid_samples': eval_info.get('valid_samples', 0),
            'skipped_samples': eval_info.get('skipped_samples', 0)
        }
    except Exception as e:
        results[dataset_name] = {'error': str(e)}
        validation_summary[dataset_name] = {'error': str(e)}

# Calculate aggregated metrics
if results:
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        # Calculate means across datasets
        all_maps = [v.get('mAP', 0) for v in valid_results.values()]
        all_mars = [v.get('mAR', 0) for v in valid_results.values()]
        all_f1s = [v.get('mF1', 0) for v in valid_results.values()]
        
        aggregated = {
            'mean_mAP': sum(all_maps) / len(all_maps) if all_maps else 0,
            'mean_mAR': sum(all_mars) / len(all_mars) if all_mars else 0,
            'mean_mF1': sum(all_f1s) / len(all_f1s) if all_f1s else 0,
            'datasets_evaluated': len(valid_results),
            'datasets_failed': len(results) - len(valid_results)
        }
    else:
        aggregated = {'error': 'No valid results found'}
else:
    aggregated = {'error': 'No results found'}

# Aggregate validation info
total_samples = sum(v.get('total_samples', 0) for v in validation_summary.values() if 'error' not in v)
total_valid = sum(v.get('valid_samples', 0) for v in validation_summary.values() if 'error' not in v)
total_skipped = sum(v.get('skipped_samples', 0) for v in validation_summary.values() if 'error' not in v)

summary = {
    'exp_name': exp_name,
    'experiment_id': '$EXPERIMENT_ID',
    'evaluation_timestamp': '$(date -Iseconds)',
    'aggregated_metrics': aggregated,
    'validation_summary': {
        'total_samples': total_samples,
        'valid_samples': total_valid,
        'skipped_samples': total_skipped,
        'per_dataset': validation_summary
    },
    'per_dataset_metrics': results,
    'evaluation_config': {
        'iou_threshold': $IOU_THRESHOLD,
        'semantic_threshold': $SEMANTIC_THRESHOLD,
        'enable_soft_matching': $ENABLE_SOFT_MATCHING,
        'enable_hierarchical': $ENABLE_HIERARCHICAL,
        'enable_novel_detection': $ENABLE_NOVEL_DETECTION,
        'minimal_metrics': $MINIMAL_METRICS,
        'log_level': '$LOG_LEVEL'
    }
}

with open('$SUMMARY_FILE', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print()
print('📊 EVALUATION SUMMARY FOR: $EXP_NAME')
print('=' * 60)
if 'error' not in aggregated:
    print(f'Mean mAP: {aggregated[\"mean_mAP\"]:.4f}')
    print(f'Mean mAR: {aggregated[\"mean_mAR\"]:.4f}')
    print(f'Mean mF1: {aggregated[\"mean_mF1\"]:.4f}')
    print(f'Datasets evaluated: {aggregated[\"datasets_evaluated\"]}')
    if aggregated[\"datasets_failed\"] > 0:
        print(f'Datasets failed: {aggregated[\"datasets_failed\"]}')
else:
    print(f'Error: {aggregated[\"error\"]}')

print()
print('📋 VALIDATION SUMMARY:')
print(f'Total samples processed: {total_samples}')
print(f'Valid samples: {total_valid}')
if total_skipped > 0:
    print(f'⚠️  Skipped samples: {total_skipped} ({total_skipped/total_samples*100:.1f}%)')
    print('   Check evaluation logs for details on skipped samples')
else:
    print('✅ No validation issues found')
"

# Final summary
echo ""
echo "🏁 Evaluation completed for experiment: ${EXP_NAME}"
echo "✅ Successful: ${successful_count}/${#INFERENCE_FILES[@]} datasets"
if [ ${#failed_datasets[@]} -gt 0 ]; then
    echo "❌ Failed: ${failed_datasets[*]}"
fi
echo "📁 Results: ${EVALUATION_DIR}"
echo "📋 Summary: ${SUMMARY_FILE}"
echo ""
echo "🎯 Next step: Compare experiments"
echo "   Run: python eval/compare_experiments.py"