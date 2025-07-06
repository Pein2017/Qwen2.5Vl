#!/bin/bash
# Convenient wrapper for running complete experiments
set -e

###############################################################################
# USAGE EXAMPLES
###############################################################################
# Run inference and evaluation for a specific experiment and dataset:
#   bash eval/run_experiment.sh 1_teacher val
#   bash eval/run_experiment.sh no_teacher train
#
# Run for all datasets:
#   bash eval/run_experiment.sh 1_teacher all
###############################################################################

if [ $# -lt 2 ]; then
    echo "Usage: $0 <exp_name> <dataset>"
    echo ""
    echo "Examples:"
    echo "  $0 1_teacher val       # Run inference + evaluation for val dataset"
    echo "  $0 no_teacher train    # Run inference + evaluation for train dataset"  
    echo "  $0 1_teacher all       # Run inference + evaluation for both datasets"
    echo ""
    echo "Available datasets: train, val, all"
    exit 1
fi

EXP_NAME="$1"
DATASET="$2"

echo "🚀 Running complete experiment pipeline"
echo "   Experiment: $EXP_NAME"
echo "   Dataset: $DATASET"
echo ""

# Function to run single dataset
run_single_dataset() {
    local dataset="$1"
    
    echo "=== Processing dataset: $dataset ==="
    
    # Run inference
    echo "🔄 Step 1: Running inference..."
    EXP_NAME="$EXP_NAME" DATASET="$dataset" bash eval/infer_dataset_new.sh
    
    if [ $? -eq 0 ]; then
        echo "✅ Inference completed successfully"
        echo ""
        
        # Run evaluation
        echo "🔄 Step 2: Running evaluation..."
        EXP_NAME="$EXP_NAME" DATASET="$dataset" bash eval/run_evaluation_new.sh
        
        if [ $? -eq 0 ]; then
            echo "✅ Evaluation completed successfully"
        else
            echo "❌ Evaluation failed"
            return 1
        fi
    else
        echo "❌ Inference failed"
        return 1
    fi
    
    echo ""
}

# Run based on dataset parameter
if [ "$DATASET" = "all" ]; then
    echo "🔄 Running for all datasets..."
    echo ""
    
    run_single_dataset "train"
    run_single_dataset "val"
    
    echo "🏁 Complete experiment finished for: $EXP_NAME"
    echo "📁 Results in: experiments/$EXP_NAME/"
    
elif [ "$DATASET" = "train" ] || [ "$DATASET" = "val" ]; then
    run_single_dataset "$DATASET"
    
    echo "🏁 Experiment finished for: $EXP_NAME, dataset: $DATASET"
    echo "📁 Results in: experiments/$EXP_NAME/$DATASET/"
    
else
    echo "❌ Invalid dataset: $DATASET"
    echo "Valid options: train, val, all"
    exit 1
fi

echo ""
echo "🎯 Next steps:"
echo "   1. Compare experiments: python eval/compare_experiments.py"
echo "   2. View results: ls -la experiments/$EXP_NAME/"