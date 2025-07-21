#!/bin/bash

# Data Conversion Pipeline - Manual Configuration
# 
# Processes raw dataset directories into training-ready format:
# /data/{dataset_name}/
#   ├── images/*.jpeg        # Smart-resized images  
#   ├── all_samples.jsonl    # All processed samples
#   ├── train.jsonl          # Training split
#   ├── val.jsonl            # Validation split  
#   ├── teacher.jsonl        # Teacher samples
#   └── label_vocabulary.json # Statistics
#
# REQUIRED: Set all configuration variables below before running!

set -e

# Set proper locale for UTF-8 handling
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

# Environment setup
export PYTHONPATH=/data3/data_conversion:$PYTHONPATH
export MODELSCOPE_CACHE="/data3/Qwen2.5-VL-main/modelscope/hub"

# ============================================================================
# MANUAL CONFIGURATION - EDIT THESE VALUES BEFORE RUNNING
# ============================================================================

# Required paths - YOU MUST SET THESE
INPUT_DIR="ds_v2"                    # e.g., "ds_v2" or "my_dataset"
OUTPUT_DIR="data"                   # e.g., "data" or "/path/to/output"
DATASET_NAME="ds_v2"                 # e.g., "experiment_1" or leave empty to auto-detect

# Optional configuration files - SET THESE IF YOU HAVE THEM
HIERARCHY_FILE=""               # e.g., "data_conversion/label_hierarchy.json" or leave empty

# Processing parameters - YOU MUST SET THESE
VAL_RATIO="0.1"                    # e.g., "0.1" for 10% validation split
MAX_TEACHERS="10"                 # e.g., "10" for max teacher samples
RESIZE="true"                       # "true" or "false" for image resizing
RESPONSE_TYPES="object_type property extra_info"               # e.g., "object_type property" (space-separated)

# Optional settings
LOG_LEVEL="INFO"                    # e.g., "INFO", "DEBUG", "WARNING", "ERROR" or leave empty
SEED="17"                         # e.g., "17" or leave empty

# ============================================================================
# SIMPLIFIED VALIDATION - Python config handles detailed validation
# ============================================================================

echo "🔍 Basic configuration check..."

# Only check critical path existence - Python handles the rest
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Auto-detect dataset name if not provided
if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME=$(basename "$INPUT_DIR")
    echo "🔄 Auto-detected dataset name: $DATASET_NAME"
fi

echo "✅ Basic validation passed - Python will handle detailed validation"

# ============================================================================
# PROCESSING
# ============================================================================

echo ""
echo "🚀 Starting Data Conversion Pipeline"
echo "======================================"
echo "📋 Configuration:"
echo "   Input Dir: $INPUT_DIR"
echo "   Output Dir: $OUTPUT_DIR"
echo "   Dataset Name: $DATASET_NAME"
echo "   Language: Chinese (default)"
echo "   Hierarchy File: ${HIERARCHY_FILE:-'(not set)'}"
echo "   Val Ratio: $VAL_RATIO"
echo "   Max Teachers: $MAX_TEACHERS"
echo "   Smart Resize: $RESIZE"
echo "   Response Types: $RESPONSE_TYPES"
echo "   Log Level: $LOG_LEVEL"
echo "   Seed: $SEED"
echo ""

# Build command arguments
PYTHON_CMD="/root/miniconda3/envs/ms/bin/python data_conversion/processor.py"
ARGS="--input_dir \"$INPUT_DIR\""
ARGS="$ARGS --output_dir \"$OUTPUT_DIR\""
ARGS="$ARGS --dataset_name \"$DATASET_NAME\""
ARGS="$ARGS --val_ratio \"$VAL_RATIO\""
ARGS="$ARGS --max_teachers \"$MAX_TEACHERS\""
ARGS="$ARGS --seed \"$SEED\""
ARGS="$ARGS --log_level \"$LOG_LEVEL\""
ARGS="$ARGS --response_types $RESPONSE_TYPES"

# Add optional arguments if provided
if [ -n "$HIERARCHY_FILE" ]; then
    ARGS="$ARGS --hierarchy_path \"$HIERARCHY_FILE\""
fi


if [ "$RESIZE" = "true" ]; then
    ARGS="$ARGS --resize"
fi

echo "🔄 Processing dataset: $DATASET_NAME ($INPUT_DIR)"
echo "  └─ Executing: $PYTHON_CMD"

# Execute the command
eval "$PYTHON_CMD $ARGS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset $DATASET_NAME processed successfully!"
    echo "📁 Output: $OUTPUT_DIR/$DATASET_NAME/"
    echo "🚀 Ready for training!"
else
    echo ""
    echo "❌ Dataset $DATASET_NAME processing failed"
    exit 1
fi