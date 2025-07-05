#!/bin/bash

# Unified Data Conversion Pipeline Entry Point
# 
# This script provides backward compatibility while routing to the new
# Python-based pipeline manager for better error handling and functionality.

set -e

# Set proper locale for UTF-8 handling
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

# Environment setup
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export MODELSCOPE_CACHE="/data4/swift/modelscope/hub"

echo "🚀 Starting Unified Data Conversion Pipeline"
echo "=============================================="

# Build arguments from environment variables (backward compatibility)
ARGS=""

# Map environment variables to command line arguments
if [ ! -z "$INPUT_DIR" ]; then
    ARGS="$ARGS --input_dir $INPUT_DIR"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
    ARGS="$ARGS --output_dir $OUTPUT_DIR"
fi

if [ ! -z "$OUTPUT_IMAGE_DIR" ]; then
    ARGS="$ARGS --output_image_dir $OUTPUT_IMAGE_DIR"
fi

if [ ! -z "$LANGUAGE" ]; then
    ARGS="$ARGS --language $LANGUAGE"
fi

if [ ! -z "$RESPONSE_TYPES" ]; then
    ARGS="$ARGS --response_types $RESPONSE_TYPES"
fi

if [ "$RESIZE" = "true" ]; then
    ARGS="$ARGS --resize"
fi

if [ ! -z "$VAL_RATIO" ]; then
    ARGS="$ARGS --val_ratio $VAL_RATIO"
fi

if [ ! -z "$MAX_TEACHERS" ]; then
    ARGS="$ARGS --max_teachers $MAX_TEACHERS"
fi

if [ ! -z "$SEED" ]; then
    ARGS="$ARGS --seed $SEED"
fi

if [ ! -z "$TOKEN_MAP_EN" ] || [ ! -z "$TOKEN_MAP_ZH" ]; then
    if [ "$LANGUAGE" = "english" ] && [ ! -z "$TOKEN_MAP_EN" ]; then
        ARGS="$ARGS --token_map_path $TOKEN_MAP_EN"
    elif [ "$LANGUAGE" = "chinese" ] && [ ! -z "$TOKEN_MAP_ZH" ]; then
        ARGS="$ARGS --token_map_path $TOKEN_MAP_ZH"
    fi
fi

if [ ! -z "$HIERARCHY_FILE" ]; then
    ARGS="$ARGS --hierarchy_path $HIERARCHY_FILE"
fi

if [ ! -z "$LOG_LEVEL" ]; then
    ARGS="$ARGS --log_level $LOG_LEVEL"
fi

# Execute the Python pipeline manager
echo "🔄 Executing pipeline manager with args: $ARGS"
python data_conversion/pipeline_manager.py $ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pipeline completed successfully!"
    echo "🚀 Ready for training!"
else
    echo ""
    echo "❌ Pipeline failed - check logs for details"
    exit 1
fi