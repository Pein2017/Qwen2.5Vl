#!/bin/bash

# Simplified Data Conversion Pipeline for Qwen2.5-VL
# 
# This script provides a clean, single-step conversion from raw JSON/images
# directly to train.jsonl, val.jsonl, and teacher.jsonl files.
# 
# Replaces the complex multi-step pipeline with a unified modular processor.

set -e

# Set proper locale for UTF-8 handling
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

# Environment setup
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export MODELSCOPE_CACHE="/data4/swift/modelscope/hub"

# ============================================================================
# Configuration Parameters
# ============================================================================

# Data paths
INPUT_DIR="ds"                          # Raw data directory
OUTPUT_DIR="data"                       # Output directory for JSONL files
OUTPUT_IMAGE_DIR="ds_rescaled"          # Output for processed images

# Token map files
TOKEN_MAP_EN="data_conversion/token_map.json"       # English token map
TOKEN_MAP_ZH="data_conversion/token_map_zh.json"    # Chinese token map

# Processing parameters
LANGUAGE="chinese"                      # "english" or "chinese"
RESPONSE_TYPES="object_type property"   # Space-separated list
RESIZE=true                            # Enable image resizing
VAL_RATIO=0.1                          # Validation split ratio
MAX_TEACHERS=10                        # Maximum teacher samples
SEED=42                                # Random seed for reproducibility
LOG_LEVEL="INFO"                       # Logging level

# Support files
HIERARCHY_FILE="data_conversion/label_hierarchy.json"

echo "🚀 Starting Simplified Data Conversion Pipeline"
echo "=============================================="
echo "Language: $LANGUAGE"
echo "Response Types: $RESPONSE_TYPES"
echo "Input: $INPUT_DIR → Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# Build Command Arguments
# ============================================================================

# Base command
CMD="python data_conversion/processor.py \
    --input_dir \"$INPUT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --language \"$LANGUAGE\" \
    --response_types $RESPONSE_TYPES \
    --val_ratio $VAL_RATIO \
    --max_teachers $MAX_TEACHERS \
    --seed $SEED \
    --log_level $LOG_LEVEL"

# Add image processing options
if [[ "$RESIZE" == "true" ]]; then
    CMD="$CMD --resize --output_image_dir \"$OUTPUT_IMAGE_DIR\""
fi

# Add token map based on language
if [[ "$LANGUAGE" == "english" ]]; then
    if [[ -f "$TOKEN_MAP_EN" ]]; then
        CMD="$CMD --token_map_path \"$TOKEN_MAP_EN\""
    else
        echo "⚠️  Warning: English token map not found at $TOKEN_MAP_EN"
    fi
elif [[ "$LANGUAGE" == "chinese" ]]; then
    if [[ -f "$TOKEN_MAP_ZH" ]]; then
        CMD="$CMD --token_map_path \"$TOKEN_MAP_ZH\""
    fi
fi

# Add label hierarchy if available
if [[ -f "$HIERARCHY_FILE" ]]; then
    CMD="$CMD --hierarchy_path \"$HIERARCHY_FILE\""
fi

# ============================================================================
# Execute Pipeline
# ============================================================================

echo "🔄 Running unified data processor..."
echo "Command: $CMD"
echo ""

eval $CMD

# ============================================================================
# Validation
# ============================================================================

echo ""
echo "🔍 Validating output files..."

# Check if output files exist
TRAIN_FILE="$OUTPUT_DIR/train.jsonl"
VAL_FILE="$OUTPUT_DIR/val.jsonl"
TEACHER_FILE="$OUTPUT_DIR/teacher.jsonl"
COMBINED_FILE="$OUTPUT_DIR/all_samples.jsonl"

if [[ -f "$TRAIN_FILE" && -f "$VAL_FILE" && -f "$TEACHER_FILE" ]]; then
    echo "✅ All output files created successfully"
    
    # Create combined file with all samples
    echo "🔗 Creating combined file with all samples..."
    cat "$TEACHER_FILE" "$TRAIN_FILE" "$VAL_FILE" > "$COMBINED_FILE"
    
    # Run validation if script exists
    if [[ -f "data_conversion/simple_validate.py" ]]; then
        python data_conversion/simple_validate.py "$TRAIN_FILE" "$VAL_FILE" "$TEACHER_FILE"
    fi
    
    # Count samples in each file
    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    VAL_COUNT=$(wc -l < "$VAL_FILE")
    TEACHER_COUNT=$(wc -l < "$TEACHER_FILE")
    COMBINED_COUNT=$(wc -l < "$COMBINED_FILE")
    
    echo ""
    echo "📊 Final Summary:"
    echo "   Training samples: $TRAIN_COUNT → $TRAIN_FILE"
    echo "   Validation samples: $VAL_COUNT → $VAL_FILE"
    echo "   Teacher samples: $TEACHER_COUNT → $TEACHER_FILE"
    echo "   Combined samples: $COMBINED_COUNT → $COMBINED_FILE"
    echo ""
    echo "🎉 Pipeline completed successfully!"
    echo "🚀 Ready for training!"
else
    echo "❌ Error: Some output files are missing"
    exit 1
fi 