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
OUTPUT_IMAGE_DIR="ds_output"            # Output for processed images and cleaned JSON
CLEANED_JSON_DIR="ds_output"            # Output directory for cleaned JSON files (same as images)

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
echo "Cleaned JSON/Images: $OUTPUT_IMAGE_DIR"
echo ""

# ============================================================================
# Step 1: Clean Raw JSON Files
# ============================================================================

echo "🧹 Step 1: Cleaning raw JSON annotation files..."
echo "Input: $INPUT_DIR → Cleaned JSON: $CLEANED_JSON_DIR"

# Determine language parameter for clean_raw_json.py
if [[ "$LANGUAGE" == "english" ]]; then
    JSON_LANG="en"
elif [[ "$LANGUAGE" == "chinese" ]]; then
    JSON_LANG="zh"
else
    JSON_LANG="both"
fi

# Clean the raw JSON files
python data_conversion/clean_raw_json.py "$INPUT_DIR" "$CLEANED_JSON_DIR" --lang "$JSON_LANG"

# Apply token mapping to cleaned JSON files
if [[ "$LANGUAGE" == "chinese" && -f "$TOKEN_MAP_ZH" ]]; then
    echo "🔄 Applying token mapping to cleaned JSON files..."
    python data_conversion/apply_token_mapping.py "$CLEANED_JSON_DIR" "$TOKEN_MAP_ZH"
fi

if [[ $? -ne 0 ]]; then
    echo "❌ Error: JSON cleaning failed"
    exit 1
fi

# Copy images to the cleaned directory if they don't exist there
echo "📁 Copying images to cleaned directory..."
for img_file in "$INPUT_DIR"/*.{jpeg,jpg}; do
    if [[ -f "$img_file" ]]; then
        cp "$img_file" "$CLEANED_JSON_DIR/" || echo "⚠️  Warning: Could not copy $img_file"
    fi
done

echo "✅ JSON cleaning and image copying completed successfully"
echo ""

# ============================================================================
# Step 2: Build Command Arguments for Main Processing
# ============================================================================

echo "🔄 Step 2: Processing cleaned data to generate JSONL files..."

# Base command - now uses cleaned JSON directory
CMD="python data_conversion/processor.py \
    --input_dir \"$CLEANED_JSON_DIR\" \
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
# Step 3: Execute Main Processing Pipeline
# ============================================================================

echo "🔄 Step 3: Running unified data processor..."
echo "Command: $CMD"
echo ""

eval $CMD

# ============================================================================
# Step 3.5: Update ds_output Coordinates to Match Resized Images
# ============================================================================

if [[ "$RESIZE" == "true" ]]; then
    echo ""
    echo "🔧 Step 3.5: Updating ds_output coordinates to match resized images..."
    echo "Scaling coordinates in JSON files to align with resized images..."
    
    python data_conversion/update_ds_output_coordinates.py --original_dir "$INPUT_DIR" --ds_output_dir "$OUTPUT_IMAGE_DIR"
    
    if [[ $? -ne 0 ]]; then
        echo "❌ Error: ds_output coordinate update failed"
        exit 1
    fi
    
    echo "✅ ds_output coordinate update completed successfully"
    echo ""
fi

# ============================================================================
# Step 4: Validation
# ============================================================================

echo ""
echo "🔍 Step 4: Validating output files..."

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