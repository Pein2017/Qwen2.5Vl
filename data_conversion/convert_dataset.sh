#!/bin/bash
# Qwen-VL Data Conversion Pipeline with Compact Format
# 
# This script orchestrates the complete conversion pipeline:
# 1. Raw JSON annotations → Intermediate JSONL (convert_pure_json.py)
# 2. Extract candidate phrases for reference-based grounding
# 3. Intermediate JSONL → Qwen-VL format with compact descriptions
# 4. Split into train/validation sets with data leakage prevention
#
# Requirements:
# - Raw data in ds/ directory (JSON + image files)
# - Configured token_map.json for label mapping
# - Run from project root directory

# =============================================================================
# CONFIGURATION - Modify these settings as needed
# =============================================================================

# Data paths
INPUT_DIR="ds"                          # Raw data directory
RESCALED_DIR="ds_rescaled"              # Output for rescaled images
DATA_CONVERSION_DIR="data_conversion"   # Script directory

# Output files
TRAIN_OUTPUT_FILE="603_candidates_train.jsonl"
VAL_OUTPUT_FILE="603_candidates_val.jsonl"
TEMP_JSONL="${DATA_CONVERSION_DIR}/qwen_combined.jsonl"
MAP_FILE="${DATA_CONVERSION_DIR}/token_map.json"

# Training parameters
VAL_RATIO=0.1                           # Validation set ratio (10%)
RANDOM_SEED=17                          # Random seed for reproducible splits

# Response types (standardized field names)
RESPONSE_TYPES="object_type property extra_info" 

# Reference-based grounding settings
USE_CANDIDATES="true"                  # Enable reference-based grounding
CANDIDATES_FILE="data_conversion/candidate_phrases.json"  # Candidate phrases file

# Enable both multi-image few-shot learning and candidate list
MULTI_IMAGE="true"                     # Enable multi-image few-shot learning
INCLUDE_EXAMPLES="true"                # Enable examples in training data
MAX_EXAMPLES=3                         # Maximum examples per training sample
EXAMPLES_FILE="data_analysis/training_examples.json"

# Environment setup
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export MODELSCOPE_CACHE="/data4/swift/modelscope/hub"

# =============================================================================
# PIPELINE EXECUTION - Do not modify below this line
# =============================================================================

set -e

echo "🚀 Starting Qwen-VL Compact Data Conversion Pipeline"
echo "========================================================"
echo "Mode: Multi-Image Few-Shot Learning with Candidate List"
echo "Grounding Type: $(if [ "$USE_CANDIDATES" == "true" ]; then echo "Reference-Based"; else echo "Dense Captioning"; fi)"
echo "Include Examples: $INCLUDE_EXAMPLES"
echo "Max Examples per Sample: $MAX_EXAMPLES"
echo "Response Types: $RESPONSE_TYPES"
echo "Training Output: $TRAIN_OUTPUT_FILE"
echo "Validation Output: $VAL_OUTPUT_FILE"
if [ "$USE_CANDIDATES" == "true" ]; then
    echo "Candidates File: $CANDIDATES_FILE"
fi
if [ "$INCLUDE_EXAMPLES" == "true" ]; then
    echo "Examples File: $EXAMPLES_FILE"
fi
echo ""

# Step 1: Check if intermediate JSONL exists, if not create it
if [[ ! -f "$TEMP_JSONL" ]]; then
    echo "📁 Step 1: Preparing rescaled image directory: $RESCALED_DIR"
    rm -rf "$RESCALED_DIR"
    mkdir -p "$RESCALED_DIR"

    # Find and copy all .jpg/.jpeg files from INPUT_DIR to RESCALED_DIR, preserving structure
    echo "   Copying images from $INPUT_DIR to $RESCALED_DIR..."
    TOTAL_COUNT=$(find "$INPUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
    echo "   Found $TOTAL_COUNT image files to process"

    # Copy images with progress tracking
    find "$INPUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | \
      while IFS= read -r -d '' file; do
        rel_path="${file#${INPUT_DIR}/}"
        dest_sub_dir="$RESCALED_DIR/$(dirname "$rel_path")"
        dest_file="$RESCALED_DIR/$rel_path"
        
        # Create destination directory if it doesn't exist
        mkdir -p "$dest_sub_dir"
        
        # Copy the file
        cp "$file" "$dest_file"
      done

    # Verify the copy operation
    FINAL_COUNT=$(find "$RESCALED_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
    echo "   ✅ Copied $FINAL_COUNT images to $RESCALED_DIR"

    echo "✅ Image preparation complete"
    echo ""

    # Step 2: Convert raw JSONs to intermediate JSONL with standardized field names
    echo "🔄 Step 2: Converting raw JSONs to intermediate JSONL..."
    python "${DATA_CONVERSION_DIR}/convert_pure_json.py" \
        --input_folder "$INPUT_DIR" \
        --output_image_folder "$RESCALED_DIR" \
        --output_jsonl "$TEMP_JSONL" \
        --map_file "$MAP_FILE" \
        --response_types $RESPONSE_TYPES \
        --resize

    echo "✅ Raw JSON conversion complete"
    echo ""
else
    echo "📁 Step 1-2: Using existing intermediate JSONL: $TEMP_JSONL"
    echo ""
fi

# Step 3: Extract candidate phrases for reference-based grounding (if enabled)
if [[ "$USE_CANDIDATES" == "true" ]]; then
    echo "📝 Step 3: Extracting candidate phrases for reference-based grounding..."
    python "${DATA_CONVERSION_DIR}/extract_unique_phrases.py" \
        --input_jsonl "$TEMP_JSONL" \
        --output_phrases "$CANDIDATES_FILE" \
        --min_frequency 1
    echo "✅ Phrase extraction complete"
    echo ""
fi

# Step 4: Extract representative examples for few-shot learning (if enabled)
if [[ "$INCLUDE_EXAMPLES" == "true" ]]; then
    echo "📊 Step 4: Extracting representative examples for few-shot learning..."
    if [[ -f "data_analysis/extract_examples_from_conversations.py" ]]; then
        python "data_analysis/extract_examples_from_conversations.py" \
            "$TEMP_JSONL" \
            --output "$EXAMPLES_FILE" \
            --num_examples 10 \
            --response_types $RESPONSE_TYPES \
            --seed $RANDOM_SEED
        echo "✅ Example extraction complete"
    else
        echo "⚠️  Warning: extract_examples_from_conversations.py not found. Disabling examples."
        INCLUDE_EXAMPLES="false"
    fi
    echo ""
fi

# Step 5: Convert to final Qwen2.5-VL format with compact descriptions
echo "🔧 Step 5: Converting to Qwen2.5-VL compact format..."

CONVERSION_CMD="python ${DATA_CONVERSION_DIR}/qwen_converter_unified.py \
    --input_jsonl $TEMP_JSONL \
    --output_train $TRAIN_OUTPUT_FILE \
    --output_val $VAL_OUTPUT_FILE \
    --val_ratio $VAL_RATIO \
    --seed $RANDOM_SEED \
    --response_types $RESPONSE_TYPES"

# Add candidate-related options if enabled
if [[ "$USE_CANDIDATES" == "true" ]]; then
    CONVERSION_CMD="$CONVERSION_CMD \
        --use_candidates \
        --candidates_file $CANDIDATES_FILE"
fi

# Add multi-image options if enabled
if [[ "$MULTI_IMAGE" == "true" ]]; then
    CONVERSION_CMD="$CONVERSION_CMD --multi_image"
fi

# Add example-related options if enabled
if [[ "$INCLUDE_EXAMPLES" == "true" ]]; then
    CONVERSION_CMD="$CONVERSION_CMD \
        --include_examples \
        --examples_file $EXAMPLES_FILE \
        --max_examples $MAX_EXAMPLES"
fi

# Execute conversion
echo "Running: $CONVERSION_CMD"
eval $CONVERSION_CMD

echo "✅ Compact conversion complete"
echo ""

# Step 6: Validate pipeline output
echo "🔍 Step 6: Validating pipeline output..."

# Check if output files exist and are not empty
if [[ ! -f "$TRAIN_OUTPUT_FILE" || ! -s "$TRAIN_OUTPUT_FILE" ]]; then
    echo "❌ Error: Training file $TRAIN_OUTPUT_FILE is missing or empty"
    exit 1
fi

if [[ ! -f "$VAL_OUTPUT_FILE" || ! -s "$VAL_OUTPUT_FILE" ]]; then
    echo "❌ Error: Validation file $VAL_OUTPUT_FILE is missing or empty"
    exit 1
fi

# Use external validation script if available
if [[ -f "${DATA_CONVERSION_DIR}/validate_jsonl.py" ]]; then
    VALIDATION_CMD="python ${DATA_CONVERSION_DIR}/validate_jsonl.py \
        --train_file $TRAIN_OUTPUT_FILE \
        --val_file $VAL_OUTPUT_FILE"

    # Add examples flag if applicable
    if [[ "$INCLUDE_EXAMPLES" == "true" ]]; then
        VALIDATION_CMD="$VALIDATION_CMD --include_examples"
    fi

    # Run validation
    if ! eval $VALIDATION_CMD; then
        echo "❌ Validation failed"
        exit 1
    fi
fi

echo "✅ Pipeline validation complete"
echo ""

# Step 7: Display results and summary
echo "🎉 Compact Pipeline Complete!"
echo "================================="
echo "📁 Training set: $TRAIN_OUTPUT_FILE"
echo "📁 Validation set: $VAL_OUTPUT_FILE"
echo ""

echo "🔹 Pipeline Features:"
echo "   ✅ Compact comma-separated descriptions"
echo "   ✅ Multi-image few-shot learning"
echo "   ✅ Representative examples in training data"
echo "   ✅ Standardized field names (object_type, property, extra_info)"
echo "   ✅ Core modules for consistent processing"
echo "   ✅ Data leakage prevention"
echo "   ✅ Comprehensive validation and error checking"
if [ "$USE_CANDIDATES" == "true" ]; then
    echo "   ✅ Reference-based grounding with candidate phrases"
fi
if [ "$INCLUDE_EXAMPLES" == "true" ]; then
    echo "   ✅ Enhanced pattern recognition through examples"
fi

# Display statistics and sample format using external script if available
if [[ -f "${DATA_CONVERSION_DIR}/display_sample.py" ]]; then
    DISPLAY_CMD="python ${DATA_CONVERSION_DIR}/display_sample.py \
        --train_file $TRAIN_OUTPUT_FILE \
        --val_file $VAL_OUTPUT_FILE \
        --show_stats \
        --show_sample"

    # Run display script
    eval $DISPLAY_CMD
fi

echo ""
echo "🚀 Ready for training! Use the generated files with your training scripts."
echo "📚 For more details, check the generated files and core_modules.py documentation." 