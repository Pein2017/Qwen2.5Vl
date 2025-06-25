#!/bin/bash

# Clean Data Conversion Pipeline for Qwen2.5-VL
# 
# This script implements the new clean architecture:
# 1. Raw JSON annotations → Intermediate JSONL (convert_pure_json.py)
# 2. Extract candidate phrases and examples
# 3. Intermediate JSONL → Clean semantic data (no special tokens)
# 
# The training pipeline will handle special token conversion.

set -e

# Set proper locale for UTF-8 handling
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

# Data paths
INPUT_DIR="ds"                          # Raw data directory
RESCALED_DIR="ds_rescaled"              # Output for rescaled images
DATA_CONVERSION_DIR="data_conversion"   # Script directory

# Intermediate files
TEMP_JSONL="${DATA_CONVERSION_DIR}/qwen_combined.jsonl"
# Token map files for English and Chinese
MAP_FILE_EN="${DATA_CONVERSION_DIR}/token_map.json"   # English token map
MAP_FILE_ZH="${DATA_CONVERSION_DIR}/token_map_zh.json" # Chinese token map



# Support files
CANDIDATES_FILE="data_conversion/candidate_phrases.json"
TEACHER_POOL_FILE="data/teacher_pool.jsonl"

# Parameters
VAL_RATIO=0.1
SEED=42
# Allow configuring multiple response types: object_type, property, extra_info
RESPONSE_TYPES="object_type property"
USE_CANDIDATES=true
LANGUAGE="chinese" # "english" or "chinese"
RESIZE=true
MAX_TEACHERS=10

# Final outputs (dash-separated for clarity)
TRAIN_FILE="data/${LANGUAGE}-train.jsonl"
VAL_FILE="data/${LANGUAGE}-val.jsonl"

# Environment setup
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export MODELSCOPE_CACHE="/data4/swift/modelscope/hub"


echo "🚀 Starting Clean Data Conversion Pipeline"
echo "=========================================="
echo "Language: $LANGUAGE"
echo "Response Types: $RESPONSE_TYPES"
echo ""


echo "📁 Step 1: Preparing rescaled image directory: $RESCALED_DIR"
rm -rf "$RESCALED_DIR"
mkdir -p "$RESCALED_DIR"

# Copy images with progress tracking
echo "   Copying images from $INPUT_DIR to $RESCALED_DIR..."
TOTAL_COUNT=$(find "$INPUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
echo "   Found $TOTAL_COUNT image files to process"

find "$INPUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | \
  while IFS= read -r -d '' file; do
    rel_path="${file#${INPUT_DIR}/}"
    dest_sub_dir="$RESCALED_DIR/$(dirname "$rel_path")"
    dest_file="$RESCALED_DIR/$rel_path"
    
    mkdir -p "$dest_sub_dir"
    cp "$file" "$dest_file"
  done

FINAL_COUNT=$(find "$RESCALED_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
echo "   ✅ Copied $FINAL_COUNT images to $RESCALED_DIR"

echo "🔄 Step 2: Converting raw JSONs to intermediate JSONL (regenerating)..."

# Base command
CMD="python ${DATA_CONVERSION_DIR}/convert_pure_json.py \
    --input_folder \"$INPUT_DIR\" \
    --output_image_folder \"$RESCALED_DIR\" \
    --output_jsonl \"$TEMP_JSONL\" \
    --language \"$LANGUAGE\" \
    --resize $RESIZE \
    --response_types \"$RESPONSE_TYPES\" \
    --log_level DEBUG"

# Add map_file based on language
if [[ "$LANGUAGE" == "english" ]]; then
    CMD="$CMD --map_file \"$MAP_FILE_EN\""
elif [[ "$LANGUAGE" == "chinese" ]]; then
    CMD="$CMD --map_file \"$MAP_FILE_ZH\""
fi

# Execute the command
eval $CMD

echo "✅ Raw JSON conversion complete"


# Extract candidate phrases
if [[ "$USE_CANDIDATES" == "true" ]]; then
    echo "📝 Step 3: Extracting candidate phrases (regenerating)..."
    # Remove existing candidate files to force regeneration
    rm -f "$CANDIDATES_FILE"
    rm -f "${DATA_CONVERSION_DIR}/candidate_phrases.metadata.json"
    rm -f "${DATA_CONVERSION_DIR}/candidate_phrases.metadata.metadata.json"
    
    python "${DATA_CONVERSION_DIR}/extract_candidates.py" \
        --input_jsonl "$TEMP_JSONL" \
        --output_phrases "$CANDIDATES_FILE" \
        --min_frequency 1 \
        --response_types "$RESPONSE_TYPES"
    echo "✅ Phrase extraction complete"
fi

# Extract teacher pool and filter student pool
echo "📊 Step 4: Extracting teacher pool (preparing teacher-student split)"
python data_conversion/create_teacher_pool.py \
  --data_path "$TEMP_JSONL" \
  --hierarchy "${DATA_CONVERSION_DIR}/label_hierarchy.json" \
  --max_teachers $MAX_TEACHERS \
  --output "$TEACHER_POOL_FILE"
echo "✅ Teacher pool generated: $TEACHER_POOL_FILE"

echo "📊 Step 5: Filtering student pool from intermediate JSONL"
STUDENT_JSONL="${DATA_CONVERSION_DIR}/student_combined.jsonl"
# Remove old student pool to avoid stale files
rm -f "$STUDENT_JSONL"
python - <<EOF
import json
import sys
# Set UTF-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# Load teacher image paths from teacher_pool.jsonl
teacher_paths = set()
with open("$TEACHER_POOL_FILE", "r", encoding="utf-8") as tf:
    for line in tf:
        if not line.strip():
            continue
        sample = json.loads(line)
        imgs = sample.get("images", [])
        if imgs:
            teacher_paths.add(imgs[0])
# Filter out teacher samples from intermediate JSONL
with open("$TEMP_JSONL", 'r', encoding="utf-8") as fin, open("$STUDENT_JSONL", 'w', encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        img = sample.get("images", [None])[0]
        if img not in teacher_paths:
            fout.write(line + '\n')
EOF
echo "✅ Student pool ready: $STUDENT_JSONL"

echo "🔧 Step 6: Splitting student pool into train/val JSONL (clean format)..."
# Remove existing output files to force regeneration
rm -f "$TRAIN_FILE" "$VAL_FILE"

python data_conversion/split_train_val.py \
    --input_jsonl "$STUDENT_JSONL" \
    --output_train "$TRAIN_FILE" \
    --output_val "$VAL_FILE" \
    --val_ratio $VAL_RATIO \
    --seed $SEED

echo "✅ Clean semantic data created:" \
    && echo "   Training: $TRAIN_FILE" \
    && echo "   Validation: $VAL_FILE"
echo "   Teacher pool: $TEACHER_POOL_FILE"

# =============================================================================
# STEP 7: Validation
# =============================================================================

echo "🔍 Validating converted train/val data..."
python data_conversion/simple_validate.py "$TRAIN_FILE" "$VAL_FILE"

echo ""
echo "🎉 Pipeline completed successfully!"
echo "=========================================="
echo "📊 Final Output:"
echo "   Clean semantic data: $TRAIN_FILE, $VAL_FILE"
echo "   Candidate phrases: $CANDIDATES_FILE"
echo "   Teacher pool: $TEACHER_POOL_FILE"
echo ""
echo "🔧 Architecture:"
echo "   Raw JSON → Intermediate JSONL → Clean Semantic Data"
echo "   ✅ No special tokens in clean data"
echo "   ✅ Compact descriptions: 'object_type, property, extra_info'"
echo "   ✅ Training pipeline will handle special token conversion"
echo "   ✅ Always regenerates all files"
echo ""
echo "🚀 Ready for training! Use $TRAIN_FILE and $VAL_FILE with your training pipeline." 