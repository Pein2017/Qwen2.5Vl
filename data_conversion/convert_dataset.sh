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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
INPUT_DIR="ds"                          # Raw data directory
RESCALED_DIR="ds_rescaled"              # Output for rescaled images
DATA_CONVERSION_DIR="data_conversion"   # Script directory

# Intermediate files
TEMP_JSONL="${DATA_CONVERSION_DIR}/qwen_combined.jsonl"
# Token map files for English and Chinese
MAP_FILE_EN="${DATA_CONVERSION_DIR}/token_map.json"   # English token map
MAP_FILE_ZH="${DATA_CONVERSION_DIR}/token_map_zh.json" # Chinese token map

# Final clean semantic data (only output we need)
CLEAN_TRAIN="data/605_objecttype_property_train.jsonl"
CLEAN_VAL="data/605_objecttype_property_val.jsonl"

# Support files
EXAMPLES_FILE="data_analysis/training_examples.json"
CANDIDATES_FILE="data_conversion/candidate_phrases.json"

# Parameters
VAL_RATIO=0.1
SEED=42
MULTI_ROUND=true
INCLUDE_EXAMPLES=true
MAX_EXAMPLES=1
# Allow configuring multiple response types: object_type, property, extra_info
RESPONSE_TYPES="object_type property"
USE_CANDIDATES=true
LANGUAGE="chinese" # "english" or "chinese"
RESIZE=true

# Environment setup
export PYTHONPATH=/data4/Qwen2.5-VL-main:$PYTHONPATH
export MODELSCOPE_CACHE="/data4/swift/modelscope/hub"

echo "🚀 Starting Clean Data Conversion Pipeline"
echo "=========================================="
echo "Language: $LANGUAGE"
echo "Mode: Multi-Image Few-Shot Learning with Clean Architecture"
echo "Include Examples: $INCLUDE_EXAMPLES"
echo "Max Examples per Sample: $MAX_EXAMPLES"
echo "Response Types: $RESPONSE_TYPES"
echo ""

# =============================================================================
# STEP 1: Raw JSON to Intermediate JSONL (Always regenerate)
# =============================================================================

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

# =============================================================================
# STEP 3: Extract Support Files (Always regenerate)
# =============================================================================

# Extract candidate phrases
if [[ "$USE_CANDIDATES" == "true" ]]; then
    echo "📝 Step 3: Extracting candidate phrases (regenerating)..."
    # Remove existing candidate files to force regeneration
    rm -f "$CANDIDATES_FILE"
    rm -f "${DATA_CONVERSION_DIR}/candidate_phrases.metadata.json"
    rm -f "${DATA_CONVERSION_DIR}/candidate_phrases.metadata.metadata.json"
    
    python "${DATA_CONVERSION_DIR}/extract_unique_phrases.py" \
        --input_jsonl "$TEMP_JSONL" \
        --output_phrases "$CANDIDATES_FILE" \
        --min_frequency 1 \
        --response_types "$RESPONSE_TYPES"
    echo "✅ Phrase extraction complete"
fi

# Extract examples
if [[ "$INCLUDE_EXAMPLES" == "true" ]]; then
    echo "📊 Step 4: Extracting representative examples (regenerating)..."
    # Remove existing examples file to force regeneration
    rm -f "$EXAMPLES_FILE"
    
    if [[ -f "data_analysis/extract_examples_from_conversations.py" ]]; then
        python "data_analysis/extract_examples_from_conversations.py" \
            "$TEMP_JSONL" \
            --output "$EXAMPLES_FILE" \
            --num_examples 5 \
            --response_types $RESPONSE_TYPES \
            --seed $SEED
        echo "✅ Example extraction complete"
    else
        echo "⚠️  Warning: extract_examples_from_conversations.py not found. Disabling examples."
        INCLUDE_EXAMPLES="false"
    fi
fi

# =============================================================================
# STEP 5: Convert to Clean Semantic Data (Always regenerate)
# =============================================================================

echo "🔧 Step 5: Converting to clean semantic format (regenerating)..."
# Remove existing output files to force regeneration
rm -f "$CLEAN_TRAIN"
rm -f "$CLEAN_VAL"

python data_conversion/qwen_converter_unified.py \
    --input_jsonl "$TEMP_JSONL" \
    --output_train "$CLEAN_TRAIN" \
    --output_val "$CLEAN_VAL" \
    --val_ratio $VAL_RATIO \
    --seed $SEED \
    $([ "$MULTI_ROUND" = true ] && echo "--multi_round") \
    $([ "$INCLUDE_EXAMPLES" = true ] && echo "--include_examples") \
    --examples_file "$EXAMPLES_FILE" \
    --max_examples $MAX_EXAMPLES \
    --response_types "$RESPONSE_TYPES"

echo "✅ Clean semantic data created:"
echo "   Training: $CLEAN_TRAIN"
echo "   Validation: $CLEAN_VAL"

# =============================================================================
# STEP 6: Validation
# =============================================================================

echo "🔍 Step 6: Validating pipeline..."
python -c "
import json

def validate_sample(sample):
    \"\"\"Validate a single sample (handles both simple and multi-round formats).\"\"\"
    if 'target' in sample and 'examples' in sample:
        # Multi-round format
        target = sample['target']
        examples = sample['examples']
        
        # Validate target
        if not isinstance(target.get('images'), list) or not target['images']:
            return False
        if not isinstance(target.get('objects'), list):
            return False
            
        # Validate examples
        if not isinstance(examples, list):
            return False
        for example in examples:
            if not isinstance(example.get('images'), list) or not example['images']:
                return False
            if not isinstance(example.get('objects'), list):
                return False
        return True
    else:
        # Simple format
        if not isinstance(sample.get('images'), list) or not sample['images']:
            return False
        if not isinstance(sample.get('objects'), list):
            return False
        return True

def load_jsonl(file_path):
    \"\"\"Load samples from JSONL file.\"\"\"
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

# Validate clean data
print('Validating clean semantic data...')
clean_samples = load_jsonl('$CLEAN_TRAIN')
valid_count = sum(1 for sample in clean_samples if validate_sample(sample))
print(f'✅ Clean data: {valid_count}/{len(clean_samples)} samples valid')

# Show sample format
if clean_samples:
    print('\\n📋 Sample clean semantic data:')
    sample = clean_samples[0]
    
    if 'target' in sample and 'examples' in sample:
        print('Format: Multi-round with examples')
        print(f'Examples: {len(sample[\"examples\"])} examples')
        print(f'Target Images: {sample[\"target\"].get(\"images\", [])}')
        print(f'Target Objects: {len(sample[\"target\"].get(\"objects\", []))} objects')
        if sample['target'].get('objects'):
            obj = sample['target']['objects'][0]
            print(f'Sample target object: {obj}')
            print(f'Description format: \"{obj.get(\"desc\", \"\")}\"')
    else:
        print('Format: Simple')
        print(f'Images: {sample.get(\"images\", [])}')
        print(f'Objects: {len(sample.get(\"objects\", []))} objects')
        if sample.get('objects'):
            obj = sample['objects'][0]
            print(f'Sample object: {obj}')
            print(f'Description format: \"{obj.get(\"desc\", \"\")}\"')
"

echo ""
echo "🎉 Pipeline completed successfully!"
echo "=========================================="
echo "📊 Final Output:"
echo "   Clean semantic data: $CLEAN_TRAIN, $CLEAN_VAL"
echo "   Candidate phrases: $CANDIDATES_FILE"
echo ""
echo "🔧 Architecture:"
echo "   Raw JSON → Intermediate JSONL → Clean Semantic Data"
echo "   ✅ No special tokens in clean data"
echo "   ✅ Compact descriptions: 'object_type, property, extra_info'"
echo "   ✅ Training pipeline will handle special token conversion"
echo "   ✅ Always regenerates all files"
echo ""
echo "🚀 Ready for training! Use $CLEAN_TRAIN and $CLEAN_VAL with your training pipeline." 