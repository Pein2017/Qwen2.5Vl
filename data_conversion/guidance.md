# Qwen2.5-VL Data Conversion Guide

## Overview

This guide documents the complete data conversion pipeline for Qwen2.5-VL reference-based grounding, incorporating all latest improvements and the compact format. The pipeline converts raw telecom equipment inspection data into training-ready format with enhanced prompts and few-shot learning.

## Key Features

### ✅ **Ultra-Compact Format**
- **Natural language descriptions**: Comma-separated instead of structured format
- **Minimal tokens**: `{bbox:[x1,y1,x2,y2],desc:'object_type, quality_details'}`
- **Unquoted field names**: `bbox` and `desc` (not `"bbox"` and `"desc"`)
- **English-only enforcement**: Explicit requirement to prevent Chinese output
- **Strict format compliance**: Clear instructions about exact output format

### ✅ **Enhanced System Prompts**
- **Clear modular structure**: Distinct sections for format, instructions, and phrases
- **Categorized phrase organization**: Logical grouping by equipment type
- **Language control**: "Respond ONLY in English (never use Chinese characters)"
- **Format enforcement**: "Output EXACTLY this format" with examples
- **Process guidance**: Step-by-step inspection process for multi-image mode

### ✅ **Proper Few-Shot Structure**
- **Split examples**: Each example as separate user ↔ assistant turns
- **Natural conversation flow**: Compatible with Jinja template processing
- **Real data examples**: Uses actual training data for better learning
- **Minimal user prompts**: Just `<image>` after system prompt establishes context

### ✅ **Reference-Based Grounding**
- **System-level candidate integration**: Phrases in system prompt for consistency
- **Global configuration**: Easy toggle between dense captioning and reference modes
- **Token efficiency**: Shared system prompt across conversation turns
- **Better few-shot learning**: Consistent candidate context across examples

## System Prompt Structure

### Multi-Image Mode with Candidates
```text
You are Q-Vision-QC, an expert assistant specialized in telecom-equipment inspection.
Your task: produce exactly one JSON array of detected objects for each input image.

OUTPUT FORMAT:
- A JSON array where each element has:
    bbox: [x1, y1, x2, y2],
    desc: 'comma-separated object_type and details'
- Sort by top-to-bottom (increasing y), then left-to-right (increasing x).
- Use unquoted keys: bbox and desc.
- Wrap string in single quotes. No whitespace or comments outside the JSON.
- Always respond in English only.
- Output only the JSON array (no extra text or explanations).

MULTI-ROUND INSTRUCTIONS:
1) You will see K example rounds. Each round has:
   - A user turn with `<image>`
   - An assistant turn with the correct JSON array.

2) Then you will see one final user turn with `<image>`. Your job is to reply with the JSON array for that image.

AVAILABLE PHRASES FOR REFERENCE (choose only those matching visible objects):

1. BBU Types:
   - huawei bbu
   - zte bbu
   - ericsson bbu

2. Shield/Baffle Equipment:
   - bbu shield installed
   - bbu shield not installed
   - shield orientation correct
   - shield unobstructed
   - shield obstructed
   - shield brand mismatch
   - shield installed in wrong position
   - shield screws not fully installed

3. Cabinet Status:
   - cabinet fully occupied
   - cabinet not fully occupied
   - cabinet grounding correct
   - cabinet grounding incorrect

4. Screw/Installation:
   - install screw correct
   - install screw incorrect
   - floor screw installed
   - not tightened
   - installation position incorrect

5. Cable/Connection:
   - fiber cable
   - non-fiber cable
   - fibre bend radius proper
   - fibre bend radius improper
   - snake tube protection
   - armour protection
   - no snake tube or armour protection
   - fibre is protected by both armour and snake tube
   - cpri connection correct
   - cpri connection incorrect
   - odf connection correct
   - binding aligned horizontally and vertically
   - binding not aligned horizontally and vertically
   - only part of the fibre is visible
   - copper exposed

6. Label/Marking:
   - label matches
   - label does not match
   - match
   - not match

7. Other Abnormal:
   - rust
   - bbu not inserted
   - foreign object above bbu
   - unable to assess bend radius
   - which is usually unnecessary
   - other case

Select phrases that apply to objects visible in the image. You may use multiple phrases per object (e.g., 'huawei bbu, shield orientation correct, shield unobstructed').
```

## Conversation Structure

### Proper Few-Shot Format
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "[MULTI_IMAGE_SYSTEM_PROMPT with candidates]"
    },
    // Example 1
    {
      "role": "user",
      "content": "<image>"
    },
    {
      "role": "assistant",
      "content": "[{bbox:[0,0,85,140],desc:'install screw correct'},{bbox:[0,0,699,879],desc:'huawei bbu'},{bbox:[68,696,220,823],desc:'install screw incorrect, rust'}]"
    },
    // Example 2
    {
      "role": "user",
      "content": "<image>"
    },
    {
      "role": "assistant",
      "content": "[{bbox:[16,0,672,1038],desc:'cabinet fully occupied'},{bbox:[0,22,672,563],desc:'zte bbu'},{bbox:[223,213,542,397],desc:'cpri connection correct'}]"
    },
    // Example 3
    {
      "role": "user",
      "content": "<image>"
    },
    {
      "role": "assistant",
      "content": "[{bbox:[114,98,337,1400],desc:'bbu shield installed, shield orientation correct, shield unobstructed'},{bbox:[183,714,483,886],desc:'fiber cable, fibre bend radius proper, snake tube protection'}]"
    },
    // Query
    {
      "role": "user",
      "content": "<image>"
    }
  ],
  "images": ["example1.jpg", "example2.jpg", "example3.jpg", "query.jpg"]
}
```

## Response Format Examples

### Simple Objects
```json
{bbox:[336,0,698,1392],desc:'zte bbu'}
{bbox:[279,924,656,1092],desc:'label matches'}
{bbox:[616,3,677,52],desc:'install screw correct'}
```

### Objects with Multiple Attributes
```json
{bbox:[114,98,337,1400],desc:'bbu shield installed, shield orientation correct, shield unobstructed'}
{bbox:[183,714,483,886],desc:'fiber cable, fibre bend radius proper, snake tube protection'}
{bbox:[68,696,220,823],desc:'install screw incorrect, rust'}
```

### Complex Scenes
```json
[
  {bbox:[0,0,85,140],desc:'install screw correct'},
  {bbox:[0,0,699,879],desc:'huawei bbu'},
  {bbox:[183,714,483,886],desc:'fiber cable, fibre bend radius proper, snake tube protection'},
  {bbox:[355,755,436,866],desc:'cpri connection correct'},
  {bbox:[378,917,498,1112],desc:'label matches'}
]
```

## Pipeline Usage

### Enable Reference-Based Grounding
```bash
cd /data4/Qwen2.5-VL-main
./data_conversion/toggle_candidates.sh enable
./data_conversion/convert_dataset.sh
```

### Disable (Return to Dense Captioning)
```bash
./data_conversion/toggle_candidates.sh disable
./data_conversion/convert_dataset.sh
```

### Manual Conversion
```bash
# Extract candidate phrases
python data_conversion/extract_unique_phrases.py \
    --input_jsonl data_conversion/qwen_combined.jsonl \
    --output_phrases data_conversion/candidate_phrases.json

# Convert with candidates
python data_conversion/qwen_converter_unified.py \
    --input_jsonl data_conversion/qwen_combined.jsonl \
    --output_train 603_candidates_train.jsonl \
    --output_val 603_candidates_val.jsonl \
    --use_candidates \
    --candidates_file data_conversion/candidate_phrases.json \
    --use_few_shot \
    --examples_file data_analysis/training_examples.json
```

## Key Benefits

### 🎯 **Improved Model Performance**
- **Simpler target format**: Easier for models to learn and generate
- **Natural language patterns**: More intuitive comma-separated descriptions
- **Reduced hallucination**: Clear categorized phrase lists provide better guidance
- **Better few-shot learning**: Cleaner prompt structure teaches better patterns

### 🚀 **Enhanced Efficiency**
- **Reduced token count**: Compact format uses fewer tokens
- **Faster training**: Simplified complexity speeds up learning
- **Better compliance**: Clear format instructions improve output consistency
- **Token efficiency**: System-level candidate integration avoids repetition

### 🔧 **Superior Maintainability**
- **Modular prompt structure**: Easy to modify and extend
- **Clear separation of concerns**: System vs user prompt responsibilities
- **Backward compatibility**: Automatic conversion from old formats
- **Global configuration**: Easy mode switching and tuning

## Migration from Old Format

### Format Transformation
**Before** (Structured):
```json
{bbox:[114,98,337,1400],desc:'object_type:bbu shield installed;property:shield orientation correct, shield unobstructed;extra_info:none'}
```

**After** (Compact):
```json
{bbox:[114,98,337,1400],desc:'bbu shield installed, shield orientation correct, shield unobstructed'}
```

### Automatic Conversion
The pipeline automatically converts old structured formats to the new compact format:
- Removes `object_type:`, `property:`, `extra_info:` prefixes
- Filters out "none" values
- Joins components with comma separation
- Deduplicates repeated phrases
- Maintains semantic meaning

## File Structure

```
data_conversion/
├── convert_dataset.sh              # Main pipeline script
├── toggle_candidates.sh            # Mode switching utility
├── qwen_converter_unified.py       # Main converter with all features
├── core_modules.py                 # Response formatting and utilities
├── extract_unique_phrases.py       # Candidate phrase extraction
├── guidance.md                     # This comprehensive guide
└── candidate_phrases.json          # Generated candidate phrases

data_analysis/
├── extract_examples_from_conversations.py  # Few-shot example extraction
└── training_examples.json                  # Generated few-shot examples
```

## Expected Performance Improvements

1. **Better Format Compliance**: Simpler target format is easier for models to learn
2. **Reduced Hallucination**: Clear categorized phrase lists provide better guidance  
3. **Improved Few-Shot Learning**: Cleaner prompt structure teaches better patterns
4. **Faster Training**: Reduced token count and complexity
5. **More Natural Output**: Comma-separated descriptions are more readable
6. **Enhanced Consistency**: System-level candidate integration ensures uniform context

This enhanced pipeline provides a solid foundation for high-quality reference-based grounding with your real telecom equipment inspection data, incorporating all the latest improvements and best practices.

