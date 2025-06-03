# Data Analysis Tools

This folder contains essential analysis tools for the telecommunications quality inspection dataset. These scripts help extract representative samples for few-shot learning in the training pipeline.

## Core Scripts

### 1. `extract_examples_from_conversations.py`
**Purpose**: Extract representative samples from conversation-format training data for few-shot learning

**Features**:
- Analyzes conversation-format JSONL files (with `conversations` structure)
- Categorizes samples by complexity: sparse, medium, dense, diverse, rare
- Calculates educational value scores for optimal sample selection
- Exports examples in format compatible with `qwen_converter.py`

**Usage**:
```bash
# Extract 5 representative examples from training data
python data_analysis/extract_examples_from_conversations.py \
  521_qwen_train.jsonl \
  --output data_analysis/training_examples.json \
  --num_examples 5
```

### 2. `unified_analyzer.py`
**Purpose**: Comprehensive dataset analysis and sample inspection (optional, for detailed analysis)

**Features**:
- Analyzes both raw JSON files and processed JSONL files
- Calculates complexity scores and statistics
- Provides detailed sample inspection capabilities
- Generates comprehensive analysis reports

**Usage**:
```bash
# Analyze dataset and extract samples
python data_analysis/unified_analyzer.py 521_qwen_train.jsonl \
  --analyze --summary --extract-samples 5 \
  --export-analysis analysis_report.json
```

## Integration with Training Pipeline

The main integration point is through the end-to-end pipeline:

```bash
# Complete pipeline: extract examples + convert with few-shot learning
python create_training_pipeline.py \
  --input_jsonl 521_qwen_train.jsonl \
  --output_train final_train.jsonl \
  --output_val final_val.jsonl \
  --num_examples 5
```

This pipeline:
1. **Extracts representative examples** using `extract_examples_from_conversations.py`
2. **Converts dataset with few-shot prompts** using `qwen_converter.py`
3. **Excludes example images from training set** to prevent data leakage
4. **Creates final train/val files** ready for model training

## Output Files

### Generated Examples
- **`training_examples.json`**: Selected representative samples for few-shot learning
  - Format: `{"sparse": {...}, "medium": {...}, "dense": {...}, "diverse": {...}, "rare": {...}}`
  - Each category contains: `{"image": "path", "objects": [{"bbox": [...], "description": "..."}]}`

### Example Categories
- **Sparse**: 1-3 objects (simple cases)
- **Medium**: 4-10 objects (typical complexity)  
- **Dense**: 11+ objects (complex scenes)
- **Diverse**: High variety of object types
- **Rare**: Uncommon object types or combinations

## Dataset Statistics Summary

Based on the telecommunications quality inspection dataset:

- **Total Samples**: 266 (training set)
- **Total Objects**: 2,143 objects
- **Average Objects/Sample**: 8.06
- **Unique Object Types**: 18
- **Unique Questions**: 30
- **Complexity Score Range**: 0-50+ (median ~20)

### Most Common Object Types
1. `install screw correct` (475 instances)
2. `label matches` (456 instances)
3. `cpri connection correct` (394 instances)
4. `fiber cable` (340 instances)
5. `huawei bbu` (245 instances)

### Rarest Object Types
1. `cpri connection incorrect` (1 instance)
2. `cabinet grounding incorrect` (4 instances)
3. `ericsson bbu` (6 instances)
4. `install screw incorrect` (8 instances)

## End-to-End Workflow

### 1. Raw Data Processing
```bash
# Convert raw JSON to intermediate format
bash data_conversion/convert_dataset.sh
```

### 2. Training Data with Few-Shot Examples
```bash
# Extract examples and create final training files
python create_training_pipeline.py --input_jsonl 521_qwen_train.jsonl
```

### 3. Model Training
```bash
# Use generated files for training
bash scripts/train_3b.sh  # Uses final_train.jsonl and final_val.jsonl
```

## Dependencies

- Python 3.8+
- Standard library: `json`, `argparse`, `logging`, `statistics`, `collections`, `pathlib`
- No external dependencies required

## Notes

- All scripts follow fail-fast principles with transparent error handling
- Comprehensive logging for debugging and monitoring
- JSON output format for easy integration with training pipeline
- Modular design allows independent use of each component
- Examples are automatically excluded from training set to prevent data leakage 