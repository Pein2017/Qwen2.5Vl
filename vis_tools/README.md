# Qwen2.5-VL Visualization Tools

This directory contains visualization tools for analyzing Qwen2.5-VL model performance on dense captioning and grounding tasks.

## Overview

The visualization tools help you create clear, side-by-side comparisons of ground truth annotations versus model predictions, with different colors for different object categories.

## Features

- **Side-by-side comparison**: Ground truth vs predictions on the same image
- **Color-coded labels**: Each object category gets a distinct, consistent color
- **Count statistics**: Shows object counts for each category in ground truth vs predictions
- **Chinese font support**: Properly displays Chinese labels
- **Batch processing**: Process multiple samples at once
- **Flexible input**: Handles various JSON formats and malformed files

## Files

- `vis_generation.py` - Main visualization script
- `fix_json.py` - Utility to fix malformed JSON inference result files
- `example_usage.py` - Example script showing how to use the tools
- `vis_train.py` - Training visualization utilities

## Quick Start

### 1. Basic Usage

```bash
# Visualize inference results
python vis_tools/vis_generation.py --input infer_result/chinese-val.json --output_dir visualizations --max_samples 5
```

### 2. Fix Malformed JSON First (if needed)

```bash
# Fix malformed JSON files
python vis_tools/fix_json.py infer_result/chinese-val.json --output infer_result/chinese-val_fixed.json

# Then visualize the fixed file
python vis_tools/vis_generation.py --input infer_result/chinese-val_fixed.json --output_dir visualizations
```

### 3. Visualize Specific Samples

```bash
# Visualize specific sample indices
python vis_tools/vis_generation.py --input infer_result/chinese-val.json --output_dir visualizations --sample_indices "0,5,10,15"
```

## Command Line Options

### vis_generation.py

```bash
python vis_tools/vis_generation.py [OPTIONS]

Required Arguments:
  --input INPUT_FILE          Path to inference results JSON file

Optional Arguments:
  --output_dir OUTPUT_DIR     Output directory for visualizations (default: visualizations)
  --base_path BASE_PATH       Base path for image files (default: .)
  --max_samples MAX_SAMPLES   Maximum number of samples to visualize
  --sample_indices INDICES    Comma-separated list of specific sample indices to visualize
```

### fix_json.py

```bash
python vis_tools/fix_json.py INPUT_FILE [--output OUTPUT_FILE]

Arguments:
  INPUT_FILE                  Path to malformed JSON file
  --output OUTPUT_FILE        Path to save fixed JSON (default: INPUT_FILE_fixed.json)
```

## Input Data Format

The visualization script expects JSON files with the following format:

```json
[
  {
    "image": "path/to/image.jpeg",
    "ground_truth": "[{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category_name\"}, ...]",
    "pred_result": "[{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category_name\"}, ...]",
    "height": 728,
    "width": 532
  },
  ...
]
```

Where:
- `image`: Relative or absolute path to the image file
- `ground_truth`: JSON string containing list of ground truth bounding boxes
- `pred_result`: JSON string containing list of predicted bounding boxes
- `bbox_2d`: Bounding box coordinates [x1, y1, x2, y2]
- `label`: Object category label (supports Chinese text)

## Output Format

For each input sample, the script generates a PNG file with:
- **Left subplot**: Ground truth annotations
- **Right subplot**: Model predictions  
- **Legend**: Color-coded categories with count statistics
- **Title**: Image filename and comparison info

Output files are named: `{image_basename}_comparison.png`

## Examples

### Example 1: Process all samples in a file

```bash
python vis_tools/vis_generation.py \
    --input infer_result/chinese-train.json \
    --output_dir model_analysis \
    --base_path .
```

### Example 2: Quick preview of first 10 samples

```bash
python vis_tools/vis_generation.py \
    --input infer_result/chinese-val.json \
    --output_dir quick_preview \
    --max_samples 10
```

### Example 3: Analyze specific challenging cases

```bash
python vis_tools/vis_generation.py \
    --input infer_result/chinese-val.json \
    --output_dir error_analysis \
    --sample_indices "2,7,15,23,31"
```

## Understanding the Visualizations

### Color Coding
- Each unique label gets a consistent, deterministic color
- Colors are generated using a predefined palette for consistency
- Same categories always get the same color across different images

### Legend Information
- Format: `Category Name (GT: X, Pred: Y)`
- `GT`: Number of ground truth objects of this category
- `Pred`: Number of predicted objects of this category
- Helps identify over-detection, under-detection, and category confusion

### Visual Analysis Tips
1. **Missing predictions**: Categories present in GT but not in predictions
2. **False positives**: Categories in predictions but not in GT
3. **Localization errors**: Same category but different bounding box locations
4. **Category confusion**: Different categories predicted for same objects

## Troubleshooting

### Common Issues

1. **JSON parsing errors**
   ```bash
   # Fix malformed JSON files first
   python vis_tools/fix_json.py your_file.json
   ```

2. **Image not found errors**
   ```bash
   # Use --base_path to specify correct image directory
   python vis_tools/vis_generation.py --input data.json --base_path /path/to/images
   ```

3. **Chinese font warnings**
   - The script includes Chinese font support
   - Warnings about missing fonts are cosmetic and don't affect functionality
   - Images will still be generated correctly

4. **Memory issues with large datasets**
   ```bash
   # Process in smaller batches
   python vis_tools/vis_generation.py --input data.json --max_samples 50
   ```

## Integration with Evaluation Pipeline

The visualization tools integrate seamlessly with the evaluation pipeline:

```bash
# 1. Run inference
./eval/infer_dataset.sh

# 2. Fix JSON if needed  
python vis_tools/fix_json.py infer_result/chinese-val.json

# 3. Run evaluation
./eval/run_evaluation.sh

# 4. Visualize results
python vis_tools/vis_generation.py --input infer_result/chinese-val_fixed.json --output_dir analysis
```

## Performance Notes

- Processing time: ~0.5-1 second per image
- Output file size: ~0.8-1.5MB per visualization (high quality, 150 DPI)
- Memory usage: Minimal, processes one image at a time
- Supports batch processing of thousands of samples

## Contributing

When adding new features:
1. Follow the existing code style
2. Add appropriate logging
3. Handle edge cases gracefully
4. Update this README with new options 