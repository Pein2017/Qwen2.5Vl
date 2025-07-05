# Enhanced Scaling Comparison Visualization

## Overview

The enhanced `vis_scaling_comparison.py` script creates side-by-side comparisons between:
- **Raw cleaned images** from `ds_output/` with polished annotations 
- **Training rescaled images** from the final training data with scaled annotations

## How to Use

### 1. Simple Usage - Just Run It!

```bash
python vis_tools/vis_scaling_comparison.py
```

The script is configured like a bash file - just modify the settings at the top and run!

### 2. Configuration (Top of Script)

```python
# Sample ID to visualize (modify this to change which sample to visualize)
SAMPLE_ID = "QC-20230225-0000414_19823"

# Input paths
TRAINING_DATA_PATH = "data/all_samples.jsonl"  # Training data with rescaled annotations
RAW_CLEANED_DIR = "ds_output"                  # Directory with raw cleaned images and JSON
BASE_DIR = "."                                 # Base directory

# Output settings
OUTPUT_DIR = "scaling_comparisons"             # Where to save visualizations
OUTPUT_FILENAME = f"{SAMPLE_ID}_training_vs_raw.jpeg"  # Output file name

# Visualization settings
FIGURE_SIZE = (20, 10)                        # Figure size (width, height)
DPI = 300                                     # Output resolution
FONT_SIZE_TITLE = 14                          # Title font size
FONT_SIZE_LABEL = 8                           # Label font size
```

### 3. What It Does

1. **Loads training data** from `data/all_samples.jsonl` for the specified sample ID
2. **Loads raw cleaned data** from `ds_output/{SAMPLE_ID}.json` 
3. **Extracts objects** from both sources with proper coordinate handling
4. **Creates a 2-subplot visualization**:
   - Left: Raw cleaned image with polished annotations
   - Right: Training rescaled image with final training annotations
5. **Saves high-quality output** with consistent color coding and legend

## Features

### ✅ **Consistent Color Mapping**
- Pre-assigned colors for main object types (螺丝连接点, 标签贴纸, etc.)
- Same labels get same colors across both panels
- Comprehensive legend showing all object types

### ✅ **Chinese Font Support**
- Proper rendering of Chinese labels
- Fallback handling if fonts not available

### ✅ **Detailed Summary Output**
```
TRAINING VS RAW CLEANED COMPARISON
================================================================================
Sample ID: QC-20230225-0000414_19823

Raw Cleaned:
  Image: ds_output/QC-20230225-0000414_19823.jpeg (532×728)
  Objects: 8
  Format: Polished annotations from ds_output

Training Rescaled:
  Image: ds_output/QC-20230225-0000414_19823.jpeg (532×728)
  Objects: 8
  Format: Final training data

Scale factors: x=1.0000, y=1.0000
Output: scaling_comparisons/QC-20230225-0000414_19823_training_vs_raw.jpeg
================================================================================
```

### ✅ **High-Quality Output**
- 300 DPI resolution by default
- Professional layout with numbered objects
- Comprehensive legend and statistics

## Example Usage Scenarios

### Compare Different Samples
```python
# Change sample ID at top of script
SAMPLE_ID = "QC-20230225-0000414_19822"  # Different sample
```

### Different Output Settings
```python
# Custom output location
OUTPUT_DIR = "my_visualizations"
DPI = 150  # Lower resolution for faster processing
FIGURE_SIZE = (16, 8)  # Smaller figure
```

### Batch Processing Multiple Samples
You can easily modify the script to loop through multiple samples by changing the main function.

## Output

The script generates:
- High-quality JPEG visualization showing both images side-by-side
- Detailed console output with statistics and paths
- Consistent color coding and professional layout
- Legend showing all object types present

## Benefits

1. **Validates coordinate scaling** - Ensure annotations align properly with images
2. **Quality control** - Compare raw vs processed annotations visually  
3. **Debugging tool** - Identify issues in the data pipeline
4. **Documentation** - Create visual documentation of data processing
5. **Easy to use** - No command line arguments needed, just modify and run!

Perfect for verifying that your data conversion pipeline is working correctly! 🎯