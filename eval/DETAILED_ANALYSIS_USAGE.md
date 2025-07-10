# Detailed Performance Analysis for BBU Detection Model

This document describes how to use the detailed performance analysis pipeline for the BBU equipment detection model.

## Overview

The detailed analysis pipeline provides comprehensive insights into model performance beyond standard metrics like mAP and mAR. It analyzes:

- **Per-class Performance**: Detailed metrics for each of the 11 BBU object types
- **Vendor-specific Analysis**: Performance comparison across 中兴/华为/爱立信 equipment
- **Spatial Analysis**: Performance patterns across different image regions
- **Failure Pattern Detection**: Common types of errors and misclassifications
- **Success Pattern Analysis**: What the model learned well
- **Semantic Confusion**: Label confusion matrices and common mistakes

## Quick Start

### Automatic Integration (Recommended)

The detailed analysis is automatically integrated into the evaluation pipeline. Simply run:

```bash
bash eval/run_evaluation.sh
```

By default, `ENABLE_DETAILED_ANALYSIS=true` in the script, so detailed analysis will run automatically after standard evaluation.

### Manual Analysis

To run detailed analysis independently:

```bash
python eval/detailed_analysis.py \
    --predictions_file experiments/1_teacher/val/inference/predictions.json \
    --output_dir experiments/1_teacher/val/detailed_analysis \
    --label_vocab data/label_vocabulary.json \
    --log_level INFO
```

## Output Files

The analysis generates several output files in the `detailed_analysis/` directory:

### 1. `analysis_summary.txt`
Human-readable summary with:
- Overall performance metrics
- Per-class performance breakdown
- Vendor and object type comparisons
- Failure and success patterns
- Actionable recommendations

### 2. `visual_analysis.txt`
Text-based visualizations including:
- Performance bar charts by class
- Vendor comparison charts
- Confusion matrix highlights
- Spatial performance heatmap
- Size-based performance analysis

### 3. `detailed_analysis.json`
Complete raw analysis data including:
- Class-wise metrics (precision, recall, F1, IoU)
- Spatial analysis grid performance
- Confusion matrices
- Failure/success pattern details
- Recommendations with explanations

### 4. `confusion_matrix.csv`
Confusion matrix in CSV format for further analysis or import into other tools.

## Key Insights Provided

### Performance Analysis
- **Class Performance**: Which BBU components are detected well vs poorly
- **Vendor Analysis**: How well the model performs on different equipment brands
- **Size Sensitivity**: Performance correlation with object size
- **Spatial Patterns**: Which image regions have better/worse detection

### Error Analysis
- **Localization Errors**: Correct label but poor bounding box (IoU < 0.5)
- **Classification Errors**: Good localization but wrong label (IoU > 0.5)
- **Common Misclassifications**: Most frequent label confusions
- **False Positives/Negatives**: Missing detections and hallucinations

### Success Patterns
- **High-Quality Predictions**: What the model does exceptionally well
- **Best Performing Classes**: Classes with highest F1 scores
- **Size Categories**: Which object sizes work best

## Understanding the Reports

### Performance Metrics
- **Precision**: How many predicted objects are correct
- **Recall**: How many actual objects are detected
- **F1 Score**: Harmonic mean of precision and recall
- **Average IoU**: Localization quality for correct predictions
- **Center Error**: Average pixel distance error for object centers

### Spatial Analysis
The spatial analysis divides images into a 3x3 grid and shows performance in each region:
- 🟢 ≥80% accuracy (excellent)
- 🟡 ≥60% accuracy (good)  
- 🟠 ≥40% accuracy (fair)
- 🔴 <40% accuracy (poor)

### Recommendations
The system automatically generates actionable recommendations based on analysis:
- Low performance classes that need more training data
- Localization vs classification issues
- Vendor-specific problems
- Suggested improvements for training pipeline

## Configuration

### In `run_evaluation.sh`
```bash
# Enable/disable detailed analysis
ENABLE_DETAILED_ANALYSIS=true

# Log level affects detail of analysis output
LOG_LEVEL="debug"  # or "info"
```

### Command Line Options
```bash
python eval/detailed_analysis.py \
    --predictions_file <path_to_predictions.json> \
    --output_dir <output_directory> \
    --label_vocab <path_to_label_vocabulary.json> \
    --log_level <DEBUG|INFO|WARNING|ERROR>
```

## Example Workflow

1. **Run Inference**: Generate predictions using your trained model
2. **Run Evaluation**: `bash eval/run_evaluation.sh` (includes detailed analysis)
3. **Review Reports**: 
   - Quick overview: `visual_analysis.txt`
   - Detailed insights: `analysis_summary.txt`
   - Raw data: `detailed_analysis.json`
4. **Act on Recommendations**: Implement suggested improvements
5. **Compare Experiments**: Use reports to track improvement across training iterations

## Troubleshooting

### Common Issues

**JSON Parsing Errors**: Some prediction samples may have malformed JSON. The analysis will skip these with a warning and continue.

**Empty Analysis**: If no valid predictions are found, check:
- Predictions file format matches expected structure
- Label vocabulary file exists and is correctly formatted
- Predictions contain the required fields: `ground_truth`, `pred_result`, `width`, `height`

**Memory Issues**: For very large datasets, consider processing in batches or increasing available memory.

## Integration with Training Pipeline

The detailed analysis is designed to integrate seamlessly with the existing evaluation pipeline:

1. **During Training**: Run analysis on validation set to monitor learning progress
2. **After Training**: Comprehensive analysis of final model performance
3. **Model Comparison**: Compare detailed reports across different experiments
4. **Debugging**: Identify specific failure modes for targeted improvements

## Next Steps

After reviewing the detailed analysis:

1. **Address Low-Performance Classes**: Add more training data or adjust class weights
2. **Fix Localization Issues**: Tune bbox regression loss or anchor parameters  
3. **Improve Classification**: Enhance discriminative features or add hard negative mining
4. **Vendor-Specific Issues**: Balance training data across equipment types
5. **Spatial Biases**: Ensure training data covers all image regions evenly

The detailed analysis provides the insights needed to systematically improve your BBU detection model's performance.