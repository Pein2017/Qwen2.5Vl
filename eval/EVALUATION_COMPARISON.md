# Evaluation Methods Comparison: Basic vs COCO-Style Metrics

## Overview

This document compares the evaluation approaches for your multimodal telecommunications equipment detection task, explaining how to handle **open-vocabulary raw string responses** and estimate performance.

## Current Basic Metrics (`metrics.py`)

### **What it does:**
- **Single IoU threshold** evaluation (default 0.5)
- **Binary matching**: Each prediction matches at most one ground truth
- **Spatial-only**: Only considers bounding bbox_2d overlap (IoU)
- **No semantic understanding**: Ignores description content

### **Limitations for Open-Vocabulary:**
1. **No semantic evaluation** of descriptions
2. **Single threshold** doesn't show performance across difficulty levels
3. **No category-specific analysis** for different equipment types
4. **Limited insight** into model's understanding vs pure localization

```python
# Current approach - spatial only
def calculate_sample_metrics(pred_objects, gt_objects, iou_threshold=0.5):
    # Only checks: IoU >= threshold
    # Ignores: semantic content, equipment type, description quality
```

## Enhanced COCO-Style Metrics (`coco_metrics.py`)

### **Key Improvements:**

#### 1. **Multi-Threshold Evaluation** (Like COCO)
```python
# COCO-style IoU thresholds: 0.5, 0.55, 0.60, ..., 0.95
iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

# Provides metrics:
# - AP@0.5: Easy detections (loose matching)
# - AP@0.75: Hard detections (precise matching)  
# - mAP: Average across all thresholds (overall performance)
```

#### 2. **Semantic Similarity Evaluation** (For Open-Vocabulary)
```python
class SemanticMatcher:
    def calculate_semantic_similarity(self, desc1: str, desc2: str) -> float:
        # Extract object types: "object_type:huawei bbu;question:..."
        type1 = self.extract_object_type(desc1)  # "huawei bbu"
        type2 = self.extract_object_type(desc2)  # "zte bbu"
        
        # Method 1: Sentence transformer embeddings
        embeddings = self.model.encode([type1, type2])
        similarity = cosine_similarity(embeddings)
        
        # Method 2: Rule-based fallback
        # "huawei bbu" vs "zte bbu" -> both are 'bbu' category -> 0.6 similarity
```

#### 3. **Per-Category Analysis** (Equipment-Specific)
```python
equipment_categories = {
    'bbu': ['huawei bbu', 'zte bbu', 'ericsson bbu'],
    'cable': ['fiber cable', 'non-fiber cable'],
    'connection': ['cpri connection', 'odf connection'],
    'screw': ['install screw', 'floor screw'],
    'shield': ['bbu shield'],
    'cabinet': ['cabinet'],
    'label': ['label matches'],
    'grounding': ['grounding']
}

# Provides per-category metrics:
# - BBU detection performance
# - Cable detection performance  
# - etc.
```

#### 4. **Combined Spatial + Semantic Matching**
```python
def match_predictions_to_gt(pred_objects, gt_objects, iou_threshold, use_semantic=True):
    # For each prediction-GT pair:
    iou_score = calculate_iou(pred_bbox, gt_bbox)
    semantic_score = calculate_semantic_similarity(pred_desc, gt_desc)
    
    # Match criteria:
    iou_ok = iou_score >= iou_threshold          # Spatial accuracy
    semantic_ok = semantic_score >= 0.7          # Semantic accuracy
    
    # Both must be satisfied for a match
    if iou_ok and semantic_ok:
        matches.append((pred_idx, gt_idx, iou_score, semantic_score))
```

## Handling Open-Vocabulary Raw Strings

### **Challenge:**
Your model outputs raw text like:
```json
[
  {
    "bbox_2d": [100, 200, 300, 400],
    "description": "object_type:huawei bbu;question:is the shield properly installed;extra question:none"
  }
]
```

### **Solution Approach:**

#### 1. **Structured Parsing** (Already Implemented)
```python
class ResponseParser:
    def parse_prediction(self, prediction_text: str) -> List[Dict]:
        # Handles multiple formats:
        # - Direct JSON arrays
        # - JSON in markdown code blocks  
        # - Truncated responses
        # - Malformed JSON with regex extraction
```

#### 2. **Semantic Evaluation Methods**

**Method A: Rule-Based Matching**
```python
def _rule_based_similarity(self, desc1: str, desc2: str) -> float:
    type1 = extract_object_type(desc1).lower()  # "huawei bbu"
    type2 = extract_object_type(desc2).lower()  # "zte bbu"
    
    # Exact match
    if type1 == type2: return 1.0
    
    # Partial match  
    if type1 in type2 or type2 in type1: return 0.8
    
    # Category match (both are BBUs)
    if both_in_same_category(type1, type2): return 0.6
    
    return 0.0
```

**Method B: Embedding-Based Matching** (Requires sentence-transformers)
```python
def calculate_semantic_similarity(self, desc1: str, desc2: str) -> float:
    # Extract key terms
    type1 = extract_object_type(desc1)  # "huawei bbu"
    type2 = extract_object_type(desc2)  # "base band unit"
    
    # Get semantic embeddings
    embeddings = sentence_transformer.encode([type1, type2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return similarity  # 0.85 (high similarity despite different wording)
```

## Performance Estimation Strategies

### 1. **Multi-Dimensional Evaluation**

```python
# Spatial Performance (Traditional)
spatial_metrics = {
    "mAP_spatial": 0.75,      # Pure IoU-based matching
    "AP@0.5_spatial": 0.85,   # Easy spatial matching
    "AP@0.75_spatial": 0.65,  # Hard spatial matching
}

# Semantic Performance (Open-Vocabulary)  
semantic_metrics = {
    "mAP_semantic": 0.68,     # Semantic similarity matching
    "category_accuracy": 0.82, # Correct equipment type identification
    "description_quality": 0.71, # Overall description relevance
}

# Combined Performance (Both Required)
combined_metrics = {
    "mAP_combined": 0.62,     # Both spatial AND semantic correct
    "strict_accuracy": 0.58,  # High threshold for both
}
```

### 2. **Category-Specific Analysis**

```python
per_category_results = {
    "bbu": {
        "mAP": 0.78,           # BBU detection performance
        "semantic_acc": 0.85,  # BBU type identification (Huawei vs ZTE vs Ericsson)
        "spatial_acc": 0.82,   # BBU localization accuracy
    },
    "cable": {
        "mAP": 0.65,
        "semantic_acc": 0.72,  # Fiber vs non-fiber identification
        "spatial_acc": 0.78,
    },
    "connection": {
        "mAP": 0.58,
        "semantic_acc": 0.68,  # CPRI vs ODF identification  
        "spatial_acc": 0.71,
    }
}
```

### 3. **Error Analysis Dimensions**

```python
error_analysis = {
    "localization_errors": {
        "false_positives": 15,    # Detected non-existent objects
        "false_negatives": 8,     # Missed existing objects
        "poor_localization": 12,  # Correct detection, wrong bbox
    },
    "semantic_errors": {
        "wrong_category": 6,      # BBU detected as cable
        "wrong_subtype": 9,       # Huawei BBU detected as ZTE BBU
        "incomplete_desc": 14,    # Missing question/extra question fields
    },
    "combined_errors": {
        "spatial_ok_semantic_wrong": 18,  # Right location, wrong type
        "semantic_ok_spatial_wrong": 11,  # Right type, wrong location
        "both_wrong": 5,                  # Both location and type wrong
    }
}
```

## Usage Examples

### **Basic Evaluation** (Current)
```bash
python eval/metrics.py \
    --responses_file eval_results/raw_responses.json \
    --output_file eval_results/basic_metrics.json \
    --iou_threshold 0.5
```

### **COCO-Style Evaluation** (Enhanced)
```bash
# Full evaluation with semantic matching
python eval/coco_metrics.py \
    --responses_file eval_results/raw_responses.json \
    --output_file eval_results/coco_metrics.json \
    --semantic_threshold 0.7

# Spatial-only evaluation (no semantic)
python eval/coco_metrics.py \
    --responses_file eval_results/raw_responses.json \
    --output_file eval_results/spatial_only.json \
    --no_semantic
```

## Expected Output Comparison

### **Basic Metrics Output:**
```json
{
  "overall_metrics": {
    "precision": 0.75,
    "recall": 0.68,
    "f1": 0.71
  }
}
```

### **COCO-Style Metrics Output:**
```json
{
  "overall_metrics": {
    "mAP": 0.65,
    "AP@0.5": 0.78,
    "AP@0.75": 0.52,
    "mAR": 0.71
  },
  "category_metrics": {
    "bbu": {"mAP": 0.78, "AP@0.5": 0.85},
    "cable": {"mAP": 0.58, "AP@0.5": 0.72},
    "connection": {"mAP": 0.51, "AP@0.5": 0.65}
  },
  "semantic_metrics": {
    "mAP": 0.68,
    "semantic_accuracy": 0.74
  }
}
```

## Recommendations

### **For Development/Debugging:**
1. Use **basic metrics** for quick iteration
2. Focus on **AP@0.5** for initial model validation
3. Monitor **per-category performance** to identify weak areas

### **For Final Evaluation:**
1. Use **COCO-style metrics** for comprehensive assessment
2. Report **mAP (0.5:0.95)** as primary metric
3. Include **semantic similarity** scores for open-vocabulary evaluation
4. Analyze **per-category performance** for domain-specific insights

### **For Production Deployment:**
1. Set **minimum thresholds** based on use case:
   - Safety-critical: AP@0.75 > 0.8 (high precision)
   - General inspection: AP@0.5 > 0.7 (balanced)
   - Screening: mAR > 0.8 (high recall)

This multi-dimensional evaluation approach provides much richer insights into your model's performance on the open-vocabulary multimodal task compared to simple IoU-based metrics. 