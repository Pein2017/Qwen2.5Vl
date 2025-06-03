# Telecommunications Dataset Distribution Analysis

## Overview

This document provides a comprehensive analysis of the telecommunications quality inspection dataset, including object types, quality checks, and distribution patterns that inform the system prompt design.

**Dataset Statistics:**
- **Total Samples**: 333
- **Total Objects**: 2,687
- **Average Objects per Sample**: 8.07
- **Object Count Range**: 1-43 objects per image
- **Unique Object Types**: 19
- **Unique Questions**: 31
- **Extra Questions**: Mostly empty (used rarely)

## Object Type Distribution

### Most Common Object Types (Top 10)

| Object Type                  | Count | Percentage | Description                    |
| ---------------------------- | ----- | ---------- | ------------------------------ |
| `install screw correct`      | 475   | 17.7%      | Properly installed screws      |
| `label matches`              | 456   | 17.0%      | Correct and matching labels    |
| `cpri connection correct`    | 394   | 14.7%      | Proper CPRI connections        |
| `fiber cable`                | 340   | 12.7%      | Fiber optic cables             |
| `huawei bbu`                 | 245   | 9.1%       | Huawei Base Band Units         |
| `bbu shield installed`       | 143   | 5.3%       | BBU shields properly installed |
| `non-fiber cable`            | 126   | 4.7%       | Non-fiber cables               |
| `cabinet not fully occupied` | 121   | 4.5%       | Cabinets with available space  |
| `cabinet fully occupied`     | 85    | 3.2%       | Fully occupied cabinets        |
| `floor screw installed`      | 79    | 2.9%       | Floor mounting screws          |

### Equipment Categories

#### 1. BBU (Base Band Unit) Equipment
- **`huawei bbu`**: 245 instances (9.1%)
- **`zte bbu`**: 38 instances (1.4%)
- **`ericsson bbu`**: 6 instances (0.2%)

**Insight**: Huawei BBUs dominate the dataset, representing 85% of all BBU equipment.

#### 2. Cabinet Status
- **`cabinet not fully occupied`**: 121 instances (4.5%)
- **`cabinet fully occupied`**: 85 instances (3.2%)

**Insight**: More images show cabinets with available space than fully occupied ones.

#### 3. Installation Components
- **`install screw correct`**: 475 instances (17.7%)
- **`install screw incorrect`**: 8 instances (0.3%)
- **`floor screw installed`**: 79 instances (2.9%)

**Insight**: Installation screws are the most frequently detected objects, with 98.3% being correctly installed.

#### 4. Connections
- **`cpri connection correct`**: 394 instances (14.7%)
- **`cpri connection incorrect`**: 1 instance (0.04%)
- **`odf connection correct`**: 52 instances (1.9%)

**Insight**: CPRI connections are very reliable with 99.7% correct installation rate.

#### 5. Cables
- **`fiber cable`**: 340 instances (12.7%)
- **`non-fiber cable`**: 126 instances (4.7%)

**Insight**: Fiber cables are 2.7x more common than non-fiber cables.

#### 6. Shields
- **`bbu shield installed`**: 143 instances (5.3%)
- **`bbu shield not installed`**: 41 instances (1.5%)

**Insight**: 77.7% of BBU shields are properly installed.

#### 7. Labels
- **`label matches`**: 456 instances (17.0%)
- **`label does not match`**: 11 instances (0.4%)

**Insight**: Labels have a 97.6% accuracy rate.

#### 8. Grounding
- **`cabinet grounding correct`**: 62 instances (2.3%)
- **`cabinet grounding incorrect`**: 4 instances (0.1%)

**Insight**: Grounding has a 93.9% correct installation rate.

## Quality Check Distribution

### Most Common Quality Checks

| Quality Check                                                                 | Count | Context                           |
| ----------------------------------------------------------------------------- | ----- | --------------------------------- |
| `none`                                                                        | 1,944 | No specific quality check needed  |
| `fibre bend radius proper, snake tube protection`                             | 208   | Fiber cable quality check         |
| `shield orientation correct, shield orientation correct, shield unobstructed` | 132   | BBU shield quality check          |
| `binding aligned horizontally and vertically`                                 | 124   | Cable alignment check             |
| `match`                                                                       | 116   | Label matching verification       |
| `fibre bend radius proper, no snake tube or armour protection`                | 65    | Fiber cable without protection    |
| `fibre bend radius proper, armour protection`                                 | 46    | Fiber cable with armor protection |
| `not tightened`                                                               | 8     | Screw tightness check             |
| `not match`                                                                   | 5     | Label mismatch                    |

### Quality Check Categories

#### 1. Fiber Cable Quality (319 total checks)
- **With snake tube protection**: 208 instances (65.2%)
- **No protection**: 65 instances (20.4%)
- **With armor protection**: 46 instances (14.4%)

#### 2. Shield Quality (132 total checks)
- **Orientation and obstruction checks**: 132 instances
- Focus on proper orientation and unobstructed installation

#### 3. Cable Alignment (124 total checks)
- **Horizontal and vertical alignment**: 124 instances
- Critical for proper cable management

#### 4. Label Verification (121 total checks)
- **Match verification**: 116 instances (95.9%)
- **Mismatch detection**: 5 instances (4.1%)

## Sample Complexity Distribution

### By Object Count
- **Sparse (≤3 objects)**: 68 samples (20.4%)
- **Medium (4-10 objects)**: 183 samples (55.0%)
- **Dense (>10 objects)**: 82 samples (24.6%)

### Representative Examples
1. **Sparse**: 1 object (simple label check)
2. **Medium**: 6 objects (typical installation with BBU, screws, cables)
3. **Dense**: 15-43 objects (complex installations with multiple components)

## Data Quality Insights

### High Reliability Components
1. **CPRI Connections**: 99.7% correct rate
2. **Installation Screws**: 98.3% correct rate
3. **Labels**: 97.6% accuracy rate
4. **Grounding**: 93.9% correct rate

### Areas Requiring Attention
1. **BBU Shields**: 22.3% not installed (needs monitoring)
2. **Fiber Cable Protection**: 20.4% without proper protection
3. **Cabinet Utilization**: 58.7% not fully occupied

### Rare Error Cases
- **CPRI connection incorrect**: Only 1 instance
- **Cabinet grounding incorrect**: Only 4 instances
- **Ericsson BBU**: Only 6 instances (rare equipment type)

## System Prompt Optimization

Based on this analysis, the system prompts have been updated to:

1. **Include all 19 actual object types** found in the dataset
2. **Specify exact quality check phrases** used in the data
3. **Prioritize common object types** in the prompt structure
4. **Use correct field format** (`"extra question:"` not `"extra:"`)

### Updated Object Type Categories
```
1. BBU Equipment: huawei bbu, zte bbu, ericsson bbu
2. Cabinet Status: cabinet fully occupied, cabinet not fully occupied  
3. Installation: install screw correct, install screw incorrect, floor screw installed
4. Connections: cpri connection correct, cpri connection incorrect, odf connection correct
5. Cables: fiber cable, non-fiber cable
6. Shields: bbu shield installed, bbu shield not installed
7. Labels: label matches, label does not match
8. Grounding: cabinet grounding correct, cabinet grounding incorrect
```

### Updated Quality Checks
```
- Fiber cables: "fibre bend radius proper, snake tube protection" / "fibre bend radius proper, armour protection" / "fibre bend radius proper, no snake tube or armour protection"
- Shields: "shield orientation correct, shield orientation correct, shield unobstructed"
- Cables: "binding aligned horizontally and vertically"
- Labels: "match" / "not match"
- Screws: "not tightened"
```

## Recommendations

### For Training
1. **Focus on rare cases**: Include examples with incorrect installations and rare equipment types
2. **Balance complexity**: Ensure training data includes sparse, medium, and dense samples
3. **Quality check emphasis**: Train on specific quality check phrases found in the data

### For Evaluation
1. **Monitor rare object types**: Pay special attention to Ericsson BBU and error cases
2. **Quality check accuracy**: Evaluate model's ability to use exact quality check phrases
3. **Complex scene handling**: Test performance on dense samples (>15 objects)

### For Data Collection
1. **Increase rare equipment**: Collect more Ericsson BBU and error case examples
2. **Error case documentation**: Document more incorrect installation examples
3. **Quality check standardization**: Ensure consistent quality check phrase usage

This analysis provides the foundation for accurate system prompts and effective model training on telecommunications equipment quality inspection tasks. 