# Line Directional Normalization

## Problem Statement

Multi-point line annotations (LineString geometry with >2 points) have an inherent directional ambiguity. A line path representing a physical cable/wire can be traced in either direction by different annotators, producing semantically equivalent but different coordinate sequences.

### Example of Directional Ambiguity

```python
# Same physical cable traced in different directions
forward_path = [10, 50, 30, 40, 50, 30, 70, 20]  # Left-to-right
reverse_path = [70, 20, 50, 30, 30, 40, 10, 50]  # Right-to-left

# Both represent the same physical cable but with different coordinate sequences
```

This ambiguity can cause:
- **Training inconsistency**: Same physical object with different representations
- **Model confusion**: Learning different patterns for identical objects
- **Annotation variance**: Different annotators producing different results

## Solution: Canonical Direction Normalization

We implement a normalization strategy that establishes a canonical direction for multi-point lines while preserving the path structure.

### Normalization Strategy

1. **Preserve path structure**: Don't reorder intermediate points
2. **Establish consistent direction**: Choose a canonical starting point
3. **Use deterministic rule**:
   - Start from the topmost point (lowest y-coordinate)
   - If multiple points have the same y-coordinate, start from the leftmost (lowest x-coordinate)
   - If the path should be reversed to meet this criteria, reverse the entire coordinate sequence

### Implementation

The normalization is implemented in `CoordinateManager._normalize_polyline_direction()`:

```python
def _normalize_polyline_direction(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Normalize multi-point line direction to establish canonical ordering.
    
    Strategy:
    1. Find canonical starting point (topmost, then leftmost)
    2. If current start is canonical, keep as-is
    3. If current end is canonical, reverse entire path
    4. If canonical point is in middle, compare endpoints and choose more canonical direction
    """
```

### Algorithm Details

1. **Find canonical point**: `min(points, key=lambda p: (p[1], p[0]))`
2. **Check current path**:
   - If path already starts with canonical point → keep as-is
   - If path ends with canonical point → reverse entire path
   - If canonical point is in middle → compare endpoints and choose more canonical direction

### Examples

#### Case 1: Canonical Point at End
```python
# Input: (40,60) -> (30,50) -> (10,20)
# Canonical point: (10,20) at end
# Output: (10,20) -> (30,50) -> (40,60)  [REVERSED]
```

#### Case 2: Canonical Point at Start
```python
# Input: (10,20) -> (30,50) -> (40,60)
# Canonical point: (10,20) at start
# Output: (10,20) -> (30,50) -> (40,60)  [UNCHANGED]
```

#### Case 3: Canonical Point in Middle
```python
# Input: (30,50) -> (10,20) -> (40,60)
# Canonical point: (10,20) in middle
# Compare endpoints: (30,50) vs (40,60)
# (30,50) is more canonical, so keep original direction
# Output: (30,50) -> (10,20) -> (40,60)  [UNCHANGED]
```

## Benefits

1. **Consistent Training Data**: Same physical cables always produce same coordinate sequences
2. **Reduced Model Confusion**: Eliminates directional ambiguity in training
3. **Annotation Independence**: Results independent of annotator's tracing direction
4. **Path Preservation**: Maintains correct curved path structure

## Integration

The normalization is automatically applied during coordinate processing:

- **Data Conversion**: Applied in `normalize_line_coordinates()`
- **Training Pipeline**: Integrated into coordinate transformation pipeline
- **Token Generation**: Works seamlessly with existing token system

## Testing

Comprehensive tests verify the implementation:

- **Directional Consistency**: Same path traced in opposite directions produces identical results
- **Path Preservation**: Intermediate points maintain correct order
- **Edge Cases**: Handles canonical points at start, end, or middle of path
- **Backward Compatibility**: 2-point lines still use existing canonical ordering

## Usage

The normalization is applied automatically during data processing. No manual intervention required.

```python
# Automatic application during coordinate normalization
normalized_obj = CoordinateManager.normalize_object_coordinates(obj, width, height)

# The line coordinates in normalized_obj will have canonical direction
```
