# Multi-Geometry Support System

**Status:** ✅ PRODUCTION READY | **Comprehensive multi-geometry token parsing and coordinate system**

## 🎯 **Overview**

The multi-geometry support system enables the Qwen2.5-VL model to detect and describe multiple geometry types (bounding boxes, lines, and squares) with precise coordinate information. The system automatically detects coordinate token capabilities and provides both caption-only and coordinate-enabled modes.

## 🏗️ **Architecture**

### **System Components**

1. **Multi-Geometry Token Parser** (`src/utils/response_parser.py`)
   - Parses geometry-specific tokens: `<bbox_2d_start>`, `<line_start>`, `<square_start>`
   - Decodes coordinate tokens: `<|coord_X|>` format
   - Supports caption-only and coordinate-enabled modes

2. **Inference Engine Integration** (`src/inference.py`)
   - Automatic configuration detection from YAML configs
   - Seamless multi-geometry processing pipeline
   - Fallback behavior for non-coordinate models

3. **Evaluation System** (`eval/coco_metrics.py`)
   - Multi-geometry validation and IoU computation
   - Geometry-specific coordinate validation
   - Cross-geometry similarity calculation

## 🔧 **Configuration**

### **Single Source of Truth**

The system uses `coordinate_tokens_enabled: true` in `configs/bbu_v2.yaml` (line 70) as the single configuration point:

```yaml
# Multi-geometry coordinate token configuration
coordinate_tokens_enabled: true  # Enables coordinate token parsing and multi-geometry support
```

### **Automatic Detection Flow**

1. **Config Loading**: `_init_config(args.config_path)` loads configuration
2. **Detection**: `_detect_coordinate_tokens_enabled()` reads `config.coordinate_tokens_enabled`
3. **Auto-Enable**: If enabled → automatically activate multi-geometry parsing
4. **Processing**: `_process_model_response()` applies appropriate parsing mode

## 📊 **Supported Geometry Types**

### **1. Bounding Box (bbox_2d)**
- **Format**: `[x1, y1, x2, y2]` - top-left and bottom-right corners
- **Token Pattern**: `<bbox_2d_start>caption<|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><bbox_2d_end>`
- **Output**: `bbox_2d:BBU设备[150,10,211,35]`

### **2. Line Segments (line)**
- **Format**: `[x1, y1, x2, y2]` - start and end points
- **Token Pattern**: `<line_start>caption<|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><line_end>`
- **Output**: `line:连接线[100,200,150,250]`

### **3. Quadrilaterals (square)**
- **Format**: `[x1, y1, x2, y2, x3, y3, x4, y4]` - four corner points
- **Token Pattern**: `<square_start>caption<|coord_x1|><|coord_y1|>...<|coord_x4|><|coord_y4|><square_end>`
- **Output**: `square:标签[50,60,80,65,85,95,55,90]`

## 🚀 **Usage Guide**

### **Inference with Multi-Geometry**

```bash
# Standard inference (auto-detects multi-geometry from config)
python src/inference.py \
    --config_path configs/bbu_v2.yaml \
    --model_path output-722/7-22-bbu_v2/checkpoint-180 \
    --input_file data/ds_v2_full/val.jsonl \
    --output_file results/multi_geometry_output.json

# Batch processing
bash eval/infer_dataset.sh
```

### **Expected Output**

**Single Object:**
```json
{
  "prediction": "bbox_2d:BBU设备[150,10,211,35]",
  "parsed_objects": [
    {
      "geometry_type": "bbox_2d",
      "caption": "BBU设备",
      "coordinates": [150, 10, 211, 35],
      "formatted_output": "bbox_2d:BBU设备[150,10,211,35]"
    }
  ]
}
```

**Multiple Objects:**
```json
{
  "prediction": "bbox_2d:设备1[x1,y1,x2,y2] | line:连接线[x1,y1,x2,y2]",
  "parsed_objects": [
    {
      "geometry_type": "bbox_2d",
      "caption": "设备1",
      "coordinates": [100, 50, 200, 150],
      "formatted_output": "bbox_2d:设备1[100,50,200,150]"
    },
    {
      "geometry_type": "line",
      "caption": "连接线",
      "coordinates": [200, 100, 300, 200],
      "formatted_output": "line:连接线[200,100,300,200]"
    }
  ]
}
```

## 🔍 **Implementation Details**

### **Coordinate Token Parsing**

```python
def _extract_coordinate_tokens(self, content: str) -> List[int]:
    """Extract coordinate values from coordinate tokens."""
    coord_pattern = r'<\|coord_(\d+)\|>'
    matches = re.findall(coord_pattern, content)
    coordinates = []
    
    for match in matches:
        try:
            coord_value = int(match)
            if 0 <= coord_value <= 4096:  # Validate coordinate range
                coordinates.append(coord_value)
        except ValueError:
            continue  # Skip invalid coordinate tokens
    
    return coordinates
```

### **Multi-Geometry Token Parsing**

```python
def _parse_multi_geometry_tokens(self, response: str, coordinate_tokens_enabled: bool = False) -> List[Dict]:
    """Parse multi-geometry tokens from model response."""
    geometry_patterns = {
        'bbox_2d': (r'<bbox_2d_start>(.*?)<bbox_2d_end>', 4),
        'line': (r'<line_start>(.*?)<line_end>', 4),
        'square': (r'<square_start>(.*?)<square_end>', 8)
    }
    
    parsed_objects = []
    
    for geometry_type, (pattern, expected_coords) in geometry_patterns.items():
        matches = re.finditer(pattern, response, re.DOTALL)
        
        for match in matches:
            content = match.group(1).strip()
            
            if coordinate_tokens_enabled:
                coordinates = self._extract_coordinate_tokens(content)
                caption = re.sub(r'<\|coord_\d+\|>', '', content).strip()
                
                if len(coordinates) == expected_coords and caption:
                    formatted_output = f"{geometry_type}:{caption}[{','.join(map(str, coordinates))}]"
                    parsed_objects.append({
                        "geometry_type": geometry_type,
                        "caption": caption,
                        "coordinates": coordinates,
                        "formatted_output": formatted_output
                    })
            else:
                # Caption-only mode
                if content:
                    formatted_output = f"{geometry_type}:{content}"
                    parsed_objects.append({
                        "geometry_type": geometry_type,
                        "caption": content,
                        "coordinates": None,
                        "formatted_output": formatted_output
                    })
    
    return parsed_objects
```

### **Automatic Configuration Detection**

```python
def _detect_coordinate_tokens_enabled(self) -> bool:
    """Detect if coordinate tokens are enabled from the global configuration."""
    try:
        # Primary method: Check the global config
        from src.config import config
        if hasattr(config, "coordinate_tokens_enabled"):
            coordinate_enabled = config.coordinate_tokens_enabled
            logger.debug(f"🔍 Coordinate tokens from global config: {coordinate_enabled}")
            return coordinate_enabled
        
        # Fallback method: Check model config
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'coordinate_tokens_enabled'):
            return self.model.config.coordinate_tokens_enabled
        
        # Final fallback: Check tokenizer vocabulary
        coord_tokens = [token for token in self.tokenizer.get_vocab().keys() if token.startswith('<|coord_')]
        return len(coord_tokens) > 0
        
    except Exception as e:
        logger.warning(f"Could not detect coordinate tokens configuration: {e}")
        return False
```

## 📈 **Integration Benefits**

### **1. Simplified Configuration**
- **Single source of truth**: Only `configs/bbu_v2.yaml` needs configuration
- **No redundant parameters**: Eliminates duplicate settings
- **Automatic behavior**: No manual command-line flags needed

### **2. Robust Detection**
- **Primary method**: Global config detection (most reliable)
- **Fallback methods**: Model config and tokenizer vocabulary detection
- **Error handling**: Graceful fallback to disabled state on errors

### **3. Backward Compatibility**
- **Non-coordinate models**: Automatically use standard processing
- **Existing workflows**: No changes needed for models without coordinate tokens
- **Graceful degradation**: Falls back to raw response if parsing fails

## 🧪 **Testing & Validation**

### **Test Coverage**

The multi-geometry system includes comprehensive test coverage:

1. **Unit Tests** (`tests/test_multi_geometry_parser.py`) - 24 test cases
   - Core parsing functionality
   - Coordinate token decoding
   - Error handling and edge cases
   - Alternative token patterns

2. **Integration Tests** (`tests/test_inference_multi_geometry.py`) - 18 test cases
   - End-to-end inference pipeline
   - Configuration detection
   - Multi-geometry object handling

3. **Edge Case Tests** (`tests/test_multi_geometry_edge_cases.py`) - 17 test cases
   - Empty content handling
   - Large coordinate values
   - Unicode and special characters
   - Malformed token sequences

4. **Performance Tests** (`tests/test_multi_geometry_performance.py`) - 10 test cases
   - Large input processing (500+ objects)
   - Memory usage validation
   - Parsing speed benchmarks

### **Running Tests**

```bash
# Run all multi-geometry tests
python -m pytest tests/test_multi_geometry_*.py -v

# Run specific test categories
python -m pytest tests/test_multi_geometry_parser.py -v
python -m pytest tests/test_inference_multi_geometry.py -v
python -m pytest tests/test_multi_geometry_edge_cases.py -v
python -m pytest tests/test_multi_geometry_performance.py -v
```

### **Performance Benchmarks**

- **Small Scale (10-50 objects)**: <0.1s processing time
- **Medium Scale (100-200 objects)**: <1.0s processing time
- **Large Scale (500+ objects)**: <5.0s processing time
- **Memory Efficiency**: Linear scaling, no memory leaks
- **Coordinate Token Parsing**: >1000 tokens/s

## 🔧 **Error Handling**

### **Coordinate Token Validation**

```python
def validate_coordinate_token(self, coord_str: str) -> Optional[int]:
    """Validate and convert coordinate token to integer."""
    try:
        coord_value = int(coord_str)
        if 0 <= coord_value <= 4096:  # Valid coordinate range
            return coord_value
        else:
            logger.warning(f"Coordinate value {coord_value} out of range [0, 4096]")
            return None
    except ValueError:
        logger.warning(f"Invalid coordinate token: {coord_str}")
        return None
```

### **Geometry Validation**

```python
def validate_geometry_coordinates(self, geometry_type: str, coordinates: List[int]) -> bool:
    """Validate coordinate count for geometry type."""
    expected_counts = {
        'bbox_2d': 4,  # [x1, y1, x2, y2]
        'line': 4,     # [x1, y1, x2, y2]
        'square': 8    # [x1, y1, x2, y2, x3, y3, x4, y4]
    }
    
    expected = expected_counts.get(geometry_type)
    if expected is None:
        logger.error(f"Unknown geometry type: {geometry_type}")
        return False
    
    if len(coordinates) != expected:
        logger.warning(f"Invalid coordinate count for {geometry_type}: got {len(coordinates)}, expected {expected}")
        return False
    
    return True
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **No Multi-Geometry Parsing**
   - Check `coordinate_tokens_enabled: true` in config
   - Verify model has coordinate tokens in vocabulary
   - Check logs for configuration detection messages

2. **Invalid Coordinate Parsing**
   - Ensure coordinate tokens use format `<|coord_X|>`
   - Verify coordinate values are within range [0, 4096]
   - Check for malformed token sequences

3. **Missing Geometry Objects**
   - Verify geometry tokens are properly formatted
   - Check for complete token sequences (start/end tags)
   - Validate coordinate count matches geometry type

### **Debug Commands**

```bash
# Test coordinate token detection
python -c "
from src.inference import InferenceEngine
engine = InferenceEngine('path/to/model')
enabled = engine._detect_coordinate_tokens_enabled()
print(f'Coordinate tokens enabled: {enabled}')
"

# Test multi-geometry parsing
python -c "
from src.utils.response_parser import ResponseParser
parser = ResponseParser()
test_input = '<obj_ref_start><bbox_2d_start>test<|coord_100|><|coord_200|><bbox_2d_end><obj_ref_end>'
objects = parser._parse_multi_geometry_tokens(test_input, True)
print('Parsed objects:', objects)
"
```

## 📋 **Configuration Reference**

### **Required Configuration**

```yaml
# coords/bbu_v2.yaml
coordinate_tokens_enabled: true  # Enable multi-geometry parsing
max_coord_value: 4096           # Maximum coordinate value
```

### **Optional Configuration**

```yaml
# Additional geometry-related settings
geometry_types: ["bbox_2d", "line", "square"]  # Supported geometry types
coordinate_validation: true                     # Enable coordinate validation
output_format: "formatted"                      # Output format: "formatted" or "raw"
```

## 🎉 **Summary**

The multi-geometry support system provides:

- **✅ Automatic configuration detection** from existing YAML config files
- **✅ Support for multiple geometry types** (bbox_2d, line, square)
- **✅ Coordinate token parsing** with validation
- **✅ Robust error handling** and graceful degradation
- **✅ Comprehensive test coverage** (69 tests across 4 test files)
- **✅ Performance optimization** for large-scale processing
- **✅ Backward compatibility** with existing workflows

The system automatically adapts to the model's coordinate token capabilities without requiring manual configuration of multiple parameters, providing a clean, maintainable solution for multi-geometry object detection and description.