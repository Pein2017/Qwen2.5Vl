# BBU Training Pipeline Test Suite

Professional test suite for comprehensive validation of the BBU (Base-Band Unit) equipment detection training pipeline using Qwen2.5-VL vision-language model.

## Overview

This test suite provides comprehensive coverage of all training pipeline components with realistic synthetic data, ensuring robustness and reliability before production training runs.

### Test Coverage

- **Data Pipeline**: BBU dataset loading, both collator types, chat processing
- **Model Loading**: Coordinate token enabled/disabled modes, vocabulary validation
- **Training Components**: Forward pass, loss computation, evaluation pipeline
- **Integration**: End-to-end pipeline testing with performance benchmarking

## Quick Start

### Prerequisites

- Conda environment: `ms` (activated)
- CUDA-compatible GPU (A100 recommended)
- Qwen2.5-VL-3B model cached at `/data3/Qwen2.5-VL-main/model_cache/`

### Running Tests

```bash
cd /data3/Qwen2.5-VL-main

# Run all tests
/root/miniconda3/envs/ms/bin/python -m pytest tests/ -v

# Run specific test modules
/root/miniconda3/envs/ms/bin/python -m pytest tests/test_data_pipeline.py -v
/root/miniconda3/envs/ms/bin/python -m pytest tests/test_model_loading.py -v
/root/miniconda3/envs/ms/bin/python -m pytest tests/test_training_components.py -v
/root/miniconda3/envs/ms/bin/python -m pytest tests/test_integration.py -v

# Run with specific markers
/root/miniconda3/envs/ms/bin/python -m pytest tests/ -m "not slow" -v    # Skip slow tests
/root/miniconda3/envs/ms/bin/python -m pytest tests/ -m "gpu" -v         # GPU tests only

# Run as unittest (alternative)
/root/miniconda3/envs/ms/bin/python -m unittest tests.test_data_pipeline -v
```

**Expected runtime**: 5-10 minutes for complete test suite.

## Test Architecture

### Test Modules

#### 1. `test_data_pipeline.py` - Data Pipeline Tests
- **BBU Dataset Loading**: Flat sample validation, teacher-student pairing
- **Data Collators**: StandardDataCollator vs PackedDataCollator efficiency comparison
- **Chat Processing**: Coordinate token replacement, Chinese/English prompts
- **Teacher-Student Integration**: Conversation structuring and span validation
- **Error Handling**: Malformed data rejection and graceful error recovery

#### 2. `test_model_loading.py` - Model Loading Tests
- **Standard Mode**: Base model loading without coordinate tokens
- **Coordinate Mode**: Detection wrapper with extended vocabulary
- **Flash Attention**: Compatibility testing with both model types
- **Vocabulary Extension**: Token range validation and embedding consistency
- **Inference Mode**: Generation capability and cache configuration

#### 3. `test_training_components.py` - Training Components Tests
- **Forward Pass**: Both standard and coordinate modes with loss validation
- **Backward Pass**: Gradient computation and parameter updates
- **Evaluation Mode**: Model.eval() behavior and validation pipeline
- **Memory Management**: GPU memory usage tracking and optimization
- **Trainer Creation**: BBUTrainer instantiation with proper component integration

#### 4. `test_integration.py` - Integration Tests
- **End-to-End Pipeline**: Complete training workflow from data to loss
- **Collator Comparison**: Performance and memory efficiency benchmarking
- **Multi-Geometry Support**: bbox_2d, square, line coordinate validation
- **Error Recovery**: Edge case handling and robustness testing
- **Performance Benchmarking**: Timing and memory usage across configurations

### Test Fixtures

#### `tests/fixtures/synthetic_data.py` - SyntheticDataGenerator
- Generates realistic BBU equipment samples with Chinese labels
- Multi-geometry support (bbox_2d, square, line)
- Proper image dimensions (420×924, multiples of 28)
- Teacher-student conversation structures

#### `tests/fixtures/config_factory.py` - TestConfigFactory
- Creates test configurations for different modes
- Coordinate enabled/disabled variants
- Standard/packed collator configurations
- Performance-optimized settings for fast testing

#### `tests/fixtures/test_utils.py` - TestUtils
- Tensor validation utilities
- Memory efficiency calculations
- Performance measurement context managers
- Error validation and cleanup helpers

## Synthetic Data

### Equipment Labels (Chinese)
```
BBU设备/显示完整，华为/无需安装
标签/5G-BBU-接地线
螺丝、光纤插头/显示完整/BBU安装螺丝/符合要求
电线/无遮挡，捆扎整齐
标签/4G-RRU3-光纤
螺丝、光纤插头/只显示部分/机柜处接地螺丝/符合要求
```

### Geometry Types
- **bbox_2d** (60%): `[x1, y1, x2, y2]`
- **square** (20%): `[x1, y1, x2, y2, x3, y3, x4, y4]`
- **line** (20%): `[x1, y1, ..., xn, yn]`

### Dataset Split
- **Training**: 60% of samples
- **Validation**: 20% of samples  
- **Teacher Pool**: 20% of samples

## Configuration Testing

### Test Configurations

#### Coordinate Token Modes
```yaml
# Coordinate Enabled
coordinate_tokens_enabled: true
max_coord_value: 2048
coordinate_loss_weight: 0.05

# Coordinate Disabled  
coordinate_tokens_enabled: false
max_coord_value: 0
coordinate_loss_weight: 0.0
```

#### Collator Types
```yaml
# Standard Collator (with padding)
collator_type: "standard"

# Packed Collator (memory efficient)
collator_type: "packed"
```

## Performance Benchmarks

### Expected Results

#### Memory Efficiency
- **Standard Collator**: ~80% efficiency (20% padding waste)
- **Packed Collator**: ~98% efficiency (minimal waste)

#### Model Loading
- **Base Model**: ~30-45 seconds
- **Coordinate Model**: ~45-60 seconds (with vocabulary extension)

#### Forward Pass
- **Standard Mode**: <1.5 seconds per batch
- **Coordinate Mode**: <2.0 seconds per batch

#### Memory Usage
- **Model Loading**: ~15-20GB GPU memory
- **Training Batch**: +2-5GB additional

## Validation Criteria

### Data Pipeline
- ✅ Dataset loading with proper validation
- ✅ Collator memory efficiency comparison
- ✅ Chat processor token replacement
- ✅ Teacher-student conversation structure
- ✅ Multi-geometry coordinate processing

### Model Loading
- ✅ Vocabulary size consistency
- ✅ Coordinate token range validation
- ✅ Flash attention compatibility
- ✅ Model-tokenizer embedding alignment
- ✅ Inference vs training mode differences

### Training Components
- ✅ Forward pass loss computation
- ✅ Gradient computation and backpropagation
- ✅ Evaluation mode validation
- ✅ Memory management and cleanup
- ✅ Trainer component integration

### Integration
- ✅ End-to-end pipeline execution
- ✅ Performance benchmarking
- ✅ Error recovery and edge cases
- ✅ Multi-configuration consistency

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in test configurations
# Or run on smaller test dataset
export CUDA_VISIBLE_DEVICES=0
```

#### Model Not Found
```bash
# Ensure model is cached
ls /data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct/
```

#### Import Errors
```bash
# Verify conda environment
/root/miniconda3/envs/ms/bin/python -c "import torch; print(torch.__version__)"
```

### Debug Mode

Enable detailed logging:
```python
# In test files, modify logging level
configure_global_logging(rank=0, world_size=1, level="DEBUG")
```

## Development Workflow

### Adding New Tests

1. **Create Test Class**: Inherit from `unittest.TestCase`
2. **Use Test Fixtures**: Import from `tests.fixtures`
3. **Follow Naming**: `test_specific_functionality`
4. **Add Validation**: Use `TestUtils` for assertions
5. **Clean Resources**: Add cleanup in `tearDown`

### Test Organization

```python
class TestNewFeature(unittest.TestCase):
    """Test new feature with comprehensive validation."""
    
    @classmethod
    def setUpClass(cls):
        """One-time test setup."""
        pass
    
    def setUp(self):
        """Per-test setup."""
        pass
    
    def test_specific_functionality(self):
        """Test specific aspect with descriptive name."""
        pass
    
    def tearDown(self):
        """Per-test cleanup."""
        pass
```

### Performance Testing

Use built-in measurement utilities:
```python
with self.test_utils.measure_time("Operation name"):
    with self.test_utils.measure_memory("Operation name"):
        # Code to benchmark
        pass
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run BBU Tests
  run: |
    cd /data3/Qwen2.5-VL-main
    /root/miniconda3/envs/ms/bin/python -m pytest tests/ -v --tb=short
```

### Performance Regression Detection
Monitor key metrics:
- Model loading time < 60s
- Memory efficiency > 95% (packed collator)
- Forward pass time < 2s per batch
- GPU memory usage < 25GB

## Contributing

### Guidelines
1. **Test First**: Write tests before implementing features
2. **Comprehensive Coverage**: Test both success and failure cases
3. **Realistic Data**: Use synthetic data that matches production format
4. **Performance Aware**: Include timing and memory benchmarks
5. **Documentation**: Update README for new test modules

### Code Standards
- **Type Hints**: Annotate all function parameters and returns
- **Docstrings**: Document test purpose and expected behavior
- **Error Messages**: Provide descriptive assertion messages
- **Resource Cleanup**: Always clean up temporary files and GPU memory

This professional test suite ensures reliable development and deployment of the BBU training pipeline with comprehensive validation and performance monitoring.