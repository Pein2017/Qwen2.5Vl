# Testing and Validation Guide

This document provides comprehensive testing procedures and validation workflows for the Qwen2.5-VL BBU fine-tuning project.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure](#test-structure)
3. [Data Pipeline Testing](#data-pipeline-testing)
4. [Model Training Testing](#model-training-testing)
5. [Inference Testing](#inference-testing)
6. [Integration Testing](#integration-testing)
7. [Performance Testing](#performance-testing)
8. [Regression Testing](#regression-testing)
9. [Validation Procedures](#validation-procedures)
10. [Continuous Testing](#continuous-testing)

---

## Testing Philosophy

The project follows a **fail-fast testing approach** with comprehensive validation at every stage:

### Core Principles
1. **Explicit Error Handling**: No silent failures or bare `except` clauses
2. **Early Validation**: Catch issues before they compound
3. **Comprehensive Coverage**: Test all critical paths and edge cases
4. **Regression Prevention**: Ensure changes don't break existing functionality
5. **Performance Monitoring**: Track performance regressions

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **System Tests**: End-to-end workflow testing
- **Regression Tests**: Ensure stability across changes
- **Performance Tests**: Benchmark critical operations

---

## Test Structure

### Test Organization
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_data_processing.py
│   ├── test_model_components.py
│   ├── test_coordinate_transforms.py
│   └── test_utilities.py
├── integration/             # Integration tests
│   ├── test_pipeline_integration.py
│   ├── test_training_integration.py
│   └── test_inference_integration.py
├── system/                  # End-to-end system tests
│   ├── test_complete_workflow.py
│   └── test_performance_benchmarks.py
├── fixtures/                # Test data and fixtures
│   ├── sample_data/
│   ├── mock_configs/
│   └── reference_outputs/
└── conftest.py             # Pytest configuration
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/system/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov=data_conversion --cov-report=html

# Run performance tests
python -m pytest tests/system/test_performance_benchmarks.py -v
```

---

## Data Pipeline Testing

### 1. Pipeline Component Tests

#### JSON Cleaning Tests
```bash
# Test JSON cleaning with various input formats
python -m pytest tests/unit/test_json_cleaning.py -v

# Test cases covered:
# - Valid JSON structures
# - Malformed JSON handling
# - Metadata stripping
# - Content preservation
```

#### Coordinate Transformation Tests
```bash
# Test 3-stage coordinate transformation
python -m pytest tests/unit/test_coordinate_transforms.py -v

# Test cases covered:
# - EXIF orientation handling
# - Dimension mismatch rescaling  
# - Smart resize scaling
# - Boundary condition validation
# - Precision preservation
```

#### Sample Processing Tests
```bash
# Test individual sample processing
python -m pytest tests/unit/test_sample_processing.py -v

# Test cases covered:
# - Object extraction from different JSON formats
# - Label hierarchy filtering
# - Field standardization
# - Image processing workflow
```

### 2. Pipeline Validation Tests

#### Data Validation
```bash
# Validate pipeline output
python data_conversion/simple_validate.py

# Comprehensive validation
python -c "
from data_conversion.utils.validators import DataValidator
validator = DataValidator()
result = validator.validate_pipeline_output('data/')
print(f'Validation result: {result}')
"
```

#### Integration Tests
```bash
# Test complete pipeline
python data_conversion/test_pipeline.py

# Test with minimal dataset
python -c "
from data_conversion.pipeline_manager import PipelineManager
manager = PipelineManager()
result = manager.run_pipeline('tests/fixtures/minimal_data/')
assert result.success
"
```

### 3. Edge Case Testing

#### Boundary Conditions
```python
# Test coordinate edge cases
def test_coordinate_boundaries():
    # Test coordinates at image boundaries
    coords = [0, 0, 100, 100]
    result = transform_coordinates(coords, (100, 100), (50, 50))
    assert all(0 <= c <= 50 for c in result)
    
    # Test invalid coordinates
    invalid_coords = [-10, -10, 110, 110]
    with pytest.raises(ValueError):
        validate_coordinates(invalid_coords, (100, 100))
```

#### Data Format Variations
```python
# Test different annotation formats
def test_annotation_formats():
    # Test dataList format
    datalist_sample = {"dataList": [...]}
    result = process_sample(datalist_sample)
    assert result.success
    
    # Test markResult format  
    markresult_sample = {"markResult": [...]}
    result = process_sample(markresult_sample)
    assert result.success
```

---

## Model Training Testing

### 1. Configuration Testing

#### Configuration Validation
```bash
# Test configuration loading
python -c "
from src.config.config_manager import ConfigManager
config = ConfigManager('tests/fixtures/test_config.yaml')
assert config.is_valid()
"

# Test domain-specific configs
python -m pytest tests/unit/test_configuration.py -v
```

#### Parameter Validation
```python
# Test parameter interdependencies
def test_config_dependencies():
    config = load_test_config()
    
    # Test incompatible parameter combinations
    config.pack_sequences = True
    config.max_seq_length = 16384
    
    with pytest.raises(ConfigValidationError):
        config.validate_cross_dependencies()
```

### 2. Model Component Testing

#### Model Loading Tests
```bash
# Test model loading and patching
python -m pytest tests/unit/test_model_components.py -v

# Test specific components
python -c "
from src.models.model_loader import ModelLoader
from src.models.patches import apply_model_patches
loader = ModelLoader('tests/fixtures/mock_checkpoint/')
model = loader.load_model()
patched_model = apply_model_patches(model, config)
assert patched_model is not None
"
```

#### Detection Head Tests
```python
# Test DETR-style detection head
def test_detection_head():
    from src.detection.detection_head import DetectionHead
    
    config = get_test_config()
    head = DetectionHead(config)
    
    # Test forward pass
    vision_features = torch.randn(2, 256, 768)
    language_features = torch.randn(2, 512, 768)
    
    outputs = head(vision_features, language_features)
    
    assert outputs.boxes.shape == (2, 100, 4)
    assert outputs.objectness.shape == (2, 100)
    assert outputs.captions.shape == (2, 100, config.vocab_size)
```

### 3. Loss Function Testing

#### Multi-Task Loss Tests
```python
# Test loss computation
def test_loss_computation():
    from src.training.loss_manager import LossManager
    
    loss_manager = LossManager(config)
    
    # Mock model outputs
    model_outputs = create_mock_outputs()
    inputs = create_mock_inputs()
    
    total_loss, loss_dict = loss_manager.compute_total_loss(model_outputs, inputs)
    
    assert 'lm_loss' in loss_dict
    assert 'detection_loss' in loss_dict
    assert 'teacher_loss' in loss_dict
    assert 'student_loss' in loss_dict
    assert total_loss.requires_grad
```

#### Hungarian Matching Tests
```python
# Test Hungarian matching algorithm
def test_hungarian_matching():
    from src.detection.detection_loss import hungarian_matching
    
    predictions = create_mock_predictions(batch_size=2, num_queries=100)
    targets = create_mock_targets(batch_size=2)
    
    indices = hungarian_matching(predictions, targets)
    
    # Validate matching results
    assert len(indices) == 2  # batch_size
    assert all(len(idx[0]) == len(idx[1]) for idx in indices)
```

---

## Inference Testing

### 1. Inference Pipeline Tests

#### Single Image Inference
```bash
# Test single image inference
python -c "
from src.inference import Inference
inference = Inference('tests/fixtures/mock_checkpoint/')
result = inference.predict_detection('tests/fixtures/test_image.jpg')
assert 'boxes' in result
assert 'captions' in result
"
```

#### Batch Inference Tests
```python
# Test batch processing
def test_batch_inference():
    from src.inference import Inference
    
    inference = Inference('tests/fixtures/mock_checkpoint/')
    images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
    
    results = inference.batch_predict(images)
    
    assert len(results) == 3
    assert all('boxes' in result for result in results)
```

### 2. Response Parser Testing

#### Format Compatibility Tests
```python
# Test response parsing with different formats
def test_response_parsing():
    from src.utils.response_parser import ResponseParser
    
    parser = ResponseParser()
    
    # Test JSON format
    json_response = '{"objects": [{"type": "螺丝", "bbox": [10, 20, 30, 40]}]}'
    result = parser.parse_response(json_response)
    assert len(result.objects) == 1
    
    # Test unquoted format
    unquoted_response = "螺丝/BBU安装螺丝/连接正确"
    result = parser.parse_response(unquoted_response)
    assert result.object_type == "螺丝"
```

### 3. Performance Testing

#### Inference Speed Tests
```python
# Test inference performance
def test_inference_performance():
    import time
    from src.inference import Inference
    
    inference = Inference('tests/fixtures/mock_checkpoint/')
    
    start_time = time.time()
    result = inference.predict_detection('tests/fixtures/test_image.jpg')
    inference_time = time.time() - start_time
    
    # Assert reasonable inference time (adjust threshold as needed)
    assert inference_time < 5.0  # 5 seconds max
```

---

## Integration Testing

### 1. End-to-End Workflow Tests

#### Complete Pipeline Test
```python
# Test complete data processing to training workflow
def test_complete_workflow():
    # Stage 1: Data processing
    from data_conversion.pipeline_manager import PipelineManager
    pipeline = PipelineManager()
    processing_result = pipeline.run_pipeline('tests/fixtures/minimal_data/')
    assert processing_result.success
    
    # Stage 2: Training setup
    from src.training.trainer_factory import create_trainer_with_coordinator
    trainer = create_trainer_with_coordinator('tests/fixtures/test_config.yaml')
    assert trainer is not None
    
    # Stage 3: Single training step
    trainer.train_one_step()  # Mock single step
    
    # Stage 4: Inference
    from src.inference import Inference
    inference = Inference(trainer.model)
    result = inference.predict_detection('tests/fixtures/test_image.jpg')
    assert result is not None
```

### 2. Component Integration Tests

#### Model-Data Integration
```python
# Test model and data compatibility
def test_model_data_integration():
    # Load processed data
    from src.core.data_processor import DataProcessor
    data_processor = DataProcessor()
    batch = data_processor.load_batch('tests/fixtures/processed_data/')
    
    # Load model
    from src.models.model_loader import ModelLoader
    model = ModelLoader('tests/fixtures/mock_checkpoint/').load_model()
    
    # Test forward pass
    outputs = model(**batch)
    assert outputs.logits.shape[0] == batch['input_ids'].shape[0]
```

---

## Performance Testing

### 1. Benchmark Tests

#### Data Processing Benchmarks
```python
# Benchmark data processing performance
def test_data_processing_performance():
    import time
    from data_conversion.unified_processor import UnifiedProcessor
    
    processor = UnifiedProcessor()
    
    start_time = time.time()
    result = processor.process_dataset('tests/fixtures/benchmark_data/')
    processing_time = time.time() - start_time
    
    # Performance assertions (adjust thresholds based on hardware)
    samples_per_second = result.total_samples / processing_time
    assert samples_per_second > 100  # At least 100 samples/second
```

#### Training Performance Benchmarks
```python
# Benchmark training performance
def test_training_performance():
    from src.training.trainer import BBUTrainer
    
    trainer = create_mock_trainer()
    
    # Benchmark single training step
    start_time = time.time()
    loss = trainer.train_one_step()
    step_time = time.time() - start_time
    
    # Performance assertions
    assert step_time < 10.0  # Max 10 seconds per step
    assert not torch.isnan(loss)  # Loss should be valid
```

### 2. Memory Testing

#### Memory Usage Validation
```python
# Test memory usage during training
def test_memory_usage():
    import torch
    from src.training.trainer import BBUTrainer
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # Create trainer and run training step
    trainer = create_mock_trainer()
    trainer.train_one_step()
    
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = peak_memory - initial_memory
    
    # Memory assertions (adjust based on model size)
    assert memory_used < 24 * 1024**3  # Less than 24GB
```

---

## Regression Testing

### 1. Output Consistency Tests

#### Data Processing Consistency
```python
# Test that data processing produces consistent outputs
def test_processing_consistency():
    from data_conversion.unified_processor import UnifiedProcessor
    
    processor = UnifiedProcessor()
    
    # Process same data twice
    result1 = processor.process_dataset('tests/fixtures/reference_data/')
    result2 = processor.process_dataset('tests/fixtures/reference_data/')
    
    # Results should be identical
    assert result1.train_samples == result2.train_samples
    assert result1.val_samples == result2.val_samples
```

#### Model Output Consistency
```python
# Test model output consistency
def test_model_consistency():
    from src.inference import Inference
    
    inference = Inference('tests/fixtures/reference_checkpoint/')
    
    # Same input should produce same output
    result1 = inference.predict_detection('tests/fixtures/test_image.jpg')
    result2 = inference.predict_detection('tests/fixtures/test_image.jpg')
    
    # Compare outputs (allow small numerical differences)
    assert torch.allclose(result1.boxes, result2.boxes, atol=1e-6)
```

### 2. Reference Testing

#### Golden Output Tests
```python
# Test against reference outputs
def test_golden_outputs():
    # Load reference outputs
    with open('tests/fixtures/reference_outputs/expected_result.json') as f:
        expected = json.load(f)
    
    # Generate current output
    current = run_inference_pipeline('tests/fixtures/test_data/')
    
    # Compare with tolerance
    assert abs(current['accuracy'] - expected['accuracy']) < 0.01
    assert current['object_count'] == expected['object_count']
```

---

## Validation Procedures

### 1. Pre-Training Validation

#### Data Quality Checks
```bash
# Comprehensive data validation
python -c "
from data_conversion.utils.validators import DataValidator
validator = DataValidator()

# Validate data format
validator.validate_format('data/train.jsonl')
validator.validate_format('data/val.jsonl')

# Validate coordinates
validator.validate_coordinates('data/')

# Check for data leakage
validator.check_data_leakage('data/train.jsonl', 'data/val.jsonl')
"
```

#### Configuration Validation
```bash
# Validate training configuration
python -c "
from src.config.config_manager import ConfigManager
config = ConfigManager('configs/base_flat_v2.yaml')
config.validate_all_domains()
config.validate_cross_dependencies()
print('Configuration validation passed')
"
```

### 2. Post-Training Validation

#### Model Validation
```bash
# Validate trained model
python -c "
from src.models.model_loader import ModelLoader
loader = ModelLoader('path/to/checkpoint')
model = loader.load_model()
loader.validate_model_integrity(model)
print('Model validation passed')
"
```

#### Performance Validation
```bash
# Validate performance metrics
python eval/validate_results.py --checkpoint path/to/checkpoint --data data/val.jsonl
```

---

## Continuous Testing

### 1. Automated Testing Pipeline

#### Pre-commit Tests
```bash
# Run before each commit
python -m pytest tests/unit/ -x  # Stop on first failure
python data_conversion/simple_validate.py
python -c "from src.config.config_manager import ConfigManager; ConfigManager('configs/base_flat_v2.yaml')"
```

#### CI/CD Pipeline Tests
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: python -m pytest tests/unit/ -v
      - name: Run integration tests
        run: python -m pytest tests/integration/ -v
      - name: Validate data pipeline
        run: python data_conversion/test_pipeline.py
```

### 2. Monitoring and Alerts

#### Performance Monitoring
```python
# Monitor performance regressions
def monitor_performance():
    current_metrics = run_performance_tests()
    baseline_metrics = load_baseline_metrics()
    
    for metric, value in current_metrics.items():
        baseline = baseline_metrics[metric]
        regression_threshold = baseline * 1.1  # 10% regression tolerance
        
        if value > regression_threshold:
            raise PerformanceRegressionError(
                f"Performance regression in {metric}: {value} > {regression_threshold}"
            )
```

---

## Test Data Management

### 1. Test Fixtures

#### Creating Test Data
```bash
# Create minimal test dataset
python -c "
from tests.utils.fixture_generator import create_test_fixtures
create_test_fixtures(
    output_dir='tests/fixtures/',
    num_samples=10,
    image_size=(224, 224)
)
"
```

#### Mock Data Generation
```python
# Generate mock training data
def create_mock_training_batch():
    return {
        'input_ids': torch.randint(0, 1000, (4, 512)),
        'attention_mask': torch.ones(4, 512),
        'labels': torch.randint(0, 1000, (4, 512)),
        'pixel_values': torch.randn(4, 3, 224, 224),
        'bbox_labels': torch.randn(4, 100, 4)
    }
```

### 2. Test Environment Management

#### Test Database
```python
# Maintain test result database
class TestResultDatabase:
    def record_test_result(self, test_name, result, duration):
        # Store test results for trend analysis
        pass
    
    def get_performance_trend(self, test_name, days=30):
        # Retrieve performance trends
        pass
```

---

## Best Practices

### 1. Test Writing Guidelines

- **Clear Test Names**: Use descriptive names that explain what is being tested
- **Single Responsibility**: Each test should verify one specific behavior
- **Independent Tests**: Tests should not depend on each other
- **Deterministic Results**: Tests should produce consistent results
- **Fast Execution**: Optimize test execution time

### 2. Test Maintenance

- **Regular Updates**: Keep tests in sync with code changes
- **Performance Monitoring**: Track test execution time
- **Coverage Analysis**: Maintain high test coverage
- **Documentation**: Document complex test scenarios

### 3. Debugging Failed Tests

```bash
# Debug test failures
python -m pytest tests/failing_test.py -v -s --pdb  # Drop into debugger
python -m pytest tests/ --lf  # Run only last failed tests
python -m pytest tests/ --tb=long  # Detailed traceback
```

---

This comprehensive testing guide ensures the reliability, performance, and maintainability of the Qwen2.5-VL BBU fine-tuning project. Regular testing is essential for maintaining system quality and preventing regressions.