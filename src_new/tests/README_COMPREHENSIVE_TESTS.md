# Comprehensive Test Suite for Qwen2.5-VL Coordinate Token System

This directory contains a comprehensive test suite for the Qwen2.5-VL coordinate token system implementation. The tests are organized into a clear hierarchical structure that provides thorough coverage of all components while maintaining clarity and maintainability.

## 🎯 Test Coverage Goals

The comprehensive test suite provides complete coverage of the Qwen2.5-VL coordinate token system:

1. **Unit Testing** - Individual component testing with mocks and fixtures
2. **Integration Testing** - Component interaction and end-to-end pipeline validation
3. **Real Data Validation** - Testing with actual dataset files and real-world scenarios
4. **Coordinate Token System** - Complete testing of coordinate token processing and loss computation
5. **Teacher-Student Training** - Multi-turn conversation handling and span-based loss masking
6. **Configuration Validation** - Comprehensive configuration loading and validation testing

## 📁 Directory Structure

```
src_new/tests/
├── unit/                           # Unit tests for individual components
│   ├── configuration/              # Configuration validation tests
│   │   ├── test_config_validation.py       # Comprehensive config loading and validation
│   │   └── test_config_*.py               # Specific configuration component tests
│   ├── data_processing/            # Data processing and dataset tests
│   │   ├── test_multimodal_processing.py  # Multi-modal data processing tests
│   │   ├── test_conversation_flow.py      # Conversation format handling
│   │   └── test_dataset_*.py              # Dataset component tests
│   ├── loss_computation/           # Loss manager and computation tests
│   │   ├── test_loss_manager.py           # Comprehensive loss computation tests
│   │   └── test_coordinate_loss.py        # Coordinate-specific loss tests
│   ├── models/                     # Model wrapper and component tests
│   │   ├── test_detection_model.py        # DetectionModel wrapper tests
│   │   ├── test_debug_logging_*.py        # Debug logging integration tests
│   │   └── test_rank_aware_logging.py     # Distributed training logging tests
│   ├── token_processing/           # Coordinate token system tests
│   │   ├── test_coordinate_system.py      # Complete coordinate token system tests
│   │   ├── test_coordinate_modes.py       # Coordinate mode switching tests
│   │   ├── test_coordinate_converter.py   # Coordinate conversion tests
│   │   └── test_token_processor.py        # Token processor tests
│   └── training_pipeline/          # Training pipeline component tests
│       ├── test_token_masking.py          # Token masking and span validation
│       └── test_trainer_*.py              # Trainer component tests
├── integration/                    # Integration and end-to-end tests
│   ├── end_to_end/                 # Complete pipeline integration tests
│   │   ├── test_training_pipeline_integration.py  # Full training pipeline tests
│   │   └── test_real_data_pipeline.py             # Real dataset integration tests
│   ├── inference/                 # Inference pipeline and eval integration
│   │   ├── test_comprehensive_inference.py        # Comprehensive inference tests
│   │   ├── test_end_to_end_pipeline.py            # End-to-end inference validation
│   │   └── test_path_resolution.py                # PathManager integration tests
│   └── real_data/                 # Real data validation tests
│       └── test_edge_cases.py               # Malformed data and boundary testing
├── fixtures/                      # Test fixtures and mock objects
│   ├── mock_objects.py                    # Mock model and tokenizer objects
│   └── sample_data.py                     # Sample data for testing
├── conftest.py                    # Pytest configuration and shared fixtures
└── README_COMPREHENSIVE_TESTS.md  # This documentation
```

## 🧪 Test Categories

### Unit Tests (`unit/`)

**Purpose**: Test individual components in isolation using mocks and fixtures.

- **Configuration Tests**: Validate configuration loading, field validation, and default values
- **Data Processing Tests**: Test dataset loading, conversation processing, and multi-modal data handling
- **Loss Computation Tests**: Validate dual-loss system (LLM + coordinate) with mathematical correctness
- **Model Tests**: Test DetectionModel wrapper, coordinate mode switching, and model integration
- **Token Processing Tests**: Test complete coordinate token system, conversion, and processing
- **Training Pipeline Tests**: Test training components, token masking, and span handling

### Integration Tests (`integration/`)

**Purpose**: Test component interactions and complete workflows.

- **End-to-End Tests**: Complete training pipeline from data loading to model training
- **Real Data Tests**: Validation using actual dataset files and real-world scenarios

## 🚀 Running Tests

### Run All Tests
```bash
cd /data3/Qwen2.5-VL-main
python -m pytest src_new/tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest src_new/tests/unit/ -v

# Integration tests only
python -m pytest src_new/tests/integration/ -v

# Specific component tests
python -m pytest src_new/tests/unit/token_processing/ -v
python -m pytest src_new/tests/unit/loss_computation/ -v
```

### Run Tests with Markers
```bash
# Run only unit tests
python -m pytest src_new/tests/ -m unit -v

# Run only integration tests
python -m pytest src_new/tests/ -m integration -v

# Run tests requiring real data
python -m pytest src_new/tests/ -m real_data -v

# Skip slow tests
python -m pytest src_new/tests/ -m "not slow" -v
```

## 🔧 Test Configuration

### Pytest Markers

The test suite uses the following pytest markers for categorization:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, component interaction)
- `@pytest.mark.real_data` - Tests requiring real data files
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.performance` - Performance benchmarking tests
- `@pytest.mark.regression` - Regression testing

### Fixtures and Mock Objects

The test suite provides comprehensive fixtures in `conftest.py`:

- `mock_config` - Mock configuration with latest architecture settings
- `mock_extended_tokenizer` - Mock tokenizer with coordinate tokens
- `mock_loss_components` - Mock loss components for testing
- `sample_teacher_student_spans` - Sample span data for teacher-student training
- `sample_coordinate_data` - Sample coordinate data for testing

## 📊 Test Coverage

### Core Components Covered

1. **Coordinate Token System**
   - Token conversion and validation
   - Tokenizer vocabulary extension
   - Coordinate mask creation and processing
   - Coordinate extraction from tokens

2. **Loss Computation**
   - Dual-loss system (LLM + coordinate)
   - Soft expectation coordinate loss
   - Teacher-student loss splitting
   - Loss aggregation and weighting

3. **Data Processing**
   - Multi-modal data handling
   - Conversation format processing
   - Teacher-student data preparation
   - Data collation and batching

4. **Model Integration**
   - DetectionModel wrapper functionality
   - Coordinate mode switching
   - Forward pass with coordinate tokens
   - Model saving and loading

5. **Training Pipeline**
   - Complete training workflow
   - Token masking and span handling
   - Distributed training support
   - Checkpoint management

## 🐛 Debugging Tests

### Running Individual Tests
```bash
# Run a specific test file
python -m pytest src_new/tests/unit/token_processing/test_coordinate_system.py -v

# Run a specific test method
python -m pytest src_new/tests/unit/loss_computation/test_loss_manager.py::TestLossManager::test_loss_manager_initialization -v
```

### Debug Output
```bash
# Run with detailed output
python -m pytest src_new/tests/ -v -s

# Run with coverage report
python -m pytest src_new/tests/ --cov=src_new --cov-report=html

# Run with profiling
python -m pytest src_new/tests/ --profile
```

## 📝 Adding New Tests

### Test File Naming Convention
- Unit tests: `test_<component_name>.py`
- Integration tests: `test_<workflow_name>_integration.py`
- Real data tests: `test_<scenario>_validation.py`

### Test Class and Method Naming
```python
class TestComponentName:
    """Test suite for ComponentName."""

    def test_component_initialization(self):
        """Test component initialization."""
        pass

    def test_component_functionality(self):
        """Test specific functionality."""
        pass
```

### Using Fixtures
```python
def test_with_fixtures(self, mock_config, mock_tokenizer):
    """Test using shared fixtures."""
    # Test implementation using fixtures
    pass
```

## 🎯 Best Practices

1. **Use Descriptive Test Names** - Test names should clearly describe what is being tested
2. **Test One Thing at a Time** - Each test should focus on a single aspect of functionality
3. **Use Appropriate Fixtures** - Leverage shared fixtures to reduce code duplication
4. **Mock External Dependencies** - Use mocks for external services and heavy dependencies
5. **Test Edge Cases** - Include tests for boundary conditions and error scenarios
6. **Keep Tests Fast** - Unit tests should run quickly; use integration tests for slower scenarios
7. **Document Complex Tests** - Add docstrings explaining the purpose of complex test scenarios

## 🔄 Maintenance

### Updating Tests After Code Changes

1. **Import Changes** - Update imports when modules are moved or renamed
2. **API Changes** - Update test calls when method signatures change
3. **Configuration Changes** - Update mock configurations when new fields are added
4. **New Features** - Add tests for new functionality as it's implemented

### Regular Test Maintenance

- Run the full test suite regularly to catch regressions
- Update fixtures when the underlying architecture changes
- Remove obsolete tests when features are deprecated
- Add performance tests for critical paths
├── test_validation/           # Edge cases and boundary conditions
│   └── test_edge_cases.py              # Malformed data and boundary testing
└── run_comprehensive_tests.py          # Test runner script
```

## 🚀 Quick Start

### Prerequisites

- Conda environment `ms` activated
- Real dataset files in `data/ds_v2_full/`:
  - `teacher_pool.jsonl`
  - `train.jsonl`
  - `val.jsonl`
  - `images/` directory
- Qwen2.5-VL model cached at configured path

### Running Tests

```bash
# Run all comprehensive tests
python src_new/tests/run_comprehensive_tests.py

# Run specific test categories
python src_new/tests/run_comprehensive_tests.py --category integration
python src_new/tests/run_comprehensive_tests.py --category inference
python src_new/tests/run_comprehensive_tests.py --category training
python src_new/tests/run_comprehensive_tests.py --category validation

# Run with verbose output
python src_new/tests/run_comprehensive_tests.py --verbose

# Run individual test files
python -m pytest src_new/tests/integration/end_to_end/test_real_data_pipeline.py -v
python -m pytest src_new/tests/unit/training_pipeline/test_loss_computation.py -v
python -m pytest src_new/tests/integration/real_data/test_edge_cases.py -v
```

## 📋 Test Categories

### 1. Integration Tests (`integration/`)

**Real Data Pipeline Tests** (`test_real_data_pipeline.py`):
- ✅ Real dataset file loading and validation
- ✅ Teacher pool manager with actual data
- ✅ Coordinate conversion with real geometry data
- ✅ Conversation flow with real samples
- ✅ Teacher-student conversation creation

**Complete Pipeline Tests** (`test_complete_pipeline.py`):
- ✅ End-to-end workflow validation
- ✅ Performance monitoring and benchmarking
- ✅ Memory usage validation
- ✅ Throughput measurement
- ✅ Integration component testing

### 1.1 Inference Tests (`integration/inference/`)

These tests validate the end-to-end inference pipeline, path resolution, and eval script compatibility without requiring heavy HF components:

- `test_comprehensive_inference.py`: Single/multi-image, teacher-student, coordinate token parsing, edge cases, and performance.
- `test_eval_script_simulation.py`: Simulated eval script integration, batch processing, and output format.
- `test_path_resolution.py`: `PathManager` behavior in realistic scenarios.

Run only inference tests:

```bash
python -m pytest src_new/tests/integration/inference -v
```

### 2. Training Tests (`unit/training_pipeline/`)

**Loss Computation Tests** (`test_loss_computation.py`):
- ✅ Coordinate token masking with real data
- ✅ Teacher-student loss breakdown
- ✅ Dual-loss system validation (LLM + coordinate)
- ✅ Loss component mathematical correctness
- ✅ Span-based loss computation

**Token Masking Tests** (`test_token_masking.py`):
- ✅ Conversation token structure logging
- ✅ Label masking with real conversations
- ✅ System/user prompt masking (-100 indices)
- ✅ Assistant token identification
- ✅ Image pad token handling

### 3. Validation Tests (`integration/real_data/`)

**Edge Cases Tests** (`test_edge_cases.py`):
- ✅ Empty objects list handling
- ✅ Coordinate boundary conditions
- ✅ Malformed data error handling
- ✅ Template parsing edge cases
- ✅ Unicode and special character support

## 🔍 Key Test Features

### Real Data Integration
- Uses actual `data/ds_v2_full/` files instead of mock data
- Tests with real BBU equipment samples and Chinese descriptions
- Validates actual coordinate geometries (bbox_2d, quad, line)
- Tests with real image dimensions and file references

### Detailed Step-by-Step Logging
```python
logger.info("🧪 Testing conversation flow with real data...")
logger.info("📝 Testing simple conversation (student only)...")
logger.info(f"📊 Input shape: {simple_inputs['input_ids'].shape}")
logger.info(f"🔤 First 20 tokens: {first_tokens}")
logger.info("✅ Simple conversation created successfully")
```

### Comprehensive Validation
- Token ID assignment and structure validation
- Conversation boundary identification
- Loss computation mathematical correctness
- Memory and performance monitoring
- Error handling and graceful degradation

### Edge Case Coverage
- Boundary conditions with extreme coordinates
- Malformed data patterns from real scenarios
- Template parsing robustness
- Unicode and special character handling
- Memory efficiency under stress conditions

## 📊 Expected Test Results

When all tests pass, you should see:

```
📊 COMPREHENSIVE TEST SUMMARY
============================================================
Total tests: 12
Passed: 12 ✅
Failed: 0 ❌
Success rate: 100.0%
Total time: 45.2s
🎉 ALL COMPREHENSIVE TESTS PASSED!
✅ Training pipeline is ready for production use
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**
   ```
   ❌ Real data file not found: /data3/Qwen2.5-VL-main/data/ds_v2_full/train.jsonl
   ```
   **Solution**: Ensure all required JSONL files exist in `data/ds_v2_full/`

2. **Model Path Issues**
   ```
   ❌ Real config file not found
   ```
   **Solution**: Verify `configs/bbu_v2_debug.yaml` exists and model path is correct

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   **Solution**: Run tests on GPU with sufficient memory or reduce batch sizes

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src_new'
   ```
   **Solution**: Run from project root directory where `src_new/` is visible

### Debug Mode

For detailed debugging, run with maximum verbosity:

```bash
python src_new/tests/run_comprehensive_tests.py --verbose --category integration
```

This will show detailed token-level logging and step-by-step processing information.

## 🔗 Integration with Existing Tests

These comprehensive tests complement the existing unit tests:

- **Existing tests**: Mock data, isolated component testing
- **Comprehensive tests**: Real data, end-to-end integration, production scenarios

Both test suites should be run to ensure complete coverage:

```bash
# Run existing unit tests
python -m pytest src_new/tests/unit/configuration/ -v
python -m pytest src_new/tests/unit/data_processing/ -v
python -m pytest src_new/tests/unit/token_processing/ -v

# Run comprehensive integration tests
python src_new/tests/run_comprehensive_tests.py
```

## 📈 Performance Benchmarks

The comprehensive tests also provide performance benchmarks:

- **Data loading time**: ~2-5 seconds for full dataset
- **Processing throughput**: ~2-5 samples/second
- **Memory usage**: Monitored per batch
- **Token generation rate**: Measured tokens/second

These benchmarks help identify performance regressions and optimization opportunities.
