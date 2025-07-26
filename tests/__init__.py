"""
BBU Training Pipeline Test Suite

Professional test suite for comprehensive validation of the BBU training pipeline.
Tests all components from data loading to model training with both coordinate
token modes and different collator types.

Test Modules:
- test_data_pipeline: Data loading, collators, chat processing
- test_model_loading: Model loading in different modes
- test_training_components: Trainer, loss manager, forward pass
- test_integration: End-to-end pipeline tests

Usage:
    cd /data3/Qwen2.5-VL-main
    /root/miniconda3/envs/ms/bin/python -m pytest tests/ -v
"""

__version__ = "1.0.0"
__author__ = "BBU Development Team"

# Test configuration constants
TEST_CONFIG_DEFAULTS = {
    "num_samples": 20,
    "image_size": (420, 924),  # Width, Height (multiples of 28)
    "batch_size": 2,
    "max_total_length": 2048,
    "num_train_epochs": 2,
}

# Test data paths
TEST_DATA_ROOT = "test_data_temp"
TEST_MODEL_CACHE = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"