"""
Pytest configuration and fixtures for src_new testing.

This file provides common fixtures and configuration used across all test modules.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml


# Test data constants
SAMPLE_CONFIG_DATA = {
    # Model settings
    "model_path": "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
    "model_size": "3B",
    "model_max_length": 120000,
    "attn_implementation": "flash_attention_2",
    "torch_dtype": "bfloat16",
    # Training settings
    "num_train_epochs": 30,
    "per_device_train_batch_size": 1,
    "learning_rate": 5e-6,
    "vision_lr": 5e-7,
    "merger_lr": 5e-5,
    "llm_lr": 5e-6,
    # Data settings
    "train_data_path": "data/ds_v2_full/train.jsonl",
    "val_data_path": "data/ds_v2_full/val.jsonl",
    "teacher_pool_file": "data/ds_v2_full/teacher_pool.jsonl",
    "teacher_ratio": 0.5,
    "collator_type": "packed",
    "language": "chinese",
    # Coordinate token settings
    "coordinate_tokens_enabled": True,
    "max_coord_value": 2048,
    "coordinate_loss_weight": 0.05,
    "regular_loss_weight": 1.0,
    # Teacher-student settings
    "teacher_loss_weight": 0.3,
    "student_loss_weight": 1.0,
    # Output settings
    "output_dir": "test_output",
    "logging_steps": 10,
}

SAMPLE_JSONL_DATA = [
    {
        "images": ["test_image_1.jpg"],
        "objects": [
            {
                "bbox_2d": [290, 375, 310, 424],
                "desc": "螺丝连接点/光纤插头连接点,显示完整,符合要求",
            },
            {
                "square": [209, 477, 254, 486, 252, 500, 211, 490],
                "desc": "标签贴纸/GPS信号线标识",
            },
        ],
        "width": 532,
        "height": 728,
    },
    {
        "images": ["test_image_2.jpg"],
        "objects": [
            {
                "line": [237, 572, 225, 612, 245, 627, 298, 622, 360, 632, 419, 657],
                "desc": "线缆/有遮挡,捆扎整齐",
            }
        ],
        "width": 800,
        "height": 600,
    },
]


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="qwen25vl_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config_dict():
    """Provide sample configuration data."""
    return SAMPLE_CONFIG_DATA.copy()


@pytest.fixture
def sample_config_yaml(temp_dir, sample_config_dict):
    """Create a temporary YAML config file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def sample_jsonl_data():
    """Provide sample JSONL data."""
    return SAMPLE_JSONL_DATA.copy()


@pytest.fixture
def sample_jsonl_file(temp_dir, sample_jsonl_data):
    """Create a temporary JSONL file."""
    jsonl_path = temp_dir / "test_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sample_jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return jsonl_path


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test decoded text"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 151665
    tokenizer.model_max_length = 120000
    return tokenizer


@pytest.fixture
def mock_image_processor():
    """Create a mock image processor for testing."""
    processor = Mock()
    processor.preprocess.return_value = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "image_grid_thw": torch.tensor([[1, 224, 224]]),
    }
    return processor


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.config = Mock()
    model.config.vocab_size = 151665
    model.config.hidden_size = 2048

    # Mock forward pass
    mock_output = Mock()
    mock_output.loss = torch.tensor(1.5)
    mock_output.logits = torch.randn(1, 100, 151665)
    mock_output.hidden_states = torch.randn(1, 100, 2048)

    model.forward.return_value = mock_output
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    return model


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5]]),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "image_grid_thw": torch.tensor([[1, 224, 224]]),
    }


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state before each test."""
    # Clear any global variables that might affect tests
    yield
    # Cleanup after test
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line(
        "markers", "compatibility: mark test as a compatibility test"
    )
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "regression: mark test as a regression test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
