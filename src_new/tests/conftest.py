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

# Import real component fixtures
from src_new.tests.fixtures.real_components import *  # noqa: F401,F403


# Test data constants - Updated for latest architecture
SAMPLE_CONFIG_DATA = {
    # Model settings
    "model_path": "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024",
    "model_size": "3B",
    "model_max_length": 32000,  # Updated to match documentation
    "attn_implementation": "flash_attention_2",
    "torch_dtype": "bfloat16",
    "use_cache": False,
    "model_hidden_size": 2048,
    # Training parameters
    "num_train_epochs": 20,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-6,
    "vision_lr": 5e-7,
    "merger_lr": 1e-5,  # Updated value
    "llm_lr": 5e-6,
    "warmup_ratio": 0.1,
    "weight_decay": 0.0001,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": True,
    "bf16": True,
    # Data settings
    "train_data_path": "data/ds_v2_full/train.jsonl",
    "val_data_path": "data/ds_v2_full/val.jsonl",
    "teacher_pool_file": "data/ds_v2_full/teacher_pool.jsonl",
    "max_total_length": 12000,  # Updated from documentation
    "collator_type": "packed",
    "language": "chinese",
    "max_pixels": 401408,  # Added from documentation
    # Coordinate token system
    "coordinate_tokens_enabled": True,
    "max_coord_value": 1024,  # Updated to match documentation
    "coordinate_loss_weight": 0.05,
    "regular_loss_weight": 1.0,
    "coordinate_temperature": 1.0,  # Fixed: use coordinate_temperature
    # Teacher-student training
    "teacher_ratio": 0.5,
    "num_teacher_samples": 1,
    "teacher_loss_weight": 0.3,
    "student_loss_weight": 1.0,
    # Performance optimization
    "use_flash_attention": True,
    "mixed_precision": "bf16",
    "dataloader_num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    "save_safetensors": True,
    # Output settings
    "output_dir": "test_output",
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
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
                "quad": [209, 477, 254, 486, 252, 500, 211, 490],
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
        "pixel_values": torch.randn(4, 1024),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
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
        "pixel_values": torch.randn(4, 1024),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mock_config():
    """Create a mock configuration object with latest architecture settings."""
    config = Mock()
    # Copy all settings from SAMPLE_CONFIG_DATA
    for key, value in SAMPLE_CONFIG_DATA.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def extended_tokenizer(real_base_tokenizer):
    """Return the already extended tokenizer from the pre-expanded cache.
    Kept for backwards compatibility with tests expecting this fixture.
    """
    return real_base_tokenizer


@pytest.fixture
def mock_extended_tokenizer():
    """Create a mock extended tokenizer with coordinate tokens."""
    tokenizer = Mock()
    tokenizer.vocab_size = 151665 + 1027  # Base + coordinate tokens + line tokens
    tokenizer.model_max_length = 32000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    # Create vocabulary with coordinate tokens
    vocab = {"<|pad|>": 0, "<|eos|>": 1}
    # Add line tokens
    vocab["<|line_start|>"] = 151665
    vocab["<|line_end|>"] = 151666
    # Add coordinate tokens
    for i in range(1025):  # 0 to 1024
        vocab[f"<|coord_{i}|>"] = 151667 + i

    tokenizer.get_vocab.return_value = vocab
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test decoded text"

    return tokenizer


@pytest.fixture
def mock_loss_components():
    """Create mock loss components for testing."""
    from src_new.models.loss_manager import LossComponents

    return LossComponents(
        loss=torch.tensor(1.5),
        teacher_llm_loss=torch.tensor(0.8),
        student_llm_loss=torch.tensor(0.6),
        teacher_l1_loss=torch.tensor(0.05),
        student_l1_loss=torch.tensor(0.05),
    )


@pytest.fixture
def sample_teacher_student_spans():
    """Create sample teacher-student spans for testing."""
    return {
        "teacher_spans": [[(5, 10), (15, 20)], [(8, 12)]],
        "student_spans": [[(25, 30)], [(18, 25)]],
    }


@pytest.fixture
def sample_coordinate_data():
    """Create sample coordinate data for testing."""
    return [
        {
            "images": ["test_image_1.jpg"],
            "objects": [
                {"bbox_2d": [100, 150, 200, 250], "desc": "BBU设备"},
                {"quad": [50, 60, 70, 80, 90, 100, 110, 120], "desc": "标签贴纸"},
            ],
            "width": 800,
            "height": 600,
        },
        {
            "images": ["test_image_2.jpg"],
            "objects": [{"line": [10, 20, 30, 40, 50, 60], "desc": "光纤线缆"}],
            "width": 640,
            "height": 480,
        },
    ]


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
    config.addinivalue_line(
        "markers", "real_data: mark test as requiring real data files"
    )
