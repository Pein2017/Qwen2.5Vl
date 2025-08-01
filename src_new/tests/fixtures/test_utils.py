"""
Testing utilities and helper functions.

This module provides utility functions for:
- Configuration comparison and validation
- Batch structure validation
- Tensor shape assertions
- File creation and cleanup
- Test data generation
"""

import json
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml


def compare_configs(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
    ignore_keys: Optional[List[str]] = None,
) -> bool:
    """
    Compare two configuration dictionaries, optionally ignoring certain keys.

    Args:
        config1: First configuration
        config2: Second configuration
        ignore_keys: Keys to ignore during comparison

    Returns:
        True if configurations are equivalent
    """
    if ignore_keys is None:
        ignore_keys = []

    # Create copies to avoid modifying originals
    c1 = {k: v for k, v in config1.items() if k not in ignore_keys}
    c2 = {k: v for k, v in config2.items() if k not in ignore_keys}

    return c1 == c2


def validate_batch_structure(
    batch: Dict[str, torch.Tensor], expected_keys: Optional[List[str]] = None
) -> bool:
    """
    Validate that a batch has the expected structure and tensor properties.

    Args:
        batch: Batch dictionary to validate
        expected_keys: List of required keys

    Returns:
        True if batch structure is valid
    """
    if expected_keys is None:
        expected_keys = ["input_ids", "attention_mask", "labels"]

    # Check required keys exist
    for key in expected_keys:
        if key not in batch:
            return False

    # Check that values are tensors
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            return False

    # Check batch consistency - handle packed vs standard collation
    if "cu_seqlens" in batch:
        # Packed collation mode - different validation rules
        # Text tensors should have batch size 1 (concatenated)
        text_keys = ["input_ids", "attention_mask", "labels"]
        for key in text_keys:
            if key in batch and batch[key].size(0) != 1:
                return False

        # Image tensors can have different batch sizes
        # cu_seqlens should be 1D
        if batch["cu_seqlens"].dim() != 1:
            return False
    else:
        # Standard collation mode - all tensors should have same batch size
        batch_sizes = []
        for key, tensor in batch.items():
            if tensor.dim() >= 1:
                batch_sizes.append(tensor.size(0))

        if len(set(batch_sizes)) > 1:  # Not all batch sizes are the same
            return False

    return True


def assert_tensor_shapes(
    tensors: Dict[str, torch.Tensor], expected_shapes: Dict[str, tuple]
):
    """
    Assert that tensors have expected shapes.

    Args:
        tensors: Dictionary of tensors to check
        expected_shapes: Dictionary of expected shapes (can include -1 for variable dimensions)
    """
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            actual = tensor.shape

            # Check dimension count
            assert len(actual) == len(expected), (
                f"Tensor {name}: expected {len(expected)} dims, got {len(actual)}"
            )

            # Check each dimension (allowing -1 for variable)
            for i, (exp_dim, act_dim) in enumerate(zip(expected, actual)):
                if exp_dim != -1:  # -1 means variable dimension
                    assert act_dim == exp_dim, (
                        f"Tensor {name} dim {i}: expected {exp_dim}, got {act_dim}"
                    )


def create_temp_files(
    files: Dict[str, Union[str, Dict, List]], base_dir: Optional[Path] = None
) -> Path:
    """
    Create temporary files for testing.

    Args:
        files: Dictionary mapping file paths to content
        base_dir: Optional base directory (creates temp dir if None)

    Returns:
        Path to the temporary directory
    """
    if base_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="test_files_"))

    for file_path, content in files.items():
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, str):
            # Text file
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif isinstance(content, dict):
            # JSON/YAML file
            if full_path.suffix.lower() in [".yaml", ".yml"]:
                with open(full_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(content, f)
            else:
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
        elif isinstance(content, list):
            # JSONL file
            with open(full_path, "w", encoding="utf-8") as f:
                for item in content:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return base_dir


def cleanup_temp_files(temp_dir: Path):
    """Clean up temporary files and directories."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def generate_coordinate_tokens(max_coord: int = 2048) -> List[str]:
    """Generate coordinate token strings for testing."""
    return [f"<|coord_{i}|>" for i in range(max_coord)]


def create_mock_conversation(include_teacher: bool = False) -> List[Dict[str, str]]:
    """Create a mock conversation for testing."""
    conversation = [
        {
            "role": "system",
            "content": "你是通信机房设备检测AI助手。请识别图像中的所有目标并输出位置与类别。",
        }
    ]

    if include_teacher:
        # Add teacher example
        conversation.extend(
            [
                {"role": "user", "content": "📚 参考示例: <image>"},
                {
                    "role": "assistant",
                    "content": '[{"bbox_2d":[50,60,70,80], "desc":"示例设备/参考目标"}]',
                },
                {"role": "user", "content": "现在请根据参考示例检测以下图像: <image>"},
            ]
        )
    else:
        conversation.append(
            {"role": "user", "content": "请检测图像中的设备和部件: <image>"}
        )

    conversation.append(
        {
            "role": "assistant",
            "content": '[{"bbox_2d":[100,150,200,250], "desc":"测试设备/基础检测目标"}]',
        }
    )

    return conversation


def assert_config_validity(
    config: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
    check_converted: bool = False,
):
    """Assert that a configuration contains required fields and valid values."""
    if required_fields is None:
        required_fields = [
            "model_path",
            "model_size",
            "learning_rate",
            "train_data_path",
            "val_data_path",
        ]

    # Check required fields
    for field in required_fields:
        assert field in config, f"Missing required config field: {field}"
        assert config[field] is not None, f"Config field {field} cannot be None"

    # Handle scientific notation conversion if needed
    if check_converted:
        from src_new.config.config import _convert_scientific_notation

        config = _convert_scientific_notation(config)

    # Check specific field types and ranges
    if "learning_rate" in config:
        lr_value = config["learning_rate"]
        # Handle both string (raw YAML) and converted float values
        if isinstance(lr_value, str):
            try:
                lr_value = float(lr_value)
            except ValueError:
                assert False, f"learning_rate must be numeric, got string: {lr_value}"

        assert isinstance(lr_value, (int, float)), (
            f"learning_rate must be numeric, got {type(lr_value)}: {lr_value}"
        )
        assert lr_value > 0, f"learning_rate must be positive, got {lr_value}"

    if "num_train_epochs" in config:
        assert isinstance(config["num_train_epochs"], int), (
            "num_train_epochs must be integer"
        )
        assert config["num_train_epochs"] > 0, "num_train_epochs must be positive"

    if "coordinate_tokens_enabled" in config:
        assert isinstance(config["coordinate_tokens_enabled"], bool), (
            "coordinate_tokens_enabled must be boolean"
        )


def compare_model_outputs(
    output1: torch.Tensor, output2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Compare model outputs with tolerance for numerical differences."""
    return torch.allclose(output1, output2, rtol=rtol, atol=atol)


def generate_test_images(
    count: int = 3, size: tuple = (224, 224)
) -> List[torch.Tensor]:
    """Generate random test images as tensors."""
    return [torch.randn(3, size[0], size[1]) for _ in range(count)]


def create_coordinate_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for coordinate token functionality."""
    return [
        {
            "name": "bbox_2d_basic",
            "geometry_type": "bbox_2d",
            "coordinates": [100, 150, 200, 250],
            "expected_tokens": [
                "<|coord_100|>",
                "<|coord_150|>",
                "<|coord_200|>",
                "<|coord_250|>",
            ],
        },
        {
            "name": "quad_complex",
            "geometry_type": "quad",
            "coordinates": [300, 400, 350, 410, 348, 425, 302, 415],
            "expected_tokens": [
                f"<|coord_{c}|>" for c in [300, 400, 350, 410, 348, 425, 302, 415]
            ],
        },
        {
            "name": "line_variable_length",
            "geometry_type": "line",
            "coordinates": [50, 100, 150, 120, 250, 140, 350, 160, 450, 180],
            "expected_tokens": [
                f"<|coord_{c}|>"
                for c in [50, 100, 150, 120, 250, 140, 350, 160, 450, 180]
            ],
        },
        {
            "name": "edge_case_zero",
            "geometry_type": "bbox_2d",
            "coordinates": [0, 0, 10, 10],
            "expected_tokens": [
                "<|coord_0|>",
                "<|coord_0|>",
                "<|coord_10|>",
                "<|coord_10|>",
            ],
        },
        {
            "name": "edge_case_max",
            "geometry_type": "bbox_2d",
            "coordinates": [2040, 2041, 2047, 2047],
            "expected_tokens": [
                "<|coord_2040|>",
                "<|coord_2041|>",
                "<|coord_2047|>",
                "<|coord_2047|>",
            ],
        },
    ]


def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass instances to dictionaries for easier comparison."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    else:
        return obj


class TestMetrics:
    """Simple metrics tracking for tests."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0

        self.metrics[name] += value
        self.counts[name] += 1

    def get_average(self, name: str) -> float:
        if name not in self.metrics or self.counts[name] == 0:
            return 0.0
        return self.metrics[name] / self.counts[name]

    def get_all_averages(self) -> Dict[str, float]:
        return {name: self.get_average(name) for name in self.metrics.keys()}


def skip_if_no_gpu():
    """Decorator to skip tests if GPU is not available."""
    import pytest

    return pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


def skip_if_no_model_cache():
    """Decorator to skip tests if model cache is not available."""
    import pytest

    model_path = Path("/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct")
    return pytest.mark.skipif(
        not model_path.exists(), reason="Model cache not available"
    )


# Context managers for testing
class capture_logs:
    """Context manager to capture log output during testing."""

    def __init__(self):
        self.logs = []
        self.handler = None

    def __enter__(self):
        import logging

        # Create a custom handler that captures logs
        class TestLogHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.logs = log_list

            def emit(self, record):
                self.logs.append(self.format(record))

        self.handler = TestLogHandler(self.logs)
        logging.getLogger().addHandler(self.handler)
        return self.logs

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            import logging

            logging.getLogger().removeHandler(self.handler)


class temporary_config:
    """Context manager for temporarily setting configuration values."""

    def __init__(self, **config_overrides):
        self.overrides = config_overrides
        self.original_values = {}

    def __enter__(self):
        # This would integrate with the actual config system
        # For now, just return the overrides
        return self.overrides

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        pass
