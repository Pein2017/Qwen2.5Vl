"""Pytest fixtures for GRPO diagnostics tests.

Feature: 004-grpo-post-training
Constitution: v4.1.1
"""

from unittest.mock import MagicMock, Mock

import pytest
import torch


@pytest.fixture
def mock_tokenizer():
    """Mock Qwen2.5-VL tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.model_max_length = 2048
    tokenizer.pad_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.decode = Mock(
        return_value="<|im_start|>user\nDescribe this image.<|im_end|><|im_start|>assistant\nA test caption<|im_end|>"
    )
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    return tokenizer


@pytest.fixture
def mock_image_processor():
    """Mock Qwen2.5-VL image processor for testing."""
    processor = MagicMock()
    processor.merge_size = 2
    processor.patch_size = 14
    processor.image_mean = [0.5, 0.5, 0.5]
    return processor


@pytest.fixture
def mock_model():
    """Mock Qwen2.5-VL model for testing."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.num_hidden_layers = 28
    model.config.vocab_size = 152064
    model.device = torch.device("cuda:0")

    # Mock generate method
    def mock_generate(**kwargs):
        batch_size = kwargs.get("input_ids").shape[0]
        seq_len = 50
        return torch.randint(0, 152064, (batch_size, seq_len))

    model.generate = Mock(side_effect=mock_generate)

    # Mock forward method
    def mock_forward(**kwargs):
        output = MagicMock()
        batch_size = kwargs.get("input_ids").shape[0]
        seq_len = kwargs.get("input_ids").shape[1]
        vocab_size = 152064
        output.logits = torch.randn(batch_size, seq_len, vocab_size)
        output.loss = torch.tensor(2.5)
        return output

    model.forward = Mock(side_effect=mock_forward)
    model.__call__ = Mock(side_effect=mock_forward)

    return model


@pytest.fixture
def mock_image_grid_thw():
    """Mock image_grid_thw tensor for testing."""
    # Single image: 1568×1176 smart-resized → 56×42 grid → THW = [1, 56, 42]
    return torch.tensor([[1, 56, 42]], dtype=torch.int64)


@pytest.fixture
def mock_pixel_values(mock_image_grid_thw):
    """Mock pixel_values tensor for testing."""
    # Row count should match sum(t*h*w) from image_grid_thw
    t, h, w = mock_image_grid_thw[0].tolist()
    num_patches = t * h * w  # 1 * 56 * 42 = 2352
    channels = 3
    patch_size = 14
    return torch.randn(num_patches, channels, patch_size, patch_size)


@pytest.fixture
def mock_reward_tensor():
    """Mock reward tensor [prompt_batch_size, K] for testing."""
    prompt_batch_size = 4
    K = 4
    # Create rewards with some variance for diversity testing
    return torch.randn(prompt_batch_size, K) * 0.1 + 0.5


@pytest.fixture
def mock_ratio_tensor():
    """Mock GRPO ratio tensor for trust region testing."""
    # Healthy ratios should be close to 1.0 with std > 0.1
    ratios = torch.randn(64) * 0.2 + 1.0  # mean=1.0, std≈0.2
    return torch.clamp(ratios, 0.5, 1.5)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for diagnostic exports."""
    output_dir = tmp_path / "diagnostics_test"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_accelerator():
    """Mock Accelerate accelerator for distributed training tests."""
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.process_index = 0
    accelerator.num_processes = 8
    accelerator.device = torch.device("cuda:0")
    accelerator.backward = Mock()
    return accelerator
