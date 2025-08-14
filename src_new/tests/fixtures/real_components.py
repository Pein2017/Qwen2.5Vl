"""
Real component fixtures for testing with actual Qwen2.5-VL components.

This module provides pytest fixtures that load real tokenizers, processors, and models
instead of using Mock objects, making tests more reliable and realistic.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from src_new.config.config import load_config
from src_new.processing.token_processor import TokenConfig, TokenProcessor


@pytest.fixture(scope="session")
def real_config():
    """Load real configuration from bbu_v2_use_coord.yaml."""
    config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")
    config = load_config(config_path)

    # Override some settings for testing
    config.num_train_epochs = 1
    config.max_dataset_size = 2  # Small dataset for testing
    config.logging_steps = 1
    config.eval_steps = 1
    config.save_steps = 1

    return config


@pytest.fixture(scope="session")
def real_base_tokenizer(real_config):
    """Load the real tokenizer from model_path (expected pre-expanded cache).

    Validates that the tokenizer already includes line and coordinate tokens with the
    strict ID range derived from documentation:
      - <|line_start|> == 151665
      - <|line_end|>   == 151666
      - <|coord_0|>.. <|coord_max|> starting at 151667 and ending at 151667+max_coord_value
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            real_config.model_path, trust_remote_code=True, use_fast=False
        )
        vocab = tokenizer.get_vocab()
        # Sanity: expanded vocab must be larger than the base 151665
        assert len(vocab) > 151665, (
            f"Expanded tokenizer vocab too small: {len(vocab)}; expected > 151665."
        )
        # Validate line tokens
        assert vocab.get("<|line_start|>") == 151665, "<|line_start|> id mismatch"
        assert vocab.get("<|line_end|>") == 151666, "<|line_end|> id mismatch"
        # Validate coordinate tokens exist for configured max range (default 1024)
        max_c = int(getattr(real_config, "max_coord_value", 1024))
        coord_ids = [vocab.get(f"<|coord_{i}|>") for i in range(max_c + 1)]
        assert None not in coord_ids, "Missing coordinate tokens in expanded tokenizer"
        assert min(coord_ids) == 151667, "Coordinate ID start mismatch"
        assert max(coord_ids) == 151667 + max_c, "Coordinate ID end mismatch"
        return tokenizer
    except Exception as e:
        pytest.skip(f"Could not load expanded tokenizer: {e}")


@pytest.fixture(scope="session")
def real_extended_tokenizer(real_base_tokenizer, real_config):
    """Return the already-expanded tokenizer and strictly verify final size and IDs.

    Keeps fixture name for backwards compatibility; does not perform re-extension.
    """
    tokenizer = real_base_tokenizer
    vocab = tokenizer.get_vocab()

    # Exact id checks
    assert vocab.get("<|line_start|>") == 151665, "<|line_start|> id mismatch"
    assert vocab.get("<|line_end|>") == 151666, "<|line_end|> id mismatch"

    max_c = int(getattr(real_config, "max_coord_value", 1024))
    coord_ids = [vocab.get(f"<|coord_{i}|>") for i in range(max_c + 1)]
    assert None not in coord_ids, "Missing coordinate tokens after extension"
    assert min(coord_ids) == 151667 and max(coord_ids) == 151667 + max_c, (
        "Coordinate ID range mismatch"
    )

    return tokenizer


@pytest.fixture(scope="session")
def real_base_model(real_config):
    """Load the real base model (lightweight for testing)."""
    try:
        # Load model with minimal resources for testing
        model = AutoModel.from_pretrained(
            real_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()  # Set to eval mode for testing
        return model
    except Exception as e:
        pytest.skip(f"Could not load real base model: {e}")


@pytest.fixture(scope="session")
def realistic_base_model(real_extended_tokenizer):
    """Create a realistic base model mock that behaves like the real model."""
    from unittest.mock import Mock

    # Create a mock that behaves like a real model
    model = Mock()
    model.config = Mock()
    model.config.vocab_size = len(real_extended_tokenizer)
    model.config.hidden_size = 2048
    # Provide image token id to avoid mock equality quirks
    model.config.image_token_id = 151655

    # Create a proper output class that behaves like ModelOutput
    class RealisticModelOutput:
        def __init__(self, batch_size, seq_len, vocab_size):
            self.loss = torch.tensor(1.5, requires_grad=True)
            self.logits = torch.randn(
                batch_size, seq_len, vocab_size, requires_grad=True
            )
            self.hidden_states = torch.randn(batch_size, seq_len, 2048)

        def __getitem__(self, key):
            return getattr(self, key)

        def __contains__(self, key):
            return hasattr(self, key)

        def keys(self):
            return ["loss", "logits", "hidden_states"]

    # Mock forward method that returns proper tensors
    def realistic_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape[:2]
        else:
            batch_size, seq_len = 1, 10
        vocab_size = len(real_extended_tokenizer)

        return RealisticModelOutput(batch_size, seq_len, vocab_size)

    # Set up all possible call methods
    model.forward = realistic_forward
    model.__call__ = realistic_forward
    model.side_effect = realistic_forward  # For direct Mock() calls

    # Ensure the model behaves like a proper PyTorch module
    model.train.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = []
    model.named_parameters.return_value = []

    # Override any default Mock behavior that might interfere
    model._mock_name = "RealisticBaseModel"
    model._spec_class = None

    return model


@pytest.fixture(scope="session")
def real_processor(real_config):
    """Load the real image/vision processor."""
    try:
        processor = AutoProcessor.from_pretrained(
            real_config.model_path, trust_remote_code=True
        )
        return processor
    except Exception as e:
        pytest.skip(f"Could not load real processor: {e}")


@pytest.fixture
def temp_test_data_dir():
    """Create temporary directory with real test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample training data with real structure
        train_data = [
            {
                "images": ["test_image_1.jpg"],
                "conversations": [
                    {"from": "user", "value": "请识别图中的设备位置。"},
                    {
                        "from": "assistant",
                        "value": "图中有一个BBU设备<|object_ref_start|>BBU设备<|object_ref_end|><|box_start|>[<|coord_100|>, <|coord_150|>, <|coord_200|>, <|coord_250|>]<|box_end|>。",
                    },
                ],
                "objects": [{"bbox_2d": [100, 150, 200, 250], "desc": "BBU设备"}],
                "width": 800,
                "height": 600,
            },
            {
                "images": ["test_image_2.jpg"],
                "conversations": [
                    {"from": "user", "value": "这个设备是什么？"},
                    {
                        "from": "assistant",
                        "value": "这是一个光纤设备<|object_ref_start|>光纤设备<|object_ref_end|><|quad_start|>[<|coord_50|>, <|coord_60|>, <|coord_70|>, <|coord_80|>, <|coord_90|>, <|coord_100|>, <|coord_110|>, <|coord_120|>]<|quad_end|>。",
                    },
                ],
                "objects": [
                    {"quad": [50, 60, 70, 80, 90, 100, 110, 120], "desc": "光纤设备"}
                ],
                "width": 640,
                "height": 480,
            },
        ]

        # Write training data
        train_file = temp_path / "train.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Create teacher pool data
        teacher_data = [
            {
                "images": ["teacher_image_1.jpg"],
                "conversations": [
                    {"from": "user", "value": "识别设备。"},
                    {"from": "teacher", "value": "这是专业的设备识别结果。"},
                    {
                        "from": "assistant",
                        "value": "设备位于<|object_ref_start|>设备<|object_ref_end|><|box_start|>[<|coord_300|>, <|coord_400|>, <|coord_500|>, <|coord_600|>]<|box_end|>。",
                    },
                ],
                "objects": [{"bbox_2d": [300, 400, 500, 600], "desc": "设备"}],
                "width": 1024,
                "height": 768,
            }
        ]

        # Write teacher pool data
        teacher_file = temp_path / "teacher_pool.jsonl"
        with open(teacher_file, "w", encoding="utf-8") as f:
            for item in teacher_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Create real test images
        image_names = ["test_image_1.jpg", "test_image_2.jpg", "teacher_image_1.jpg"]
        for img_name in image_names:
            # Create a simple colored image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(temp_path / img_name)

        yield temp_path


@pytest.fixture
def real_test_config(real_config, temp_test_data_dir):
    """Create test configuration with real components and temporary data paths."""
    import copy

    # Create a deep copy of the real config
    test_config = copy.deepcopy(real_config)

    # Override paths to use temporary test data
    test_config.data_root = str(temp_test_data_dir)
    test_config.train_data_path = str(temp_test_data_dir / "train.jsonl")
    test_config.teacher_pool_file = str(temp_test_data_dir / "teacher_pool.jsonl")
    test_config.max_dataset_size = 2
    test_config.output_dir = str(temp_test_data_dir / "output")

    return test_config


@pytest.fixture
def sample_real_images():
    """Create sample real images for testing."""
    images = []
    for i in range(2):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images


@pytest.fixture
def real_token_processor(real_config):
    """Create real token processor with actual configuration."""
    token_config = TokenConfig(
        coordinate_tokens_enabled=real_config.coordinate_tokens_enabled,
        max_coord_value=real_config.max_coord_value,
        coordinate_init_mode="fourier_ramp",
    )
    return TokenProcessor(token_config)
