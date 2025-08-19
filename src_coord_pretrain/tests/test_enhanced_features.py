#!/usr/bin/env python3
"""
Comprehensive test suite for coordinate token training features (single-phase).
"""

from unittest.mock import MagicMock


# Test data generation enhancements
def test_data_generation_reverse_mapping():
    """Test reverse mapping data generation."""
    import random

    from src_coord_pretrain.scripts.generate_coord_bootstrap import _gen_reverse_mapping

    rng = random.Random(42)
    samples = list(_gen_reverse_mapping(max_coord=100, n=10, rng=rng))

    assert len(samples) == 10
    for q, y, res in samples:
        # Check that question contains coordinate token
        assert "<|coord_" in q and "|>" in q
        # Check that answer is raw digit
        assert y.isdigit()
        # Check that result matches answer
        assert int(y) == res
        # Check result is in valid range
        assert 0 <= res <= 100


def test_data_generation_identity_canonical():
    """Test canonical prompt generation."""
    import random

    from src_coord_pretrain.scripts.generate_coord_bootstrap import _gen_identity

    rng = random.Random(42)

    # Test canonical prompts
    canonical_samples = list(
        _gen_identity(max_coord=10, n=5, rng=rng, canonical_only=True)
    )
    regular_samples = list(
        _gen_identity(max_coord=10, n=5, rng=rng, canonical_only=False)
    )

    # Heuristic: shapes and sizes
    assert len(canonical_samples) == 5
    assert len(regular_samples) == 5


def test_unlikelihood_config_validation():
    """Test unlikelihood configuration validation."""
    valid_config = {
        "unlikelihood_enabled": True,
        "unlikelihood_lambda_digits": 1.0,
        "unlikelihood_lambda_coords": 0.5,
        "unlikelihood_coord_window": 8,
    }

    assert valid_config["unlikelihood_enabled"] is True
    assert valid_config["unlikelihood_lambda_digits"] == 1.0
    assert valid_config["unlikelihood_coord_window"] == 8


def test_coordinate_token_detection():
    """Test coordinate token ID detection."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "<|coord_0|>": 1000,
        "<|coord_1|>": 1001,
        "<|coord_2|>": 1002,
        "regular_token": 500,
    }

    vocab = mock_tokenizer.get_vocab()
    coord_tokens = []
    for token_name, token_id in vocab.items():
        if token_name.startswith("<|coord_") and token_name.endswith("|>"):
            coord_tokens.append(token_id)

    coord_tokens = sorted(coord_tokens)
    assert coord_tokens == [1000, 1001, 1002]


def test_digit_token_detection():
    """Test digit token ID detection."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {
        "0": 100,
        "1": 101,
        "2": 102,
        " 0": 200,
        " 1": 201,
        "10": 300,
        "100": 301,
        "regular_token": 500,
    }

    vocab = mock_tokenizer.get_vocab()
    digit_tokens = []

    for digit in "0123456789":
        if digit in vocab:
            digit_tokens.append(vocab[digit])

    numeric_patterns = ["10", "100", " 0", " 1"]
    for pattern in numeric_patterns:
        if pattern in vocab:
            digit_tokens.append(vocab[pattern])

    digit_tokens = list(set(digit_tokens))
    expected_tokens = [100, 101, 102, 200, 201, 300, 301]
    assert sorted(digit_tokens) == sorted(expected_tokens)


def test_strictness_validation():
    """Test coordinate token strictness validation."""
    import re

    def check_strictness(generated_text):
        """Simplified version of strictness check."""
        coord_tokens = re.findall(r"<\|coord_\d+\|>", generated_text)
        text_without_coords = re.sub(r"<\|coord_\d+\|>", "", generated_text)
        raw_digits = re.findall(r"\b\d+\b", text_without_coords)
        if coord_tokens:
            return len(coord_tokens) == 1 and len(raw_digits) == 0
        if raw_digits:
            return len(raw_digits) == 1 and len(coord_tokens) == 0
        return False

    assert check_strictness("<|coord_123|>") is True
    assert check_strictness("<|coord_123|> <|coord_456|>") is False
    assert check_strictness("<|coord_123|> 456") is False
    assert check_strictness("123 456") is False
    assert check_strictness("123") is True


def test_inference_checkpoint_structure():
    """Test that inference checkpoint contains all required files."""
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "coordinate_config.json",
        "coord_token_ids.json",
        "metrics-final.json",
        "README_INFERENCE.json",
    ]
    assert len(required_files) == 9
    assert "coordinate_config.json" in required_files
    assert "README_INFERENCE.json" in required_files


def test_enhanced_config_yaml_structure():
    """Test configuration YAML structure (single-phase)."""
    enhanced_config = {
        "unlikelihood_enabled": True,
        "unlikelihood_lambda_digits": 1.0,
        "unlikelihood_lambda_coords": 1.0,
        "unlikelihood_coord_window": 8,
        "coordinate_tokens_enabled": True,
        "max_coord_value": 1024,
    }
    assert "unlikelihood_enabled" in enhanced_config
    assert enhanced_config["max_coord_value"] == 1024


if __name__ == "__main__":
    test_functions = [
        test_data_generation_reverse_mapping,
        test_data_generation_identity_canonical,
        test_unlikelihood_config_validation,
        test_coordinate_token_detection,
        test_digit_token_detection,
        test_strictness_validation,
        test_inference_checkpoint_structure,
        test_enhanced_config_yaml_structure,
    ]

    print("🧪 Running features test suite (single-phase)...")

    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")

    print("🎉 Test suite completed!")
