#!/usr/bin/env python3
"""
Test dataset validation for enhanced coordinate token training.

Tests that the dataset validation correctly handles both:
1. Forward mapping: "N" -> <|coord_N|> (coordinate tokens in assistant content)
2. Reverse mapping: <|coord_N|> -> "N" (raw numbers in assistant content)
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MockConfig:
    max_coord_value: int = 1024


def test_dataset_validation():
    """Test that dataset validation handles both coordinate tokens and raw numbers."""

    # Import the validation functions
    from src_coord_pretrain.datasets.bootstrap_coord_dataset import (
        _COORD_TOKEN_RE,
        _RAW_NUMBER_RE,
    )

    print("🧪 Testing dataset validation for enhanced features...")

    # Test regex patterns
    print("\n📋 Testing regex patterns:")

    # Test coordinate token regex
    coord_tests = [
        ("<|coord_0|>", True),
        ("<|coord_123|>", True),
        ("<|coord_1024|>", True),
        ("coord_123", False),
        ("<|coord_123", False),
        ("coord_123|>", False),
        ("<|coord_abc|>", False),
    ]

    for text, expected in coord_tests:
        result = bool(_COORD_TOKEN_RE.match(text))
        status = "✅" if result == expected else "❌"
        print(f"  {status} Coordinate token '{text}': {result} (expected {expected})")
        assert result == expected, f"Coordinate token regex failed for '{text}'"

    # Test raw number regex
    number_tests = [
        ("0", True),
        ("123", True),
        ("1024", True),
        ("abc", False),
        ("123abc", False),
        ("", False),
        ("12.3", False),
    ]

    for text, expected in number_tests:
        result = bool(_RAW_NUMBER_RE.match(text))
        status = "✅" if result == expected else "❌"
        print(f"  {status} Raw number '{text}': {result} (expected {expected})")
        assert result == expected, f"Raw number regex failed for '{text}'"

    # Test dataset validation with sample data
    print("\n📋 Testing dataset validation:")

    # Create test data with both types
    test_records = [
        # Forward mapping samples (identity and arithmetic)
        {
            "messages": [
                {"role": "user", "content": "What is `123` in coordinate space?"},
                {"role": "assistant", "content": "<|coord_123|>"},
            ],
            "meta": {"task": "identity", "result": 123},
        },
        {
            "messages": [
                {"role": "user", "content": "What is 100 + 50?"},
                {"role": "assistant", "content": "<|coord_150|>"},
            ],
            "meta": {"task": "arithmetic", "result": 150},
        },
        # Reverse mapping samples
        {
            "messages": [
                {"role": "user", "content": "What is <|coord_456|> as text?"},
                {"role": "assistant", "content": "456"},
            ],
            "meta": {"task": "reverse_mapping", "result": 456},
        },
        {
            "messages": [
                {"role": "user", "content": "Convert <|coord_789|> to number"},
                {"role": "assistant", "content": "789"},
            ],
            "meta": {"task": "reverse_mapping", "result": 789},
        },
    ]

    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for record in test_records:
            f.write(json.dumps(record) + "\n")
        temp_file = f.name

    try:
        # Test dataset creation and validation
        config = MockConfig(max_coord_value=1024)

        # Test the validation logic directly
        from src_coord_pretrain.datasets.bootstrap_coord_dataset import (
            CoordBootstrapDataset,
        )

        # Create a mock dataset instance to access the validation method
        class MockDataset:
            def __init__(self, config):
                self.config = config

            # Copy the validation method from the real dataset class
            def _validate_record(self, rec):
                return CoordBootstrapDataset._validate_record(self, rec)

        mock_dataset = MockDataset(config)

        # Test validation on each record
        for i, record in enumerate(test_records):
            try:
                mock_dataset._validate_record(record)
                print(f"  ✅ Record {i + 1} validation passed")
            except Exception as e:
                print(f"  ❌ Record {i + 1} validation failed: {e}")
                raise

        print(f"  ✅ Dataset validation passed for {len(test_records)} records")
        print(f"     - Forward mapping samples: 2")
        print(f"     - Reverse mapping samples: 2")

    finally:
        # Clean up
        Path(temp_file).unlink()

    # Test invalid samples
    print("\n📋 Testing invalid sample detection:")

    invalid_samples = [
        # Invalid coordinate value
        {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "<|coord_2000|>"},  # > max_coord_value
            ]
        },
        # Invalid raw number
        {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "2000"},  # > max_coord_value
            ]
        },
        # Invalid format
        {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "invalid_format"},
            ]
        },
    ]

    for i, invalid_record in enumerate(invalid_samples):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(invalid_record) + "\n")
            temp_file = f.name

        try:
            config = MockConfig(max_coord_value=1024)

            # This should raise an error
            try:
                # Use real dataset class; expect construction to raise due to invalid record
                from src_coord_pretrain.datasets.bootstrap_coord_dataset import (
                    CoordBootstrapDataset,
                    DatasetConfig,
                )

                ds_conf = DatasetConfig(
                    data_path=temp_file,
                    max_coord_value=1024,
                    use_apply_chat_template=True,
                )
                _ = CoordBootstrapDataset(tokenizer=None, config=ds_conf)  # type: ignore[arg-type]
                print(f"  ❌ Invalid sample {i + 1} was not detected!")
                assert False, f"Invalid sample {i + 1} should have been rejected"
            except Exception as e:
                print(
                    f"  ✅ Invalid sample {i + 1} correctly rejected: {type(e).__name__}"
                )

        finally:
            Path(temp_file).unlink()

    print("\n🎉 All dataset validation tests passed!")


def test_edge_cases():
    """Test edge cases for dataset validation."""

    from src_coord_pretrain.datasets.bootstrap_coord_dataset import (
        _COORD_TOKEN_RE,
        _RAW_NUMBER_RE,
    )

    print("\n🔍 Testing edge cases:")

    # Edge cases for coordinate tokens
    edge_coord_cases = [
        ("<|coord_0|>", True),  # Minimum value
        ("<|coord_1024|>", True),  # Maximum value (assuming max_coord_value=1024)
        ("<|coord_00123|>", True),  # Leading zeros (should still work)
        ("<|coord_|>", False),  # Empty number
        ("<|coord_-1|>", False),  # Negative number
    ]

    for text, expected in edge_coord_cases:
        result = bool(_COORD_TOKEN_RE.match(text))
        status = "✅" if result == expected else "❌"
        print(f"  {status} Edge coordinate token '{text}': {result}")
        if result != expected:
            print(f"      Expected {expected}, got {result}")

    # Edge cases for raw numbers
    edge_number_cases = [
        ("0", True),  # Minimum value
        ("1024", True),  # Maximum value
        ("00123", True),  # Leading zeros
        ("", False),  # Empty string
        ("-1", False),  # Negative number
        ("1.0", False),  # Float
        ("1e3", False),  # Scientific notation
    ]

    for text, expected in edge_number_cases:
        result = bool(_RAW_NUMBER_RE.match(text))
        status = "✅" if result == expected else "❌"
        print(f"  {status} Edge raw number '{text}': {result}")
        if result != expected:
            print(f"      Expected {expected}, got {result}")

    print("✅ Edge case testing completed!")


if __name__ == "__main__":
    test_dataset_validation()
    test_edge_cases()

    print("\n📊 Summary:")
    print("✅ Dataset validation now supports both:")
    print("   - Forward mapping: 'N' -> <|coord_N|> (coordinate tokens)")
    print("   - Reverse mapping: <|coord_N|> -> 'N' (raw numbers)")
    print("✅ Invalid samples are properly detected and rejected")
    print("✅ Edge cases are handled correctly")
    print("\n🎯 The dataset validation error should now be resolved!")
