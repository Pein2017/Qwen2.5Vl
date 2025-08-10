#!/usr/bin/env python3
"""
Test coordinate token pipeline in both enabled and disabled modes.
"""

import sys
from pathlib import Path

import torch


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer

from src_new.models.wrapper import CoordinateProcessor


def test_coordinate_modes():
    """Test coordinate processor in both enabled and disabled modes."""
    print("🧪 Testing Coordinate Token Modes")
    print("=" * 60)

    try:
        # Load tokenizer
        model_path = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"📝 Base tokenizer vocab size: {len(tokenizer.get_vocab())}")

        # Test 1: Coordinate tokens disabled
        print(f"\n🔧 Test 1: Coordinate tokens DISABLED")
        print("-" * 40)

        # Create mock config with coordinate tokens disabled
        class MockConfig:
            def __init__(self, coordinate_tokens_enabled):
                self.coordinate_tokens_enabled = coordinate_tokens_enabled
                self.max_coord_value = 1024

        config_disabled = MockConfig(coordinate_tokens_enabled=False)
        coord_processor_disabled = CoordinateProcessor(config_disabled)
        coord_processor_disabled.set_tokenizer(tokenizer)

        print(f"  ✅ Coordinate processor created (disabled)")
        print(
            f"  📊 Original vocab size: {coord_processor_disabled.original_vocab_size}"
        )
        print(
            f"  🎯 Coordinate token range: {coord_processor_disabled.coordinate_token_range}"
        )
        print(
            f"  🔧 Coordinate tokens enabled: {coord_processor_disabled.coordinate_tokens_enabled}"
        )

        # Test 2: Coordinate tokens enabled with base tokenizer (should handle gracefully)
        print(f"\n🔧 Test 2: Coordinate tokens ENABLED (base tokenizer)")
        print("-" * 40)

        config_enabled = MockConfig(coordinate_tokens_enabled=True)
        coord_processor_enabled = CoordinateProcessor(config_enabled)
        coord_processor_enabled.set_tokenizer(tokenizer)

        print(f"  ✅ Coordinate processor created (enabled, base tokenizer)")
        print(
            f"  📊 Original vocab size: {coord_processor_enabled.original_vocab_size}"
        )
        print(
            f"  🎯 Coordinate token range: {coord_processor_enabled.coordinate_token_range}"
        )
        print(
            f"  🔧 Coordinate tokens enabled: {coord_processor_enabled.coordinate_tokens_enabled}"
        )

        # Test 3: Simulate tokenizer extension
        print(f"\n🔧 Test 3: Simulate tokenizer extension")
        print("-" * 40)

        # Create a mock extended tokenizer
        class MockExtendedTokenizer:
            def __init__(self, base_tokenizer):
                self.base_vocab = base_tokenizer.get_vocab().copy()
                # Add some mock coordinate tokens
                for i in range(10):
                    self.base_vocab[f"<|coord_{i}|>"] = 151665 + i

            def get_vocab(self):
                return self.base_vocab

        mock_extended_tokenizer = MockExtendedTokenizer(tokenizer)
        print(
            f"  📝 Mock extended tokenizer vocab size: {len(mock_extended_tokenizer.get_vocab())}"
        )

        # Test with extended tokenizer
        coord_processor_extended = CoordinateProcessor(config_enabled)
        coord_processor_extended.set_tokenizer(mock_extended_tokenizer)

        print(f"  ✅ Coordinate processor created (enabled, extended tokenizer)")
        print(
            f"  📊 Original vocab size: {coord_processor_extended.original_vocab_size}"
        )
        print(
            f"  🎯 Coordinate token range: {coord_processor_extended.coordinate_token_range}"
        )
        print(
            f"  🔧 Coordinate tokens enabled: {coord_processor_extended.coordinate_tokens_enabled}"
        )

        # Test 4: Test update_after_extension method
        print(f"\n🔧 Test 4: Test update_after_extension method")
        print("-" * 40)

        # Start with base tokenizer
        coord_processor_update = CoordinateProcessor(config_enabled)
        coord_processor_update.set_tokenizer(tokenizer)
        print(
            f"  📊 Before extension - range: {coord_processor_update.coordinate_token_range}"
        )

        # Simulate extension
        coord_processor_update.update_after_extension(mock_extended_tokenizer)
        print(
            f"  📊 After extension - range: {coord_processor_update.coordinate_token_range}"
        )

        # Test coordinate mask creation
        print(f"\n🔧 Test 5: Test coordinate mask creation")
        print("-" * 40)

        # Create test input with coordinate tokens
        test_input_ids = torch.tensor(
            [[151644, 151665, 151666, 151667, 151645]]
        )  # Some mock tokens

        # Test with disabled processor
        mask_disabled = coord_processor_disabled.get_coordinate_mask(test_input_ids)
        print(
            f"  🔧 Disabled processor mask: {mask_disabled.sum().item()} coordinate tokens"
        )

        # Test with enabled processor
        mask_enabled = coord_processor_extended.get_coordinate_mask(test_input_ids)
        print(
            f"  🔧 Enabled processor mask: {mask_enabled.sum().item()} coordinate tokens"
        )

        print(f"\n🎉 All coordinate mode tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_coordinate_modes()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
