#!/usr/bin/env python3
"""
Test trainer tokenizer access fix.

Tests that the trainer can properly access tokenizer and get coordinate/digit token IDs.
"""

import torch
from unittest.mock import MagicMock


def test_coordinate_token_ids_with_known_ranges():
    """Test coordinate token ID retrieval using known ranges."""
    
    from src_coord_pretrain.training.trainer import PhaseATrainer
    
    # Create a mock trainer instance
    class MockTrainer:
        def __init__(self):
            # Mock the methods we need
            pass
        
        def _get_coordinate_token_ids(self):
            return PhaseATrainer._get_coordinate_token_ids(self)
    
    trainer = MockTrainer()
    
    print("🧪 Testing coordinate token ID retrieval...")
    
    # Test without tokenizer (should use known ranges)
    coord_ids = trainer._get_coordinate_token_ids()
    
    # Verify the known range from src_new: 151667 to 152691 (1025 tokens)
    expected_start = 151667
    expected_end = 152691
    expected_count = 1025  # coord_0 to coord_1024
    
    assert len(coord_ids) == expected_count, f"Expected {expected_count} coordinate tokens, got {len(coord_ids)}"
    assert min(coord_ids) == expected_start, f"Expected start ID {expected_start}, got {min(coord_ids)}"
    assert max(coord_ids) == expected_end, f"Expected end ID {expected_end}, got {max(coord_ids)}"
    
    print(f"✅ Coordinate token IDs: {len(coord_ids)} tokens")
    print(f"   Range: {min(coord_ids)} to {max(coord_ids)}")
    print(f"   Expected: {expected_start} to {expected_end}")


def test_digit_token_ids_with_mock_tokenizer():
    """Test digit token ID retrieval with mock tokenizer."""
    
    from src_coord_pretrain.training.trainer import PhaseATrainer
    
    # Create a mock trainer with mock tokenizer
    class MockTrainer:
        def __init__(self):
            # Mock tokenizer
            self.tokenizer = MagicMock()
            
            # Mock tokenization results for digits
            def mock_tokenize(text, add_special_tokens=False, return_tensors=None):
                # Simulate single-token digits
                digit_to_id = {
                    "0": 15, "1": 16, "2": 17, "3": 18, "4": 19,
                    "5": 20, "6": 21, "7": 22, "8": 23, "9": 24
                }
                if text in digit_to_id:
                    return {"input_ids": [digit_to_id[text]]}
                else:
                    return {"input_ids": [999]}  # Unknown token
            
            self.tokenizer.side_effect = mock_tokenize
        
        def _get_digit_token_ids(self):
            return PhaseATrainer._get_digit_token_ids(self)
    
    trainer = MockTrainer()
    
    print("\n🧪 Testing digit token ID retrieval...")
    
    # Test digit token retrieval
    digit_ids = trainer._get_digit_token_ids()
    
    # Should get 10 digit tokens (0-9)
    expected_count = 10
    expected_ids = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    
    assert len(digit_ids) == expected_count, f"Expected {expected_count} digit tokens, got {len(digit_ids)}"
    assert set(digit_ids) == set(expected_ids), f"Expected IDs {expected_ids}, got {digit_ids}"
    
    print(f"✅ Digit token IDs: {digit_ids}")


def test_tokenizer_fallback_mechanisms():
    """Test tokenizer access fallback mechanisms."""
    
    from src_coord_pretrain.training.trainer import PhaseATrainer
    
    print("\n🧪 Testing tokenizer fallback mechanisms...")
    
    # Test 1: No tokenizer available
    class MockTrainerNoTokenizer:
        def _get_coordinate_token_ids(self):
            return PhaseATrainer._get_coordinate_token_ids(self)
        
        def _get_digit_token_ids(self):
            return PhaseATrainer._get_digit_token_ids(self)
    
    trainer1 = MockTrainerNoTokenizer()
    
    # Should gracefully handle missing tokenizer
    coord_ids = trainer1._get_coordinate_token_ids()
    digit_ids = trainer1._get_digit_token_ids()
    
    # Coordinate IDs should use known ranges
    assert len(coord_ids) == 1025, f"Expected 1025 coordinate tokens, got {len(coord_ids)}"
    
    # Digit IDs should be empty without tokenizer
    assert len(digit_ids) == 0, f"Expected 0 digit tokens without tokenizer, got {len(digit_ids)}"
    
    print("✅ No tokenizer fallback works correctly")
    
    # Test 2: Tokenizer via data_collator
    class MockTrainerDataCollator:
        def __init__(self):
            self.data_collator = MagicMock()
            self.data_collator.tokenizer = MagicMock()
            
            # Mock tokenization for digits
            def mock_tokenize(text, add_special_tokens=False, return_tensors=None):
                if text in "0123456789":
                    return {"input_ids": [ord(text) - ord('0') + 100]}  # Simple mapping
                return {"input_ids": [999]}
            
            self.data_collator.tokenizer.side_effect = mock_tokenize
        
        def _get_digit_token_ids(self):
            return PhaseATrainer._get_digit_token_ids(self)
    
    trainer2 = MockTrainerDataCollator()
    digit_ids2 = trainer2._get_digit_token_ids()
    
    # Should get digit tokens via data_collator.tokenizer
    assert len(digit_ids2) == 10, f"Expected 10 digit tokens via data_collator, got {len(digit_ids2)}"
    
    print("✅ data_collator.tokenizer fallback works correctly")
    
    # Test 3: Tokenizer via train_dataset
    class MockTrainerDataset:
        def __init__(self):
            self.train_dataset = MagicMock()
            self.train_dataset.tokenizer = MagicMock()
            
            # Mock tokenization for digits
            def mock_tokenize(text, add_special_tokens=False, return_tensors=None):
                if text in "0123456789":
                    return {"input_ids": [ord(text) - ord('0') + 200]}  # Different mapping
                return {"input_ids": [999]}
            
            self.train_dataset.tokenizer.side_effect = mock_tokenize
        
        def _get_digit_token_ids(self):
            return PhaseATrainer._get_digit_token_ids(self)
    
    trainer3 = MockTrainerDataset()
    digit_ids3 = trainer3._get_digit_token_ids()
    
    # Should get digit tokens via train_dataset.tokenizer
    assert len(digit_ids3) == 10, f"Expected 10 digit tokens via train_dataset, got {len(digit_ids3)}"
    
    print("✅ train_dataset.tokenizer fallback works correctly")


def test_coordinate_token_validation():
    """Test coordinate token validation with mock vocabulary."""
    
    from src_coord_pretrain.training.trainer import PhaseATrainer
    
    print("\n🧪 Testing coordinate token validation...")
    
    # Create trainer with mock tokenizer that has coordinate tokens
    class MockTrainerWithCoordTokens:
        def __init__(self):
            self.tokenizer = MagicMock()
            
            # Mock vocabulary with coordinate tokens
            vocab = {}
            for i in range(1025):  # 0 to 1024
                vocab[f"<|coord_{i}|>"] = 151667 + i
            
            self.tokenizer.get_vocab.return_value = vocab
        
        def _get_coordinate_token_ids(self):
            return PhaseATrainer._get_coordinate_token_ids(self)
    
    trainer = MockTrainerWithCoordTokens()
    coord_ids = trainer._get_coordinate_token_ids()
    
    # Should validate against known ranges and use vocabulary
    assert len(coord_ids) == 1025, f"Expected 1025 coordinate tokens, got {len(coord_ids)}"
    assert min(coord_ids) == 151667, f"Expected min ID 151667, got {min(coord_ids)}"
    assert max(coord_ids) == 152691, f"Expected max ID 152691, got {max(coord_ids)}"
    
    print("✅ Coordinate token validation works correctly")


if __name__ == "__main__":
    print("🧪 Running Trainer Tokenizer Fix Tests...")
    print("=" * 50)
    
    test_functions = [
        test_coordinate_token_ids_with_known_ranges,
        test_digit_token_ids_with_mock_tokenizer,
        test_tokenizer_fallback_mechanisms,
        test_coordinate_token_validation,
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            raise
    
    print(f"\n🎉 All trainer tokenizer fix tests passed!")
    print("\n📊 Fix Summary:")
    print("✅ Coordinate token IDs use known ranges from src_new (151667-152691)")
    print("✅ Digit token IDs use tokenization approach for single digits")
    print("✅ Multiple tokenizer access fallbacks (self.tokenizer, data_collator, train_dataset)")
    print("✅ Graceful degradation when no tokenizer available")
    print("✅ Vocabulary validation for coordinate token ranges")
    print("\n🚀 Trainer tokenizer access is now robust and reliable!")
