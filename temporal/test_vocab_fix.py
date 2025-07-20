#!/usr/bin/env python3
"""
Test script to verify the vocabulary size fix eliminates CUDA index out of bounds error.
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
from transformers import AutoTokenizer
from src.config import config, init_config, reset_config

def test_vocabulary_fix():
    """Test that the vocabulary size fix works."""
    
    print("🧪 TESTING VOCABULARY SIZE FIX")
    print("=" * 50)
    
    # Initialize with fixed config
    reset_config()
    init_config("configs/base_flat_det.yaml")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    # Get sizes
    actual_vocab_size = len(tokenizer.get_vocab())
    config_vocab_size = config.model_vocab_size
    max_coord_value = config.coordinate_config_max_coord_value
    expected_extended_size = actual_vocab_size + max_coord_value
    
    print(f"📊 VOCABULARY SIZE VERIFICATION:")
    print(f"   Actual tokenizer vocab size: {actual_vocab_size}")
    print(f"   Config model_vocab_size: {config_vocab_size}")
    print(f"   Max coordinate value: {max_coord_value}")
    print(f"   Expected extended size: {expected_extended_size}")
    
    # Check if fix is correct
    if config_vocab_size == expected_extended_size:
        print(f"✅ VOCAB SIZE FIX VERIFIED: Config matches expected extended size")
        vocab_fix_ok = True
    else:
        print(f"❌ VOCAB SIZE FIX FAILED: Config ({config_vocab_size}) != Expected ({expected_extended_size})")
        vocab_fix_ok = False
    
    # Test coordinate token range safety
    print(f"\n🎯 COORDINATE TOKEN RANGE SAFETY TEST:")
    coord_start = actual_vocab_size
    coord_end = expected_extended_size
    
    print(f"   Coordinate token range: [{coord_start}, {coord_end})")
    print(f"   Logits tensor size: {config_vocab_size}")
    
    if coord_end <= config_vocab_size:
        print(f"✅ COORDINATE RANGE SAFE: coord_end ({coord_end}) <= logits_size ({config_vocab_size})")
        range_safe = True
    else:
        print(f"❌ COORDINATE RANGE UNSAFE: coord_end ({coord_end}) > logits_size ({config_vocab_size})")
        range_safe = False
    
    # Test coordinate token access pattern
    print(f"\n🔍 COORDINATE TOKEN ACCESS PATTERN TEST:")
    print(f"   coord_only_logits = coord_logits[:, {coord_start}:{coord_end}]")
    
    if coord_start < config_vocab_size and coord_end <= config_vocab_size:
        print(f"✅ ACCESS PATTERN SAFE: Both indices within bounds")
        access_safe = True
    else:
        print(f"❌ ACCESS PATTERN UNSAFE: Indices would cause out of bounds")
        access_safe = False
    
    # Create mock tensor to test actual access
    print(f"\n🧪 MOCK TENSOR ACCESS TEST:")
    try:
        mock_logits = torch.randn(1, 10, config_vocab_size)  # Mock logits tensor
        coord_only_logits = mock_logits[:, :, coord_start:coord_end]
        
        print(f"   Created mock logits: {mock_logits.shape}")
        print(f"   Extracted coord_only_logits: {coord_only_logits.shape}")
        print(f"   Expected coord tokens: {coord_end - coord_start}")
        
        if coord_only_logits.shape[-1] == (coord_end - coord_start):
            print(f"✅ MOCK ACCESS SUCCESSFUL: Correct coordinate token extraction")
            mock_test_ok = True
        else:
            print(f"❌ MOCK ACCESS FAILED: Wrong coordinate token count")
            mock_test_ok = False
            
    except Exception as e:
        print(f"❌ MOCK ACCESS FAILED: {e}")
        mock_test_ok = False
    
    # Overall test result
    print(f"\n📋 TEST SUMMARY:")
    print(f"   Vocab size fix: {'✅ PASS' if vocab_fix_ok else '❌ FAIL'}")
    print(f"   Range safety: {'✅ PASS' if range_safe else '❌ FAIL'}")
    print(f"   Access pattern: {'✅ PASS' if access_safe else '❌ FAIL'}")
    print(f"   Mock tensor test: {'✅ PASS' if mock_test_ok else '❌ FAIL'}")
    
    all_tests_pass = vocab_fix_ok and range_safe and access_safe and mock_test_ok
    
    if all_tests_pass:
        print(f"\n🎉 ALL TESTS PASSED: CUDA index out of bounds fix verified!")
        print(f"   The coordinate token system should now work without errors.")
    else:
        print(f"\n❌ SOME TESTS FAILED: Fix needs adjustment")
    
    return all_tests_pass

def test_coordinate_loss_computer_compatibility():
    """Test that the fix is compatible with coordinate loss computer."""
    
    print(f"\n🔧 COORDINATE LOSS COMPUTER COMPATIBILITY TEST:")
    print(f"=" * 55)
    
    # Import after config initialization
    from src.utils.coordinate_token_manager import create_coordinate_token_manager
    from src.utils.coordinate_loss_computer import create_coordinate_loss_computer
    
    # Create coordinate system
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    original_vocab_size = len(tokenizer.get_vocab())
    
    coordinate_config_dict = {
        "enable_coordinate_tokens": True,
        "max_coord_value": config.coordinate_config_max_coord_value,
        "coordinate_loss_weight": 1.0,
        "regular_loss_weight": 1.0,
        "soft_expectation_temperature": 1.0,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
        "box_start_id": 151648,
        "box_end_id": 151649,
    }
    
    coordinate_manager = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=original_vocab_size,
        coordinate_config=coordinate_config_dict
    )
    
    coordinate_loss_computer = create_coordinate_loss_computer(coordinate_manager)
    
    print(f"   Coordinate manager created: ✅")
    print(f"   Original vocab size: {coordinate_manager.original_vocab_size}")
    print(f"   Extended vocab size: {coordinate_manager.extended_vocab_size}")
    print(f"   Coord start ID: {coordinate_manager.coord_start_id}")
    print(f"   Coord end ID: {coordinate_manager.coord_end_id}")
    
    # Test coordinate range compatibility
    coord_start = coordinate_manager.coord_start_id
    coord_end = coordinate_manager.coord_end_id
    
    if coord_end <= config.model_vocab_size:
        print(f"   ✅ COORDINATE RANGE COMPATIBLE: [{coord_start}, {coord_end}) fits in logits({config.model_vocab_size})")
        return True
    else:
        print(f"   ❌ COORDINATE RANGE INCOMPATIBLE: [{coord_start}, {coord_end}) exceeds logits({config.model_vocab_size})")
        return False

if __name__ == "__main__":
    try:
        # Test the vocabulary fix
        vocab_fix_ok = test_vocabulary_fix()
        
        # Test coordinate loss computer compatibility
        coord_compat_ok = test_coordinate_loss_computer_compatibility()
        
        # Final result
        print(f"\n🏁 FINAL RESULT:")
        print(f"=" * 30)
        
        if vocab_fix_ok and coord_compat_ok:
            print(f"✅ ALL TESTS PASSED: The CUDA index out of bounds fix is verified!")
            print(f"   The coordinate token system should now work correctly.")
            print(f"   No more index out of bounds errors expected.")
        else:
            print(f"❌ TESTS FAILED: Fix needs more work")
            if not vocab_fix_ok:
                print(f"   - Vocabulary size fix failed")
            if not coord_compat_ok:
                print(f"   - Coordinate loss computer compatibility failed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()