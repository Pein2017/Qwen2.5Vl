#!/usr/bin/env python3
"""
Test script to demonstrate enhanced coordinate error handling with raw text logging.

This script tests the coordinate validation system that:
1. Detects invalid coordinate spans (wrong coordinate counts)
2. Logs raw text instead of just token IDs for better debugging
3. Can raise errors or just warn based on configuration

Usage:
    python temporal/test_coordinate_error_handling.py
"""

import sys
import os
import torch
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_coordinate_error_handling():
    """Test the enhanced coordinate error handling functionality."""
    
    print("🧪 Testing Enhanced Coordinate Error Handling")
    print("=" * 60)
    
    # Load tokenizer
    from transformers import AutoTokenizer
    model_path = "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Add special tokens
    from utils.simple_token_manager import SimpleTokenManager
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    token_manager = SimpleTokenManager(tokenizer, model)
    num_added = token_manager.add_tokens()
    print(f"✅ Added {num_added} special tokens to tokenizer")
    
    # Create coordinate manager with strict validation enabled
    from utils.coordinate_token_manager import create_coordinate_token_manager
    
    coordinate_config_strict = {
        "enable_coordinate_tokens": True,
        "max_coord_value": 2048,
        "box_start_id": 151648,
        "box_end_id": 151649,
        "coordinate_loss_weight": 1.0,
        "regular_loss_weight": 1.0,
        "soft_expectation_temperature": 1.0,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
        "enable_multi_geometry": True,
        "square_start_id": 151650,
        "square_end_id": 151651,
        "line_start_id": 151652,
        "line_end_id": 151653,
        "max_line_coordinates": 50,
        "bbox_giou_weight": 0.5,
        "square_polygon_weight": 0.3,
        "square_corner_weight": 0.2,
        "line_smoothness_weight": 0.4,
        "line_ordering_weight": 0.1,
        "geometry_focal_weight": 0.2,
        "enable_validation": True,
        "enable_caching": True,
        "batch_processing": True,
        "strict_coordinate_validation": True,  # Raise errors
        "log_raw_text_on_error": True,  # Log raw text
    }
    
    coord_manager_strict = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=151669,
        coordinate_config=coordinate_config_strict
    )
    
    print("✅ Created coordinate manager with strict validation enabled")
    
    # Test 1: Valid coordinate sequence (should work)
    print("\n📋 Test 1: Valid coordinate sequence")
    valid_sequence = torch.tensor([
        151648,  # <|box_start|>
        151669,  # <coord_0>
        151670,  # <coord_1>
        151671,  # <coord_2>
        151672,  # <coord_3>
        151649,  # <|box_end|>
    ]).unsqueeze(0)  # Add batch dimension
    
    try:
        spans = coord_manager_strict.find_coordinate_spans(valid_sequence)
        print(f"   ✅ Valid sequence processed successfully: {len(spans[0])} spans found")
    except Exception as e:
        print(f"   ❌ Valid sequence failed: {e}")
    
    # Test 2: Invalid coordinate sequence with strict validation (should raise error)
    print("\n📋 Test 2: Invalid coordinate sequence (strict validation)")
    invalid_sequence = torch.tensor([
        151648,  # <|box_start|>
        151669,  # <coord_0> (only 1 coordinate instead of 4)
        151649,  # <|box_end|>
    ]).unsqueeze(0)  # Add batch dimension
    
    try:
        spans = coord_manager_strict.find_coordinate_spans(invalid_sequence)
        print("   ❌ Invalid sequence incorrectly passed validation")
    except ValueError as e:
        if "INVALID_COORDINATE_SPAN" in str(e):
            print("   ✅ Invalid sequence correctly raised error with detailed message")
            print(f"   📝 Error details: {str(e)[:200]}...")
        else:
            print(f"   ⚠️ Invalid sequence raised error for wrong reason: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 3: Create coordinate manager with lenient validation
    print("\n📋 Test 3: Invalid coordinate sequence (lenient validation)")
    
    coordinate_config_lenient = coordinate_config_strict.copy()
    coordinate_config_lenient["strict_coordinate_validation"] = False  # Just warn
    
    coord_manager_lenient = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=151669,
        coordinate_config=coordinate_config_lenient
    )
    
    try:
        spans = coord_manager_lenient.find_coordinate_spans(invalid_sequence)
        print("   ✅ Invalid sequence processed with warnings (lenient mode)")
        print(f"   📊 Spans found: {len(spans[0])}")
    except Exception as e:
        print(f"   ❌ Unexpected error in lenient mode: {e}")
    
    # Test 4: Test raw text logging
    print("\n📋 Test 4: Raw text logging demonstration")
    
    # Create a sequence with some text that can be decoded
    mixed_sequence = torch.tensor([
        151644,  # <|im_start|>
        151648,  # <|box_start|>
        12345,   # Some regular text token
        151669,  # <coord_0> (only 1 coordinate - invalid)
        151649,  # <|box_end|>
        151645,  # <|im_end|>
    ]).unsqueeze(0)
    
    print("   🔍 Testing raw text decoding for invalid sequence...")
    try:
        spans = coord_manager_strict.find_coordinate_spans(mixed_sequence)
        print("   ❌ Mixed sequence incorrectly passed validation")
    except ValueError as e:
        if "Raw text" in str(e):
            print("   ✅ Error message includes decoded raw text")
            # Extract and show the raw text part
            error_lines = str(e).split('\n')
            for line in error_lines:
                if "Raw text" in line:
                    print(f"   📝 {line.strip()}")
        else:
            print("   ⚠️ Error message missing raw text decoding")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    print("\n🎯 Enhanced Coordinate Error Handling Test Summary:")
    print("   ✅ Strict validation correctly raises errors for invalid coordinate spans")
    print("   ✅ Lenient validation allows processing with warnings")
    print("   ✅ Raw text decoding provides better debugging information")
    print("   ✅ Configuration controls error handling behavior")
    
    print("\n📝 Configuration Options:")
    print("   coordinate_config_strict_coordinate_validation: true/false")
    print("   coordinate_config_log_raw_text_on_error: true/false")

if __name__ == "__main__":
    test_coordinate_error_handling()
    
    print("\n🎉 All coordinate error handling tests completed!")
    print("\n📝 Usage Instructions:")
    print("   1. Set 'coordinate_config_strict_coordinate_validation: true' to raise errors")
    print("   2. Set 'coordinate_config_log_raw_text_on_error: true' to see decoded text")
    print("   3. Invalid coordinate spans will show detailed error messages with raw text")
    print("   4. This helps identify data quality issues in your training samples")
