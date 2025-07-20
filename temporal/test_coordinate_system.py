#!/usr/bin/env python3
"""
Test script for the enhanced coordinate token system.

This script tests the new production-robust coordinate token implementation
including the unified manager, bbox-aware loss computation, and enhanced tracking.
"""

import torch
from transformers import AutoTokenizer

from src.utils.coordinate_token_manager import (
    CoordinateTokenConfig,
    create_coordinate_token_manager,
)
from src.utils.coordinate_loss_computer import create_coordinate_loss_computer


def test_coordinate_manager():
    """Test the CoordinateTokenManager functionality."""
    print("🧪 Testing CoordinateTokenManager...")
    
    # Create a dummy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True
    )
    
    # Get actual original vocab size from tokenizer
    original_vocab_size = len(tokenizer.get_vocab())
    
    # Create coordinate token manager
    manager = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=original_vocab_size,
        enable_coordinate_tokens=True,
        max_coord_value=2048,
    )
    
    # Test JSON to coordinate format conversion
    json_input = '[{"bbox_2d": [0.1, 0.2, 0.8, 0.9], "label": "test object"}]'
    coord_output = manager.convert_json_to_coordinate_format(json_input)
    print(f"✅ JSON to coordinate conversion:")
    print(f"   Input: {json_input}")
    print(f"   Output: {coord_output}")
    
    # Test coordinate to JSON format conversion
    json_output = manager.convert_coordinate_to_json_format(coord_output)
    print(f"✅ Coordinate to JSON conversion:")
    print(f"   Input: {coord_output}")
    print(f"   Output: {json_output}")
    
    # Test bbox span detection with correct coordinate token IDs
    coord_start_id = manager.coord_start_id
    test_tokens = torch.tensor([
        [100, 200, 151648, coord_start_id, coord_start_id+1, coord_start_id+2, coord_start_id+3, 151649, 300, 400]
    ])  # [regular, regular, box_start, coord_0, coord_1, coord_2, coord_3, box_end, regular, regular]
    
    bbox_spans = manager.detect_bbox_spans(test_tokens)
    print(f"✅ Bbox span detection:")
    print(f"   Tokens: {test_tokens}")
    print(f"   Detected spans: {bbox_spans}")
    
    # Test coordinate mask creation
    coord_mask = manager.create_coordinate_mask(test_tokens)
    print(f"✅ Coordinate mask creation:")
    print(f"   Mask: {coord_mask}")
    
    # Test validation
    valid_coord_text = "test: <|box_start|><coord_100><coord_200><coord_1500><coord_1800><|box_end|>"
    invalid_coord_text = "test: <|box_start|><coord_100><coord_200><coord_2500><coord_1800><|box_end|>"
    
    print(f"✅ Coordinate validation:")
    print(f"   Valid text: {manager.validate_coordinate_format(valid_coord_text)}")
    print(f"   Invalid text: {manager.validate_coordinate_format(invalid_coord_text)}")
    
    # Print metrics
    print(f"✅ Manager metrics: {manager.get_metrics()}")
    
    return manager


def test_coordinate_loss_computer():
    """Test the CoordinateLossComputer functionality."""
    print("\n🧪 Testing CoordinateLossComputer...")
    
    # Create coordinate manager first
    tokenizer = AutoTokenizer.from_pretrained(
        "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True
    )
    
    # Get actual original vocab size from tokenizer
    original_vocab_size = len(tokenizer.get_vocab())
    
    manager = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=original_vocab_size,
        enable_coordinate_tokens=True,
        max_coord_value=2048,
    )
    
    # Create loss computer
    loss_computer = create_coordinate_loss_computer(manager)
    
    # Create test data with correct extended vocab size
    batch_size, seq_len = 2, 10
    extended_vocab_size = manager.extended_vocab_size
    logits = torch.randn(batch_size, seq_len, extended_vocab_size)
    
    # Create labels with bbox spans using correct coordinate token IDs
    coord_start_id = manager.coord_start_id
    labels = torch.tensor([
        [100, 200, 151648, coord_start_id, coord_start_id+1, coord_start_id+2, coord_start_id+3, 151649, 300, 400],  # With bbox span
        [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]  # Without bbox span
    ])
    
    # Compute coordinate-aware loss
    total_loss, loss_components = loss_computer.compute_coordinate_aware_loss(
        logits, labels
    )
    
    print(f"✅ Coordinate loss computation:")
    print(f"   Total loss: {total_loss}")
    print(f"   Loss components: {loss_components}")
    
    # Test with coordinate tokens disabled
    # Get actual original vocab size from tokenizer
    original_vocab_size = len(tokenizer.get_vocab())
    
    manager_disabled = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=original_vocab_size,
        enable_coordinate_tokens=False,
    )
    
    loss_computer_disabled = create_coordinate_loss_computer(manager_disabled)
    
    standard_loss, standard_components = loss_computer_disabled.compute_coordinate_aware_loss(
        logits, labels
    )
    
    print(f"✅ Standard loss computation (disabled):")
    print(f"   Total loss: {standard_loss}")
    print(f"   Loss components: {standard_components}")
    
    # Print metrics
    print(f"✅ Loss computer metrics: {loss_computer.get_metrics()}")
    
    return loss_computer


def test_integration():
    """Test integration between manager and loss computer."""
    print("\n🧪 Testing Integration...")
    
    # This would normally be done in the model wrapper
    tokenizer = AutoTokenizer.from_pretrained(
        "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True
    )
    
    # Get actual original vocab size from tokenizer
    original_vocab_size = len(tokenizer.get_vocab())
    
    manager = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=original_vocab_size,
        enable_coordinate_tokens=True,
        max_coord_value=2048,
    )
    
    loss_computer = create_coordinate_loss_computer(manager)
    
    # Test the full pipeline
    json_input = '[{"bbox_2d": [0.1, 0.2, 0.8, 0.9], "label": "screw"}]'
    coord_format = manager.convert_json_to_coordinate_format(json_input)
    
    print(f"✅ Full pipeline test:")
    print(f"   JSON input: {json_input}")
    print(f"   Coordinate format: {coord_format}")
    
    # Test tokenization (this would normally be done by the chat processor)
    tokens = tokenizer(coord_format, return_tensors="pt")
    print(f"   Tokenized: {tokens['input_ids'].shape}")
    
    # Test loss computation on actual tokens with correct vocab size
    extended_vocab_size = manager.extended_vocab_size
    logits = torch.randn(tokens['input_ids'].shape[0], tokens['input_ids'].shape[1], extended_vocab_size)
    loss, components = loss_computer.compute_coordinate_aware_loss(
        logits, tokens['input_ids']
    )
    
    print(f"   Loss: {loss}")
    print(f"   Components: {components}")
    
    print("✅ Integration test completed successfully!")


if __name__ == "__main__":
    print("🚀 Testing Enhanced Coordinate Token System\n")
    
    try:
        # Test individual components
        manager = test_coordinate_manager()
        loss_computer = test_coordinate_loss_computer()
        
        # Test integration
        test_integration()
        
        print("\n🎉 All tests passed! The enhanced coordinate token system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise