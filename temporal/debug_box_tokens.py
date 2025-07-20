#!/usr/bin/env python3
"""
Debug box token IDs and coordinate token detection
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
from src.config import init_config, config
from src.models.model_loader import load_model_and_processor_unified

def debug_box_tokens():
    """Debug box token IDs and coordinate token detection."""
    
    print("🔍 Debugging Box Token IDs")
    print("=" * 50)
    
    # Initialize config
    init_config("configs/base_flat_det.yaml")
    
    # Load tokenizer
    print("🤖 Loading tokenizer...")
    model, tokenizer, image_processor = load_model_and_processor_unified(config.model_path)
    print(f"✅ Tokenizer loaded, vocab size: {len(tokenizer.get_vocab())}")
    
    # Check box token IDs
    print(f"\n🏷️ Box Token Analysis:")
    
    # Get actual token IDs from tokenizer
    box_start_text = "<|box_start|>"
    box_end_text = "<|box_end|>"
    
    box_start_id = tokenizer.convert_tokens_to_ids(box_start_text)
    box_end_id = tokenizer.convert_tokens_to_ids(box_end_text)
    
    print(f"   {box_start_text} actual ID: {box_start_id}")
    print(f"   {box_end_text} actual ID: {box_end_id}")
    
    # Check what the config expects
    print(f"\n📋 Config Expected Values:")
    print(f"   coordinate_config_box_start_id: {getattr(config, 'coordinate_config_box_start_id', 'NOT FOUND')}")
    print(f"   coordinate_config_box_end_id: {getattr(config, 'coordinate_config_box_end_id', 'NOT FOUND')}")
    
    # Check coordinate tokens
    print(f"\n🎯 Coordinate Token Analysis:")
    for i in range(5):  # Test first 5 coordinate tokens
        coord_text = f"<coord_{i}>"
        coord_id = tokenizer.convert_tokens_to_ids(coord_text)
        print(f"   {coord_text} ID: {coord_id}")
    
    # Test tokenization of coordinate format
    print(f"\n🧪 Test Coordinate Format Tokenization:")
    test_text = "bbu基带处理单元/华为: <|box_start|><coord_3><coord_259><coord_295><coord_653><|box_end|>"
    token_ids = tokenizer.encode(test_text)
    print(f"   Text: {test_text}")
    print(f"   Token IDs: {token_ids}")
    
    # Find box token positions
    box_start_positions = [i for i, token_id in enumerate(token_ids) if token_id == box_start_id]
    box_end_positions = [i for i, token_id in enumerate(token_ids) if token_id == box_end_id]
    
    print(f"   <|box_start|> positions: {box_start_positions}")
    print(f"   <|box_end|> positions: {box_end_positions}")
    
    # Find coordinate token positions
    coord_positions = []
    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.convert_ids_to_tokens(token_id)
        if token_text.startswith('<coord_'):
            coord_positions.append((i, token_text, token_id))
    
    print(f"   Coordinate token positions: {coord_positions}")
    
    # Create a test tensor and check bbox span detection
    print(f"\n🔍 Test Bbox Span Detection:")
    
    # Create a dummy coordinate loss computer
    from src.utils.coordinate_loss_computer import CoordinateLossComputer
    from src.utils.coordinate_token_manager import create_coordinate_token_manager
    
    coordinate_manager = create_coordinate_token_manager(
        tokenizer=tokenizer,
        original_vocab_size=len(tokenizer.get_vocab()),
        coordinate_config={
            "enable_coordinate_tokens": True,
            "max_coord_value": 2048,
            "coordinate_loss_weight": 1.0,
            "regular_loss_weight": 1.0,
            "soft_expectation_temperature": 1.0,
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
        }
    )
    
    coord_computer = CoordinateLossComputer(coordinate_manager)
    
    # Create test labels tensor
    test_labels = torch.tensor([token_ids], dtype=torch.long)
    print(f"   Test labels shape: {test_labels.shape}")
    print(f"   Test labels: {test_labels}")
    
    # Test bbox span detection
    bbox_spans = coord_computer._detect_bbox_spans_batch_enhanced(test_labels)
    print(f"   Detected bbox spans: {bbox_spans}")
    
    # Check config values used by coordinate loss computer
    print(f"\n⚙️ Coordinate Loss Computer Config:")
    print(f"   box_start_id: {coord_computer.config.box_start_id}")
    print(f"   box_end_id: {coord_computer.config.box_end_id}")
    print(f"   max_coord_value: {coord_computer.config.max_coord_value}")
    
    # Check if they match
    if coord_computer.config.box_start_id == box_start_id:
        print("   ✅ box_start_id matches tokenizer")
    else:
        print(f"   ❌ box_start_id mismatch: config={coord_computer.config.box_start_id}, tokenizer={box_start_id}")
    
    if coord_computer.config.box_end_id == box_end_id:
        print("   ✅ box_end_id matches tokenizer")
    else:
        print(f"   ❌ box_end_id mismatch: config={coord_computer.config.box_end_id}, tokenizer={box_end_id}")

if __name__ == "__main__":
    debug_box_tokens()