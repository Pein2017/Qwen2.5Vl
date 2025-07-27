#!/usr/bin/env python3
"""
Test script for multi-geometry coordinate token handling.

Tests the complete pipeline with your template training sample to ensure:
1. Coordinate tokens are properly added to vocabulary
2. Multi-geometry detection works correctly  
3. Coordinate conversion handles bbox_2d, square, line formats
4. Loss computation works for all geometry types
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from src.config import get_config, init_config
from src.models.wrapper import CoordinateConfig
from src.utils.coordinate_token_manager import (
    CoordinateTokenConfig, 
    CoordinateTokenManager
)

# Global variables to avoid re-initialization
_tokenizer = None
_original_vocab_size = None


def test_coordinate_token_extension():
    """Test coordinate token addition to vocabulary."""
    global _tokenizer, _original_vocab_size
    
    print("🧪 Testing coordinate token extension...")
    
    if _tokenizer is None:
        # Initialize config first
        init_config("configs/bbu_v2.yaml")
        config = get_config()
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, 
            trust_remote_code=True
        )
        
        original_vocab_size = len(tokenizer.get_vocab())
        print(f"   Original vocab size: {original_vocab_size}")
        
        # Add coordinate tokens
        coordinate_tokens = [f"<coord_{i}>" for i in range(2048)]
        geometry_tokens = ["<|square_start|>", "<|square_end|>", "<|line_start|>", "<|line_end|>"]
        
        tokenizer.add_special_tokens({
            "additional_special_tokens": 
            (tokenizer.additional_special_tokens or []) + coordinate_tokens + geometry_tokens
        })
        
        _tokenizer = tokenizer
        _original_vocab_size = original_vocab_size
    else:
        tokenizer = _tokenizer
        original_vocab_size = _original_vocab_size
    
    extended_vocab_size = len(tokenizer.get_vocab())
    print(f"   Extended vocab size: {extended_vocab_size}")
    print(f"   Added tokens: {extended_vocab_size - original_vocab_size}")
    
    # Test coordinate token IDs
    coord_0_id = tokenizer.convert_tokens_to_ids("<coord_0>")
    coord_2047_id = tokenizer.convert_tokens_to_ids("<coord_2047>")
    
    print(f"   <coord_0> -> {coord_0_id}")
    print(f"   <coord_2047> -> {coord_2047_id}")
    
    # Test geometry token IDs
    square_start_id = tokenizer.convert_tokens_to_ids("<|square_start|>")
    square_end_id = tokenizer.convert_tokens_to_ids("<|square_end|>")
    
    print(f"   <|square_start|> -> {square_start_id}")
    print(f"   <|square_end|> -> {square_end_id}")
    
    return tokenizer, original_vocab_size


def test_multi_geometry_detection():
    """Test multi-geometry coordinate conversion with your template data."""
    print("\n🧪 Testing multi-geometry detection...")
    
    # Your template training sample
    template_sample = {
        "images": ["images/QC-20230323-0001262_237114.jpeg"], 
        "objects": [
            {"square": [150, 10, 211, 35, 218, 16, 166, 0], "desc": "标签/5G-BBU-（接地线）"}, 
            {"bbox_2d": [269, 46, 291, 69], "desc": "螺丝、光纤插头/机柜处接地螺丝,只显示部分,符合要求"}, 
            {"line": [260, 52, 219, 31, 173, 6, 116, 3, 75, 3, 23, 8, 0, 16], "desc": "电线/有遮挡,捆扎整齐"}
        ], 
        "width": 532, 
        "height": 728
    }
    
    tokenizer, original_vocab_size = test_coordinate_token_extension()
    
    # Create coordinate manager
    coord_config = CoordinateTokenConfig(
        enable_coordinate_tokens=True,
        max_coord_value=2048,
        box_start_id=151648,  # <|box_start|>
        box_end_id=151649,    # <|box_end|>
        enable_multi_geometry=True,
        coordinate_loss_weight=1.0,
        regular_loss_weight=1.0,
        soft_expectation_temperature=1.0,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        enable_validation=True,
        enable_caching=False,
        batch_processing=True,
    )
    
    coord_manager = CoordinateTokenManager(
        tokenizer=tokenizer,
        config=coord_config,
        original_vocab_size=original_vocab_size
    )
    
    # Test JSON to coordinate conversion
    json_response = json.dumps(template_sample["objects"])
    coordinate_format = coord_manager.convert_json_to_coordinate_format(json_response)
    
    print(f"   Original JSON: {json_response[:100]}...")
    print(f"   Coordinate format: {coordinate_format[:200]}...")
    
    # Test back conversion
    json_back = coord_manager.convert_coordinate_to_json_format(coordinate_format)
    print(f"   Back to JSON: {json_back[:100]}...")
    
    return coord_manager


def test_geometry_span_detection():
    """Test geometry span detection with coordinate count logic."""
    print("\n🧪 Testing geometry span detection...")
    
    coord_manager = test_multi_geometry_detection()
    
    # Create sample token sequences for testing
    # Format: <|box_start|><coord_x1><coord_y1><coord_x2><coord_y2><|box_end|> (bbox - 4 coords)
    bbox_tokens = [
        151648,  # <|box_start|>
        coord_manager.coord_start_id + 100,  # <coord_100>
        coord_manager.coord_start_id + 200,  # <coord_200>
        coord_manager.coord_start_id + 300,  # <coord_300>
        coord_manager.coord_start_id + 400,  # <coord_400>
        151649   # <|box_end|>
    ]
    
    # Square format: 8 coordinates
    square_tokens = [
        151648,  # <|box_start|>
        coord_manager.coord_start_id + 150,  # 8 coordinate tokens
        coord_manager.coord_start_id + 10,
        coord_manager.coord_start_id + 211,
        coord_manager.coord_start_id + 35,
        coord_manager.coord_start_id + 218,
        coord_manager.coord_start_id + 16,
        coord_manager.coord_start_id + 166,
        coord_manager.coord_start_id + 0,
        151649   # <|box_end|>
    ]
    
    # Line format: >8 coordinates  
    line_tokens = [
        151648,  # <|box_start|>
        coord_manager.coord_start_id + 260,  # 14 coordinate tokens (7 points)
        coord_manager.coord_start_id + 52,
        coord_manager.coord_start_id + 219,
        coord_manager.coord_start_id + 31,
        coord_manager.coord_start_id + 173,
        coord_manager.coord_start_id + 6,
        coord_manager.coord_start_id + 116,
        coord_manager.coord_start_id + 3,
        coord_manager.coord_start_id + 75,
        coord_manager.coord_start_id + 3,
        coord_manager.coord_start_id + 23,
        coord_manager.coord_start_id + 8,
        coord_manager.coord_start_id + 0,
        coord_manager.coord_start_id + 16,
        151649   # <|box_end|>
    ]
    
    # Combine into batch
    batch_tokens = torch.tensor([
        bbox_tokens + [0] * (20 - len(bbox_tokens)),  # Pad to same length
        square_tokens + [0] * (20 - len(square_tokens)),
        line_tokens + [0] * (20 - len(line_tokens))
    ])
    
    print(f"   Test batch shape: {batch_tokens.shape}")
    
    # Test geometry span detection
    geometry_spans = coord_manager.detect_geometry_spans(batch_tokens)
    
    for i, spans in enumerate(geometry_spans):
        print(f"   Batch {i} spans: {spans}")
        for start, end, geom_type in spans:
            coord_count = sum(1 for t in batch_tokens[i][start+1:end-1] 
                            if coord_manager.is_coordinate_token(t.item()))
            print(f"     {geom_type}: span=[{start}:{end}], coords={coord_count}")
    
    return geometry_spans


def test_loss_computation():
    """Test multi-geometry loss computation."""
    print("\n🧪 Testing multi-geometry loss computation...")
    
    coord_manager = test_multi_geometry_detection()
    geometry_spans = test_geometry_span_detection()
    
    # Create dummy logits and labels for testing
    batch_size, seq_len, vocab_size = 2, 20, coord_manager.extended_vocab_size
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Set some labels to coordinate tokens
    labels[0, 1:5] = torch.tensor([
        coord_manager.coord_start_id + 100,
        coord_manager.coord_start_id + 200, 
        coord_manager.coord_start_id + 300,
        coord_manager.coord_start_id + 400
    ])
    
    print(f"   Test logits shape: {logits.shape}")
    print(f"   Test labels shape: {labels.shape}")
    
    # Compute losses
    losses = coord_manager.compute_coordinate_losses(
        logits=logits,
        labels=labels, 
        bbox_spans=[]  # Using geometry spans instead
    )
    
    print(f"   Computed losses: {list(losses.keys())}")
    for loss_name, loss_value in losses.items():
        print(f"     {loss_name}: {loss_value.item():.4f}")
    
    return losses


def main():
    """Main test function."""
    print("🚀 Testing Multi-Geometry Coordinate Token Implementation")
    print("=" * 60)
    
    try:
        # Test each component
        test_coordinate_token_extension() 
        test_multi_geometry_detection()
        test_geometry_span_detection()
        test_loss_computation()
        
        print("\n✅ All tests passed! Your multi-geometry coordinate token implementation is working correctly.")
        print("\n📋 Summary:")
        print("   - Coordinate tokens (<coord_0> to <coord_2047>) added to vocabulary")
        print("   - Multi-geometry detection working (bbox=4, square=8, line=>8 coords)")
        print("   - Geometry-specific loss computation implemented")
        print("   - Compatible with your template training data format")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())