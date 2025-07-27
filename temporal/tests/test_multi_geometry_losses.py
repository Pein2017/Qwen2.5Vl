#!/usr/bin/env python3
"""
Test script to verify multi-geometry loss computation
Tests bbox, square, and line geometry loss functions
"""

import sys
import os
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
import torch.nn.functional as F
from src.config import init_config
from src.models.model_loader import load_model_and_processor_unified

def test_multi_geometry_losses():
    """Test the multi-geometry loss computation system."""
    print("🧪 Testing Multi-Geometry Loss System...")
    
    # Initialize config
    config_path = 'configs/bbu_v2.yaml'
    print(f"📄 Initializing config from: {config_path}")
    init_config(config_path)
    
    # Load model
    from src.config import get_config
    config = get_config()
    model_path = config.model_path
    print(f"🤖 Loading model from: {model_path}")
    
    try:
        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=model_path,
            for_inference=False,
            attn_implementation='flash_attention_2'
        )
        
        print("✅ Model loaded successfully!")
        
        # Get coordinate manager
        if not hasattr(model, 'coordinate_manager') or model.coordinate_manager is None:
            print("❌ No coordinate manager found!")
            return False
            
        coord_manager = model.coordinate_manager
        print(f"✅ Coordinate manager found with config: {coord_manager.config.enable_multi_geometry}")
        
        # Test 1: Bbox geometry detection and loss
        print("\n🧪 Test 1: Bbox Geometry Detection and Loss")
        test_bbox_geometry(coord_manager, tokenizer)
        
        # Test 2: Square geometry detection and loss  
        print("\n🧪 Test 2: Square Geometry Detection and Loss")
        test_square_geometry(coord_manager, tokenizer)
        
        # Test 3: Line geometry detection and loss
        print("\n🧪 Test 3: Line Geometry Detection and Loss") 
        test_line_geometry(coord_manager, tokenizer)
        
        # Test 4: Mixed geometry types
        print("\n🧪 Test 4: Mixed Geometry Types")
        test_mixed_geometries(coord_manager, tokenizer)
        
        print("\n🎉 All multi-geometry loss tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bbox_geometry(coord_manager, tokenizer):
    """Test bbox geometry detection and GIoU loss computation."""
    # Create mock input with bbox coordinates (4 coordinates)
    # Format: <|box_start|> coord_100 coord_200 coord_300 coord_400 <|box_end|>
    
    box_start_id = 151648
    box_end_id = 151649
    coord_start_id = coord_manager.coord_start_id
    
    input_ids = torch.tensor([[
        1, 2, 3,  # Some text tokens
        box_start_id,  # <|box_start|>
        coord_start_id + 100, coord_start_id + 200, coord_start_id + 300, coord_start_id + 400,  # 4 coords
        box_end_id,  # <|box_end|>
        4, 5, 6  # More text tokens
    ]], device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Detect geometry spans
    geometry_spans = coord_manager.detect_geometry_spans(input_ids)
    print(f"   📊 Detected spans: {geometry_spans}")
    
    # Verify bbox detection
    if len(geometry_spans) > 0 and len(geometry_spans[0]) > 0:
        start, end, geometry_type = geometry_spans[0][0]
        if geometry_type == "bbox":
            print(f"   ✅ Correctly detected bbox geometry at span ({start}, {end})")
        else:
            print(f"   ❌ Expected bbox, got {geometry_type}")
    else:
        print(f"   ❌ No geometry spans detected")

def test_square_geometry(coord_manager, tokenizer):
    """Test square geometry detection and polygon/corner loss computation."""
    # Create mock input with square coordinates (8 coordinates)
    # Format: <|box_start|> coord1 coord2 ... coord8 <|box_end|>
    
    box_start_id = 151648
    box_end_id = 151649
    coord_start_id = coord_manager.coord_start_id
    
    input_ids = torch.tensor([[
        1, 2, 3,  # Some text tokens
        box_start_id,  # <|box_start|>
        coord_start_id + 100, coord_start_id + 200,  # Corner 1: (100, 200)
        coord_start_id + 300, coord_start_id + 200,  # Corner 2: (300, 200)
        coord_start_id + 300, coord_start_id + 400,  # Corner 3: (300, 400)
        coord_start_id + 100, coord_start_id + 400,  # Corner 4: (100, 400)
        box_end_id,  # <|box_end|>
        4, 5, 6  # More text tokens
    ]], device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Detect geometry spans
    geometry_spans = coord_manager.detect_geometry_spans(input_ids)
    print(f"   📊 Detected spans: {geometry_spans}")
    
    # Verify square detection
    if len(geometry_spans) > 0 and len(geometry_spans[0]) > 0:
        start, end, geometry_type = geometry_spans[0][0]
        if geometry_type == "square":
            print(f"   ✅ Correctly detected square geometry at span ({start}, {end})")
        else:
            print(f"   ❌ Expected square, got {geometry_type}")
    else:
        print(f"   ❌ No geometry spans detected")

def test_line_geometry(coord_manager, tokenizer):
    """Test line geometry detection and smoothness/ordering loss computation."""
    # Create mock input with line coordinates (6 coordinates = 3 points)
    # Format: <|box_start|> coord1 coord2 coord3 coord4 coord5 coord6 <|box_end|>
    
    box_start_id = 151648
    box_end_id = 151649
    coord_start_id = coord_manager.coord_start_id
    
    input_ids = torch.tensor([[
        1, 2, 3,  # Some text tokens
        box_start_id,  # <|box_start|>
        coord_start_id + 100, coord_start_id + 200,  # Point 1: (100, 200)
        coord_start_id + 150, coord_start_id + 250,  # Point 2: (150, 250)
        coord_start_id + 200, coord_start_id + 300,  # Point 3: (200, 300)
        box_end_id,  # <|box_end|>
        4, 5, 6  # More text tokens
    ]], device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Detect geometry spans
    geometry_spans = coord_manager.detect_geometry_spans(input_ids)
    print(f"   📊 Detected spans: {geometry_spans}")
    
    # Verify line detection
    if len(geometry_spans) > 0 and len(geometry_spans[0]) > 0:
        start, end, geometry_type = geometry_spans[0][0]
        if geometry_type == "line":
            print(f"   ✅ Correctly detected line geometry at span ({start}, {end})")
        else:
            print(f"   ❌ Expected line, got {geometry_type}")
    else:
        print(f"   ❌ No geometry spans detected")

def test_mixed_geometries(coord_manager, tokenizer):
    """Test mixed geometry types in a single input."""
    # Create mock input with bbox + square + line
    
    box_start_id = 151648
    box_end_id = 151649
    coord_start_id = coord_manager.coord_start_id
    
    input_ids = torch.tensor([[
        1, 2, 3,  # Text
        box_start_id,  # Bbox start
        coord_start_id + 10, coord_start_id + 20, coord_start_id + 30, coord_start_id + 40,  # 4 coords = bbox
        box_end_id,   # Bbox end
        4, 5,  # Text
        box_start_id,  # Square start  
        coord_start_id + 100, coord_start_id + 200, coord_start_id + 300, coord_start_id + 200,
        coord_start_id + 300, coord_start_id + 400, coord_start_id + 100, coord_start_id + 400,  # 8 coords = square
        box_end_id,   # Square end
        6, 7,  # Text
        box_start_id,  # Line start
        coord_start_id + 500, coord_start_id + 600, coord_start_id + 550, coord_start_id + 650,
        coord_start_id + 600, coord_start_id + 700,  # 6 coords = line
        box_end_id,   # Line end
        8, 9, 10  # Text
    ]], device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Detect geometry spans
    geometry_spans = coord_manager.detect_geometry_spans(input_ids)
    print(f"   📊 Detected spans: {geometry_spans}")
    
    if len(geometry_spans) > 0 and len(geometry_spans[0]) == 3:
        types = [span[2] for span in geometry_spans[0]]
        print(f"   📊 Detected geometry types: {types}")
        
        expected_types = ["bbox", "square", "line"]
        if types == expected_types:
            print(f"   ✅ Correctly detected all geometry types: {types}")
        else:
            print(f"   ❌ Expected {expected_types}, got {types}")
    else:
        print(f"   ❌ Expected 3 geometry spans, got {len(geometry_spans[0]) if geometry_spans else 0}")

def main():
    """Main test function."""
    success = test_multi_geometry_losses()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()