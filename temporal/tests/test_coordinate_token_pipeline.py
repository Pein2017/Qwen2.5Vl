#!/usr/bin/env python3
"""
Test coordinate token generation pipeline.
This script focuses on testing the coordinate token manager and coordinate processing.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

from src.config import init_config, get_config
from src.utils.coordinate_token_manager import create_coordinate_token_manager
from src.utils.simple_token_manager import SimpleTokenManager

def load_sample_data(limit: int = 3) -> List[Dict]:
    """Load a few samples for testing."""
    data_file = "data/ds_v2_full/all_samples.jsonl"
    samples = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            samples.append(json.loads(line.strip()))
    
    return samples

def test_coordinate_token_manager():
    """Test the coordinate token manager functionality."""
    print("🔍 Testing Coordinate Token Manager")
    print("-" * 40)
    
    try:
        config = get_config()
        print(f"✅ Config loaded - coordinate tokens enabled: {config.coordinate_config_enable_coordinate_tokens}")
        
        # Create coordinate token manager
        coord_manager = create_coordinate_token_manager(config)
        print(f"✅ Coordinate token manager created: {type(coord_manager).__name__}")
        
        # Test coordinate token ranges
        if hasattr(coord_manager, 'coordinate_token_ranges'):
            print(f"📊 Coordinate token ranges: {coord_manager.coordinate_token_ranges}")
        
        # Test coordinate vocabulary
        if hasattr(coord_manager, 'coordinate_vocab_size'):
            print(f"📊 Coordinate vocabulary size: {coord_manager.coordinate_vocab_size}")
        
        return coord_manager
        
    except Exception as e:
        print(f"❌ Error testing coordinate token manager: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_token_manager():
    """Test the simple token manager functionality."""
    print("\n🔍 Testing Simple Token Manager")
    print("-" * 40)
    
    try:
        config = get_config()
        
        # Create simple token manager
        simple_manager = SimpleTokenManager(config)
        print(f"✅ Simple token manager created")
        
        # Test coordinate conversion
        test_coordinates = {
            'bbox_2d': [100, 150, 200, 250],
            'square': [100, 150, 200, 150, 200, 250, 100, 250],
            'line': [100, 150, 120, 160, 140, 170, 160, 180]
        }
        
        for geom_type, coords in test_coordinates.items():
            try:
                tokens = simple_manager.coordinates_to_tokens(coords, geom_type)
                print(f"✅ {geom_type} coordinates {coords} -> {len(tokens) if tokens else 0} tokens")
                if tokens:
                    print(f"   Sample tokens: {tokens[:5]}...")
            except Exception as e:
                print(f"❌ Error converting {geom_type}: {e}")
        
        return simple_manager
        
    except Exception as e:
        print(f"❌ Error testing simple token manager: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_coordinate_conversion_with_real_data(samples: List[Dict]):
    """Test coordinate conversion with real sample data."""
    print("\n🔍 Testing Coordinate Conversion with Real Data")
    print("-" * 40)
    
    try:
        config = get_config()
        simple_manager = SimpleTokenManager(config)
        
        for i, sample in enumerate(samples[:2]):  # Test first 2 samples
            print(f"\n📊 Sample {i+1}:")
            print(f"   Image: {sample['images'][0]}")
            print(f"   Objects: {len(sample['objects'])}")
            
            coordinate_tokens_generated = 0
            
            for j, obj in enumerate(sample['objects'][:3]):  # Test first 3 objects
                desc = obj.get('desc', 'unknown')
                print(f"   Object {j+1}: {desc[:50]}...")
                
                for geom_type, coords in obj.items():
                    if geom_type == 'desc':
                        continue
                    
                    try:
                        tokens = simple_manager.coordinates_to_tokens(coords, geom_type)
                        if tokens:
                            coordinate_tokens_generated += len(tokens)
                            print(f"     {geom_type}: {len(coords)} coords -> {len(tokens)} tokens")
                        else:
                            print(f"     {geom_type}: {len(coords)} coords -> NO TOKENS ❌")
                    except Exception as e:
                        print(f"     {geom_type}: Error - {e}")
            
            if coordinate_tokens_generated == 0:
                print(f"   ❌ CRITICAL: No coordinate tokens generated for sample {i+1}")
            else:
                print(f"   ✅ Total coordinate tokens generated: {coordinate_tokens_generated}")
    
    except Exception as e:
        print(f"❌ Error testing coordinate conversion: {e}")
        import traceback
        traceback.print_exc()

def test_coordinate_token_id_ranges():
    """Test coordinate token ID ranges and vocabulary."""
    print("\n🔍 Testing Coordinate Token ID Ranges")
    print("-" * 40)
    
    try:
        config = get_config()
        coord_manager = create_coordinate_token_manager(config)
        
        # Test coordinate token ID generation
        test_values = [0, 50, 100, 255, 500, 1000]
        
        for val in test_values:
            try:
                if hasattr(coord_manager, 'get_coordinate_token_id'):
                    token_id = coord_manager.get_coordinate_token_id(val)
                    print(f"   Value {val} -> Token ID {token_id}")
                elif hasattr(coord_manager, '_coordinate_to_token_id'):
                    token_id = coord_manager._coordinate_to_token_id(val)
                    print(f"   Value {val} -> Token ID {token_id}")
                else:
                    print(f"   Cannot find coordinate token ID method")
                    break
            except Exception as e:
                print(f"   Value {val} -> Error: {e}")
        
        # Check if coordinate tokens are in the expected high range
        if hasattr(coord_manager, 'coordinate_token_ranges'):
            ranges = coord_manager.coordinate_token_ranges
            print(f"\n📊 Coordinate token ranges: {ranges}")
            
            # Check if the ranges are in the high token ID space (should be > 150000)
            for range_name, (start, end) in ranges.items():
                if start < 150000:
                    print(f"   ⚠️ WARNING: {range_name} range starts at {start}, which is low")
                else:
                    print(f"   ✅ {range_name} range: {start}-{end} (high range, good)")
    
    except Exception as e:
        print(f"❌ Error testing coordinate token ranges: {e}")
        import traceback
        traceback.print_exc()

def test_conversation_formatting():
    """Test how coordinates are formatted in conversations."""
    print("\n🔍 Testing Conversation Formatting")
    print("-" * 40)
    
    try:
        samples = load_sample_data(1)
        sample = samples[0]
        
        print(f"Sample has {len(sample['objects'])} objects")
        
        # Check if there are coordinate markers in the expected format
        obj = sample['objects'][0]
        desc = obj.get('desc', '')
        
        print(f"First object description: {desc}")
        
        # Look for geometry data
        for geom_type, coords in obj.items():
            if geom_type != 'desc':
                print(f"Geometry type: {geom_type}, coords: {coords}")
                
                # Check if this would generate coordinate tokens
                expected_format = f"<coordinate>{coords}</coordinate>"
                print(f"Expected coordinate format: {expected_format}")
                break
        
        # Test if the coordinate format matches what the chat processor expects
        print("\n🔍 Checking coordinate format expectations:")
        
        # This is what the training debug shows - we need coordinate tokens in the text
        expected_patterns = [
            "<coordinate>",
            "</coordinate>", 
            "坐标标记"  # Chinese coordinate markers
        ]
        
        print("Looking for coordinate patterns in conversation text...")
        # This would normally be done by chat processor, but we can check the raw data
        
    except Exception as e:
        print(f"❌ Error testing conversation formatting: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🚀 BBU Coordinate Token Pipeline Test")
    print("=" * 50)
    
    # Initialize configuration
    print("🔧 Initializing configuration...")
    try:
        init_config("configs/bbu_v2.yaml")
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Load sample data
    print("\n📂 Loading sample data...")
    try:
        samples = load_sample_data(3)
        print(f"✅ Loaded {len(samples)} samples")
    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        return
    
    # Run tests
    coord_manager = test_coordinate_token_manager()
    simple_manager = test_simple_token_manager()
    
    if simple_manager:
        test_coordinate_conversion_with_real_data(samples)
    
    if coord_manager:
        test_coordinate_token_id_ranges()
    
    test_conversation_formatting()
    
    # Final diagnosis
    print("\n🏥 DIAGNOSIS:")
    print("-" * 20)
    
    if not coord_manager:
        print("❌ CRITICAL: Coordinate token manager failed to initialize")
        print("   This would prevent coordinate tokens from being generated")
    
    if not simple_manager:
        print("❌ CRITICAL: Simple token manager failed to initialize")
        print("   This would prevent coordinate conversion in chat processor")
    
    if coord_manager and simple_manager:
        print("✅ Both coordinate managers initialized successfully")
        print("🔍 Issue likely in chat processor or data loading pipeline")
        print("📋 Recommended next steps:")
        print("   1. Check chat processor coordinate token integration")
        print("   2. Verify data loading in training pipeline")
        print("   3. Test with a single sample in training script")
    
    print(f"\n🏁 Coordinate token pipeline test complete.")

if __name__ == "__main__":
    main()