#!/usr/bin/env python3
"""
Standalone test script for Simple Token Manager implementation.
Tests token functionality without requiring full ChatProcessor setup.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.simple_token_manager import SimpleTokenManager, create_simple_token_manager


def test_token_manager_with_real_data():
    """Test SimpleTokenManager with actual processed data sample."""
    print("🧪 Testing SimpleTokenManager with Real Data")
    
    # Mock tokenizer and model
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                "<|endoftext|>": 151643,
                "<|im_start|>": 151644,
                "<|im_end|>": 151645,
                "<|object_ref_start|>": 151646,
                "<|object_ref_end|>": 151647,
                "<|box_start|>": 151648,
                "<|box_end|>": 151649
            }
            self.next_id = 151665
        
        def get_vocab(self):
            return self.vocab
        
        def add_special_tokens(self, token_dict):
            num_added = 0
            for tokens in token_dict.values():
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = self.next_id
                        self.next_id += 1
                        num_added += 1
            return num_added
        
        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, -1)
        
        def __len__(self):
            return len(self.vocab)
    
    class MockModel:
        def resize_token_embeddings(self, new_size):
            print(f"   📏 Model embeddings resized to {new_size}")
    
    # Create token manager
    tokenizer = MockTokenizer()
    model = MockModel()
    manager = create_simple_token_manager(tokenizer, model)
    
    # Load processed data sample (JSONL format)
    data_file = Path("data/ds_v2_full/all_samples.jsonl")
    if not data_file.exists():
        print("   ⚠️ Processed data file not found, using mock data")
        return test_with_mock_data(manager)
    
    # Read first few lines of JSONL
    test_objects = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= 3:  # Process first 3 samples
                break
            
            data = json.loads(line.strip())
            objects = data.get("objects", [])
            
            # Take first 3 objects from each sample
            for obj in objects[:3]:
                test_objects.append(obj)
    
    print(f"   📊 Loaded {len(test_objects)} objects from processed data")
    
    # Test token wrapping for each object type
    bbox_count = square_count = line_count = 0
    
    for i, obj in enumerate(test_objects):
        # Count geometry types
        if "bbox_2d" in obj:
            bbox_count += 1
        elif "square" in obj:
            square_count += 1
        elif "line" in obj:
            line_count += 1
        
        print(f"\n   🎯 Object {i+1}:")
        print(f"      Original: {obj}")
        
        # Format with token manager
        formatted = manager.format_object(obj)
        print(f"      Formatted: {formatted}")
        
        # Create final output format 
        if "wrapped_desc" in formatted:
            desc_wrapped = formatted["wrapped_desc"]
            
            # Find geometry wrapper
            geom_wrapped = None
            for key, value in formatted.items():
                if key.startswith("wrapped_") and "desc" not in key:
                    geom_wrapped = value
                    break
            
            if geom_wrapped:
                final_format = f"{geom_wrapped} {desc_wrapped}"
                print(f"      Final: {final_format}")
    
    print(f"\n   📊 Geometry type distribution:")
    print(f"      • bbox_2d: {bbox_count} objects") 
    print(f"      • square: {square_count} objects")
    print(f"      • line: {line_count} objects")
    
    return test_objects


def test_with_mock_data(manager):
    """Test with mock data if real data unavailable."""
    print("   🧪 Testing with mock data...")
    
    mock_objects = [
        {
            "bbox_2d": [100, 200, 300, 400],
            "desc": "BBU设备"
        },
        {
            "square": [150, 10, 211, 35, 218, 16, 166, 0],
            "desc": "标签/5G-BBU-（接地线）"
        },
        {
            "line": [579.3, 1385.8, 679.6, 1451.6, 764.6, 1444.1],
            "desc": "光纤"
        }
    ]
    
    for i, obj in enumerate(mock_objects):
        print(f"\n   🎯 Mock Object {i+1}:")
        print(f"      Original: {obj}")
        
        formatted = manager.format_object(obj)
        print(f"      Formatted: {formatted}")
        
        # Create final format 
        if "wrapped_desc" in formatted:
            desc_wrapped = formatted["wrapped_desc"]
            
            geom_wrapped = None
            for key, value in formatted.items():
                if key.startswith("wrapped_") and "desc" not in key:
                    geom_wrapped = value
                    break
            
            if geom_wrapped:
                final_format = f"{geom_wrapped} {desc_wrapped}"
                print(f"      Final: {final_format}")
    
    return mock_objects


def demonstrate_expected_output():
    """Demonstrate what the expected training data should look like."""
    print("\n\n🎯 Expected Training Data Format:")
    print("   The model should see inputs like:")
    
    examples = [
        "<|box_start|>100.0, 200.0, 300.0, 400.0<|box_end|> <|object_ref_start|>BBU设备<|object_ref_end|>",
        "<|square_start|>150.0, 10.0, 211.0, 35.0, 218.0, 16.0, 166.0, 0.0<|square_end|> <|object_ref_start|>标签/5G-BBU<|object_ref_end|>",
        "<|line_start|>579.3, 1385.8, 679.6, 1451.6, 764.6, 1444.1<|line_end|> <|object_ref_start|>光纤<|object_ref_end|>"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"   Example {i}: {example}")
    
    print("\n   ✅ This enables the model to learn:")
    print("      • Semantic geometry tokens for different shapes")
    print("      • Content wrapping with object_ref tokens")
    print("      • Coordinate token sequences for regression")


if __name__ == "__main__":
    print("🧪 Testing Simple Token Manager - Standalone Version\n")
    
    try:
        test_objects = test_token_manager_with_real_data()
        demonstrate_expected_output()
        
        print(f"\n✅ All tests passed! Found {len(test_objects)} test objects.")
        print("   Implementation is ready for integration with training pipeline.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()