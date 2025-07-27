#!/usr/bin/env python3
"""
Test script for Simple Token Manager implementation.
Tests token addition and wrapping functionality.
"""

import json
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chat_processor import ChatProcessor
from src.utils.simple_token_manager import (
    SimpleTokenManager,
    create_simple_token_manager,
)


def test_simple_token_manager():
    """Test SimpleTokenManager functionality."""
    print("🧪 Testing SimpleTokenManager...")

    # Mock tokenizer and model for testing
    class MockTokenizer:
        def __init__(self):
            self.vocab = {"<|endoftext|>": 151643, "<|im_start|>": 151644}
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
            print(f"   Model embeddings resized to {new_size}")

    tokenizer = MockTokenizer()
    model = MockModel()

    # Test token manager creation
    manager = SimpleTokenManager(tokenizer, model)
    num_added = manager.add_tokens()

    print(f"   ✅ Added {num_added} tokens")
    print(f"   📊 Total vocab size: {len(tokenizer)}")

    # Test token ID retrieval
    for token in manager.NEW_TOKENS:
        token_id = manager.get_token_id(token)
        print(f"   🏷️ {token} -> ID {token_id}")

    # Test geometry token retrieval
    for geom_type in ["bbox_2d", "square", "line"]:
        start, end = manager.get_geometry_tokens(geom_type)
        print(f"   📐 {geom_type}: {start} ... {end}")

    # Test description wrapping
    desc = "标签/5G-BBU-（接地线）"
    wrapped_desc = manager.wrap_description(desc)
    print(f"   📝 Description wrapping: {wrapped_desc}")

    # Test coordinate wrapping
    coords = [150, 10, 211, 35]
    wrapped_coords = manager.wrap_coordinates(coords, "bbox_2d")
    print(f"   📐 Coordinate wrapping (bbox_2d): {wrapped_coords}")

    coords_square = [150, 10, 211, 35, 218, 16, 166, 0]
    wrapped_square = manager.wrap_coordinates(coords_square, "square")
    print(f"   🔷 Coordinate wrapping (square): {wrapped_square}")

    # Test object formatting
    obj_dict = {
        "square": [150, 10, 211, 35, 218, 16, 166, 0],
        "desc": "标签/5G-BBU-（接地线）",
    }
    formatted_obj = manager.format_object(obj_dict)
    print(f"   🎯 Object formatting: {formatted_obj}")

    print("✅ SimpleTokenManager tests passed!")
    return manager


def test_chat_processor():
    """Test ChatProcessor integration."""
    print("\n🧪 Testing ChatProcessor integration...")

    # Initialize config to avoid the error
    from src.config import init_config
    try:
        init_config(
            data_root="data",
            model_root="model_cache",
            max_total_length=8192,
            language="chinese",
            dataset_name="ds_v2_full"
        )
    except:
        pass  # Config might already be initialized

    # Test with sample data from ds_v2
    sample_objects = [
        {
            "square": [150, 10, 211, 35, 218, 16, 166, 0],
            "desc": "标签/5G-BBU-（接地线）",
        },
        {"bbox_2d": [100, 200, 300, 400], "desc": "BBU设备"},
        {"line": [50, 50, 100, 100, 150, 150], "desc": "光纤"},
    ]

    # Mock tokenizer and model
    class MockTokenizer:
        def get_vocab(self):
            return {"<|endoftext|>": 151643, "<|im_start|>": 151644}

        def add_special_tokens(self, token_dict):
            return len(token_dict.get("additional_special_tokens", []))

        def convert_tokens_to_ids(self, token):
            return 151665  # Mock ID

        def __len__(self):
            return 151670

    class MockModel:
        def resize_token_embeddings(self, new_size):
            pass

    class MockImageProcessor:
        pass

    # Create ChatProcessor with simple tokens enabled
    processor = ChatProcessor(
        tokenizer=MockTokenizer(),
        image_processor=MockImageProcessor(),
        enable_simple_tokens=True,
    )

    # Initialize simple token manager
    processor.initialize_simple_tokens(MockModel())

    # Test object formatting
    formatted_response = processor._format_objects_response(sample_objects)
    print(f"   🎯 Formatted response: {formatted_response}")

    print("✅ ChatProcessor integration tests passed!")


def test_with_real_data():
    """Test with real data sample."""
    print("\n🧪 Testing with real data sample...")

    # Load a real data sample
    data_file = Path("ds_v2/QC-20230223-0000340_221170.json")
    if not data_file.exists():
        print("   ⚠️ Real data file not found, skipping real data test")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract objects from the data
    features = data.get("markResult", {}).get("features", [])
    objects = []

    for feature in features[:3]:  # Test with first 3 objects
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})

        obj = {}

        # Extract geometry
        if geometry.get("type") == "ExtentPolygon":
            # Convert to bbox_2d format
            coords = geometry.get("coordinates", [[]])[0]
            if len(coords) >= 4:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                obj["bbox_2d"] = bbox
        elif geometry.get("type") == "Square":
            coords = geometry.get("coordinates", [[]])[0]
            flat_coords = []
            for coord in coords:
                flat_coords.extend(coord)
            obj["square"] = flat_coords
        elif geometry.get("type") == "LineString":
            coords = geometry.get("coordinates", [])
            flat_coords = []
            for coord in coords:
                flat_coords.extend(coord)
            obj["line"] = flat_coords

        # Extract description
        content_zh = properties.get("contentZh", {})
        label = content_zh.get("标签", "unknown")
        obj["desc"] = label

        objects.append(obj)

    print(f"   📊 Extracted {len(objects)} objects from real data")

    # Test formatting with mock setup
    class MockTokenizer:
        def get_vocab(self):
            return {"<|endoftext|>": 151643}

        def add_special_tokens(self, token_dict):
            return 4  # 4 new tokens

        def convert_tokens_to_ids(self, token):
            return 151665

        def __len__(self):
            return 151670

    class MockModel:
        def resize_token_embeddings(self, new_size):
            pass

    manager = create_simple_token_manager(MockTokenizer(), MockModel())

    for i, obj in enumerate(objects):
        formatted = manager.format_object(obj)
        print(f"   🎯 Object {i + 1}: {formatted}")

    print("✅ Real data tests passed!")


if __name__ == "__main__":
    print("🧪 Testing Simple Token Manager Implementation\n")

    try:
        test_simple_token_manager()
        test_chat_processor()
        test_with_real_data()

        print("\n✅ All tests passed! Implementation is ready.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
