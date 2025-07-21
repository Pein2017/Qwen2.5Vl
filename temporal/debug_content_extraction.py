#!/usr/bin/env python3
"""
Debug why content extraction is failing for v2 data.
"""

import json
import sys
from pathlib import Path

# Add data_conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_conversion"))

def debug_content_structure():
    """Debug the content structure in v2 data."""
    
    # Load a real v2 sample
    sample_file = Path("/data3/Qwen2.5-VL-main/ds_v2_clean/QC-20230216-0000244_377872.json")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get("markResult", {}).get("features", [])
    print(f"Analyzing content structure in {len(features)} features")
    print("=" * 60)
    
    for i, feature in enumerate(features[:5]):  # Check first 5
        print(f"\nFeature {i+1}:")
        
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        
        print(f"  Geometry type: {geometry.get('type', 'unknown')}")
        print(f"  Properties keys: {list(properties.keys())}")
        
        # Check content structures
        content = properties.get("content", {})
        content_zh = properties.get("contentZh", {})
        
        print(f"  Content keys: {list(content.keys())}")
        print(f"  ContentZh keys: {list(content_zh.keys())}")
        
        if content:
            print(f"  Content.label: {content.get('label', 'N/A')}")
            for key, value in content.items():
                if key != 'label':
                    print(f"    {key}: {value}")
        
        if content_zh:
            for key, value in content_zh.items():
                print(f"  ContentZh.{key}: {value}")

def test_content_extraction():
    """Test content extraction directly."""
    
    from config import DataConversionConfig
    from unified_processor import SampleExtractor
    
    # Create config
    config = DataConversionConfig(
        input_dir="/data3/Qwen2.5-VL-main/ds_v2_clean",
        output_dir="/tmp/test_output",
        language="chinese",
        response_types=["object_type", "property"]
    )
    
    # Initialize SampleExtractor
    extractor = SampleExtractor(config)
    
    # Load sample data
    sample_file = Path("/data3/Qwen2.5-VL-main/ds_v2_clean/QC-20230216-0000244_377872.json")
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get("markResult", {}).get("features", [])
    
    print(f"\n{'='*60}")
    print("Testing content extraction:")
    
    for i, feature in enumerate(features[:3]):
        print(f"\nFeature {i+1}:")
        
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        
        print(f"  Geometry: {geometry.get('type', 'unknown')}")
        
        # Test content extraction
        content_dict = extractor.extract_content_fields(properties)
        print(f"  Extracted content: {content_dict}")
        
        # Test filtering
        is_allowed = extractor.is_allowed_object(content_dict)
        print(f"  Is allowed: {is_allowed}")
        
        # Test description formatting
        if content_dict:
            from utils.transformations import FormatConverter
            desc = FormatConverter.format_description(
                content_dict, config.response_types, config.language
            )
            print(f"  Generated description: '{desc}'")

if __name__ == "__main__":
    print("Content Extraction Debug")
    print("=" * 60)
    
    debug_content_structure()
    test_content_extraction()
    
    print("\n" + "=" * 60)
    print("Debug completed!")