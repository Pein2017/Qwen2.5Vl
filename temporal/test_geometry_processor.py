#!/usr/bin/env python3
"""
Test script for GeometryProcessor with real v2 data.
"""

import json
import sys
from pathlib import Path

# Add data_conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_conversion"))

from geometry_processor import GeometryProcessor

def test_with_real_v2_data():
    """Test GeometryProcessor with actual v2 dataset samples."""
    
    # Load a real v2 sample
    sample_file = Path("/data3/Qwen2.5-VL-main/ds_v2_clean/QC-20230216-0000244_377872.json")
    
    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        return
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get("markResult", {}).get("features", [])
    print(f"Testing with {len(features)} features from {sample_file.name}")
    print("=" * 60)
    
    for i, feature in enumerate(features[:5]):  # Test first 5 features
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        
        print(f"\nFeature {i+1}:")
        print(f"  Geometry type: {geometry.get('type', 'unknown')}")
        print(f"  Label: {properties.get('content', {}).get('label', 'unknown')}")
        
        # Test bbox extraction
        bbox = GeometryProcessor.extract_bbox_from_geometry(geometry)
        print(f"  Extracted bbox: {bbox}")
        
        # Test coordinate point extraction
        points = GeometryProcessor.get_all_coordinate_points(geometry)
        print(f"  Total coordinate points: {len(points)}")
        if points:
            print(f"  First point: {points[0]}")
            print(f"  Last point: {points[-1]}")
        
        # Test geometry info
        info = GeometryProcessor.get_geometry_info(geometry)
        print(f"  Geometry info: {info}")
        
        # Test coordinate scaling
        scaled_geometry = GeometryProcessor.scale_all_coordinates(geometry, 0.5, 0.5)
        scaled_bbox = GeometryProcessor.extract_bbox_from_geometry(scaled_geometry)
        print(f"  Scaled bbox (0.5x): {scaled_bbox}")
        
        # Test bounds validation
        image_width = data.get("info", {}).get("width", 1440)
        image_height = data.get("info", {}).get("height", 1920)
        is_valid = GeometryProcessor.validate_geometry_bounds(geometry, image_width, image_height)
        print(f"  Valid bounds ({image_width}x{image_height}): {is_valid}")

def test_simple_bbox_compatibility():
    """Test that simple bbox lists still work correctly."""
    print("\n" + "=" * 60)
    print("Testing simple bbox compatibility:")
    
    # Simple bbox test
    simple_bbox = [100.0, 150.0, 200.0, 250.0]
    
    print(f"Input bbox: {simple_bbox}")
    
    # Test bbox extraction (should return same)
    extracted = GeometryProcessor.extract_bbox_from_geometry(simple_bbox)
    print(f"Extracted bbox: {extracted}")
    
    # Test coordinate points
    points = GeometryProcessor.get_all_coordinate_points(simple_bbox)
    print(f"Coordinate points: {points}")
    
    # Test scaling
    scaled = GeometryProcessor.scale_all_coordinates(simple_bbox, 2.0, 1.5)
    print(f"Scaled bbox (2x, 1.5x): {scaled}")
    
    # Test validation
    is_valid = GeometryProcessor.validate_geometry_bounds(simple_bbox, 300, 400)
    print(f"Valid bounds (300x400): {is_valid}")
    
    # Test geometry info
    info = GeometryProcessor.get_geometry_info(simple_bbox)
    print(f"Geometry info: {info}")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("Testing edge cases:")
    
    test_cases = [
        ("Empty geometry", {}),
        ("None input", None),
        ("Empty list", []),
        ("Invalid bbox", [1, 2, 3]),
        ("Empty coordinates", {"type": "LineString", "coordinates": []}),
    ]
    
    for name, geometry in test_cases:
        print(f"\n{name}: {geometry}")
        try:
            bbox = GeometryProcessor.extract_bbox_from_geometry(geometry)
            points = GeometryProcessor.get_all_coordinate_points(geometry)
            info = GeometryProcessor.get_geometry_info(geometry)
            print(f"  Bbox: {bbox}")
            print(f"  Points: {len(points)}")
            print(f"  Info: {info}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    print("GeometryProcessor Test Suite")
    print("=" * 60)
    
    test_with_real_v2_data()
    test_simple_bbox_compatibility()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("Test completed!")