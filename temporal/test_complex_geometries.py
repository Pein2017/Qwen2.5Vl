#!/usr/bin/env python3
"""
Test GeometryProcessor with complex geometries (Square, LineString).
"""

import json
import sys
from pathlib import Path

# Add data_conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_conversion"))

from geometry_processor import GeometryProcessor

def test_complex_geometries():
    """Test with Square and LineString geometries."""
    
    # Load sample with complex geometries
    sample_file = Path("/data3/Qwen2.5-VL-main/ds_v2_clean/QC-20230216-0000244_377872.json")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get("markResult", {}).get("features", [])
    
    # Find different geometry types
    geometry_types = {}
    for feature in features:
        geom_type = feature.get("geometry", {}).get("type", "unknown")
        if geom_type not in geometry_types:
            geometry_types[geom_type] = feature
    
    print("Testing complex geometries:")
    print("=" * 60)
    
    for geom_type, feature in geometry_types.items():
        print(f"\nTesting {geom_type}:")
        
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        
        print(f"  Label: {properties.get('content', {}).get('label', 'unknown')}")
        print(f"  Raw coordinates (first 3): {str(geometry.get('coordinates', []))[:200]}...")
        
        # Test bbox extraction
        bbox = GeometryProcessor.extract_bbox_from_geometry(geometry)
        print(f"  Extracted bbox: {[round(x, 2) for x in bbox]}")
        
        # Test coordinate points
        points = GeometryProcessor.get_all_coordinate_points(geometry)
        print(f"  Total points: {len(points)}")
        if points:
            print(f"  First point: ({round(points[0][0], 2)}, {round(points[0][1], 2)})")
            print(f"  Last point: ({round(points[-1][0], 2)}, {round(points[-1][1], 2)})")
        
        # Test metadata detection
        info = GeometryProcessor.get_geometry_info(geometry)
        print(f"  Has lineType: {info.get('has_line_type', False)}")
        print(f"  Has lineMode: {info.get('has_line_mode', False)}")
        
        # Test scaling all coordinates
        scaled_geometry = GeometryProcessor.scale_all_coordinates(geometry, 0.75, 0.75)
        scaled_bbox = GeometryProcessor.extract_bbox_from_geometry(scaled_geometry)
        print(f"  Scaled bbox (0.75x): {[round(x, 2) for x in scaled_bbox]}")
        
        # Validate coordinate preservation
        original_points = GeometryProcessor.get_all_coordinate_points(geometry)
        scaled_points = GeometryProcessor.get_all_coordinate_points(scaled_geometry)
        
        if original_points and scaled_points and len(original_points) == len(scaled_points):
            # Check if scaling worked correctly
            scale_check = all(
                abs(sp[0] - op[0] * 0.75) < 0.1 and abs(sp[1] - op[1] * 0.75) < 0.1
                for op, sp in zip(original_points, scaled_points)
            )
            print(f"  Coordinate scaling correct: {scale_check}")
        
        print(f"  Bounds validation: {GeometryProcessor.validate_geometry_bounds(geometry, 1445, 1920)}")

def test_linestring_specifically():
    """Test LineString geometries specifically."""
    print("\n" + "=" * 60)
    print("LineString specific tests:")
    
    sample_file = Path("/data3/Qwen2.5-VL-main/ds_v2_clean/QC-20230216-0000244_377872.json")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get("markResult", {}).get("features", [])
    
    # Find LineString features
    linestring_features = [f for f in features if f.get("geometry", {}).get("type") == "LineString"]
    
    print(f"Found {len(linestring_features)} LineString features")
    
    for i, feature in enumerate(linestring_features[:3]):  # Test first 3
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        
        print(f"\nLineString {i+1}:")
        print(f"  Label: {properties.get('content', {}).get('label', 'unknown')}")
        print(f"  lineMode: {geometry.get('lineMode', 'N/A')}")
        print(f"  lineType: {geometry.get('lineType', 'N/A')}")
        
        # Extract path information
        coordinates = geometry.get("coordinates", [])
        points = GeometryProcessor.get_all_coordinate_points(geometry)
        bbox = GeometryProcessor.extract_bbox_from_geometry(geometry)
        
        print(f"  Path points: {len(points)}")
        print(f"  Path bbox: {[round(x, 2) for x in bbox]}")
        
        if points:
            # Calculate path length
            path_length = 0
            for j in range(1, len(points)):
                dx = points[j][0] - points[j-1][0]
                dy = points[j][1] - points[j-1][1]
                path_length += (dx*dx + dy*dy)**0.5
            
            print(f"  Approximate path length: {round(path_length, 2)} pixels")
            
            # Show path endpoints
            print(f"  Start point: ({round(points[0][0], 1)}, {round(points[0][1], 1)})")
            print(f"  End point: ({round(points[-1][0], 1)}, {round(points[-1][1], 1)})")

if __name__ == "__main__":
    test_complex_geometries()
    test_linestring_specifically()