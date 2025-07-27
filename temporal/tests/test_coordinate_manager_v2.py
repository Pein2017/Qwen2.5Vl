#!/usr/bin/env python3
"""
Test enhanced CoordinateManager with unified geometry processing.
"""

import json
import sys
from pathlib import Path

# Add data_conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_conversion"))

from coordinate_manager import CoordinateManager

def test_unified_geometry_transformation():
    """Test the new transform_geometry_complete method."""
    
    # Load a real v2 sample
    sample_file = Path("ds_v2_clean/QC-20230216-0000244_377872.json")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get image info
    image_info = data.get("info", {})
    json_width = image_info.get("width", 1445)
    json_height = image_info.get("height", 1920)
    
    # Find corresponding image file
    image_name = sample_file.stem + ".jpg"
    image_path = Path("ds_v2_clean") / image_name
    
    # If image doesn't exist, find any available image for testing
    if not image_path.exists():
        # Find any image file for testing
        image_files = list(Path("/data3/Qwen2.5-VL-main").rglob("*.jpg")) + list(Path("/data3/Qwen2.5-VL-main").rglob("*.png"))
        if image_files:
            image_path = image_files[0]
            print(f"Using test image: {image_path}")
        else:
            print("No image files found for testing")
            return
    
    features = data.get("markResult", {}).get("features", [])
    print(f"Testing unified geometry transformation with {len(features)} features")
    print(f"Image: {image_path.name}, JSON dimensions: {json_width}x{json_height}")
    print("=" * 70)
    
    # Test different geometry types
    geometry_types = {}
    for feature in features:
        geom_type = feature.get("geometry", {}).get("type", "unknown")
        if geom_type not in geometry_types:
            geometry_types[geom_type] = feature
    
    for geom_type, feature in geometry_types.items():
        print(f"\nTesting {geom_type}:")
        
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        
        print(f"  Label: {properties.get('content', {}).get('label', 'unknown')}")
        
        # Test the unified transformation
        try:
            bbox, transformed_geometry, final_width, final_height = CoordinateManager.transform_geometry_complete(
                geometry_input=geometry,
                image_path=image_path,
                json_width=json_width,
                json_height=json_height,
                enable_smart_resize=True
            )
            
            print(f"  Original geometry type: {geometry.get('type', 'unknown')}")
            print(f"  Extracted bbox: {[round(x, 2) for x in bbox]}")
            print(f"  Final dimensions: {final_width}x{final_height}")
            print(f"  Transformed geometry type: {transformed_geometry.get('type', 'simple_bbox')}")
            
            # Validate transformation
            valid = CoordinateManager.validate_geometry_bounds(
                transformed_geometry, final_width, final_height
            )
            print(f"  Geometry validation: {valid}")
            
            # Show coordinate count preservation
            from geometry_processor import GeometryProcessor
            original_points = GeometryProcessor.get_all_coordinate_points(geometry)
            transformed_points = GeometryProcessor.get_all_coordinate_points(transformed_geometry)
            
            print(f"  Coordinate points: {len(original_points)} -> {len(transformed_points)}")
            
            if original_points and transformed_points:
                print(f"  First point: {original_points[0]} -> {[round(x, 2) for x in transformed_points[0]]}")
        
        except Exception as e:
            print(f"  Error: {e}")

def test_simple_bbox_compatibility():
    """Test that simple bbox still works with new unified methods."""
    print("\n" + "=" * 70)
    print("Testing simple bbox compatibility:")
    
    # Simple bbox test
    simple_bbox = [100.0, 150.0, 300.0, 250.0]
    
    # Use any available image for testing
    image_files = list(Path("/data3/Qwen2.5-VL-main").rglob("*.jpg")) + list(Path("/data3/Qwen2.5-VL-main").rglob("*.png"))
    if not image_files:
        print("No image files found for testing")
        return
    
    image_path = image_files[0]
    json_width, json_height = 1440, 1920
    
    print(f"Input bbox: {simple_bbox}")
    print(f"Test image: {image_path.name}")
    
    try:
        bbox, transformed_geometry, final_width, final_height = CoordinateManager.transform_geometry_complete(
            geometry_input=simple_bbox,
            image_path=image_path,
            json_width=json_width,
            json_height=json_height,
            enable_smart_resize=True
        )
        
        print(f"Extracted bbox: {[round(x, 2) for x in bbox]}")
        print(f"Transformed geometry: {[round(x, 2) for x in transformed_geometry] if isinstance(transformed_geometry, list) else 'complex'}")
        print(f"Final dimensions: {final_width}x{final_height}")
        
        # Validate
        valid = CoordinateManager.validate_geometry_bounds(
            transformed_geometry, final_width, final_height
        )
        print(f"Validation: {valid}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_scaling_accuracy():
    """Test that coordinate scaling is accurate across all geometry types."""
    print("\n" + "=" * 70)
    print("Testing scaling accuracy:")
    
    # Create test geometries
    test_cases = [
        ("Simple bbox", [100, 100, 200, 200]),
        ("ExtentPolygon", {
            "type": "ExtentPolygon",
            "coordinates": [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]
        }),
        ("Square", {
            "type": "Square", 
            "lineType": ["LLLLL"],
            "coordinates": [[[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]]
        }),
        ("LineString", {
            "type": "LineString",
            "lineMode": 1,
            "lineType": "LLL",
            "coordinates": [[100, 100], [150, 150], [200, 200]]
        })
    ]
    
    # Test scaling factors
    scale_x, scale_y = 2.0, 1.5
    
    for name, geometry in test_cases:
        print(f"\n{name}:")
        
        try:
            # Apply scaling directly
            scaled = CoordinateManager.apply_smart_resize_to_geometry(
                geometry, 100, 100, 200, 150  # 2x width, 1.5x height
            )
            
            # Extract coordinate points for verification
            from geometry_processor import GeometryProcessor
            original_points = GeometryProcessor.get_all_coordinate_points(geometry)
            scaled_points = GeometryProcessor.get_all_coordinate_points(scaled)
            
            print(f"  Original points: {len(original_points)}")
            print(f"  Scaled points: {len(scaled_points)}")
            
            if original_points and scaled_points and len(original_points) == len(scaled_points):
                # Check scaling accuracy
                correct_scaling = True
                for op, sp in zip(original_points[:3], scaled_points[:3]):  # Check first 3 points
                    expected_x = op[0] * scale_x
                    expected_y = op[1] * scale_y
                    actual_x, actual_y = sp
                    
                    if abs(actual_x - expected_x) > 0.1 or abs(actual_y - expected_y) > 0.1:
                        correct_scaling = False
                        break
                    
                    print(f"    {op} -> {sp} (expected: {expected_x:.1f}, {expected_y:.1f})")
                
                print(f"  Scaling accuracy: {'✓' if correct_scaling else '✗'}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    print("Enhanced CoordinateManager Test Suite")
    print("=" * 70)
    
    test_unified_geometry_transformation()
    test_simple_bbox_compatibility() 
    test_scaling_accuracy()
    
    print("\n" + "=" * 70)
    print("Test completed!")