#!/usr/bin/env python3
"""
Test the enhanced sample_processor with unified geometry processing.
"""

import json
import sys
from pathlib import Path

# Add data_conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_conversion"))

from sample_processor import SampleProcessor

def test_geometry_scaling():
    """Test the new geometry scaling functionality."""
    
    print("Testing unified geometry scaling")
    print("=" * 50)
    
    # Create test objects with mixed geometry types
    test_objects = [
        {
            "bbox_2d": [100, 100, 200, 200],
            "desc": "Simple bbox object"
        },
        {
            "bbox_2d": [150, 150, 250, 250],
            "desc": "ExtentPolygon object",
            "geometry": {
                "type": "ExtentPolygon",
                "coordinates": [[150, 150], [250, 150], [250, 250], [150, 250], [150, 150]]
            }
        },
        {
            "bbox_2d": [200, 200, 300, 350],
            "desc": "LineString object",
            "geometry": {
                "type": "LineString",
                "lineMode": 1,
                "lineType": "LLLL",
                "coordinates": [[200, 200], [250, 250], [300, 300], [300, 350]]
            }
        }
    ]
    
    # Create a minimal sample processor
    processor = SampleProcessor(
        language="chinese",
        response_types={"object_type", "property"},
        label_hierarchy={"test": []},
        resize_enabled=True,
        output_image_dir=Path("/tmp/test_output"),
        input_dir=Path("/tmp/test_input")
    )
    
    print(f"Original objects: {len(test_objects)}")
    for i, obj in enumerate(test_objects):
        print(f"  Object {i+1}: bbox={obj['bbox_2d']}, "
              f"geometry={obj.get('geometry', {}).get('type', 'none')}")
    
    # Test scaling from 1000x1000 to 500x750 (0.5x width, 0.75x height)
    original_width, original_height = 1000, 1000
    new_width, new_height = 500, 750
    
    print(f"\nScaling from {original_width}x{original_height} to {new_width}x{new_height}")
    
    # Apply scaling
    processor._scale_geometries(test_objects, original_width, original_height, new_width, new_height)
    
    print(f"\nScaled objects:")
    for i, obj in enumerate(test_objects):
        bbox = obj['bbox_2d']
        print(f"  Object {i+1}: bbox={[round(x, 2) for x in bbox]}")
        
        if "geometry" in obj:
            from geometry_processor import GeometryProcessor
            points = GeometryProcessor.get_all_coordinate_points(obj["geometry"])
            print(f"    Geometry points: {len(points)}")
            if points:
                print(f"    First point: ({round(points[0][0], 2)}, {round(points[0][1], 2)})")
                print(f"    Last point: ({round(points[-1][0], 2)}, {round(points[-1][1], 2)})")
    
    # Verify scaling accuracy
    print(f"\nScaling verification:")
    expected_scale_x = new_width / original_width  # 0.5
    expected_scale_y = new_height / original_height  # 0.75
    
    # Check first object (simple bbox)
    obj1_bbox = test_objects[0]['bbox_2d']
    expected_bbox1 = [100 * expected_scale_x, 100 * expected_scale_y, 
                     200 * expected_scale_x, 200 * expected_scale_y]
    bbox1_correct = all(abs(actual - expected) < 0.1 
                       for actual, expected in zip(obj1_bbox, expected_bbox1))
    print(f"  Object 1 bbox scaling: {'✓' if bbox1_correct else '✗'}")
    
    # Check geometry object coordinate scaling
    if "geometry" in test_objects[1]:
        from geometry_processor import GeometryProcessor
        points = GeometryProcessor.get_all_coordinate_points(test_objects[1]["geometry"])
        if points:
            first_point = points[0]
            expected_point = (150 * expected_scale_x, 150 * expected_scale_y)
            point_correct = (abs(first_point[0] - expected_point[0]) < 0.1 and 
                           abs(first_point[1] - expected_point[1]) < 0.1)
            print(f"  Object 2 geometry scaling: {'✓' if point_correct else '✗'}")

def test_v2_object_extraction():
    """Test object extraction from v2 format data."""
    
    print(f"\n{'='*50}")
    print("Testing v2 object extraction")
    
    # Load real v2 data
    sample_file = Path("ds_v2_clean/QC-20230216-0000244_377872.json")
    
    if not sample_file.exists():
        print("Sample file not found - skipping test")
        return
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create sample processor with expanded label hierarchy for v2 data
    expanded_hierarchy = {
        "connect_point": ["螺丝", "光纤插头", "BBU安装螺丝", "机柜处接地螺丝", "fiber_bbu"],
        "label": ["标签", "5G-AAU1-光纤", "5G-AAU2-光纤", "5G-BBU-传输光纤"],
        "bbu": ["BBU设备", "华为"],
        "fiber": ["光纤", "传输光纤"],
        "wire": ["电线", "接地线"]
    }
    
    processor = SampleProcessor(
        language="chinese",
        response_types={"object_type", "property"},
        label_hierarchy=expanded_hierarchy,
        resize_enabled=False
    )
    
    # Extract objects
    features = data.get("markResult", {}).get("features", [])
    print(f"Processing {len(features)} features")
    
    objects = processor._extract_objects_from_markresult(features)
    print(f"Extracted {len(objects)} valid objects")
    
    # Analyze by geometry type
    geometry_stats = {}
    for obj in objects:
        if "geometry" in obj:
            geom_type = obj["geometry"].get("type", "unknown")
            geometry_stats[geom_type] = geometry_stats.get(geom_type, 0) + 1
        else:
            geometry_stats["bbox_only"] = geometry_stats.get("bbox_only", 0) + 1
    
    print(f"Geometry types found:")
    for geom_type, count in geometry_stats.items():
        print(f"  {geom_type}: {count} objects")
    
    # Show sample objects
    print(f"\nSample extracted objects:")
    for i, obj in enumerate(objects[:3]):
        print(f"  Object {i+1}: {obj['desc']}")
        print(f"    Bbox: {[round(x, 2) for x in obj['bbox_2d']]}")
        if "geometry" in obj:
            print(f"    Geometry: {obj['geometry'].get('type', 'unknown')}")

def test_backwards_compatibility():
    """Test that old bbox-only processing still works."""
    
    print(f"\n{'='*50}")
    print("Testing backwards compatibility")
    
    # Create old-style test data (bbox only)
    old_test_objects = [
        {"bbox_2d": [100, 100, 200, 200], "desc": "Old format object 1"},
        {"bbox_2d": [300, 300, 400, 400], "desc": "Old format object 2"},
    ]
    
    processor = SampleProcessor(
        language="chinese",
        response_types={"object_type", "property"},
        label_hierarchy={"test": []},
        resize_enabled=True,
        output_image_dir=Path("/tmp/test_output"),
        input_dir=Path("/tmp/test_input")
    )
    
    print(f"Original old-format objects: {len(old_test_objects)}")
    for i, obj in enumerate(old_test_objects):
        print(f"  Object {i+1}: {obj['bbox_2d']}")
    
    # Test scaling
    processor._scale_geometries(old_test_objects, 1000, 1000, 500, 500)
    
    print(f"Scaled old-format objects:")
    for i, obj in enumerate(old_test_objects):
        bbox = obj['bbox_2d']
        print(f"  Object {i+1}: {[round(x, 2) for x in bbox]}")
    
    # Verify 0.5x scaling
    expected_bbox1 = [50.0, 50.0, 100.0, 100.0]
    actual_bbox1 = old_test_objects[0]['bbox_2d']
    scaling_correct = all(abs(a - e) < 0.1 for a, e in zip(actual_bbox1, expected_bbox1))
    print(f"Old format scaling accuracy: {'✓' if scaling_correct else '✗'}")

if __name__ == "__main__":
    print("Enhanced SampleProcessor Test Suite")
    print("=" * 50)
    
    test_geometry_scaling()
    test_v2_object_extraction()
    test_backwards_compatibility()
    
    print("\n" + "=" * 50)
    print("Test completed!")