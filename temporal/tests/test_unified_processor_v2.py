#!/usr/bin/env python3
"""
Test the updated unified_processor with v2 geometry support.
"""

import json
import sys
from pathlib import Path

# Add data_conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_conversion"))

from config import DataConversionConfig
from unified_processor import SampleExtractor

def test_v2_geometry_extraction():
    """Test that SampleExtractor can handle v2 geometries."""
    
    # Load a real v2 sample
    sample_file = Path("ds_v2_clean/QC-20230216-0000244_377872.json")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a minimal config
    config = DataConversionConfig(
        input_dir="ds_v2_clean",
        output_dir="/tmp/test_output",
        language="chinese",
        response_types=["object_type", "property"]
    )
    
    # Initialize SampleExtractor
    extractor = SampleExtractor(config)
    
    # Test feature extraction
    features = data.get("markResult", {}).get("features", [])
    print(f"Testing geometry extraction with {len(features)} features")
    print("=" * 60)
    
    # Extract objects using new method
    objects = extractor.extract_objects_from_markresult(features)
    
    print(f"Extracted {len(objects)} valid objects")
    
    # Analyze extracted objects by geometry type
    geometry_stats = {}
    
    for i, obj in enumerate(objects[:10]):  # Show first 10
        print(f"\nObject {i+1}:")
        print(f"  Description: {obj['desc']}")
        print(f"  Bbox: {[round(x, 2) for x in obj['bbox_2d']]}")
        
        if "geometry" in obj:
            geom_type = obj["geometry"].get("type", "unknown")
            print(f"  Geometry type: {geom_type}")
            
            # Count geometry types
            geometry_stats[geom_type] = geometry_stats.get(geom_type, 0) + 1
            
            # Show geometry info
            from geometry_processor import GeometryProcessor
            info = GeometryProcessor.get_geometry_info(obj["geometry"])
            print(f"  Point count: {info['point_count']}")
            print(f"  Has lineType: {info.get('has_line_type', False)}")
            print(f"  Has lineMode: {info.get('has_line_mode', False)}")
        else:
            print(f"  Geometry: bbox only (legacy format)")
            geometry_stats["bbox_only"] = geometry_stats.get("bbox_only", 0) + 1
    
    print(f"\nGeometry type summary:")
    for geom_type, count in geometry_stats.items():
        print(f"  {geom_type}: {count} objects")

def test_coordinate_transformation():
    """Test the unified coordinate transformation pipeline."""
    
    # Find test image
    image_files = list(Path("/data3/Qwen2.5-VL-main").rglob("*.jpg")) + list(Path("/data3/Qwen2.5-VL-main").rglob("*.png"))
    if not image_files:
        print("No image files found for testing")
        return
    
    image_path = image_files[0]
    
    # Create test sample data with mixed geometry types
    test_sample = {
        "objects": [
            {
                "bbox_2d": [100, 100, 200, 200],
                "desc": "Simple bbox object",
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
                "bbox_2d": [200, 200, 300, 300],
                "desc": "LineString object",
                "geometry": {
                    "type": "LineString",
                    "lineMode": 1,
                    "lineType": "LLL",
                    "coordinates": [[200, 200], [250, 250], [300, 300]]
                }
            }
        ]
    }
    
    # Create config
    config = DataConversionConfig(
        input_dir="ds_v2_clean",
        output_dir="/tmp/test_output",
        language="chinese",
        response_types=["object_type", "property"],
        resize_enabled=True
    )
    
    # Create UnifiedProcessor
    from unified_processor import UnifiedProcessor
    processor = UnifiedProcessor(config)
    
    print(f"\n{'='*60}")
    print("Testing unified coordinate transformation:")
    print(f"Test image: {image_path.name}")
    print(f"Input sample has {len(test_sample['objects'])} objects")
    
    try:
        # Test the unified coordinate processing
        processed_sample, final_width, final_height = processor._process_sample_coordinates_unified(
            test_sample, image_path, 1440, 1920, enable_smart_resize=True
        )
        
        print(f"Final dimensions: {final_width}x{final_height}")
        print(f"Output sample has {len(processed_sample['objects'])} objects")
        
        for i, obj in enumerate(processed_sample['objects']):
            print(f"\nObject {i+1}: {obj['desc']}")
            print(f"  Transformed bbox: {[round(x, 2) for x in obj['bbox_2d']]}")
            
            if "geometry" in obj:
                geom_type = obj["geometry"].get("type", "unknown")
                print(f"  Preserved geometry: {geom_type}")
                
                # Show coordinate transformation
                from geometry_processor import GeometryProcessor
                points = GeometryProcessor.get_all_coordinate_points(obj["geometry"])
                print(f"  Coordinate points: {len(points)}")
                if points:
                    print(f"  First point: ({round(points[0][0], 2)}, {round(points[0][1], 2)})")
    
    except Exception as e:
        print(f"Error in coordinate transformation: {e}")
        import traceback
        traceback.print_exc()

def test_backwards_compatibility():
    """Test that old bbox-only format still works."""
    
    print(f"\n{'='*60}")
    print("Testing backwards compatibility with old bbox format:")
    
    # Create old-style sample (bbox only, no geometry)
    old_sample = {
        "objects": [
            {"bbox_2d": [100, 100, 200, 200], "desc": "Old format object 1"},
            {"bbox_2d": [300, 300, 400, 400], "desc": "Old format object 2"},
        ]
    }
    
    # Find test image
    image_files = list(Path("/data3/Qwen2.5-VL-main").rglob("*.jpg")) + list(Path("/data3/Qwen2.5-VL-main").rglob("*.png"))
    if not image_files:
        print("No image files found for testing")
        return
    
    image_path = image_files[0]
    
    # Create config
    config = DataConversionConfig(
        input_dir="ds_v2_clean",
        output_dir="/tmp/test_output",
        language="chinese",
        response_types=["object_type", "property"],
        resize_enabled=True
    )
    
    # Create UnifiedProcessor
    from unified_processor import UnifiedProcessor
    processor = UnifiedProcessor(config)
    
    try:
        # Test with old format
        processed_sample, final_width, final_height = processor._process_sample_coordinates_unified(
            old_sample, image_path, 1440, 1920, enable_smart_resize=True
        )
        
        print(f"✓ Old format processed successfully")
        print(f"  Final dimensions: {final_width}x{final_height}")
        print(f"  Objects processed: {len(processed_sample['objects'])}")
        
        for i, obj in enumerate(processed_sample['objects']):
            print(f"  Object {i+1}: {[round(x, 2) for x in obj['bbox_2d']]}")
    
    except Exception as e:
        print(f"✗ Error processing old format: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Unified Processor V2 Test Suite")
    print("=" * 60)
    
    test_v2_geometry_extraction()
    test_coordinate_transformation()
    test_backwards_compatibility()
    
    print("\n" + "=" * 60)
    print("Test completed!")