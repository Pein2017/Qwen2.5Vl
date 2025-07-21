#!/usr/bin/env python3
"""
Test the enhanced teacher_selector with geometry type diversity.
"""

import sys
from pathlib import Path

# Add data_conversion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_conversion"))

from teacher_selector import TeacherSelector

def create_test_samples():
    """Create test samples with various geometry types."""
    
    samples = [
        # Simple bbox samples
        {
            "objects": [
                {"bbox_2d": [100, 100, 200, 200], "desc": "connect_point"},
                {"bbox_2d": [300, 300, 400, 400], "desc": "bbu"}
            ],
            "width": 1000, "height": 1000
        },
        
        # ExtentPolygon samples
        {
            "objects": [
                {
                    "bbox_2d": [150, 150, 250, 250], 
                    "desc": "label",
                    "geometry": {
                        "type": "ExtentPolygon",
                        "coordinates": [[150, 150], [250, 150], [250, 250], [150, 250], [150, 150]]
                    }
                }
            ],
            "width": 1000, "height": 1000
        },
        
        # LineString samples
        {
            "objects": [
                {
                    "bbox_2d": [200, 200, 800, 600],
                    "desc": "fiber",
                    "geometry": {
                        "type": "LineString",
                        "lineMode": 1,
                        "lineType": "LLLLLL",
                        "coordinates": [[200, 200], [300, 250], [400, 300], [500, 400], [600, 500], [800, 600]]
                    }
                }
            ],
            "width": 1000, "height": 1000
        },
        
        # Square samples
        {
            "objects": [
                {
                    "bbox_2d": [100, 100, 300, 300],
                    "desc": "bbu",
                    "geometry": {
                        "type": "Square",
                        "lineType": ["LLLLL"],
                        "coordinates": [[[100, 100], [300, 100], [300, 300], [100, 300], [100, 100]]]
                    }
                }
            ],
            "width": 1000, "height": 1000
        },
        
        # Mixed geometry sample
        {
            "objects": [
                {"bbox_2d": [50, 50, 150, 150], "desc": "connect_point"},
                {
                    "bbox_2d": [200, 200, 300, 300],
                    "desc": "label",
                    "geometry": {
                        "type": "ExtentPolygon",
                        "coordinates": [[200, 200], [300, 200], [300, 300], [200, 300], [200, 200]]
                    }
                },
                {
                    "bbox_2d": [400, 400, 600, 500],
                    "desc": "fiber",
                    "geometry": {
                        "type": "LineString",
                        "lineMode": 1,
                        "lineType": "LLL",
                        "coordinates": [[400, 400], [500, 450], [600, 500]]
                    }
                }
            ],
            "width": 1000, "height": 1000
        },
        
        # Complex path sample
        {
            "objects": [
                {
                    "bbox_2d": [100, 100, 900, 800],
                    "desc": "fiber",
                    "geometry": {
                        "type": "LineString",
                        "lineMode": 1,
                        "lineType": "LLLLLLLLLL",
                        "coordinates": [[100, 100], [200, 150], [300, 200], [400, 180], [500, 220], 
                                      [600, 300], [700, 400], [800, 500], [850, 650], [900, 800]]
                    }
                }
            ],
            "width": 1000, "height": 1000
        },
        
        # Dense object sample
        {
            "objects": [
                {"bbox_2d": [100, 100, 150, 150], "desc": "connect_point"},
                {"bbox_2d": [200, 200, 250, 250], "desc": "connect_point"},
                {"bbox_2d": [300, 300, 350, 350], "desc": "connect_point"},
                {"bbox_2d": [400, 400, 450, 450], "desc": "connect_point"},
                {"bbox_2d": [500, 500, 600, 600], "desc": "label"},
                {"bbox_2d": [700, 700, 800, 800], "desc": "bbu"},
            ],
            "width": 1000, "height": 1000
        }
    ]
    
    return samples

def test_geometry_bucket_classification():
    """Test the new geometry bucket classification."""
    
    print("Testing geometry bucket classification")
    print("=" * 50)
    
    samples = create_test_samples()
    
    # Create teacher selector
    label_hierarchy = {
        "connect_point": [],
        "label": [],
        "fiber": [],
        "bbu": []
    }
    
    selector = TeacherSelector(label_hierarchy, max_teachers=5, seed=42)
    
    # Test bucket classification for each sample
    for i, sample in enumerate(samples):
        geometry_bucket = selector._get_geometry_types_bucket(sample)
        path_bucket = selector._get_path_complexity_bucket(sample)
        count_bucket = selector._get_object_count_bucket(sample)
        spatial_bucket = selector._get_spatial_bucket(sample)
        size_bucket = selector._get_size_bucket(sample)
        
        print(f"\nSample {i+1}:")
        print(f"  Objects: {len(sample['objects'])}")
        print(f"  Geometry bucket: {geometry_bucket}")
        print(f"  Path bucket: {path_bucket}")
        print(f"  Count bucket: {count_bucket}")
        print(f"  Spatial bucket: {spatial_bucket}")
        print(f"  Size bucket: {size_bucket}")
        
        # Show geometry types present
        geometry_types = set()
        for obj in sample["objects"]:
            if "geometry" in obj:
                geom_type = obj["geometry"].get("type", "unknown")
                geometry_types.add(geom_type)
            else:
                geometry_types.add("bbox")
        print(f"  Geometry types: {sorted(geometry_types)}")

def test_teacher_selection_with_geometry():
    """Test teacher selection with geometry diversity consideration."""
    
    print(f"\n{'='*50}")
    print("Testing teacher selection with geometry diversity")
    
    samples = create_test_samples()
    
    # Create teacher selector
    label_hierarchy = {
        "connect_point": [],
        "label": [],
        "fiber": [],
        "bbu": []
    }
    
    selector = TeacherSelector(label_hierarchy, max_teachers=4, seed=42)
    
    print(f"Input: {len(samples)} samples")
    print(f"Max teachers: {selector.max_teachers}")
    
    # Select teachers
    selected_samples, teacher_indices = selector.select_teachers(samples)
    
    print(f"Selected {len(teacher_indices)} teacher samples")
    
    # Analyze selected teachers (samples are already selected)
    
    print(f"\nSelected teacher analysis:")
    for i, (idx, sample) in enumerate(zip(teacher_indices, selected_samples)):
        geometry_bucket = selector._get_geometry_types_bucket(sample)
        path_bucket = selector._get_path_complexity_bucket(sample)
        
        print(f"  Teacher {i+1} (sample {idx}):")
        print(f"    Objects: {len(sample['objects'])}")
        print(f"    Geometry: {geometry_bucket}")
        print(f"    Path: {path_bucket}")
        
        # Show unique object descriptions
        descriptions = [obj["desc"] for obj in sample["objects"]]
        unique_descriptions = sorted(set(descriptions))
        print(f"    Labels: {unique_descriptions}")

def test_coverage_analysis():
    """Test coverage analysis of selected teachers."""
    
    print(f"\n{'='*50}")
    print("Testing coverage analysis")
    
    samples = create_test_samples()
    
    label_hierarchy = {
        "connect_point": [],
        "label": [],
        "fiber": [],
        "bbu": []
    }
    
    selector = TeacherSelector(label_hierarchy, max_teachers=6, seed=42)
    
    # Compute metadata for all samples
    metadata = selector._compute_sample_metadata(samples)
    
    # Analyze overall coverage needs
    all_geometry_buckets = set(m["geometry_bucket"] for m in metadata)
    all_path_buckets = set(m["path_bucket"] for m in metadata)
    all_count_buckets = set(m["count_bucket"] for m in metadata)
    
    print(f"Total coverage needs:")
    print(f"  Geometry buckets: {sorted(all_geometry_buckets)}")
    print(f"  Path buckets: {sorted(all_path_buckets)}")
    print(f"  Count buckets: {sorted(all_count_buckets)}")
    
    # Select teachers
    selected_samples, teacher_indices = selector.select_teachers(samples)
    selected_metadata = [metadata[i] for i in teacher_indices]
    
    # Analyze coverage achieved
    covered_geometry = set(m["geometry_bucket"] for m in selected_metadata)
    covered_path = set(m["path_bucket"] for m in selected_metadata)
    covered_count = set(m["count_bucket"] for m in selected_metadata)
    
    print(f"\nCoverage achieved by {len(teacher_indices)} teachers:")
    print(f"  Geometry coverage: {len(covered_geometry)}/{len(all_geometry_buckets)} "
          f"({100*len(covered_geometry)/len(all_geometry_buckets):.1f}%)")
    print(f"  Path coverage: {len(covered_path)}/{len(all_path_buckets)} "
          f"({100*len(covered_path)/len(all_path_buckets):.1f}%)")
    print(f"  Count coverage: {len(covered_count)}/{len(all_count_buckets)} "
          f"({100*len(covered_count)/len(all_count_buckets):.1f}%)")
    
    print(f"\nMissing coverage:")
    missing_geometry = all_geometry_buckets - covered_geometry
    missing_path = all_path_buckets - covered_path
    if missing_geometry:
        print(f"  Missing geometry: {sorted(missing_geometry)}")
    if missing_path:
        print(f"  Missing path: {sorted(missing_path)}")

if __name__ == "__main__":
    print("Enhanced TeacherSelector Test Suite")
    print("=" * 50)
    
    test_geometry_bucket_classification()
    test_teacher_selection_with_geometry()
    test_coverage_analysis()
    
    print("\n" + "=" * 50)
    print("Test completed!")