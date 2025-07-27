#!/usr/bin/env python3
"""
Simplified test script to validate training sample data structures.
Focuses on data format and coordinate patterns without chat processing.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

def load_samples(file_path: str, limit: int = 100) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"✅ Loaded {len(samples)} samples from {file_path}")
        return samples
    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        return []

def analyze_sample_structure(sample: Dict) -> Dict[str, Any]:
    """Analyze the structure of a single sample."""
    analysis = {
        'has_images': 'images' in sample and len(sample['images']) > 0,
        'has_objects': 'objects' in sample and len(sample['objects']) > 0,
        'image_count': len(sample.get('images', [])),
        'object_count': len(sample.get('objects', [])),
        'width': sample.get('width'),
        'height': sample.get('height'),
        'geometry_types': set(),
        'object_descs': [],
        'coordinate_formats': set(),
        'invalid_objects': [],
        'coordinate_counts': defaultdict(int)
    }
    
    # Analyze objects
    for i, obj in enumerate(sample.get('objects', [])):
        desc = obj.get('desc', 'unknown')
        analysis['object_descs'].append(desc)
        
        geometry_found = False
        # Check geometry types
        for key in obj.keys():
            if key != 'desc':
                analysis['geometry_types'].add(key)
                geometry_found = True
                
                # Check coordinate format
                coords = obj[key]
                if isinstance(coords, list):
                    coord_count = len(coords)
                    analysis['coordinate_counts'][key] = analysis['coordinate_counts'].get(key, 0) + 1
                    
                    if key == 'bbox_2d' and coord_count == 4:
                        analysis['coordinate_formats'].add('bbox_2d_standard')
                        # Validate bbox format: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = coords
                        if x1 >= x2 or y1 >= y2:
                            analysis['invalid_objects'].append(f"Object {i}: Invalid bbox coordinates {coords}")
                    elif key == 'square' and coord_count == 8:
                        analysis['coordinate_formats'].add('square_8points')
                        # Square should be [x1,y1,x2,y2,x3,y3,x4,y4]
                    elif key == 'line' and coord_count >= 4:
                        analysis['coordinate_formats'].add('line_multi_point')
                        # Line should have even number of coordinates (x,y pairs)
                        if coord_count % 2 != 0:
                            analysis['invalid_objects'].append(f"Object {i}: Line has odd number of coordinates {coord_count}")
                    else:
                        analysis['invalid_objects'].append(f"Object {i}: Unexpected coordinate count for {key}: {coord_count}")
                else:
                    analysis['invalid_objects'].append(f"Object {i}: Non-list coordinates for {key}")
        
        if not geometry_found:
            analysis['invalid_objects'].append(f"Object {i}: No geometry data found")
    
    return analysis

def validate_image_paths(sample: Dict, base_path: str = "data/ds_v2_full") -> Dict[str, Any]:
    """Validate that image paths exist."""
    validation = {
        'valid_images': [],
        'missing_images': [],
        'image_sizes': []
    }
    
    for img_path in sample.get('images', []):
        full_path = os.path.join(base_path, img_path)
        if os.path.exists(full_path):
            validation['valid_images'].append(img_path)
            try:
                # Get image size without loading full image
                from PIL import Image
                with Image.open(full_path) as img:
                    validation['image_sizes'].append(img.size)
            except:
                validation['image_sizes'].append((0, 0))
        else:
            validation['missing_images'].append(img_path)
    
    return validation

def check_coordinate_bounds(sample: Dict) -> Dict[str, Any]:
    """Check if coordinates are within image bounds."""
    bounds_check = {
        'out_of_bounds_objects': [],
        'valid_objects': 0,
        'coordinate_ranges': {'x_min': float('inf'), 'x_max': 0, 'y_min': float('inf'), 'y_max': 0}
    }
    
    img_width = sample.get('width', 0)
    img_height = sample.get('height', 0)
    
    if img_width == 0 or img_height == 0:
        bounds_check['out_of_bounds_objects'].append("Missing image dimensions")
        return bounds_check
    
    for i, obj in enumerate(sample.get('objects', [])):
        for geom_type, coords in obj.items():
            if geom_type == 'desc':
                continue
            
            if isinstance(coords, list):
                # Extract all x, y coordinates
                if geom_type == 'bbox_2d' and len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    x_coords = [x1, x2]
                    y_coords = [y1, y2]
                elif geom_type in ['square', 'line'] and len(coords) >= 4:
                    x_coords = coords[::2]  # Every even index is x
                    y_coords = coords[1::2]  # Every odd index is y
                else:
                    continue
                
                # Update ranges
                bounds_check['coordinate_ranges']['x_min'] = min(bounds_check['coordinate_ranges']['x_min'], min(x_coords))
                bounds_check['coordinate_ranges']['x_max'] = max(bounds_check['coordinate_ranges']['x_max'], max(x_coords))
                bounds_check['coordinate_ranges']['y_min'] = min(bounds_check['coordinate_ranges']['y_min'], min(y_coords))
                bounds_check['coordinate_ranges']['y_max'] = max(bounds_check['coordinate_ranges']['y_max'], max(y_coords))
                
                # Check bounds
                out_of_bounds = False
                for x in x_coords:
                    if x < 0 or x > img_width:
                        out_of_bounds = True
                        break
                for y in y_coords:
                    if y < 0 or y > img_height:
                        out_of_bounds = True
                        break
                
                if out_of_bounds:
                    bounds_check['out_of_bounds_objects'].append(f"Object {i} ({geom_type}): coordinates exceed image bounds")
                else:
                    bounds_check['valid_objects'] += 1
    
    return bounds_check

def main():
    """Main validation function."""
    print("🔍 BBU Training Data Structure Validation")
    print("=" * 50)
    
    # Check if data file exists
    data_file = "data/ds_v2_full/all_samples.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📂 Using data file: {data_file}")
    
    # Load and analyze samples
    samples = load_samples(data_file, limit=50)  # Test with first 50 samples
    
    if not samples:
        print("❌ No samples loaded. Exiting.")
        return
    
    print("\n📊 Detailed Analysis Results:")
    print("-" * 30)
    
    # Global statistics
    all_geometry_types = set()
    all_coordinate_formats = set()
    total_invalid_objects = 0
    total_missing_images = 0
    coordinate_type_counts = defaultdict(int)
    desc_patterns = defaultdict(int)
    
    for i, sample in enumerate(samples[:10]):  # Detailed analysis for first 10
        print(f"\n🔍 Sample {i+1}:")
        
        # Analyze structure
        analysis = analyze_sample_structure(sample)
        all_geometry_types.update(analysis['geometry_types'])
        all_coordinate_formats.update(analysis['coordinate_formats'])
        total_invalid_objects += len(analysis['invalid_objects'])
        
        for geom_type, count in analysis['coordinate_counts'].items():
            coordinate_type_counts[geom_type] += count
        
        for desc in analysis['object_descs']:
            # Extract first part of description (before /)
            desc_category = desc.split('/')[0] if '/' in desc else desc
            desc_patterns[desc_category] += 1
        
        print(f"  📷 Images: {analysis['image_count']}")
        print(f"  🎯 Objects: {analysis['object_count']}")
        print(f"  📐 Geometry types: {analysis['geometry_types']}")
        print(f"  📏 Coordinate formats: {analysis['coordinate_formats']}")
        
        if analysis['invalid_objects']:
            print(f"  ❌ Invalid objects: {len(analysis['invalid_objects'])}")  
            for invalid in analysis['invalid_objects'][:3]:  # Show first 3
                print(f"     - {invalid}")
        
        # Validate images
        img_validation = validate_image_paths(sample)
        total_missing_images += len(img_validation['missing_images'])
        if img_validation['missing_images']:
            print(f"  ❌ Missing images: {img_validation['missing_images']}")
        
        # Check coordinate bounds
        bounds_check = check_coordinate_bounds(sample)
        if bounds_check['out_of_bounds_objects']:
            print(f"  ⚠️ Out of bounds objects: {len(bounds_check['out_of_bounds_objects'])}")
            for oob in bounds_check['out_of_bounds_objects'][:2]:  # Show first 2
                print(f"     - {oob}")
        
        if i < 5:  # Show coordinate ranges for first 5 samples
            ranges = bounds_check['coordinate_ranges']
            if ranges['x_min'] != float('inf'):
                print(f"  📏 Coordinate ranges: x=[{ranges['x_min']:.0f}, {ranges['x_max']:.0f}], y=[{ranges['y_min']:.0f}, {ranges['y_max']:.0f}]")
    
    # Summary analysis
    print("\n📋 Global Summary:")
    print("-" * 20)
    
    print(f"✅ Total samples analyzed: {len(samples)}")
    print(f"📐 All geometry types: {sorted(all_geometry_types)}")
    print(f"📏 All coordinate formats: {sorted(all_coordinate_formats)}")
    print(f"❌ Total invalid objects: {total_invalid_objects}")
    print(f"🖼️ Total missing images: {total_missing_images}")
    
    print(f"\n🎯 Coordinate type distribution:")
    for coord_type, count in sorted(coordinate_type_counts.items()):
        print(f"   {coord_type}: {count}")
    
    print(f"\n📝 Top object categories:")
    for desc_cat, count in sorted(desc_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {desc_cat}: {count}")
    
    # Check for specific issues that might cause training problems
    print(f"\n🚨 Potential Training Issues:")
    issues_found = []
    
    if 'bbox_2d' not in all_geometry_types and 'square' not in all_geometry_types:
        issues_found.append("No standard bounding box formats found")
    
    if total_invalid_objects > len(samples) * 0.1:  # More than 10% invalid
        issues_found.append(f"High rate of invalid objects: {total_invalid_objects}/{len(samples)*5} (avg 5 obj/sample)")
    
    if total_missing_images > 0:
        issues_found.append(f"Missing image files: {total_missing_images}")
    
    # Check for consistent coordinate formats
    mixed_formats = len(all_coordinate_formats) > 3
    if mixed_formats:
        issues_found.append(f"Too many coordinate formats may confuse coordinate token generation: {all_coordinate_formats}")
    
    if issues_found:
        for issue in issues_found:
            print(f"   ❌ {issue}")
    else:
        print(f"   ✅ No major structural issues detected")
    
    print(f"\n🏁 Structure validation complete.")
    print(f"    Next: Test coordinate token generation with chat processor")

if __name__ == "__main__":
    main()