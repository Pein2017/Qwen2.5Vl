#!/usr/bin/env python3
"""
Demonstration of Hierarchical Learning with V2 Data

This script demonstrates the complete hierarchical learning pipeline
using actual v2 data files, showing how the system processes different
annotation types and creates progressive learning stages.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_conversion.hierarchical_processor import HierarchicalProcessor
from data_conversion.description_concatenator import (
    DescriptionConcatenator, 
    ConcatenationConfig, 
    ConcatenationStrategy
)
from data_conversion.geometry_processor import GeometryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_v2_processing():
    """Demonstrate processing of v2 data format."""
    logger.info("Demonstrating V2 Data Processing")
    
    # Load actual v2 data files
    v2_files = [
        Path("/data3/Qwen2.5-VL-main/ds_v2/QC-20230222-0000317_17423.json"),
        Path("/data3/Qwen2.5-VL-main/ds_v2/QC-20230223-0000358_18620.json")
    ]
    
    # Initialize hierarchical processor
    processor = HierarchicalProcessor(
        language="chinese",
        response_types={"object_type", "property", "extra_info"}
    )
    
    print("\n" + "="*80)
    print("V2 DATA PROCESSING DEMONSTRATION")
    print("="*80)
    
    for json_file in v2_files:
        if not json_file.exists():
            logger.warning(f"File not found: {json_file}")
            continue
            
        print(f"\nProcessing: {json_file.name}")
        print("-" * 60)
        
        # Load data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract features
        features = data.get("markResult", {}).get("features", [])
        print(f"Found {len(features)} features")
        
        # Process with hierarchical processor
        objects = processor.extract_objects_from_markresult(features)
        print(f"Extracted {len(objects)} valid objects")
        
        # Analyze geometry distribution
        geometry_stats = {}
        for obj in objects:
            for geom_type in ["bbox_2d", "square", "line"]:
                if geom_type in obj:
                    geometry_stats[geom_type] = geometry_stats.get(geom_type, 0) + 1
        
        print(f"Geometry distribution: {geometry_stats}")
        
        # Show detailed examples
        for i, obj in enumerate(objects[:3]):  # Show first 3 objects
            print(f"\n  Object {i+1}:")
            print(f"    Geometry types: {[k for k in obj.keys() if k in ['bbox_2d', 'square', 'line']]}")
            print(f"    Description: {obj.get('desc', 'N/A')}")
            
            # Show progressive descriptions
            if "progressive_descriptions" in obj:
                print("    Progressive descriptions:")
                for stage, desc in obj["progressive_descriptions"].items():
                    print(f"      {stage}: {desc}")
            
            # Show learning stage content
            if "learning_stage_content" in obj:
                print("    Learning stage content:")
                for stage, content in obj["learning_stage_content"].items():
                    if content:
                        print(f"      {stage.value}: {content}")


def demonstrate_concatenation_strategies():
    """Demonstrate different concatenation strategies."""
    logger.info("Demonstrating Concatenation Strategies")
    
    # Sample content from v2 data
    sample_content = {
        "object_type": "BBU设备",
        "property": "华为",
        "extra_info": "显示完整/机柜空间充足，需要安装/这个BBU设备按要求配备了挡风板"
    }
    
    print("\n" + "="*80)
    print("CONCATENATION STRATEGIES DEMONSTRATION")
    print("="*80)
    
    strategies = [
        (ConcatenationStrategy.SLASH_SEPARATED, "Slash-Separated (Default)"),
        (ConcatenationStrategy.NATURAL_LANGUAGE, "Natural Language"),
        (ConcatenationStrategy.STRUCTURED_JSON, "Structured JSON"),
        (ConcatenationStrategy.PROGRESSIVE_STAGES, "Progressive Stages")
    ]
    
    for strategy, name in strategies:
        print(f"\n{name}:")
        print("-" * 40)
        
        config = ConcatenationConfig(strategy=strategy, language="chinese")
        concatenator = DescriptionConcatenator(config)
        
        # Show full description
        description = concatenator.concatenate_hierarchical_content(sample_content)
        print(f"Full: {description}")
        
        # Show progressive descriptions
        if strategy == ConcatenationStrategy.SLASH_SEPARATED:
            progressive = concatenator.create_progressive_descriptions(sample_content)
            for stage, desc in progressive.items():
                print(f"  {stage}: {desc}")


def demonstrate_geometry_processing():
    """Demonstrate geometry processing for different annotation types."""
    logger.info("Demonstrating Geometry Processing")
    
    print("\n" + "="*80)
    print("GEOMETRY PROCESSING DEMONSTRATION")
    print("="*80)
    
    # Sample geometries from v2 data
    geometries = [
        {
            "name": "ExtentPolygon (bbox_2d)",
            "geometry": {
                "type": "ExtentPolygon",
                "coordinates": [[862.1434676434676, 460.89316239316236], [894.3656898656898, 460.89316239316236], [894.3656898656898, 488.6709401709401], [862.1434676434676, 488.6709401709401], [862.1434676434676, 460.89316239316236]]
            }
        },
        {
            "name": "Square (四边形)",
            "geometry": {
                "type": "Square",
                "coordinates": [[[726.5189769380947, 470.6788130170485], [636.5189769380947, 507.1494012523426], [605.3425063498595, 479.5023424288132], [714.1660357616242, 438.91410713469554], [726.5189769380947, 470.6788130170485]]],
                "lineType": ["LLLLL"]
            }
        },
        {
            "name": "LineString (line)",
            "geometry": {
                "type": "LineString",
                "coordinates": [[707.7503873405584, 1368.2228462971502], [709.9726095627807, 1432.6672907415948], [758.8614984516696, 1478.2228462971502], [849.9726095627807, 1469.3339574082615], [1042.1948317850029, 1212.6672907415948]],
                "lineMode": 1,
                "lineType": "LLLLL"
            }
        }
    ]
    
    for geom_info in geometries:
        print(f"\n{geom_info['name']}:")
        print("-" * 40)
        
        geometry = geom_info["geometry"]
        
        # Extract bbox
        bbox = GeometryProcessor.extract_bbox_from_geometry(geometry)
        print(f"Extracted bbox: {bbox}")
        
        # Extract hierarchical geometry
        hierarchical = GeometryProcessor.extract_hierarchical_geometry(geometry)
        print(f"Hierarchical formats: {list(hierarchical.keys())}")
        
        # Show coordinates for each format
        for format_type, coords in hierarchical.items():
            if format_type != "geometry" and isinstance(coords, list):
                print(f"  {format_type}: {coords[:8]}{'...' if len(coords) > 8 else ''}")
        
        # Test scaling
        scaled = GeometryProcessor.scale_hierarchical_geometry(hierarchical, 0.5, 0.5)
        print(f"Scaled (0.5x): {list(scaled.keys())}")


def demonstrate_progressive_training():
    """Demonstrate how progressive training datasets would be created."""
    logger.info("Demonstrating Progressive Training")
    
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING DEMONSTRATION")
    print("="*80)
    
    # Load a sample from v2 data
    json_file = Path("/data3/Qwen2.5-VL-main/ds_v2/QC-20230222-0000317_17423.json")
    
    if not json_file.exists():
        print("Sample data file not found")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process with hierarchical processor
    processor = HierarchicalProcessor(language="chinese")
    features = data.get("markResult", {}).get("features", [])
    objects = processor.extract_objects_from_markresult(features)
    
    if not objects:
        print("No objects found in sample data")
        return
    
    # Show how stage-specific datasets would look
    stages = ["stage_1", "stage_1_2", "stage_1_2_3", "all_stages"]
    
    for stage in stages:
        print(f"\n{stage.upper()}:")
        print("-" * 40)
        
        stage_objects = processor.create_stage_specific_samples(objects, stage)
        
        print(f"Objects: {len(stage_objects)}")
        
        # Show first object as example
        if stage_objects:
            sample_obj = stage_objects[0]
            print(f"Sample description: {sample_obj.get('desc', 'N/A')}")
            print(f"Geometry types: {[k for k in sample_obj.keys() if k in ['bbox_2d', 'square', 'line']]}")


def demonstrate_validation():
    """Demonstrate validation functionality."""
    logger.info("Demonstrating Validation")
    
    print("\n" + "="*80)
    print("VALIDATION DEMONSTRATION")
    print("="*80)
    
    # Create test objects with different validation scenarios
    test_objects = [
        {
            "name": "Valid bbox object",
            "obj": {
                "bbox_2d": [100, 100, 200, 200],
                "desc": "螺丝、光纤插头/BBU安装螺丝/符合要求",
                "progressive_descriptions": {"stage_1": "螺丝、光纤插头"},
                "content_fields": {"object_type": "螺丝、光纤插头"}
            }
        },
        {
            "name": "Valid square object",
            "obj": {
                "square": [100, 100, 200, 100, 200, 200, 100, 200],
                "bbox_2d": [100, 100, 200, 200],
                "desc": "BBU设备/华为",
                "progressive_descriptions": {"stage_1": "BBU设备"}
            }
        },
        {
            "name": "Invalid object (missing description)",
            "obj": {
                "bbox_2d": [100, 100, 200, 200],
                "desc": ""
            }
        },
        {
            "name": "Invalid object (bad coordinates)",
            "obj": {
                "bbox_2d": [100, 100, 200],  # Missing coordinate
                "desc": "test"
            }
        }
    ]
    
    processor = HierarchicalProcessor(language="chinese")
    
    for test_case in test_objects:
        print(f"\n{test_case['name']}:")
        print("-" * 40)
        
        validation = processor.validate_hierarchical_object(test_case["obj"])
        
        for check, result in validation.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}: {result}")


def main():
    """Run all demonstrations."""
    print("HIERARCHICAL LEARNING FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows the complete hierarchical learning pipeline")
    print("using actual v2 data format with multiple annotation types and")
    print("progressive learning stages.")
    print("=" * 80)
    
    demonstrations = [
        ("V2 Data Processing", demonstrate_v2_processing),
        ("Concatenation Strategies", demonstrate_concatenation_strategies),
        ("Geometry Processing", demonstrate_geometry_processing),
        ("Progressive Training", demonstrate_progressive_training),
        ("Validation", demonstrate_validation)
    ]
    
    for demo_name, demo_func in demonstrations:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"Error in {demo_name}: {e}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("The hierarchical learning framework is ready for use!")
    print("Next steps:")
    print("1. Use data_conversion/hierarchical_example.py to create datasets")
    print("2. Train models progressively using stage-specific datasets")
    print("3. Evaluate performance at each learning stage")
    print("4. Fine-tune based on validation results")


if __name__ == "__main__":
    main()
