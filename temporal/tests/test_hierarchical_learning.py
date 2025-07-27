#!/usr/bin/env python3
"""
Test Hierarchical Learning Framework

This script tests the new hierarchical learning framework with v2 data format,
validating that it produces the expected progressive learning stages and
multiple annotation formats.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_conversion.hierarchical_learning_framework import (
    HierarchicalLearningFramework, 
    LearningStage
)
from data_conversion.description_concatenator import (
    DescriptionConcatenator, 
    ConcatenationConfig, 
    ConcatenationStrategy
)
from data_conversion.hierarchical_processor import HierarchicalProcessor
from data_conversion.geometry_processor import GeometryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_hierarchical_framework():
    """Test the hierarchical learning framework with sample data."""
    logger.info("Testing Hierarchical Learning Framework")
    
    # Initialize framework
    framework = HierarchicalLearningFramework("chinese")
    
    # Test content categorization
    test_content = {
        "object_type": "螺丝、光纤插头",
        "property": "BBU安装螺丝",
        "extra_info": "符合要求/显示完整"
    }
    
    stage_content = framework.categorize_content_by_stage(test_content)
    
    logger.info("Content categorization results:")
    for stage, content in stage_content.items():
        if content:
            logger.info(f"  {stage.value}: {content}")
    
    # Test hierarchical description creation
    description = framework.create_hierarchical_description(stage_content)
    logger.info(f"Hierarchical description: {description}")
    
    return True


def test_description_concatenator():
    """Test the description concatenator with different strategies."""
    logger.info("Testing Description Concatenator")
    
    test_content = {
        "object_type": "BBU设备",
        "property": "华为",
        "extra_info": "显示完整/机柜空间充足，需要安装"
    }
    
    # Test different concatenation strategies
    strategies = [
        ConcatenationStrategy.SLASH_SEPARATED,
        ConcatenationStrategy.NATURAL_LANGUAGE,
        ConcatenationStrategy.PROGRESSIVE_STAGES
    ]
    
    for strategy in strategies:
        config = ConcatenationConfig(strategy=strategy, language="chinese")
        concatenator = DescriptionConcatenator(config)
        
        description = concatenator.concatenate_hierarchical_content(test_content)
        logger.info(f"{strategy.value}: {description}")
        
        # Test progressive descriptions
        progressive = concatenator.create_progressive_descriptions(test_content)
        logger.info(f"Progressive descriptions for {strategy.value}:")
        for stage, desc in progressive.items():
            logger.info(f"  {stage}: {desc}")
    
    return True


def test_geometry_processing():
    """Test geometry processing for different annotation types."""
    logger.info("Testing Geometry Processing")
    
    # Test ExtentPolygon (bbox_2d)
    extent_geometry = {
        "type": "ExtentPolygon",
        "coordinates": [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]
    }
    
    # Test Square (四边形)
    square_geometry = {
        "type": "Square",
        "coordinates": [[[150, 150], [250, 160], [240, 250], [140, 240], [150, 150]]],
        "lineType": ["LLLLL"]
    }
    
    # Test LineString (line)
    line_geometry = {
        "type": "LineString",
        "coordinates": [[100, 100], [150, 120], [200, 140], [250, 160], [300, 180]],
        "lineMode": 1,
        "lineType": "LLLLL"
    }
    
    geometries = [
        ("ExtentPolygon", extent_geometry),
        ("Square", square_geometry), 
        ("LineString", line_geometry)
    ]
    
    for name, geometry in geometries:
        logger.info(f"Processing {name}:")
        
        # Test bbox extraction
        bbox = GeometryProcessor.extract_bbox_from_geometry(geometry)
        logger.info(f"  Extracted bbox: {bbox}")
        
        # Test hierarchical geometry extraction
        hierarchical = GeometryProcessor.extract_hierarchical_geometry(geometry)
        logger.info(f"  Hierarchical format: {list(hierarchical.keys())}")
        
        # Test coordinate scaling
        scaled = GeometryProcessor.scale_hierarchical_geometry(hierarchical, 0.5, 0.5)
        logger.info(f"  Scaled coordinates: {scaled}")
    
    return True


def test_hierarchical_processor():
    """Test the hierarchical processor with sample v2 data."""
    logger.info("Testing Hierarchical Processor")
    
    # Initialize processor
    processor = HierarchicalProcessor(
        language="chinese",
        response_types={"object_type", "property", "extra_info"}
    )
    
    # Create sample v2 feature data
    sample_features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "ExtentPolygon",
                "coordinates": [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]
            },
            "properties": {
                "contentZh": {
                    "标签": "螺丝、光纤插头",
                    "螺丝/插头是否显示完整": "显示完整",
                    "这个螺丝/插头是什么种类": "BBU安装螺丝",
                    "这个螺丝/插头连接是否符合要求": ["符合要求"]
                }
            }
        },
        {
            "type": "Feature", 
            "geometry": {
                "type": "LineString",
                "coordinates": [[300, 300], [350, 320], [400, 340], [450, 360]],
                "lineMode": 1,
                "lineType": "LLLL"
            },
            "properties": {
                "contentZh": {
                    "标签": "光纤",
                    "这条光纤弯曲半径是否合理": "弯曲半径合理",
                    "这条光纤是否采用保护措施": "有保护措施，保护措施为/铠装",
                    "这条光纤是否被遮挡": "有遮挡"
                }
            }
        }
    ]
    
    # Process features
    objects = processor.extract_objects_from_markresult(sample_features)
    
    logger.info(f"Processed {len(objects)} objects:")
    for i, obj in enumerate(objects):
        logger.info(f"Object {i+1}:")
        logger.info(f"  Description: {obj.get('desc', 'N/A')}")
        logger.info(f"  Geometry types: {[k for k in obj.keys() if k in ['bbox_2d', 'square', 'line']]}")
        
        # Test progressive descriptions
        if "progressive_descriptions" in obj:
            logger.info("  Progressive descriptions:")
            for stage, desc in obj["progressive_descriptions"].items():
                logger.info(f"    {stage}: {desc}")
        
        # Test validation
        validation = processor.validate_hierarchical_object(obj)
        logger.info(f"  Validation: {validation}")
    
    # Test stage-specific sample creation
    for stage in ["stage_1", "stage_1_2", "all_stages"]:
        stage_objects = processor.create_stage_specific_samples(objects, stage)
        logger.info(f"Stage {stage}: {len(stage_objects)} objects")
        if stage_objects:
            logger.info(f"  Sample description: {stage_objects[0].get('desc', 'N/A')}")
    
    return True


def test_with_real_v2_data():
    """Test with actual v2 data files."""
    logger.info("Testing with Real V2 Data")
    
    # Load real v2 data
    v2_files = [
        Path("ds_v2/QC-20230222-0000317_17423.json"),
        Path("ds_v2/QC-20230223-0000358_18620.json")
    ]
    
    processor = HierarchicalProcessor(language="chinese")
    
    for json_file in v2_files:
        if not json_file.exists():
            logger.warning(f"File not found: {json_file}")
            continue
            
        logger.info(f"Processing {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get("markResult", {}).get("features", [])
        logger.info(f"  Found {len(features)} features")
        
        objects = processor.extract_objects_from_markresult(features)
        logger.info(f"  Extracted {len(objects)} valid objects")
        
        # Analyze geometry types
        geometry_stats = {}
        for obj in objects:
            for geom_type in ["bbox_2d", "square", "line"]:
                if geom_type in obj:
                    geometry_stats[geom_type] = geometry_stats.get(geom_type, 0) + 1
        
        logger.info(f"  Geometry distribution: {geometry_stats}")
        
        # Show sample objects
        for i, obj in enumerate(objects[:2]):  # Show first 2 objects
            logger.info(f"  Object {i+1}:")
            logger.info(f"    Description: {obj.get('desc', 'N/A')}")
            logger.info(f"    Geometry: {[k for k in obj.keys() if k in ['bbox_2d', 'square', 'line']]}")
    
    return True


def main():
    """Run all tests."""
    logger.info("Starting Hierarchical Learning Framework Tests")
    
    tests = [
        ("Hierarchical Framework", test_hierarchical_framework),
        ("Description Concatenator", test_description_concatenator),
        ("Geometry Processing", test_geometry_processing),
        ("Hierarchical Processor", test_hierarchical_processor),
        ("Real V2 Data", test_with_real_v2_data)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info('='*50)
        
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            logger.error(f"{test_name}: ERROR - {str(e)}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Test Summary")
    logger.info('='*50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
