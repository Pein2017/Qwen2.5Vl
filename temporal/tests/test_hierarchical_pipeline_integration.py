#!/usr/bin/env python3
"""
Integration Tests for Hierarchical Learning Pipeline

This test suite validates the complete integration of the hierarchical learning
framework with the existing data conversion pipeline, ensuring backward
compatibility and new functionality work correctly.
"""

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_conversion.config import DataConversionConfig
from data_conversion.unified_processor import UnifiedProcessor
from data_conversion.hierarchical_processor import HierarchicalProcessor
from data_conversion.utils.file_ops import FileOperations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class HierarchicalPipelineIntegrationTest:
    """Integration test suite for hierarchical learning pipeline."""
    
    def __init__(self):
        self.temp_dir = None
        self.test_data_dir = None
        self.output_dir = None
        
    def setup(self):
        """Set up test environment."""
        logger.info("Setting up test environment...")
        
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_dir = self.temp_dir / "test_data"
        self.output_dir = self.temp_dir / "output"
        
        self.test_data_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create test data
        self._create_test_data()
        
        logger.info(f"Test environment created at {self.temp_dir}")
    
    def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")
    
    def _create_test_data(self):
        """Create synthetic test data in v2 format."""
        # Test data samples with different geometry types and hierarchical content
        test_samples = [
            {
                "filename": "test_bbox.json",
                "data": {
                    "info": {"depth": 3, "width": 1000, "height": 800},
                    "markResult": {
                        "features": [
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
                            }
                        ]
                    }
                }
            },
            {
                "filename": "test_square.json",
                "data": {
                    "info": {"depth": 3, "width": 1000, "height": 800},
                    "markResult": {
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Square",
                                    "coordinates": [[[150, 150], [250, 160], [240, 250], [140, 240], [150, 150]]],
                                    "lineType": ["LLLLL"]
                                },
                                "properties": {
                                    "contentZh": {
                                        "标签": "BBU设备",
                                        "这个BBU设备是什么品牌": "华为",
                                        "这个BBU设备是否显示完整": "显示完整",
                                        "这个BBU设备是否需要安装挡风板，结合图片中机柜空间状态判断": "机柜空间充足，需要安装/这个BBU设备按要求配备了挡风板"
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            {
                "filename": "test_line.json", 
                "data": {
                    "info": {"depth": 3, "width": 1000, "height": 800},
                    "markResult": {
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": [[300, 300], [350, 320], [400, 340], [450, 360], [500, 380]],
                                    "lineMode": 1,
                                    "lineType": "LLLLL"
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
                    }
                }
            },
            {
                "filename": "test_label_ocr.json",
                "data": {
                    "info": {"depth": 3, "width": 1000, "height": 800},
                    "markResult": {
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Square",
                                    "coordinates": [[[200, 200], [300, 200], [300, 250], [200, 250], [200, 200]]],
                                    "lineType": ["LLLLL"]
                                },
                                "properties": {
                                    "contentZh": {
                                        "标签": "标签",
                                        "请输入标签上的文字内容": "5G-BBU-传输光纤"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        ]
        
        # Create JSON files
        for sample in test_samples:
            json_file = self.test_data_dir / sample["filename"]
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(sample["data"], f, ensure_ascii=False, indent=2)
        
        # Create dummy image files
        for sample in test_samples:
            image_file = self.test_data_dir / sample["filename"].replace(".json", ".jpg")
            # Create a minimal dummy image file
            image_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF')  # Minimal JPEG header
        
        logger.info(f"Created {len(test_samples)} test samples")
    
    def test_backward_compatibility(self):
        """Test that the new system maintains backward compatibility."""
        logger.info("Testing backward compatibility...")
        
        # Create config using old-style parameters
        config = DataConversionConfig(
            input_dir=str(self.test_data_dir),
            output_dir=str(self.output_dir / "backward_compat"),
            language="chinese",
            response_types=["object_type", "property", "extra_info"],
            val_ratio=0.2,
            max_teachers=2,
            seed=42
        )
        
        # Process using unified processor
        processor = UnifiedProcessor(config)
        
        # Process all samples
        samples = []
        for json_file in self.test_data_dir.glob("*.json"):
            try:
                sample = processor.process_single_sample(json_file)
                if sample:
                    samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
                return False
        
        # Validate results
        if not samples:
            logger.error("No samples processed")
            return False
        
        # Check that all samples have required fields
        for sample in samples:
            if not all(key in sample for key in ["objects", "width", "height"]):
                logger.error(f"Sample missing required fields: {sample.keys()}")
                return False
            
            # Check objects have required fields
            for obj in sample["objects"]:
                if "desc" not in obj:
                    logger.error(f"Object missing description: {obj.keys()}")
                    return False
                
                # Should have at least bbox_2d
                if "bbox_2d" not in obj:
                    logger.error(f"Object missing bbox_2d: {obj.keys()}")
                    return False
        
        logger.info(f"Backward compatibility test passed: {len(samples)} samples processed")
        return True
    
    def test_hierarchical_features(self):
        """Test new hierarchical learning features."""
        logger.info("Testing hierarchical learning features...")
        
        # Create config with hierarchical features enabled
        config = DataConversionConfig(
            input_dir=str(self.test_data_dir),
            output_dir=str(self.output_dir / "hierarchical"),
            language="chinese",
            response_types=["object_type", "property", "extra_info"],
            val_ratio=0.2,
            max_teachers=2,
            seed=42,
            hierarchy_path="data_conversion/label_hierarchy_v2.json"
        )
        
        # Process using unified processor
        processor = UnifiedProcessor(config)
        
        # Process all samples
        all_objects = []
        for json_file in self.test_data_dir.glob("*.json"):
            try:
                sample = processor.process_single_sample(json_file)
                if sample and sample.get("objects"):
                    all_objects.extend(sample["objects"])
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
                return False
        
        if not all_objects:
            logger.error("No objects processed")
            return False
        
        # Test hierarchical features
        hierarchical_features_found = {
            "progressive_descriptions": False,
            "multiple_geometries": False,
            "learning_stage_content": False,
            "square_geometry": False,
            "line_geometry": False
        }
        
        for obj in all_objects:
            # Check for progressive descriptions
            if "progressive_descriptions" in obj:
                hierarchical_features_found["progressive_descriptions"] = True
                
                # Validate progressive descriptions structure
                prog_desc = obj["progressive_descriptions"]
                expected_stages = ["stage_1", "stage_1_2", "stage_1_2_3", "all_stages"]
                for stage in expected_stages:
                    if stage not in prog_desc:
                        logger.error(f"Missing progressive description stage: {stage}")
                        return False
            
            # Check for learning stage content
            if "learning_stage_content" in obj:
                hierarchical_features_found["learning_stage_content"] = True
            
            # Check for multiple geometry types
            geometry_types = [k for k in obj.keys() if k in ["bbox_2d", "square", "line"]]
            if len(geometry_types) > 1:
                hierarchical_features_found["multiple_geometries"] = True
            
            # Check for specific geometry types
            if "square" in obj:
                hierarchical_features_found["square_geometry"] = True
                # Validate square coordinates
                square_coords = obj["square"]
                if not isinstance(square_coords, list) or len(square_coords) != 8:
                    logger.error(f"Invalid square coordinates: {square_coords}")
                    return False
            
            if "line" in obj:
                hierarchical_features_found["line_geometry"] = True
                # Validate line coordinates
                line_coords = obj["line"]
                if not isinstance(line_coords, list) or len(line_coords) < 4 or len(line_coords) % 2 != 0:
                    logger.error(f"Invalid line coordinates: {line_coords}")
                    return False
        
        # Check that we found the expected hierarchical features
        missing_features = [k for k, v in hierarchical_features_found.items() if not v]
        if missing_features:
            logger.warning(f"Some hierarchical features not found: {missing_features}")
        
        logger.info(f"Hierarchical features test passed: {len(all_objects)} objects with features {hierarchical_features_found}")
        return True
    
    def test_stage_specific_datasets(self):
        """Test creation of stage-specific datasets."""
        logger.info("Testing stage-specific dataset creation...")
        
        # Create config
        config = DataConversionConfig(
            input_dir=str(self.test_data_dir),
            output_dir=str(self.output_dir / "stages"),
            language="chinese",
            response_types=["object_type", "property", "extra_info"],
            val_ratio=0.2,
            max_teachers=2,
            seed=42
        )
        
        # Process using unified processor
        processor = UnifiedProcessor(config)
        
        # Collect all objects
        all_objects = []
        for json_file in self.test_data_dir.glob("*.json"):
            try:
                sample = processor.process_single_sample(json_file)
                if sample and sample.get("objects"):
                    all_objects.extend(sample["objects"])
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
                return False
        
        if not all_objects:
            logger.error("No objects processed")
            return False
        
        # Create hierarchical datasets
        try:
            stage_files = processor.create_hierarchical_datasets(
                all_objects, 
                self.output_dir / "stages"
            )
        except Exception as e:
            logger.error(f"Failed to create hierarchical datasets: {e}")
            return False
        
        # Validate stage files were created
        expected_stages = ["stage_1", "stage_1_2", "stage_1_2_3", "all_stages"]
        for stage in expected_stages:
            if stage not in stage_files:
                logger.error(f"Missing stage files for: {stage}")
                return False
            
            stage_info = stage_files[stage]
            for file_type in ["train", "val", "teacher"]:
                if file_type not in stage_info:
                    logger.error(f"Missing {file_type} file for stage {stage}")
                    return False
                
                file_path = Path(stage_info[file_type])
                if not file_path.exists():
                    logger.error(f"Stage file does not exist: {file_path}")
                    return False
        
        # Validate stage-specific content
        for stage in expected_stages:
            train_file = Path(stage_files[stage]["train"])
            
            # Load and check content
            with open(train_file, 'r', encoding='utf-8') as f:
                stage_samples = [json.loads(line) for line in f if line.strip()]
            
            if not stage_samples:
                logger.warning(f"No samples in stage {stage} train file")
                continue
            
            # Check that descriptions are stage-appropriate
            sample_obj = stage_samples[0]
            desc = sample_obj.get("desc", "")
            
            if stage == "stage_1":
                # Should only have object type
                if "/" in desc:
                    logger.error(f"Stage 1 should not have '/' in description: {desc}")
                    return False
            elif stage == "stage_1_2":
                # Should have object type and properties
                parts = desc.split("/")
                if len(parts) > 2:
                    logger.error(f"Stage 1_2 should have at most 2 parts: {desc}")
                    return False
        
        logger.info(f"Stage-specific datasets test passed: {len(expected_stages)} stages created")
        return True
    
    def test_geometry_scaling(self):
        """Test coordinate scaling for different geometry types."""
        logger.info("Testing geometry coordinate scaling...")
        
        # Create hierarchical processor
        processor = HierarchicalProcessor(language="chinese")
        
        # Test scaling for different geometry types
        test_objects = [
            {
                "bbox_2d": [100, 100, 200, 200],
                "desc": "test bbox"
            },
            {
                "square": [100, 100, 200, 100, 200, 200, 100, 200],
                "bbox_2d": [100, 100, 200, 200],
                "desc": "test square"
            },
            {
                "line": [100, 100, 150, 120, 200, 140, 250, 160],
                "bbox_2d": [100, 100, 250, 160],
                "desc": "test line"
            }
        ]
        
        scale_x, scale_y = 0.5, 0.8
        
        for obj in test_objects:
            scaled_obj = processor.scale_object_coordinates(obj, scale_x, scale_y)
            
            # Validate bbox scaling
            if "bbox_2d" in obj:
                original_bbox = obj["bbox_2d"]
                scaled_bbox = scaled_obj["bbox_2d"]
                
                expected_bbox = [
                    original_bbox[0] * scale_x,
                    original_bbox[1] * scale_y,
                    original_bbox[2] * scale_x,
                    original_bbox[3] * scale_y
                ]
                
                if scaled_bbox != expected_bbox:
                    logger.error(f"Bbox scaling failed: {scaled_bbox} != {expected_bbox}")
                    return False
            
            # Validate square scaling
            if "square" in obj:
                original_square = obj["square"]
                scaled_square = scaled_obj["square"]
                
                if len(scaled_square) != 8:
                    logger.error(f"Scaled square should have 8 coordinates: {len(scaled_square)}")
                    return False
                
                # Check that coordinates are properly scaled
                for i in range(0, 8, 2):
                    expected_x = original_square[i] * scale_x
                    expected_y = original_square[i+1] * scale_y
                    
                    if abs(scaled_square[i] - expected_x) > 0.001:
                        logger.error(f"Square X scaling failed: {scaled_square[i]} != {expected_x}")
                        return False
                    
                    if abs(scaled_square[i+1] - expected_y) > 0.001:
                        logger.error(f"Square Y scaling failed: {scaled_square[i+1]} != {expected_y}")
                        return False
            
            # Validate line scaling
            if "line" in obj:
                original_line = obj["line"]
                scaled_line = scaled_obj["line"]
                
                if len(scaled_line) != len(original_line):
                    logger.error(f"Scaled line length mismatch: {len(scaled_line)} != {len(original_line)}")
                    return False
                
                # Check that coordinates are properly scaled
                for i in range(0, len(original_line), 2):
                    expected_x = original_line[i] * scale_x
                    expected_y = original_line[i+1] * scale_y
                    
                    if abs(scaled_line[i] - expected_x) > 0.001:
                        logger.error(f"Line X scaling failed: {scaled_line[i]} != {expected_x}")
                        return False
                    
                    if abs(scaled_line[i+1] - expected_y) > 0.001:
                        logger.error(f"Line Y scaling failed: {scaled_line[i+1]} != {expected_y}")
                        return False
        
        logger.info("Geometry scaling test passed")
        return True
    
    def run_all_tests(self):
        """Run all integration tests."""
        logger.info("Starting hierarchical pipeline integration tests...")
        
        tests = [
            ("Backward Compatibility", self.test_backward_compatibility),
            ("Hierarchical Features", self.test_hierarchical_features),
            ("Stage-Specific Datasets", self.test_stage_specific_datasets),
            ("Geometry Scaling", self.test_geometry_scaling)
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
        
        return results


def main():
    """Run integration tests."""
    test_suite = HierarchicalPipelineIntegrationTest()
    
    try:
        # Setup test environment
        test_suite.setup()
        
        # Run all tests
        results = test_suite.run_all_tests()
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info("Integration Test Summary")
        logger.info('='*50)
        
        for test_name, result in results.items():
            logger.info(f"{test_name}: {result}")
        
        passed = sum(1 for r in results.values() if r == "PASS")
        total = len(results)
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        success = passed == total
        
    finally:
        # Cleanup
        test_suite.teardown()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
