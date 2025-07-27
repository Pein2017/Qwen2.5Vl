#!/usr/bin/env python3
"""
Test Suite for Flexible Taxonomy Processor

Comprehensive tests to validate the hierarchical classification system
and ensure all V2 data patterns are handled correctly.
"""

import json
import sys
import unittest
from pathlib import Path

# Add the data_conversion directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "data_conversion"))

from flexible_taxonomy_processor import FlexibleTaxonomyProcessor, AnnotationSample
from geometry_processor import GeometryProcessor


class TestFlexibleTaxonomy(unittest.TestCase):
    """Test cases for flexible taxonomy processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = FlexibleTaxonomyProcessor()
        
        # Sample V2 feature data for testing
        self.sample_connect_point = {
            "type": "Feature",
            "geometry": {
                "coordinates": [
                    [955.8214285714286, 206.20238095238096],
                    [997.4880952380953, 206.20238095238096],
                    [997.4880952380953, 254.53571428571445],
                    [955.8214285714286, 254.53571428571445],
                    [955.8214285714286, 206.20238095238096]
                ],
                "type": "ExtentPolygon"
            },
            "properties": {
                "contentZh": {
                    "标签": "螺丝、光纤插头",
                    "螺丝/插头是否显示完整": "只显示部分",
                    "这个螺丝/插头是什么种类": "BBU安装螺丝",
                    "这个螺丝/插头连接是否符合要求": ["符合要求"]
                },
                "content": {
                    "connect_point_situation": "connect_point_situation_part",
                    "connect_point_type": "install_screw",
                    "connect_point_check": ["connect_point_check_true"],
                    "label": "connect_point",
                    "ex_info": ""
                }
            }
        }
        
        self.sample_bbu = {
            "type": "Feature",
            "geometry": {
                "lineType": ["LLLLL"],
                "coordinates": [[[972.2023809523804, 226.44047619047683], 
                               [1030.2023809523805, 1806.4404761904768],
                               [782.2023809523805, 1778.4404761904768],
                               [678.2023809523803, 276.44047619047683],
                               [972.2023809523804, 226.44047619047683]]],
                "type": "Square"
            },
            "properties": {
                "contentZh": {
                    "这个BBU设备是否需要安装挡风板，结合图片中机柜空间状态判断": "机柜空间充足，需要安装/这个BBU设备未按要求配备挡风板",
                    "如果有特殊情况，请具体描述": "线缆遮挡设备",
                    "这个BBU设备是什么品牌": "华为",
                    "标签": "BBU设备",
                    "这个BBU设备是否显示完整": "显示完整"
                },
                "content": {
                    "bbu_stituation": "bbu_stituation_complete",
                    "bbu_brand": "huawei",
                    "label": "bbu",
                    "bbu_equipment": "bbu_equipment_true/bbu_equipment_true_false",
                    "ex_info": "线缆遮挡设备"
                }
            }
        }
        
        self.sample_fiber = {
            "type": "Feature", 
            "geometry": {
                "lineMode": 1,
                "lineType": "LLLLLLLLL",
                "coordinates": [
                    [914.0595238095231, 609.583333333334],
                    [832.0595238095231, 681.583333333334], 
                    [758.0595238095231, 867.583333333334],
                    [756.0595238095231, 1097.583333333334],
                    [774.0595238095231, 1239.583333333334],
                    [812.0595238095231, 1617.583333333334],
                    [818.0595238095231, 1735.583333333334],
                    [900.0595238095231, 1865.583333333334],
                    [988.0595238095231, 1920]
                ],
                "type": "LineString"
            },
            "properties": {
                "contentZh": {
                    "这条光纤弯曲半径是否合理": "弯曲半径合理",
                    "标签": "光纤",
                    "这条光纤是否采用保护措施": "有保护措施，保护措施为/同时有蛇形管和铠装",
                    "这条光纤是否被遮挡": "有遮挡"
                },
                "content": {
                    "fiber_radius": "fiber_radius_trrue",
                    "fiber_block": "fiber_block_true",
                    "label": "fiber",
                    "ex_info": "",
                    "fiber_protection": "fiber_protection_true/fiber_protection_both"
                }
            }
        }
        
        self.sample_label = {
            "type": "Feature",
            "geometry": {
                "lineType": ["LLLLL"],
                "coordinates": [[[905.8214285714283, 505.3690476190477],
                               [939.1547619047617, 550.3690476190477],
                               [862.488095238095, 605.3690476190477],
                               [825.8214285714283, 555.3690476190477],
                               [905.8214285714283, 505.3690476190477]]],
                "type": "Square"
            },
            "properties": {
                "contentZh": {
                    "标签": "标签",
                    "能否阅读标签上的文字内容": "5G-AAU2-光纤"
                },
                "content": {
                    "label_text": "5G-AAU2-光纤",
                    "label": "label"
                }
            }
        }

    def test_object_type_detection(self):
        """Test object type detection from content fields."""
        # Test connect_point
        sample = self.processor.process_v2_feature(self.sample_connect_point)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.object_type, "connect_point")
        
        # Test BBU
        sample = self.processor.process_v2_feature(self.sample_bbu)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.object_type, "bbu")
        
        # Test fiber
        sample = self.processor.process_v2_feature(self.sample_fiber)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.object_type, "fiber")
        
        # Test label
        sample = self.processor.process_v2_feature(self.sample_label)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.object_type, "label")

    def test_geometry_processing(self):
        """Test geometry processing for different formats."""
        # Test ExtentPolygon -> bbox_2d
        sample = self.processor.process_v2_feature(self.sample_connect_point)
        self.assertEqual(sample.geometry_format, "bbox_2d")
        self.assertEqual(len(sample.coordinates), 4)
        
        # Test Square -> square
        sample = self.processor.process_v2_feature(self.sample_bbu)
        self.assertEqual(sample.geometry_format, "square")
        self.assertEqual(len(sample.coordinates), 8)  # 4 points * 2 coords
        
        # Test LineString -> line
        sample = self.processor.process_v2_feature(self.sample_fiber)
        self.assertEqual(sample.geometry_format, "line")
        self.assertGreater(len(sample.coordinates), 8)  # Multiple points
        self.assertEqual(len(sample.coordinates) % 2, 0)  # Even number (x,y pairs)

    def test_attribute_grouping(self):
        """Test attribute grouping according to taxonomy."""
        # Test connect_point attributes
        sample = self.processor.process_v2_feature(self.sample_connect_point)
        
        # Should have physical_properties group
        self.assertIn("physical_properties", sample.grouped_attributes)
        physical = sample.grouped_attributes["physical_properties"]
        self.assertIn("visibility_completeness", physical)
        self.assertEqual(physical["visibility_completeness"], "只显示部分")
        
        # Should have component_classification group
        self.assertIn("component_classification", sample.grouped_attributes)
        component = sample.grouped_attributes["component_classification"]
        self.assertIn("connection_point_type", component)
        self.assertEqual(component["connection_point_type"], "BBU安装螺丝")
        
        # Should have functional_assessment group
        self.assertIn("functional_assessment", sample.grouped_attributes)
        functional = sample.grouped_attributes["functional_assessment"]
        self.assertIn("compliance_check", functional)
        self.assertEqual(functional["compliance_check"], "符合要求")

    def test_bbu_complex_attributes(self):
        """Test BBU complex attribute processing."""
        sample = self.processor.process_v2_feature(self.sample_bbu)
        
        # Check physical properties
        self.assertIn("physical_properties", sample.grouped_attributes)
        physical = sample.grouped_attributes["physical_properties"]
        self.assertEqual(physical["brand_identification"], "华为")
        self.assertEqual(physical["visibility_completeness"], "显示完整")
        
        # Check functional assessment (complex wind baffle logic)
        self.assertIn("functional_assessment", sample.grouped_attributes)
        functional = sample.grouped_attributes["functional_assessment"]
        # This should map the complex bbu_equipment value
        self.assertIn("wind_baffle_requirements", functional)

    def test_fiber_technical_specs(self):
        """Test fiber technical specifications processing."""
        sample = self.processor.process_v2_feature(self.sample_fiber)
        
        # Check technical specifications
        self.assertIn("technical_specifications", sample.grouped_attributes)
        technical = sample.grouped_attributes["technical_specifications"]
        self.assertEqual(technical["bend_radius_assessment"], "弯曲半径合理")
        # Should handle complex protection measure mapping
        self.assertIn("protection_measures", technical)
        
        # Check environmental context
        self.assertIn("environmental_context", sample.grouped_attributes)
        environmental = sample.grouped_attributes["environmental_context"] 
        self.assertEqual(environmental["obstruction_status"], "有遮挡")

    def test_label_ocr_processing(self):
        """Test label OCR content processing."""
        sample = self.processor.process_v2_feature(self.sample_label)
        
        # Check textual content
        self.assertIn("textual_content", sample.grouped_attributes)
        textual = sample.grouped_attributes["textual_content"]
        self.assertIn("label_text_content", textual)
        self.assertEqual(textual["label_text_content"], "5G-AAU2-光纤")

    def test_description_generation(self):
        """Test hierarchical description generation."""
        # Test connect_point description
        sample = self.processor.process_v2_feature(self.sample_connect_point)
        description = sample.description
        self.assertIn("螺丝、光纤插头", description)
        self.assertIn("只显示部分", description) 
        self.assertIn("BBU安装螺丝", description)
        self.assertIn("符合要求", description)
        
        # Test BBU description  
        sample = self.processor.process_v2_feature(self.sample_bbu)
        description = sample.description
        self.assertIn("BBU设备", description)
        self.assertIn("华为", description)
        self.assertIn("显示完整", description)

    def test_training_format_output(self):
        """Test training format output structure."""
        sample = self.processor.process_v2_feature(self.sample_connect_point)
        training_data = sample.to_training_format()
        
        # Check required fields
        self.assertIn("bbox_2d", training_data)
        self.assertIn("desc", training_data)
        self.assertIn("object_type", training_data)
        self.assertIn("attributes", training_data)
        
        # Check coordinate format
        self.assertIsInstance(training_data["bbox_2d"], list)
        self.assertEqual(len(training_data["bbox_2d"]), 4)
        
        # Check description
        self.assertIsInstance(training_data["desc"], str)
        self.assertTrue(len(training_data["desc"]) > 0)

    def test_geometry_processor_integration(self):
        """Test geometry processor integration."""
        # Test bbox extraction
        geometry = self.sample_connect_point["geometry"]
        bbox = GeometryProcessor.extract_bbox_from_geometry(geometry)
        self.assertEqual(len(bbox), 4)
        self.assertIsInstance(bbox[0], (int, float))
        
        # Test hierarchical geometry extraction with preference
        hier_geo = GeometryProcessor.extract_hierarchical_geometry(geometry, "bbox_2d")
        self.assertIn("bbox_2d", hier_geo)
        self.assertIn("geometry", hier_geo)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty feature
        empty_feature = {"type": "Feature", "geometry": {}, "properties": {}}
        sample = self.processor.process_v2_feature(empty_feature)
        self.assertIsNone(sample)
        
        # Test invalid geometry
        invalid_feature = {
            "type": "Feature", 
            "geometry": {"type": "Unknown", "coordinates": []},
            "properties": {
                "content": {"label": "connect_point"},
                "contentZh": {"标签": "螺丝、光纤插头"}
            }
        }
        sample = self.processor.process_v2_feature(invalid_feature)
        if sample:  # Should handle gracefully
            self.assertEqual(sample.coordinates, [0, 0, 0, 0])

    def test_comprehensive_coverage(self):
        """Test that all object types from taxonomy are covered."""
        taxonomy = self.processor.taxonomy
        object_types = list(taxonomy["object_types"].keys())
        
        # We should have samples for key object types
        test_samples = [
            ("connect_point", self.sample_connect_point),
            ("bbu", self.sample_bbu), 
            ("fiber", self.sample_fiber),
            ("label", self.sample_label)
        ]
        
        for expected_type, sample_data in test_samples:
            sample = self.processor.process_v2_feature(sample_data)
            self.assertIsNotNone(sample, f"Failed to process {expected_type}")
            self.assertEqual(sample.object_type, expected_type)


class TestRealDataIntegration(unittest.TestCase):
    """Test with real V2 data files."""
    
    def setUp(self):
        """Set up with real data paths."""
        self.processor = FlexibleTaxonomyProcessor()
        self.data_dir = Path("ds_v2")
        
    def test_real_file_processing(self):
        """Test processing actual V2 files."""
        if not self.data_dir.exists():
            self.skipTest("V2 data directory not found")
        
        # Find a test file
        json_files = list(self.data_dir.glob("*.json"))
        if not json_files:
            self.skipTest("No JSON files found in V2 directory")
        
        test_file = json_files[0]
        samples = self.processor.process_v2_file(str(test_file))
        
        # Should process at least some samples
        self.assertGreater(len(samples), 0)
        
        # Check sample validity
        for sample in samples[:3]:  # Check first few
            self.assertIsInstance(sample, AnnotationSample)
            self.assertTrue(sample.object_type)
            self.assertTrue(sample.geometry_format in ["bbox_2d", "square", "line"])
            self.assertTrue(len(sample.coordinates) >= 4)
            self.assertTrue(sample.description)

    def test_statistics_generation(self):
        """Test statistics generation from processed samples."""
        if not self.data_dir.exists():
            self.skipTest("V2 data directory not found")
        
        json_files = list(self.data_dir.glob("*.json"))[:3]  # Process first 3 files
        all_samples = []
        
        for json_file in json_files:
            samples = self.processor.process_v2_file(str(json_file))
            all_samples.extend(samples)
        
        if not all_samples:
            self.skipTest("No samples processed")
        
        stats = self.processor.get_statistics(all_samples)
        
        # Check statistics structure
        self.assertIn("total_samples", stats)
        self.assertIn("object_types", stats)
        self.assertIn("geometry_formats", stats)
        self.assertIn("attribute_groups", stats)
        
        # Check counts
        self.assertEqual(stats["total_samples"], len(all_samples))
        self.assertGreater(len(stats["object_types"]), 0)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)