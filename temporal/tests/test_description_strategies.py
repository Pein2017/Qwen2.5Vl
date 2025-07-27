#!/usr/bin/env python3
"""
Test Description Concatenation Strategies

This test suite validates different description concatenation strategies
for hierarchical learning, ensuring they produce appropriate outputs
for different training scenarios.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_conversion.description_concatenator import (
    DescriptionConcatenator, 
    ConcatenationConfig, 
    ConcatenationStrategy
)
from data_conversion.hierarchical_learning_framework import LearningStage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DescriptionStrategyTest:
    """Test suite for description concatenation strategies."""
    
    def __init__(self):
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict]:
        """Create comprehensive test cases for different content types."""
        return [
            {
                "name": "Connect Point - Complete",
                "content": {
                    "object_type": "螺丝、光纤插头",
                    "property": "BBU安装螺丝",
                    "extra_info": "显示完整/符合要求"
                },
                "expected_stages": {
                    "stage_1": "螺丝、光纤插头",
                    "stage_1_2": "螺丝、光纤插头/BBU安装螺丝，显示完整",
                    "stage_1_2_3": "螺丝、光纤插头/BBU安装螺丝，显示完整/符合要求",
                    "all_stages": "螺丝、光纤插头/BBU安装螺丝，显示完整/符合要求"
                }
            },
            {
                "name": "BBU Equipment - Complex",
                "content": {
                    "object_type": "BBU设备",
                    "property": "华为",
                    "extra_info": "显示完整/机柜空间充足，需要安装/这个BBU设备按要求配备了挡风板"
                },
                "expected_stages": {
                    "stage_1": "BBU设备",
                    "stage_1_2": "BBU设备/华为，显示完整",
                    "stage_1_2_3": "BBU设备/华为，显示完整/机柜空间充足，需要安装，这个BBU设备按要求配备了挡风板",
                    "all_stages": "BBU设备/华为，显示完整/机柜空间充足，需要安装，这个BBU设备按要求配备了挡风板"
                }
            },
            {
                "name": "Fiber - Protection Details",
                "content": {
                    "object_type": "光纤",
                    "property": "弯曲半径合理",
                    "extra_info": "有保护措施/保护措施为/铠装/有遮挡"
                },
                "expected_stages": {
                    "stage_1": "光纤",
                    "stage_1_2": "光纤/弯曲半径合理，有保护措施，保护措施为，铠装，有遮挡",
                    "stage_1_2_3": "光纤/弯曲半径合理，有保护措施，保护措施为，铠装，有遮挡",
                    "all_stages": "光纤/弯曲半径合理，有保护措施，保护措施为，铠装，有遮挡"
                }
            },
            {
                "name": "Label - OCR Content",
                "content": {
                    "object_type": "标签",
                    "property": "",
                    "extra_info": "5G-BBU-传输光纤"
                },
                "expected_stages": {
                    "stage_1": "标签",
                    "stage_1_2": "标签",
                    "stage_1_2_3": "标签",
                    "all_stages": "标签/5G-BBU-传输光纤"
                }
            },
            {
                "name": "Wire - Organization",
                "content": {
                    "object_type": "电线",
                    "property": "捆扎整齐",
                    "extra_info": "有遮挡"
                },
                "expected_stages": {
                    "stage_1": "电线",
                    "stage_1_2": "电线/捆扎整齐，有遮挡",
                    "stage_1_2_3": "电线/捆扎整齐，有遮挡",
                    "all_stages": "电线/捆扎整齐，有遮挡"
                }
            },
            {
                "name": "Empty Fields",
                "content": {
                    "object_type": "BBU设备",
                    "property": "",
                    "extra_info": ""
                },
                "expected_stages": {
                    "stage_1": "BBU设备",
                    "stage_1_2": "BBU设备",
                    "stage_1_2_3": "BBU设备",
                    "all_stages": "BBU设备"
                }
            }
        ]
    
    def test_slash_separated_strategy(self):
        """Test slash-separated concatenation strategy."""
        logger.info("Testing slash-separated strategy...")
        
        config = ConcatenationConfig(
            strategy=ConcatenationStrategy.SLASH_SEPARATED,
            language="chinese"
        )
        concatenator = DescriptionConcatenator(config)
        
        for test_case in self.test_cases:
            logger.info(f"  Testing: {test_case['name']}")
            
            # Test progressive descriptions
            progressive = concatenator.create_progressive_descriptions(test_case["content"])
            
            # Validate each stage
            for stage, expected in test_case["expected_stages"].items():
                actual = progressive.get(stage, "")
                if actual != expected:
                    logger.error(f"    Stage {stage} mismatch:")
                    logger.error(f"      Expected: '{expected}'")
                    logger.error(f"      Actual:   '{actual}'")
                    return False
                else:
                    logger.debug(f"    Stage {stage}: '{actual}' ✓")
        
        logger.info("Slash-separated strategy test passed")
        return True
    
    def test_natural_language_strategy(self):
        """Test natural language concatenation strategy."""
        logger.info("Testing natural language strategy...")
        
        config = ConcatenationConfig(
            strategy=ConcatenationStrategy.NATURAL_LANGUAGE,
            language="chinese"
        )
        concatenator = DescriptionConcatenator(config)
        
        # Test specific cases for natural language
        test_cases = [
            {
                "content": {
                    "object_type": "BBU设备",
                    "property": "华为",
                    "extra_info": "显示完整"
                },
                "expected_pattern": "一个华为，显示完整BBU设备"
            },
            {
                "content": {
                    "object_type": "螺丝、光纤插头",
                    "property": "BBU安装螺丝",
                    "extra_info": "符合要求"
                },
                "expected_pattern": "一个BBU安装螺丝螺丝、光纤插头"
            }
        ]
        
        for test_case in test_cases:
            description = concatenator.concatenate_hierarchical_content(
                test_case["content"]
            )
            
            # Check that it contains expected elements
            expected_pattern = test_case["expected_pattern"]
            if expected_pattern not in description:
                logger.error(f"Natural language description missing pattern:")
                logger.error(f"  Expected pattern: '{expected_pattern}'")
                logger.error(f"  Actual: '{description}'")
                return False
            
            logger.debug(f"Natural language: '{description}' ✓")
        
        logger.info("Natural language strategy test passed")
        return True
    
    def test_structured_json_strategy(self):
        """Test structured JSON concatenation strategy."""
        logger.info("Testing structured JSON strategy...")
        
        config = ConcatenationConfig(
            strategy=ConcatenationStrategy.STRUCTURED_JSON,
            language="chinese"
        )
        concatenator = DescriptionConcatenator(config)
        
        test_content = {
            "object_type": "BBU设备",
            "property": "华为",
            "extra_info": "显示完整/机柜空间充足"
        }
        
        description = concatenator.concatenate_hierarchical_content(test_content)
        
        # Parse JSON to validate structure
        try:
            parsed = json.loads(description)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON output: {e}")
            logger.error(f"Description: {description}")
            return False
        
        # Validate JSON structure
        expected_keys = ["object_type", "properties", "complex_attributes"]
        for key in expected_keys:
            if key not in parsed:
                logger.error(f"Missing key in JSON output: {key}")
                return False
        
        # Validate content
        if parsed["object_type"] != "BBU设备":
            logger.error(f"Incorrect object_type in JSON: {parsed['object_type']}")
            return False
        
        if "华为" not in parsed["properties"]:
            logger.error(f"Missing property in JSON: {parsed['properties']}")
            return False
        
        logger.info(f"Structured JSON: {description}")
        logger.info("Structured JSON strategy test passed")
        return True
    
    def test_progressive_stages_strategy(self):
        """Test progressive stages concatenation strategy."""
        logger.info("Testing progressive stages strategy...")
        
        config = ConcatenationConfig(
            strategy=ConcatenationStrategy.PROGRESSIVE_STAGES,
            language="chinese"
        )
        concatenator = DescriptionConcatenator(config)
        
        test_content = {
            "object_type": "螺丝、光纤插头",
            "property": "BBU安装螺丝",
            "extra_info": "符合要求"
        }
        
        description = concatenator.concatenate_hierarchical_content(test_content)
        
        # Check that stage identifiers are present
        expected_identifiers = [
            "[object_identification]",
            "[property_recognition]",
            "[complex_attributes]"
        ]
        
        for identifier in expected_identifiers:
            if identifier not in description:
                logger.error(f"Missing stage identifier: {identifier}")
                logger.error(f"Description: {description}")
                return False
        
        logger.info(f"Progressive stages: {description}")
        logger.info("Progressive stages strategy test passed")
        return True
    
    def test_concatenation_config_options(self):
        """Test various concatenation configuration options."""
        logger.info("Testing concatenation configuration options...")
        
        test_content = {
            "object_type": "BBU设备",
            "property": "华为",
            "extra_info": "显示完整"
        }
        
        # Test with empty fields included
        config_with_empty = ConcatenationConfig(
            strategy=ConcatenationStrategy.SLASH_SEPARATED,
            language="chinese",
            include_empty_fields=True
        )
        concatenator_with_empty = DescriptionConcatenator(config_with_empty)
        
        # Test with empty property
        test_content_empty = {
            "object_type": "BBU设备",
            "property": "",
            "extra_info": "显示完整"
        }
        
        desc_with_empty = concatenator_with_empty.concatenate_hierarchical_content(test_content_empty)
        desc_without_empty = DescriptionConcatenator().concatenate_hierarchical_content(test_content_empty)
        
        # With empty fields, should have more "/" separators
        if desc_with_empty.count("/") <= desc_without_empty.count("/"):
            logger.error("include_empty_fields option not working correctly")
            logger.error(f"  With empty: '{desc_with_empty}'")
            logger.error(f"  Without empty: '{desc_without_empty}'")
            return False
        
        # Test max length limit
        config_with_limit = ConcatenationConfig(
            strategy=ConcatenationStrategy.SLASH_SEPARATED,
            language="chinese",
            max_length=10
        )
        concatenator_with_limit = DescriptionConcatenator(config_with_limit)
        
        desc_limited = concatenator_with_limit.concatenate_hierarchical_content(test_content)
        if len(desc_limited) > 10:
            logger.error(f"Max length limit not enforced: {len(desc_limited)} > 10")
            logger.error(f"Description: '{desc_limited}'")
            return False
        
        # Test custom separators
        config_custom_sep = ConcatenationConfig(
            strategy=ConcatenationStrategy.SLASH_SEPARATED,
            language="chinese",
            stage_separators={
                "between_stages": " | ",
                "within_stage": " & ",
                "final_separator": ""
            }
        )
        concatenator_custom_sep = DescriptionConcatenator(config_custom_sep)
        
        desc_custom_sep = concatenator_custom_sep.concatenate_hierarchical_content(test_content)
        if " | " not in desc_custom_sep:
            logger.error("Custom separators not applied correctly")
            logger.error(f"Description: '{desc_custom_sep}'")
            return False
        
        logger.info("Concatenation configuration options test passed")
        return True
    
    def test_validation_functionality(self):
        """Test description validation functionality."""
        logger.info("Testing description validation...")
        
        concatenator = DescriptionConcatenator()
        
        # Test validation of different descriptions
        test_descriptions = [
            {
                "desc": "BBU设备/华为/显示完整",
                "expected": {
                    "has_content": True,
                    "has_object_type": True,
                    "has_properties": True,
                    "has_special_chars": False
                }
            },
            {
                "desc": "",
                "expected": {
                    "has_content": False,
                    "has_object_type": False,
                    "has_properties": False,
                    "has_special_chars": False
                }
            },
            {
                "desc": "标签/5G-BBU-传输光纤",
                "expected": {
                    "has_content": True,
                    "has_object_type": True,
                    "has_properties": True,
                    "has_special_chars": True
                }
            }
        ]
        
        for test_case in test_descriptions:
            validation = concatenator.validate_concatenation(test_case["desc"])
            expected = test_case["expected"]
            
            for key, expected_value in expected.items():
                if validation.get(key) != expected_value:
                    logger.error(f"Validation mismatch for '{test_case['desc']}':")
                    logger.error(f"  {key}: expected {expected_value}, got {validation.get(key)}")
                    return False
        
        logger.info("Description validation test passed")
        return True
    
    def test_batch_processing(self):
        """Test batch processing of multiple content dictionaries."""
        logger.info("Testing batch processing...")
        
        concatenator = DescriptionConcatenator()
        
        # Create batch of content
        content_batch = [
            {"object_type": "BBU设备", "property": "华为", "extra_info": "显示完整"},
            {"object_type": "光纤", "property": "弯曲半径合理", "extra_info": "有保护措施"},
            {"object_type": "标签", "property": "", "extra_info": "5G-BBU-传输光纤"}
        ]
        
        # Test batch concatenation
        results = concatenator.batch_concatenate(content_batch)
        
        if len(results) != len(content_batch):
            logger.error(f"Batch processing length mismatch: {len(results)} != {len(content_batch)}")
            return False
        
        # Validate each result
        for i, result in enumerate(results):
            if not result or not isinstance(result, str):
                logger.error(f"Invalid batch result at index {i}: {result}")
                return False
        
        logger.info(f"Batch processing test passed: {len(results)} descriptions generated")
        return True
    
    def run_all_tests(self):
        """Run all description strategy tests."""
        logger.info("Starting description concatenation strategy tests...")
        
        tests = [
            ("Slash Separated Strategy", self.test_slash_separated_strategy),
            ("Natural Language Strategy", self.test_natural_language_strategy),
            ("Structured JSON Strategy", self.test_structured_json_strategy),
            ("Progressive Stages Strategy", self.test_progressive_stages_strategy),
            ("Configuration Options", self.test_concatenation_config_options),
            ("Validation Functionality", self.test_validation_functionality),
            ("Batch Processing", self.test_batch_processing)
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
    """Run description strategy tests."""
    test_suite = DescriptionStrategyTest()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("Description Strategy Test Summary")
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
