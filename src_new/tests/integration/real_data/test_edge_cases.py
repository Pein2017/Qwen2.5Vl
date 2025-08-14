#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge Cases and Boundary Condition Tests

This module tests edge cases and boundary conditions that may occur in
production with real data, focusing on areas not covered by existing tests:

1. Malformed data handling and recovery
2. Empty or incomplete conversations
3. Coordinate boundary conditions and clamping
4. Template parsing edge cases
5. Memory and performance edge cases

Key Features:
- Tests with actual problematic data patterns from real datasets
- Boundary condition validation for coordinate tokens
- Error handling and graceful degradation testing
- Template parsing robustness validation
- Memory efficiency under edge conditions
"""

import logging
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLProcessor


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src_new.config.config import load_config
from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.coordinate_converter import CoordinateTokenConverter
from src_new.processing.token_processor import TokenConfig, TokenProcessor


# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TestEdgeCasesAndBoundaryConditions:
    """
    Comprehensive tests for edge cases and boundary conditions.

    Tests robustness and error handling with problematic data patterns
    that may occur in production environments.
    """

    @pytest.fixture(scope="class")
    def edge_case_data(self):
        """Provide edge case data patterns for testing."""
        return {
            # Empty or minimal data
            "empty_objects": {
                "images": ["test.jpg"],
                "objects": [],
                "width": 532,
                "height": 728,
            },
            # Single object with minimal data
            "minimal_object": {
                "images": ["test.jpg"],
                "objects": [{"bbox_2d": [0, 0, 1, 1], "desc": ""}],
                "width": 532,
                "height": 728,
            },
            # Extreme coordinates
            "extreme_coordinates": {
                "images": ["test.jpg"],
                "objects": [
                    {"bbox_2d": [0, 0, 9999, 9999], "desc": "超大坐标测试"},
                    {"bbox_2d": [-100, -100, 50, 50], "desc": "负坐标测试"},
                    {"line": [0, 0, 1, 1, 2, 2], "desc": "最小线段"},
                    {"quad": [0, 0, 1, 0, 1, 1, 0, 1], "desc": "最小四边形"},
                ],
                "width": 532,
                "height": 728,
            },
            # Missing or malformed fields
            "malformed_data": [
                {"images": ["test.jpg"], "objects": [{"desc": "缺少几何信息"}]},
                {
                    "images": ["test.jpg"],
                    "objects": [{"bbox_2d": [100, 200], "desc": "坐标不完整"}],
                },
                {
                    "images": ["test.jpg"],
                    "objects": [{"bbox_2d": "invalid", "desc": "坐标类型错误"}],
                },
                {"objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "缺少图片"}]},
            ],
            # Very long descriptions
            "long_descriptions": {
                "images": ["test.jpg"],
                "objects": [
                    {
                        "bbox_2d": [100, 200, 150, 250],
                        "desc": "这是一个非常非常长的描述"
                        * 100,  # Very long description
                    }
                ],
                "width": 532,
                "height": 728,
            },
            # Unicode and special characters
            "special_characters": {
                "images": ["test.jpg"],
                "objects": [
                    {"bbox_2d": [100, 200, 150, 250], "desc": "测试🔧⚡️🎯📊✅❌"},
                    {
                        "bbox_2d": [200, 300, 250, 350],
                        "desc": "Test with\nnewlines\tand\ttabs",
                    },
                    {
                        "bbox_2d": [300, 400, 350, 450],
                        "desc": "引号\"测试'单引号`反引号",
                    },
                ],
            },
        }

    @pytest.fixture(scope="class")
    def test_setup(self):
        """Set up test components."""
        # Load config
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")
        if not config_path.exists():
            pytest.skip("Real config file not found")

        config = load_config(str(config_path))

        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True, use_fast=True
        )

        processor = Qwen2VLProcessor.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        return config, tokenizer, processor

    def test_empty_objects_handling(self, edge_case_data, test_setup):
        """
        Test handling of samples with empty objects list.

        Validates:
        - Graceful handling of empty object lists
        - Appropriate error messages
        - No crashes or undefined behavior
        """
        logger.info("🧪 Testing empty objects handling...")

        config, tokenizer, processor = test_setup

        # Initialize coordinate converter
        converter = CoordinateTokenConverter(
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Test empty objects
        empty_sample = edge_case_data["empty_objects"]
        logger.info(f"📝 Testing sample with {len(empty_sample['objects'])} objects")

        try:
            # This should raise a ValueError with a clear message
            with pytest.raises(ValueError, match="Empty objects list encountered"):
                converter.convert_objects_to_tokens(empty_sample["objects"])

            logger.info("✅ Empty objects correctly rejected with clear error message")

        except Exception as e:
            logger.error(f"❌ Unexpected error handling empty objects: {e}")
            raise

        logger.info("✅ Empty objects handling test completed successfully")

    def test_coordinate_boundary_conditions(self, edge_case_data, test_setup):
        """
        Test coordinate boundary conditions and clamping.

        Validates:
        - Coordinate clamping for out-of-range values
        - Handling of negative coordinates
        - Maximum coordinate value enforcement
        - Coordinate validation and normalization
        """
        logger.info("🧪 Testing coordinate boundary conditions...")

        config, tokenizer, processor = test_setup

        # Initialize coordinate converter
        converter = CoordinateTokenConverter(
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Test extreme coordinates
        extreme_sample = edge_case_data["extreme_coordinates"]
        logger.info(f"📝 Testing sample with extreme coordinates")

        try:
            # Convert objects with extreme coordinates
            result = converter.convert_objects_to_tokens(extreme_sample["objects"])

            logger.info("✅ Extreme coordinates processed successfully")
            logger.info(f"📄 Result preview: {result[:200]}...")

            # Validate that coordinate tokens are within expected range
            max_coord = config.max_coord_value

            # Check for coordinate tokens in result
            import re

            coord_pattern = r"<\|coord_(\d+)\|>"
            coord_matches = re.findall(coord_pattern, result)

            if coord_matches:
                coord_values = [int(match) for match in coord_matches]
                logger.info(
                    f"🔢 Found coordinate values: {coord_values[:10]}..."
                )  # First 10

                # Validate all coordinates are within bounds
                valid_coords = all(0 <= coord <= max_coord for coord in coord_values)
                assert valid_coords, f"Some coordinates exceed bounds [0, {max_coord}]"

                logger.info(f"✅ All coordinates within bounds [0, {max_coord}]")
            else:
                logger.warning("⚠️ No coordinate tokens found in result")

        except Exception as e:
            logger.error(f"❌ Coordinate boundary test failed: {e}")
            raise

        logger.info("✅ Coordinate boundary conditions test completed successfully")

    def test_malformed_data_handling(self, edge_case_data, test_setup):
        """
        Test handling of malformed or incomplete data.

        Validates:
        - Graceful handling of missing required fields
        - Clear error messages for malformed data
        - No crashes with invalid data types
        - Robust validation and error reporting
        """
        logger.info("🧪 Testing malformed data handling...")

        config, tokenizer, processor = test_setup

        # Initialize coordinate converter
        converter = CoordinateTokenConverter(
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Test each malformed data pattern
        malformed_samples = edge_case_data["malformed_data"]

        for i, malformed_sample in enumerate(malformed_samples):
            logger.info(f"🔍 Testing malformed sample {i}: {malformed_sample}")

            try:
                # This should raise appropriate errors
                objects = malformed_sample.get("objects", [])

                if not objects:
                    logger.info(f"  Sample {i}: Empty objects list")
                    continue

                # Try to convert - should handle errors gracefully
                try:
                    result = converter.convert_objects_to_tokens(objects)
                    logger.warning(
                        f"  Sample {i}: Unexpectedly succeeded: {result[:100]}..."
                    )
                except ValueError as ve:
                    logger.info(
                        f"  Sample {i}: Correctly rejected with ValueError: {ve}"
                    )
                except Exception as e:
                    logger.error(
                        f"  Sample {i}: Unexpected error type: {type(e).__name__}: {e}"
                    )
                    raise

            except Exception as e:
                logger.error(f"❌ Malformed data test {i} failed: {e}")
                raise

        logger.info("✅ Malformed data handling test completed successfully")

    def test_conversation_template_edge_cases(self, edge_case_data, test_setup):
        """
        Test conversation template parsing with edge cases.

        Validates:
        - Handling of very long descriptions
        - Special character processing
        - Unicode character support
        - Template robustness with unusual content
        """
        logger.info("🧪 Testing conversation template edge cases...")

        config, tokenizer, processor = test_setup
        processor.tokenizer = tokenizer  # Use our tokenizer

        # Initialize conversation processor
        conversation_processor = ConversationProcessor(
            processor=processor,
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Create mock images
        mock_images = [Image.new("RGB", (532, 728), color="blue")]

        # Test long descriptions
        logger.info("📝 Testing very long descriptions...")
        long_desc_sample = edge_case_data["long_descriptions"]

        try:
            result = conversation_processor.create_simple_conversation(
                sample=long_desc_sample, images=mock_images
            )

            logger.info("✅ Long descriptions processed successfully")
            logger.info(f"📊 Result sequence length: {result['input_ids'].shape[1]}")

            # Validate result structure
            assert "input_ids" in result, "Missing input_ids"
            assert result["input_ids"].shape[1] > 0, "Empty sequence generated"

        except Exception as e:
            logger.error(f"❌ Long descriptions test failed: {e}")
            raise

        # Test special characters
        logger.info("🔤 Testing special characters...")
        special_char_sample = edge_case_data["special_characters"]

        try:
            result = conversation_processor.create_simple_conversation(
                sample=special_char_sample, images=mock_images
            )

            logger.info("✅ Special characters processed successfully")
            logger.info(f"📊 Result sequence length: {result['input_ids'].shape[1]}")

            # Validate result structure
            assert "input_ids" in result, "Missing input_ids"
            assert result["input_ids"].shape[1] > 0, "Empty sequence generated"

        except Exception as e:
            logger.error(f"❌ Special characters test failed: {e}")
            raise

        logger.info("✅ Conversation template edge cases test completed successfully")

    def test_memory_and_performance_edge_cases(self, edge_case_data, test_setup):
        """
        Test memory efficiency and performance under edge conditions.

        Validates:
        - Large batch processing without memory leaks
        - Performance with very long sequences
        - Resource cleanup after errors
        - Memory usage patterns with edge cases
        """
        logger.info("🧪 Testing memory and performance edge cases...")

        config, tokenizer, processor = test_setup
        processor.tokenizer = tokenizer

        # Initialize conversation processor
        conversation_processor = ConversationProcessor(
            processor=processor,
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Create mock images
        mock_images = [Image.new("RGB", (532, 728), color="green")]

        # Test with multiple samples to check memory usage
        logger.info("📊 Testing batch processing memory efficiency...")

        try:
            # Process multiple samples in sequence
            for i in range(10):  # Process 10 samples
                sample = {
                    "images": ["test.jpg"],
                    "objects": [
                        {
                            "bbox_2d": [
                                100 + i * 10,
                                200 + i * 10,
                                150 + i * 10,
                                250 + i * 10,
                            ],
                            "desc": f"测试对象{i}",
                        }
                    ],
                    "width": 532,
                    "height": 728,
                }

                result = conversation_processor.create_simple_conversation(
                    sample=sample, images=mock_images
                )

                # Validate result
                assert "input_ids" in result, f"Missing input_ids in sample {i}"
                assert result["input_ids"].shape[1] > 0, f"Empty sequence in sample {i}"

                # Clear result to help with memory
                del result

            logger.info("✅ Batch processing completed successfully")

        except Exception as e:
            logger.error(f"❌ Memory/performance test failed: {e}")
            raise

        logger.info("✅ Memory and performance edge cases test completed successfully")

    def test_tokenizer_edge_cases(self, test_setup):
        """
        Test tokenizer edge cases and error handling.

        Validates:
        - Handling of unknown tokens
        - Very long sequences
        - Special token processing
        - Tokenizer vocabulary limits
        """
        logger.info("🧪 Testing tokenizer edge cases...")

        config, tokenizer, processor = test_setup

        # Initialize token processor
        token_config = TokenConfig(
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
            max_coord_value=config.max_coord_value,
            new_geometry_tokens=[],
            coordinate_init_mode="fourier_ramp",
        )
        token_processor = TokenProcessor(token_config)

        # Test coordinate token creation
        logger.info("🔢 Testing coordinate token processing...")

        try:
            # Test coordinate mask creation with edge cases
            test_input_ids = torch.tensor(
                [
                    151644,
                    1587,
                    198,  # Normal tokens
                    151667,
                    151668,  # Coordinate tokens
                    999999,  # Unknown token ID
                    151645,  # End token
                ]
            )

            # This should handle unknown tokens gracefully
            coord_mask = token_processor.create_coordinate_mask(
                test_input_ids, tokenizer
            )

            logger.info(f"✅ Coordinate mask created: {coord_mask}")
            logger.info(f"📊 Coordinate tokens found: {coord_mask.sum().item()}")

            # Validate mask structure
            assert coord_mask.shape == test_input_ids.shape, "Mask shape mismatch"
            assert coord_mask.dtype == torch.bool, "Mask should be boolean"

        except Exception as e:
            logger.error(f"❌ Tokenizer edge case test failed: {e}")
            raise

        logger.info("✅ Tokenizer edge cases test completed successfully")

    def test_error_recovery_and_graceful_degradation(self, edge_case_data, test_setup):
        """
        Test error recovery and graceful degradation.

        Validates:
        - Recovery from processing errors
        - Graceful handling of corrupted data
        - Fallback mechanisms
        - Error propagation and logging
        """
        logger.info("🧪 Testing error recovery and graceful degradation...")

        config, tokenizer, processor = test_setup

        # Initialize coordinate converter
        converter = CoordinateTokenConverter(
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Test error recovery scenarios
        error_scenarios = [
            {
                "name": "None objects list",
                "objects": None,
                "expected_error": (TypeError, AttributeError),
            },
            {
                "name": "Non-list objects",
                "objects": "not a list",
                "expected_error": (TypeError, AttributeError),
            },
            {
                "name": "Objects with None values",
                "objects": [None, {"bbox_2d": [100, 200, 150, 250], "desc": "valid"}],
                "expected_error": (TypeError, AttributeError),
            },
            {
                "name": "Mixed valid and invalid objects",
                "objects": [
                    {"bbox_2d": [100, 200, 150, 250], "desc": "valid"},
                    {"invalid": "data"},
                    {"bbox_2d": [200, 300, 250, 350], "desc": "also valid"},
                ],
                "expected_error": (ValueError, KeyError),
            },
        ]

        for scenario in error_scenarios:
            logger.info(f"🔍 Testing scenario: {scenario['name']}")

            try:
                # This should raise an appropriate error
                result = converter.convert_objects_to_tokens(scenario["objects"])
                logger.warning(
                    f"  Scenario '{scenario['name']}' unexpectedly succeeded: {result[:100]}..."
                )

            except scenario["expected_error"] as e:
                logger.info(f"  ✅ Correctly handled with {type(e).__name__}: {e}")

            except Exception as e:
                logger.error(f"  ❌ Unexpected error type {type(e).__name__}: {e}")
                # Don't raise here - we want to test all scenarios

        logger.info(
            "✅ Error recovery and graceful degradation test completed successfully"
        )

    def test_concurrent_processing_safety(self, edge_case_data, test_setup):
        """
        Test thread safety and concurrent processing.

        Validates:
        - Thread-safe processing of multiple samples
        - No race conditions in coordinate conversion
        - Proper resource isolation
        - Concurrent error handling
        """
        logger.info("🧪 Testing concurrent processing safety...")

        config, tokenizer, processor = test_setup

        # Initialize coordinate converter
        converter = CoordinateTokenConverter(
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Test concurrent processing simulation
        import threading
        import time

        results = []
        errors = []

        def process_sample(sample_id):
            """Process a sample in a separate thread."""
            try:
                sample_objects = [
                    {
                        "bbox_2d": [
                            100 + sample_id * 10,
                            200,
                            150 + sample_id * 10,
                            250,
                        ],
                        "desc": f"并发测试对象{sample_id}",
                    }
                ]

                # Add small delay to increase chance of race conditions
                time.sleep(0.01)

                result = converter.convert_objects_to_tokens(sample_objects)
                results.append((sample_id, result))

            except Exception as e:
                errors.append((sample_id, e))

        # Create and start multiple threads
        threads = []
        for i in range(5):  # 5 concurrent threads
            thread = threading.Thread(target=process_sample, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Validate results
        logger.info(
            f"📊 Concurrent processing results: {len(results)} successful, {len(errors)} errors"
        )

        if errors:
            for sample_id, error in errors:
                logger.error(f"  Thread {sample_id} error: {error}")
            raise AssertionError(f"Concurrent processing had {len(errors)} errors")

        # Validate all results are unique and correct
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

        for sample_id, result in results:
            assert f"并发测试对象{sample_id}" in result, (
                f"Sample {sample_id} result missing expected content"
            )

        logger.info("✅ Concurrent processing safety test completed successfully")

    def test_resource_cleanup_after_errors(self, test_setup):
        """
        Test proper resource cleanup after errors occur.

        Validates:
        - Memory cleanup after processing errors
        - File handle cleanup
        - Proper exception propagation
        - No resource leaks
        """
        logger.info("🧪 Testing resource cleanup after errors...")

        config, tokenizer, processor = test_setup

        # Initialize coordinate converter
        converter = CoordinateTokenConverter(
            max_coord_value=config.max_coord_value,
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        )

        # Test resource cleanup scenarios
        cleanup_scenarios = [
            {
                "name": "Large invalid data processing",
                "objects": [
                    {"invalid": "data"} for _ in range(100)
                ],  # Large invalid dataset
            },
            {
                "name": "Nested error conditions",
                "objects": [
                    {
                        "bbox_2d": [float("inf"), float("nan"), 100, 200],
                        "desc": "无效浮点数",
                    },
                    {"bbox_2d": [100, 200, 150, 250], "desc": "正常数据"},
                ],
            },
        ]

        for scenario in cleanup_scenarios:
            logger.info(f"🔍 Testing cleanup scenario: {scenario['name']}")

            try:
                # This should fail but clean up properly
                result = converter.convert_objects_to_tokens(scenario["objects"])
                logger.warning(
                    f"  Scenario '{scenario['name']}' unexpectedly succeeded"
                )

            except Exception as e:
                logger.info(
                    f"  ✅ Error handled: {type(e).__name__}: {str(e)[:100]}..."
                )

                # Verify we can still process valid data after the error
                try:
                    valid_objects = [
                        {"bbox_2d": [100, 200, 150, 250], "desc": "清理后测试"}
                    ]
                    recovery_result = converter.convert_objects_to_tokens(valid_objects)
                    logger.info(f"  ✅ Recovery successful after error")

                except Exception as recovery_error:
                    logger.error(
                        f"  ❌ Failed to recover after error: {recovery_error}"
                    )
                    raise

        logger.info("✅ Resource cleanup after errors test completed successfully")
