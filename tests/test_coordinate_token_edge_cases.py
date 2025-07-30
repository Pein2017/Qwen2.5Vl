#!/usr/bin/env python3
"""
Coordinate Token Edge Cases and Error Scenarios Tests

Tests edge cases, error scenarios, and boundary conditions for the coordinate token system
to ensure robust operation and proper error handling.

Key test scenarios:
- Boundary value testing (min/max coordinates)
- Invalid input handling and error messages
- Memory and performance edge cases
- Concurrent access patterns
- Malformed data handling
- Recovery from error states
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import shutil
import threading
import time

import torch

# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton
patch_torch_library_wrap_triton()

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

from src.logger_utils import configure_global_logging, get_logger
from src.config import init_config, load_config
from src.utils.tokens.special_tokens import (
    UnifiedTokenManager,
    SimpleCoordinateManager,
    has_coordinate_tokens_static,
    get_coordinate_token_range_static,
)
from src.chat_processor import ChatProcessor
from tests.fixtures import SyntheticDataGenerator, ConfigFactory
from tests.fixtures.gpu_test_base import GPUAwareTestCase
from tests.fixtures.test_utils import TestUtils

logger = get_logger("test_coordinate_token_edge_cases")


class TestCoordinateTokenEdgeCases(GPUAwareTestCase):
    """Tests for coordinate token edge cases and error scenarios."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        super().setUpClass()
        configure_global_logging(rank=0, world_size=1)

        # Create test utilities
        cls.test_utils = TestUtils()
        cls.config_factory = ConfigFactory()
        cls.data_generator = SyntheticDataGenerator(num_samples=3)
        cls.temp_dir = tempfile.mkdtemp(prefix="coordinate_edge_cases_tests_")
        cls.test_files_to_cleanup = []

        # Base model path
        cls.model_path = "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        
        logger.info(f"✅ Test setup complete in {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up test configs
        cls.config_factory.cleanup_test_configs()

        # Clean up temporary directory
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

        # Clean up test files
        cls.test_utils.cleanup_test_files(cls.test_files_to_cleanup)

        logger.info("🧹 Test teardown complete")

    def setUp(self):
        """Set up for each individual test."""
        self.test_dir = os.path.join(self.temp_dir, f"test_{hash(self)}")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after each test."""
        # Force aggressive GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

    def test_coordinate_value_boundary_conditions(self):
        """Test boundary conditions for coordinate values."""
        logger.info("🔬 Testing coordinate value boundary conditions...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Test with small coordinate range
        max_coord_value = 10
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Test boundary values
        boundary_tests = [
            {"coord": 0, "should_pass": True, "desc": "minimum value"},
            {"coord": max_coord_value - 1, "should_pass": True, "desc": "maximum valid value"},
            {"coord": max_coord_value, "should_pass": False, "desc": "exactly at limit"},
            {"coord": max_coord_value + 1, "should_pass": False, "desc": "over limit"},
            {"coord": -1, "should_pass": False, "desc": "negative value"},
            {"coord": 0.5, "should_pass": False, "desc": "float value"},  # Should fail as int() conversion
        ]
        
        for test in boundary_tests:
            coord = test["coord"]
            should_pass = test["should_pass"]
            desc = test["desc"]
            
            with self.subTest(coord=coord, desc=desc):
                if should_pass:
                    try:
                        token_id = token_manager.get_coordinate_token_id(coord)
                        self.assertIsInstance(token_id, int)
                        self.assertGreater(token_id, 0)
                    except (ValueError, TypeError) as e:
                        self.fail(f"Valid coordinate {coord} ({desc}) should not raise error: {e}")
                else:
                    with self.assertRaises((ValueError, TypeError)) as context:
                        token_manager.get_coordinate_token_id(coord)
                    
                    error_msg = str(context.exception).lower()
                    self.assertTrue(
                        any(keyword in error_msg for keyword in ["coordinate", "range", "value", "invalid"]),
                        f"Error message should mention coordinate/range/value: {error_msg}"
                    )
        
        logger.info("✅ Coordinate value boundary conditions test passed")

    def test_invalid_geometry_types_and_coordinates(self):
        """Test handling of invalid geometry types and coordinate arrays."""
        logger.info("🔬 Testing invalid geometry types and coordinates...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        max_coord_value = 50
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Test invalid geometry types
        invalid_geometry_tests = [
            {"coords": [10, 20, 30, 40], "geometry": "invalid_type"},
            {"coords": [10, 20, 30, 40], "geometry": ""},
            {"coords": [10, 20, 30, 40], "geometry": None},
            {"coords": [10, 20, 30, 40], "geometry": 123},
        ]
        
        for test in invalid_geometry_tests:
            coords = test["coords"]
            geometry = test["geometry"]
            
            with self.subTest(geometry=geometry):
                with self.assertRaises((ValueError, TypeError, AttributeError)) as context:
                    token_manager.wrap_coordinates(coords, geometry)
                
                error_msg = str(context.exception).lower()
                # Should mention geometry or type in error
                self.assertTrue(
                    any(keyword in error_msg for keyword in ["geometry", "type", "invalid"]),
                    f"Error should mention geometry/type: {error_msg}"
                )
        
        # Test invalid coordinate arrays
        invalid_coordinate_tests = [
            {"coords": [], "geometry": "bbox", "desc": "empty coordinates"},
            {"coords": [10], "geometry": "bbox", "desc": "too few coordinates for bbox"},
            {"coords": [10, 20, 30], "geometry": "bbox", "desc": "odd number of coordinates for bbox"},
            {"coords": [10, 20, 30, 40, 50], "geometry": "line", "desc": "wrong number for line"},
            {"coords": None, "geometry": "bbox", "desc": "None coordinates"},
            {"coords": "invalid", "geometry": "bbox", "desc": "string coordinates"},
        ]
        
        for test in invalid_coordinate_tests:
            coords = test["coords"]
            geometry = test["geometry"]
            desc = test["desc"]
            
            with self.subTest(desc=desc):
                with self.assertRaises((ValueError, TypeError, AttributeError)) as context:
                    token_manager.wrap_coordinates(coords, geometry)
        
        logger.info("✅ Invalid geometry types and coordinates test passed")

    def test_extreme_coordinate_ranges(self):
        """Test behavior with extreme coordinate ranges."""
        logger.info("🔬 Testing extreme coordinate ranges...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Test very small coordinate range
        small_token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=1,  # Only coord_0
        )
        
        # Verify small range works
        self.assertTrue(small_token_manager.has_coordinate_tokens())
        
        coord_start, coord_end = small_token_manager.get_coordinate_token_range()
        self.assertEqual(coord_end - coord_start, 1, "Small range should have 1 token")
        
        # Test coordinate wrapping with clamping
        test_coords = [0, 5, 10]  # Values beyond range should be clamped
        wrapped = small_token_manager.wrap_coordinates(test_coords[:2], "bbox")  # Only use first 2
        
        # Should contain coord_0 (clamped values)
        self.assertIn("<|coord_0|>", wrapped)
        
        # Test moderately large coordinate range
        try:
            large_token_manager = UnifiedTokenManager(
                tokenizer=tokenizer,
                model=model,
                max_coord_value=1000,  # Large but reasonable
            )
            
            # If it succeeds, test functionality
            self.assertTrue(large_token_manager.has_coordinate_tokens())
            
            large_start, large_end = large_token_manager.get_coordinate_token_range()
            self.assertEqual(large_end - large_start, 1000, "Large range should have 1000 tokens")
            
            # Test some coordinates
            for coord in [0, 500, 999]:
                token_id = large_token_manager.get_coordinate_token_id(coord)
                self.assertIsInstance(token_id, int)
                self.assertGreaterEqual(token_id, large_start)
                self.assertLess(token_id, large_end)
            
        except (ValueError, RuntimeError, MemoryError) as e:
            logger.info(f"Large coordinate range properly rejected: {e}")
            # This is acceptable - system should reject overly large ranges
        
        logger.info("✅ Extreme coordinate ranges test passed")

    def test_malformed_object_data_handling(self):
        """Test handling of malformed object data."""
        logger.info("🔬 Testing malformed object data handling...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        max_coord_value = 50
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Test malformed objects
        malformed_objects = [
            # Missing required fields
            {"desc": "Object without geometry"},
            {"bbox_2d": [10, 20, 30, 40]},  # Missing desc
            {},  # Empty object
            
            # Invalid field types
            {"bbox_2d": "not_a_list", "desc": "Invalid bbox type"},
            {"bbox_2d": [10, 20, 30, 40], "desc": None},  # Invalid desc type
            {"bbox_2d": [10, 20, 30, 40], "desc": 123},  # Invalid desc type
            
            # Invalid geometry data
            {"bbox_2d": [10, 20], "desc": "Too few coordinates"},  # Not enough coords
            {"bbox_2d": [10, 20, 30, 40, 50], "desc": "Too many coordinates"},  # Too many coords
            {"line": [10, 20, 30], "desc": "Invalid line coords"},  # Wrong number for line
            
            # Coordinates out of range
            {"bbox_2d": [10, 20, 60, 40], "desc": "Coords out of range"},  # 60 > max_coord_value
            {"bbox_2d": [-5, 20, 30, 40], "desc": "Negative coordinates"},
            
            # Mixed invalid data
            {"bbox_2d": [10, "invalid", 30, 40], "desc": "Mixed coordinate types"},
            {"invalid_geometry": [10, 20, 30, 40], "desc": "Unknown geometry type"},
        ]
        
        for i, obj in enumerate(malformed_objects):
            with self.subTest(obj_index=i, obj_keys=list(obj.keys())):
                with self.assertRaises((ValueError, TypeError, KeyError, RuntimeError)) as context:
                    token_manager.format_object(obj)
        
        # Test valid object for comparison
        valid_object = {"bbox_2d": [10, 20, 30, 40], "desc": "Valid test object"}
        try:
            formatted = token_manager.format_object(valid_object)
            self.assertIn("<|object_ref_start|>", formatted)
            self.assertIn("<|object_ref_end|>", formatted)
        except Exception as e:
            self.fail(f"Valid object should not raise error: {e}")
        
        logger.info("✅ Malformed object data handling test passed")

    def test_concurrent_token_manager_access(self):
        """Test concurrent access to token manager (thread safety)."""
        logger.info("🔬 Testing concurrent token manager access...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        max_coord_value = 100
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Test concurrent coordinate token retrieval
        results = {}
        errors = {}
        
        def worker_coordinate_access(worker_id):
            """Worker function for testing coordinate access."""
            try:
                worker_results = []
                for i in range(10):  # Test 10 coordinate accesses per worker
                    coord = (worker_id * 10 + i) % max_coord_value
                    token_id = token_manager.get_coordinate_token_id(coord)
                    worker_results.append((coord, token_id))
                
                results[worker_id] = worker_results
                
            except Exception as e:
                errors[worker_id] = e
        
        # Create and start worker threads
        num_workers = 5
        threads = []
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=worker_coordinate_access, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Check results
        self.assertEqual(len(errors), 0, f"Workers should not have errors: {errors}")
        self.assertEqual(len(results), num_workers, f"All workers should complete: {len(results)} vs {num_workers}")
        
        # Verify result consistency
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        
        for worker_id, worker_results in results.items():
            for coord, token_id in worker_results:
                expected_token_id = coord_start_id + coord
                self.assertEqual(
                    token_id, expected_token_id,
                    f"Worker {worker_id}: coord {coord} should have token_id {expected_token_id}, got {token_id}"
                )
        
        # Test concurrent object formatting
        test_objects = [
            {"bbox_2d": [10, 20, 30, 40], "desc": f"Test object {i}"}
            for i in range(10)
        ]
        
        format_results = {}
        format_errors = {}
        
        def worker_object_formatting(worker_id):
            """Worker function for testing object formatting."""
            try:
                formatted_objects = []
                for obj in test_objects:
                    # Modify object slightly for each worker
                    worker_obj = obj.copy()
                    worker_obj["desc"] = f"Worker {worker_id}: {obj['desc']}"
                    
                    formatted = token_manager.format_object(worker_obj)
                    formatted_objects.append(formatted)
                
                format_results[worker_id] = formatted_objects
                
            except Exception as e:
                format_errors[worker_id] = e
        
        # Create and start formatting worker threads
        format_threads = []
        
        for worker_id in range(3):  # Fewer workers for formatting test
            thread = threading.Thread(target=worker_object_formatting, args=(worker_id,))
            format_threads.append(thread)
            thread.start()
        
        # Wait for all formatting threads to complete
        for thread in format_threads:
            thread.join(timeout=15)  # 15 second timeout
        
        # Check formatting results
        self.assertEqual(len(format_errors), 0, f"Formatting workers should not have errors: {format_errors}")
        self.assertEqual(len(format_results), 3, f"All formatting workers should complete")
        
        # Verify formatting consistency
        for worker_id, worker_formatted in format_results.items():
            self.assertEqual(len(worker_formatted), len(test_objects), f"Worker {worker_id} should format all objects")
            
            for formatted in worker_formatted:
                self.assertIn("<|object_ref_start|>", formatted)
                self.assertIn("<|object_ref_end|>", formatted)
                self.assertIn(f"Worker {worker_id}:", formatted)
        
        logger.info("✅ Concurrent token manager access test passed")

    def test_memory_usage_with_large_coordinate_batches(self):
        """Test memory usage patterns with large coordinate batches."""
        logger.info("🔬 Testing memory usage with large coordinate batches...")
        
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory usage test")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).cuda()
        
        max_coord_value = 200  # Moderate size for memory testing
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Get initial memory usage
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create large batch of coordinate tokens
        batch_size = 10
        seq_len = 50
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        
        # Create batch with mixed coordinate and regular tokens
        batch_inputs = []
        batch_labels = []
        
        for batch_idx in range(batch_size):
            # Create sequence with coordinate tokens
            input_ids = []
            
            # Add some regular tokens
            regular_tokens = tokenizer.encode("Object at coordinates", add_special_tokens=False)
            input_ids.extend(regular_tokens[:10])  # Use first 10 tokens
            
            # Add coordinate tokens
            for i in range(min(20, seq_len - len(input_ids))):  # Add up to 20 coordinate tokens
                coord_value = (batch_idx * 20 + i) % max_coord_value
                coord_token_id = coord_start_id + coord_value
                input_ids.append(coord_token_id)
            
            # Pad or trim to seq_len
            if len(input_ids) < seq_len:
                input_ids.extend([tokenizer.eos_token_id] * (seq_len - len(input_ids)))
            else:
                input_ids = input_ids[:seq_len]
            
            batch_inputs.append(input_ids)
            
            # Create labels (shifted for next token prediction)
            labels = input_ids[1:] + [-100]
            batch_labels.append(labels)
        
        # Convert to tensors
        input_tensor = torch.tensor(batch_inputs, device=model.device)
        labels_tensor = torch.tensor(batch_labels, device=model.device)
        
        # Test forward pass
        try:
            with torch.no_grad():
                outputs = model(input_ids=input_tensor)
                
                # Verify output shape
                expected_shape = (batch_size, seq_len, len(tokenizer))
                self.assertEqual(outputs.logits.shape, expected_shape)
                
                # Test coordinate loss computation
                losses = token_manager.compute_coordinate_losses(outputs.logits, labels_tensor)
                
                # Verify losses
                self.assertIn("coordinate_loss", losses)
                self.assertIn("total_coordinate_tokens", losses)
                
                coordinate_loss = losses["coordinate_loss"]
                total_coord_tokens = losses["total_coordinate_tokens"]
                
                # Should find coordinate tokens in the batch
                self.assertGreater(total_coord_tokens, 0, "Should find coordinate tokens in batch")
                self.assertIsInstance(coordinate_loss, torch.Tensor)
                
            # Check memory usage after processing
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 1GB for this test)
            max_reasonable_increase = 1024 * 1024 * 1024  # 1GB
            self.assertLess(
                memory_increase, max_reasonable_increase,
                f"Memory increase should be reasonable: {memory_increase / (1024**2):.1f} MB"
            )
            
            logger.info(f"Memory usage: initial={initial_memory / (1024**2):.1f} MB, "
                      f"final={final_memory / (1024**2):.1f} MB, "
                      f"increase={memory_increase / (1024**2):.1f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest(f"Insufficient GPU memory for batch test: {e}")
            else:
                raise
        
        logger.info("✅ Memory usage with large coordinate batches test passed")

    def test_coordinate_token_corruption_recovery(self):
        """Test recovery from coordinate token corruption scenarios."""
        logger.info("🔬 Testing coordinate token corruption recovery...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        max_coord_value = 50
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Verify token manager is working
        self.assertTrue(token_manager.has_coordinate_tokens())
        
        # Test static detection methods still work
        has_tokens = has_coordinate_tokens_static(tokenizer)
        self.assertTrue(has_tokens, "Static detection should work")
        
        start_id, end_id = get_coordinate_token_range_static(tokenizer, max_coord_value)
        self.assertIsNotNone(start_id)
        self.assertIsNotNone(end_id)
        self.assertEqual(end_id - start_id, max_coord_value)
        
        # Test that we can create a new token manager with the same tokenizer
        # (simulating recovery after some corruption)
        try:
            recovery_token_manager = UnifiedTokenManager(
                tokenizer=tokenizer,
                model=model,
                max_coord_value=max_coord_value,
            )
            
            # Should detect existing coordinate tokens and not add duplicates
            recovery_vocab_size = len(tokenizer.get_vocab())
            
            # Verify recovery manager works
            self.assertTrue(recovery_token_manager.has_coordinate_tokens())
            
            recovery_start, recovery_end = recovery_token_manager.get_coordinate_token_range()
            self.assertEqual(recovery_start, start_id)
            self.assertEqual(recovery_end, end_id)
            
            # Test coordinate functionality in recovery manager
            for i in range(min(5, max_coord_value)):
                token_id = recovery_token_manager.get_coordinate_token_id(i)
                expected_id = start_id + i
                self.assertEqual(token_id, expected_id)
            
        except Exception as e:
            self.fail(f"Recovery token manager creation should not fail: {e}")
        
        # Test detection with fresh tokenizer (should not find coordinate tokens)
        fresh_tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        has_tokens_fresh = has_coordinate_tokens_static(fresh_tokenizer)
        self.assertFalse(has_tokens_fresh, "Fresh tokenizer should not have coordinate tokens")
        
        # Test error handling for invalid max_coord_value in detection
        try:
            invalid_start, invalid_end = get_coordinate_token_range_static(fresh_tokenizer, max_coord_value)
            # If no coordinate tokens, should return None or raise appropriate error
            if invalid_start is not None or invalid_end is not None:
                self.fail("Should not find coordinate tokens in fresh tokenizer")
        except (ValueError, RuntimeError):
            # This is expected when no coordinate tokens are present
            pass
        
        logger.info("✅ Coordinate token corruption recovery test passed")

    def test_edge_case_loss_computation_scenarios(self):
        """Test edge cases in coordinate loss computation."""
        logger.info("🔬 Testing edge case loss computation scenarios...")
        
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for loss computation test")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).cuda()
        
        max_coord_value = 30
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        vocab_size = len(tokenizer)
        
        # Test case 1: No coordinate tokens in input
        text_only_input = tokenizer.encode("Hello world", add_special_tokens=False)
        text_input_tensor = torch.tensor([text_only_input], device=model.device)
        text_labels_tensor = torch.roll(text_input_tensor, -1)
        text_labels_tensor[:, -1] = -100
        
        text_logits = torch.randn((1, len(text_only_input), vocab_size), device=model.device)
        
        no_coord_losses = token_manager.compute_coordinate_losses(text_logits, text_labels_tensor)
        
        self.assertEqual(no_coord_losses["total_coordinate_tokens"], 0)
        self.assertEqual(no_coord_losses["coordinate_loss"].item(), 0.0)
        
        # Test case 2: All coordinate tokens (extreme case)
        all_coord_input = [coord_start_id + i for i in range(min(10, max_coord_value))]
        all_coord_tensor = torch.tensor([all_coord_input], device=model.device)
        all_coord_labels = torch.roll(all_coord_tensor, -1)
        all_coord_labels[:, -1] = -100
        
        all_coord_logits = torch.randn((1, len(all_coord_input), vocab_size), device=model.device)
        
        all_coord_losses = token_manager.compute_coordinate_losses(all_coord_logits, all_coord_labels)
        
        self.assertEqual(all_coord_losses["total_coordinate_tokens"], len(all_coord_input) - 1)  # -1 for masked token
        self.assertGreater(all_coord_losses["coordinate_loss"].item(), 0.0)
        
        # Test case 3: Mixed input with padding/masking
        mixed_input = (
            tokenizer.encode("Object", add_special_tokens=False)[:2] +  # Regular tokens
            [coord_start_id, coord_start_id + 1, coord_start_id + 2] +  # Coordinate tokens
            [tokenizer.pad_token_id] * 3  # Padding tokens
        )
        mixed_tensor = torch.tensor([mixed_input], device=model.device)
        mixed_labels = torch.roll(mixed_tensor, -1)
        mixed_labels[:, -1] = -100
        # Mask padding tokens in labels
        for i in range(len(mixed_input)):
            if mixed_input[i] == tokenizer.pad_token_id:
                mixed_labels[:, i] = -100
        
        mixed_logits = torch.randn((1, len(mixed_input), vocab_size), device=model.device)
        
        mixed_losses = token_manager.compute_coordinate_losses(mixed_logits, mixed_labels)
        
        # Should find coordinate tokens but not padding tokens
        expected_coord_tokens = 2  # 3 coordinate tokens, but 1 is shifted/masked
        self.assertEqual(mixed_losses["total_coordinate_tokens"], expected_coord_tokens)
        
        # Test case 4: Empty input
        empty_input = torch.tensor([[]], device=model.device, dtype=torch.long)
        empty_labels = torch.tensor([[]], device=model.device, dtype=torch.long)
        empty_logits = torch.randn((1, 0, vocab_size), device=model.device)
        
        empty_losses = token_manager.compute_coordinate_losses(empty_logits, empty_labels)
        
        self.assertEqual(empty_losses["total_coordinate_tokens"], 0)
        self.assertEqual(empty_losses["coordinate_loss"].item(), 0.0)
        
        # Test case 5: Large batch with varying coordinate token counts
        batch_size = 5
        max_seq_len = 15
        
        batch_inputs = []
        batch_labels = []
        
        for batch_idx in range(batch_size):
            # Create sequence with different numbers of coordinate tokens
            seq_len = 10 + batch_idx  # Varying sequence lengths
            input_ids = []
            
            # Add some regular tokens
            regular_tokens = tokenizer.encode("Test", add_special_tokens=False)
            input_ids.extend(regular_tokens[:3])
            
            # Add varying numbers of coordinate tokens
            num_coord_tokens = batch_idx + 1  # 1 to 5 coordinate tokens
            for i in range(num_coord_tokens):
                coord_token_id = coord_start_id + (i % max_coord_value)
                input_ids.append(coord_token_id)
            
            # Pad to max_seq_len
            while len(input_ids) < max_seq_len:
                input_ids.append(tokenizer.pad_token_id)
            
            batch_inputs.append(input_ids[:max_seq_len])
            
            # Create labels
            labels = input_ids[1:max_seq_len] + [-100]
            # Mask padding in labels
            for i, token_id in enumerate(input_ids):
                if token_id == tokenizer.pad_token_id:
                    labels[i] = -100
            
            batch_labels.append(labels)
        
        batch_input_tensor = torch.tensor(batch_inputs, device=model.device)
        batch_labels_tensor = torch.tensor(batch_labels, device=model.device)
        batch_logits = torch.randn((batch_size, max_seq_len, vocab_size), device=model.device)
        
        batch_losses = token_manager.compute_coordinate_losses(batch_logits, batch_labels_tensor)
        
        # Should find coordinate tokens across the batch
        self.assertGreater(batch_losses["total_coordinate_tokens"], 0)
        self.assertGreater(batch_losses["coordinate_loss"].item(), 0.0)
        
        logger.info("✅ Edge case loss computation scenarios test passed")


if __name__ == "__main__":
    unittest.main()