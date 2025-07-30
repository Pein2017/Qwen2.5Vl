#!/usr/bin/env python3
"""
Coordinate Token Conversion Tests

Tests for conversion between standard mode (geometry tokens only) and 
coordinate mode (geometry + coordinate tokens) to validate the 
coordinate token vocabulary-embedding mismatch fix.

Key test scenarios:
- Standard mode → Coordinate mode conversion
- Coordinate mode → Standard mode conversion (if needed)
- Conversion validation and error handling
- Edge cases in mode switching
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import shutil

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

logger = get_logger("test_coordinate_token_conversion")


class TestCoordinateTokenConversion(GPUAwareTestCase):
    """Tests for coordinate token mode conversion and validation."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        super().setUpClass()
        configure_global_logging(rank=0, world_size=1)

        # Create test utilities
        cls.test_utils = TestUtils()
        cls.config_factory = ConfigFactory()
        cls.data_generator = SyntheticDataGenerator(num_samples=3)
        cls.temp_dir = tempfile.mkdtemp(prefix="coordinate_conversion_tests_")
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

    def test_fresh_tokenizer_to_standard_mode_conversion(self):
        """Test conversion from fresh tokenizer to standard mode (geometry tokens only)."""
        logger.info("🔬 Testing fresh tokenizer to standard mode conversion...")
        
        # Load fresh tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Record original state
        original_vocab = tokenizer.get_vocab().copy()
        original_vocab_size = len(original_vocab)
        original_model_size = model.get_input_embeddings().weight.shape[0]
        
        # Verify no coordinate tokens initially
        has_coord_tokens_initial = has_coordinate_tokens_static(tokenizer)
        self.assertFalse(
            has_coord_tokens_initial,
            "Fresh tokenizer should not have coordinate tokens"
        )
        
        # Convert to standard mode (add geometry tokens only)
        token_manager_std = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=0,  # No coordinate tokens
        )
        
        # Verify conversion to standard mode
        std_vocab = tokenizer.get_vocab()
        std_vocab_size = len(std_vocab)
        std_model_size = model.get_input_embeddings().weight.shape[0]
        
        # Check vocabulary size increased by 4 (geometry tokens)
        expected_std_vocab_size = original_vocab_size + 4
        self.assertEqual(
            std_vocab_size, expected_std_vocab_size,
            f"Standard mode vocabulary should increase by 4: "
            f"original={original_vocab_size}, std={std_vocab_size}, expected={expected_std_vocab_size}"
        )
        
        # Check model size matches vocabulary
        self.assertEqual(
            std_model_size, std_vocab_size,
            f"Standard mode model size should match vocabulary: model={std_model_size}, vocab={std_vocab_size}"
        )
        
        # Verify geometry tokens are present
        geometry_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
        for token in geometry_tokens:
            self.assertIn(
                token, std_vocab,
                f"Geometry token {token} should be in standard mode vocabulary"
            )
        
        # Verify still no coordinate tokens
        has_coord_tokens_after = has_coordinate_tokens_static(tokenizer)
        self.assertFalse(
            has_coord_tokens_after,
            "Standard mode should not have coordinate tokens"
        )
        
        # Test geometry token functionality
        for token in geometry_tokens:
            token_id = token_manager_std.get_token_id(token.replace("<|", "").replace("|>", ""))
            self.assertIsInstance(token_id, int)
            self.assertGreater(token_id, 0)
        
        logger.info("✅ Fresh to standard mode conversion test passed")

    def test_standard_mode_to_coordinate_mode_conversion(self):
        """Test conversion from standard mode to coordinate mode."""
        logger.info("🔬 Testing standard mode to coordinate mode conversion...")
        
        # Step 1: Create model in standard mode
        tokenizer_std = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model_std = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Add geometry tokens only (standard mode)
        token_manager_std = UnifiedTokenManager(
            tokenizer=tokenizer_std,
            model=model_std,
            max_coord_value=0,
        )
        
        # Record standard mode state
        std_vocab = tokenizer_std.get_vocab().copy()
        std_vocab_size = len(std_vocab)
        std_model_size = model_std.get_input_embeddings().weight.shape[0]
        
        # Verify standard mode characteristics
        has_coord_tokens_std = has_coordinate_tokens_static(tokenizer_std)
        self.assertFalse(
            has_coord_tokens_std,
            "Standard mode should not have coordinate tokens"
        )
        
        # Step 2: Convert to coordinate mode
        max_coord_value = 100
        
        # Create new tokenizer and model for coordinate mode
        tokenizer_coord = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model_coord = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # First add geometry tokens (simulate standard mode state)
        token_manager_pre_coord = UnifiedTokenManager(
            tokenizer=tokenizer_coord,
            model=model_coord,
            max_coord_value=0,
        )
        
        # Then add coordinate tokens
        token_manager_coord = UnifiedTokenManager(
            tokenizer=tokenizer_coord,
            model=model_coord,
            max_coord_value=max_coord_value,
        )
        
        # Verify coordinate mode characteristics
        coord_vocab = tokenizer_coord.get_vocab()
        coord_vocab_size = len(coord_vocab)
        coord_model_size = model_coord.get_input_embeddings().weight.shape[0]
        
        # Check vocabulary size increased by coordinate tokens
        expected_coord_vocab_size = std_vocab_size + max_coord_value
        self.assertEqual(
            coord_vocab_size, expected_coord_vocab_size,
            f"Coordinate mode vocabulary should be larger: "
            f"std={std_vocab_size}, coord={coord_vocab_size}, expected={expected_coord_vocab_size}"
        )
        
        # Check model size matches vocabulary
        self.assertEqual(
            coord_model_size, coord_vocab_size,
            f"Coordinate mode model size should match vocabulary: model={coord_model_size}, vocab={coord_vocab_size}"
        )
        
        # Verify coordinate tokens are present
        has_coord_tokens_coord = has_coordinate_tokens_static(tokenizer_coord)
        self.assertTrue(
            has_coord_tokens_coord,
            "Coordinate mode should have coordinate tokens"
        )
        
        # Verify all standard tokens still exist
        missing_std_tokens = []
        for token in std_vocab.keys():
            if token not in coord_vocab:
                missing_std_tokens.append(token)
        
        self.assertEqual(
            len(missing_std_tokens), 0,
            f"Standard tokens should be preserved in coordinate mode: {missing_std_tokens[:5]}..."
        )
        
        # Verify coordinate tokens are functional
        coord_start_id, coord_end_id = token_manager_coord.get_coordinate_token_range()
        self.assertEqual(
            coord_end_id - coord_start_id, max_coord_value,
            f"Coordinate token range should be {max_coord_value}"
        )
        
        # Test coordinate token retrieval
        for i in range(min(10, max_coord_value)):
            token_id = token_manager_coord.get_coordinate_token_id(i)
            expected_id = coord_start_id + i
            self.assertEqual(
                token_id, expected_id,
                f"Coordinate token {i} should have sequential ID: {token_id} vs {expected_id}"
            )
        
        logger.info(f"✅ Standard to coordinate mode conversion test passed with {max_coord_value} coordinate tokens")

    def test_coordinate_mode_functionality_validation(self):
        """Test coordinate mode functionality after conversion."""
        logger.info("🔬 Testing coordinate mode functionality after conversion...")
        
        # Create model in coordinate mode
        max_coord_value = 75
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Add coordinate tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Test coordinate wrapping functionality
        test_coordinates = [
            {"coords": [10, 20, 30, 40], "geometry": "bbox"},
            {"coords": [5, 15, 25, 35, 45, 55], "geometry": "line"},
            {"coords": [10, 20, 30, 20, 30, 40, 10, 40], "geometry": "square"},
        ]
        
        for test_case in test_coordinates:
            coords = test_case["coords"]
            geometry = test_case["geometry"]
            
            with self.subTest(geometry=geometry):
                # Test coordinate wrapping
                wrapped = token_manager.wrap_coordinates(coords, geometry)
                
                # Verify format
                geometry_name = geometry if geometry != "bbox" else "box"
                self.assertIn(f"<|{geometry_name}_start|>", wrapped)
                self.assertIn(f"<|{geometry_name}_end|>", wrapped)
                
                # Verify coordinate tokens are used
                for coord in coords:
                    coord_clamped = max(0, min(int(coord), max_coord_value - 1))
                    coord_token = f"<|coord_{coord_clamped}|>"
                    self.assertIn(coord_token, wrapped)
        
        # Test object formatting functionality
        test_objects = [
            {"bbox_2d": [15, 25, 35, 45], "desc": "Test object in coordinate mode"},
            {"line": [10, 20, 30, 40, 50, 60], "desc": "Test line in coordinate mode"},
        ]
        
        for obj in test_objects:
            with self.subTest(obj_type=list(obj.keys())[0]):
                formatted = token_manager.format_object(obj)
                
                # Verify object structure
                self.assertIn("<|object_ref_start|>", formatted)
                self.assertIn("<|object_ref_end|>", formatted)
                self.assertIn("desc:", formatted)
                self.assertIn(obj["desc"], formatted)
                
                # Verify coordinate tokens are present
                geometry_key = list(obj.keys())[0]
                for coord in obj[geometry_key]:
                    coord_clamped = max(0, min(int(coord), max_coord_value - 1))
                    coord_token = f"<|coord_{coord_clamped}|>"
                    self.assertIn(coord_token, formatted)
        
        logger.info("✅ Coordinate mode functionality validation passed")

    def test_chat_processor_mode_conversion_integration(self):
        """Test ChatProcessor integration with mode conversion."""
        logger.info("🔬 Testing ChatProcessor mode conversion integration...")
        
        # Load image processor
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        # Test 1: Standard mode ChatProcessor
        test_config_std = self.config_factory.create_test_config(
            coordinate_tokens_enabled=False,
            max_coord_value=0,
            language="en"
        )
        
        tokenizer_std = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        chat_processor_std = ChatProcessor(
            tokenizer=tokenizer_std,
            image_processor=image_processor,
            config=test_config_std,
            coordinate_tokens_enabled=False,
            max_coord_value=0,
            language="en"
        )
        
        # Initialize standard mode (should not have coordinate manager)
        chat_processor_std._update_coordinate_token_ranges()
        
        # Should not have coordinate manager or should indicate no coordinate tokens
        if chat_processor_std.coordinate_manager is not None:
            self.assertFalse(
                chat_processor_std.coordinate_manager.has_coordinate_tokens(),
                "Standard mode ChatProcessor should not have coordinate tokens"
            )
        
        # Test 2: Convert to coordinate mode
        max_coord_value = 80
        test_config_coord = self.config_factory.create_test_config(
            coordinate_tokens_enabled=True,
            max_coord_value=max_coord_value,
            language="en"
        )
        
        tokenizer_coord = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        chat_processor_coord = ChatProcessor(
            tokenizer=tokenizer_coord,
            image_processor=image_processor,
            config=test_config_coord,
            coordinate_tokens_enabled=True,
            max_coord_value=max_coord_value,
            language="en"
        )
        
        # Initialize coordinate mode
        chat_processor_coord._update_coordinate_token_ranges()
        
        # Should have coordinate manager
        self.assertIsNotNone(
            chat_processor_coord.coordinate_manager,
            "Coordinate mode ChatProcessor should have coordinate manager"
        )
        
        self.assertTrue(
            chat_processor_coord.coordinate_manager.has_coordinate_tokens(),
            "Coordinate mode ChatProcessor should have coordinate tokens"
        )
        
        # Test coordinate functionality
        test_objects_coord = [
            {"bbox_2d": [20, 30, 40, 50], "desc": "Test object with coordinates"},
            {"line": [10, 15, 20, 25, 30, 35], "desc": "Test line with coordinates"},
        ]
        
        formatted_response = chat_processor_coord._format_objects_response(test_objects_coord)
        
        # Verify coordinate tokens are used
        import re
        coord_token_pattern = r'<\|coord_\d+\|>'
        coord_tokens = re.findall(coord_token_pattern, formatted_response)
        
        self.assertGreater(
            len(coord_tokens), 0,
            f"Coordinate mode should generate coordinate tokens: {formatted_response[:200]}..."
        )
        
        # Verify coordinate token values are within range
        for coord_token in coord_tokens:
            coord_match = re.match(r'<\|coord_(\d+)\|>', coord_token)
            if coord_match:
                coord_value = int(coord_match.group(1))
                self.assertGreaterEqual(coord_value, 0)
                self.assertLess(coord_value, max_coord_value)
        
        logger.info(f"✅ ChatProcessor mode conversion integration test passed with {len(coord_tokens)} coordinate tokens")

    def test_mode_conversion_error_handling(self):
        """Test error handling during mode conversion."""
        logger.info("🔬 Testing mode conversion error handling...")
        
        # Test 1: Invalid max_coord_value during conversion
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Test invalid negative max_coord_value
        with self.assertRaises(ValueError) as context:
            UnifiedTokenManager(
                tokenizer=tokenizer,
                model=model,
                max_coord_value=-10,  # Invalid
            )
        
        self.assertIn("max_coord_value", str(context.exception).lower())
        
        # Test 2: Very large max_coord_value (should warn or fail gracefully)
        try:
            large_token_manager = UnifiedTokenManager(
                tokenizer=tokenizer,
                model=model,
                max_coord_value=5000,  # Very large - might succeed but should be logged
            )
            logger.warning("Large coordinate value conversion succeeded - consider adding validation")
            
            # If it succeeds, verify it's functional
            self.assertTrue(large_token_manager.has_coordinate_tokens())
            
        except (ValueError, RuntimeError, MemoryError) as e:
            logger.info(f"Large coordinate value properly rejected: {e}")
        
        # Test 3: Conversion with corrupted tokenizer state
        # Create valid token manager first
        valid_token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=50,
        )
        
        # Verify it works
        self.assertTrue(valid_token_manager.has_coordinate_tokens())
        
        # Test coordinate access errors
        with self.assertRaises(ValueError):
            valid_token_manager.get_coordinate_token_id(-1)
        
        with self.assertRaises(ValueError):
            valid_token_manager.get_coordinate_token_id(50)  # Out of range
        
        # Test 4: Mode conversion consistency checks
        coord_start_id, coord_end_id = valid_token_manager.get_coordinate_token_range()
        
        # Verify range is correct
        self.assertEqual(
            coord_end_id - coord_start_id, 50,
            "Coordinate token range should match max_coord_value"
        )
        
        # Verify static detection methods work
        has_tokens = has_coordinate_tokens_static(tokenizer)
        self.assertTrue(has_tokens, "Static detection should find coordinate tokens")
        
        static_start, static_end = get_coordinate_token_range_static(tokenizer, 50)
        self.assertEqual(static_start, coord_start_id)
        self.assertEqual(static_end, coord_end_id)
        
        logger.info("✅ Mode conversion error handling test passed")

    def test_vocabulary_consistency_across_conversions(self):
        """Test vocabulary consistency across multiple mode conversions."""
        logger.info("🔬 Testing vocabulary consistency across conversions...")
        
        # Step 1: Fresh tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        original_vocab = tokenizer.get_vocab().copy()
        original_vocab_size = len(original_vocab)
        
        # Step 2: Convert to standard mode
        token_manager_std = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=0,
        )
        
        std_vocab = tokenizer.get_vocab().copy()
        std_vocab_size = len(std_vocab)
        
        # Verify standard mode adds 4 geometry tokens
        self.assertEqual(
            std_vocab_size, original_vocab_size + 4,
            f"Standard mode should add 4 geometry tokens: {std_vocab_size} vs {original_vocab_size + 4}"
        )
        
        # Step 3: Convert to coordinate mode
        max_coord_value = 60
        token_manager_coord = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        coord_vocab = tokenizer.get_vocab().copy()
        coord_vocab_size = len(coord_vocab)
        
        # Verify coordinate mode adds coordinate tokens
        expected_coord_vocab_size = std_vocab_size + max_coord_value
        self.assertEqual(
            coord_vocab_size, expected_coord_vocab_size,
            f"Coordinate mode should add {max_coord_value} coordinate tokens: {coord_vocab_size} vs {expected_coord_vocab_size}"
        )
        
        # Step 4: Verify all original tokens are preserved
        missing_original_tokens = []
        for token in original_vocab.keys():
            if token not in coord_vocab:
                missing_original_tokens.append(token)
        
        self.assertEqual(
            len(missing_original_tokens), 0,
            f"Original tokens should be preserved: {missing_original_tokens[:10]}..."
        )
        
        # Step 5: Verify all standard mode tokens are preserved
        missing_std_tokens = []
        for token in std_vocab.keys():
            if token not in coord_vocab:
                missing_std_tokens.append(token)
        
        self.assertEqual(
            len(missing_std_tokens), 0,
            f"Standard mode tokens should be preserved: {missing_std_tokens[:10]}..."
        )
        
        # Step 6: Verify vocabulary integrity
        # Check that all coordinate tokens are present
        missing_coord_tokens = []
        for i in range(max_coord_value):
            coord_token = f"<|coord_{i}|>"
            if coord_token not in coord_vocab:
                missing_coord_tokens.append(coord_token)
        
        self.assertEqual(
            len(missing_coord_tokens), 0,
            f"All coordinate tokens should be present: {missing_coord_tokens[:10]}..."
        )
        
        # Check that all geometry tokens are present
        geometry_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
        missing_geometry_tokens = []
        for token in geometry_tokens:
            if token not in coord_vocab:
                missing_geometry_tokens.append(token)
        
        self.assertEqual(
            len(missing_geometry_tokens), 0,
            f"All geometry tokens should be present: {missing_geometry_tokens}"
        )
        
        # Step 7: Verify model embedding size matches vocabulary
        final_model_size = model.get_input_embeddings().weight.shape[0]
        self.assertEqual(
            final_model_size, coord_vocab_size,
            f"Final model size should match vocabulary: model={final_model_size}, vocab={coord_vocab_size}"
        )
        
        logger.info(f"✅ Vocabulary consistency test passed: {original_vocab_size} → {std_vocab_size} → {coord_vocab_size}")

    def test_embedding_weight_preservation_during_conversion(self):
        """Test that embedding weights are properly preserved during mode conversion."""
        logger.info("🔬 Testing embedding weight preservation during conversion...")
        
        # Load fresh model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Get original embedding weights for a few tokens
        original_vocab = tokenizer.get_vocab()
        test_token_ids = list(original_vocab.values())[:10]  # First 10 original tokens
        
        embedding_layer = model.get_input_embeddings()
        original_embeddings = embedding_layer.weight[test_token_ids].detach().clone()
        
        # Convert to standard mode
        token_manager_std = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=0,
        )
        
        # Check that original embeddings are preserved
        std_embeddings = embedding_layer.weight[test_token_ids].detach()
        original_preserved = torch.allclose(original_embeddings, std_embeddings, rtol=1e-6, atol=1e-6)
        
        self.assertTrue(
            original_preserved,
            "Original token embeddings should be preserved in standard mode"
        )
        
        # Get geometry token embeddings
        geometry_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
        geometry_token_ids = []
        std_vocab = tokenizer.get_vocab()
        
        for token in geometry_tokens:
            if token in std_vocab:
                geometry_token_ids.append(std_vocab[token])
        
        geometry_embeddings_std = embedding_layer.weight[geometry_token_ids].detach().clone()
        
        # Convert to coordinate mode
        max_coord_value = 40
        token_manager_coord = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Check that both original and geometry embeddings are preserved
        coord_embeddings_original = embedding_layer.weight[test_token_ids].detach()
        coord_embeddings_geometry = embedding_layer.weight[geometry_token_ids].detach()
        
        original_still_preserved = torch.allclose(original_embeddings, coord_embeddings_original, rtol=1e-6, atol=1e-6)
        geometry_preserved = torch.allclose(geometry_embeddings_std, coord_embeddings_geometry, rtol=1e-6, atol=1e-6)
        
        self.assertTrue(
            original_still_preserved,
            "Original token embeddings should be preserved in coordinate mode"
        )
        
        self.assertTrue(
            geometry_preserved,
            "Geometry token embeddings should be preserved in coordinate mode"
        )
        
        # Check that coordinate token embeddings are initialized (not zero)
        coord_start_id, coord_end_id = token_manager_coord.get_coordinate_token_range()
        coord_token_embeddings = embedding_layer.weight[coord_start_id:coord_end_id].detach()
        
        is_all_zeros = torch.all(coord_token_embeddings == 0.0)
        self.assertFalse(
            is_all_zeros,
            "Coordinate token embeddings should be initialized (not all zeros)"
        )
        
        # Check coordinate embeddings have reasonable variance
        coord_embedding_std = torch.std(coord_token_embeddings)
        self.assertGreater(
            coord_embedding_std.item(), 1e-6,
            f"Coordinate token embeddings should have reasonable variance: {coord_embedding_std.item()}"
        )
        
        logger.info("✅ Embedding weight preservation test passed")


if __name__ == "__main__":
    unittest.main()