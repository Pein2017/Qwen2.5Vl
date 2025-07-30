#!/usr/bin/env python3
"""
Comprehensive Coordinate Token Persistence Tests

Tests cover the complete coordinate token lifecycle:
- Token creation and vocabulary integration
- Checkpoint saving and loading with coordinate tokens
- Embedding persistence across save/load cycles
- Conversion between standard and coordinate modes
- Edge cases and error scenarios
- Integration with training pipeline components

This test suite validates the fix for coordinate token vocabulary-embedding mismatch issues.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F

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
    create_unified_token_manager,
    has_coordinate_tokens_static,
    get_coordinate_token_range_static,
)
from src.chat_processor import ChatProcessor
from src.models.wrapper import BBUModelWrapper
from tests.fixtures import SyntheticDataGenerator, ConfigFactory
from tests.fixtures.gpu_test_base import GPUAwareTestCase
from tests.fixtures.test_utils import TestUtils

logger = get_logger("test_coordinate_token_persistence")


class TestCoordinateTokenPersistence(GPUAwareTestCase):
    """Comprehensive tests for coordinate token persistence and validation."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        super().setUpClass()
        configure_global_logging(rank=0, world_size=1)

        # Create test utilities
        cls.test_utils = TestUtils()
        cls.config_factory = ConfigFactory()
        cls.data_generator = SyntheticDataGenerator(num_samples=5)
        cls.temp_dir = tempfile.mkdtemp(prefix="coordinate_persistence_tests_")
        cls.test_files_to_cleanup = []

        # Load base model and tokenizer for testing
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

    def test_coordinate_token_vocabulary_persistence(self):
        """Test that coordinate tokens persist correctly in tokenizer vocabulary."""
        logger.info("🔬 Testing coordinate token vocabulary persistence...")
        
        # Load fresh tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Record original vocabulary state
        original_vocab = tokenizer.get_vocab().copy()
        original_vocab_size = len(original_vocab)
        
        # Create token manager with coordinate tokens
        max_coord_value = 100  # Use smaller value for faster testing
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Verify coordinate tokens were added
        updated_vocab = tokenizer.get_vocab()
        updated_vocab_size = len(updated_vocab)
        
        # Check vocabulary size increased correctly
        expected_new_tokens = max_coord_value + 4  # coordinate tokens + geometry tokens
        expected_vocab_size = original_vocab_size + expected_new_tokens
        
        self.assertEqual(
            updated_vocab_size, expected_vocab_size,
            f"Vocabulary size should increase by {expected_new_tokens}: "
            f"original={original_vocab_size}, updated={updated_vocab_size}, expected={expected_vocab_size}"
        )
        
        # Verify all coordinate tokens are present
        missing_tokens = []
        for i in range(max_coord_value):
            coord_token = f"<|coord_{i}|>"
            if coord_token not in updated_vocab:
                missing_tokens.append(coord_token)
        
        self.assertEqual(
            len(missing_tokens), 0,
            f"Missing coordinate tokens from vocabulary: {missing_tokens[:10]}..."
        )
        
        # Verify geometry tokens are present
        geometry_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
        for token in geometry_tokens:
            self.assertIn(
                token, updated_vocab,
                f"Geometry token {token} should be in vocabulary"
            )
        
        # Test coordinate token ID retrieval consistency
        for i in range(min(10, max_coord_value)):  # Test first 10 tokens
            expected_id = updated_vocab[f"<|coord_{i}|>"]
            retrieved_id = token_manager.get_coordinate_token_id(i)
            self.assertEqual(
                expected_id, retrieved_id,
                f"Coordinate token ID mismatch for coord_{i}: vocab={expected_id}, manager={retrieved_id}"
            )
        
        logger.info(f"✅ Vocabulary persistence test passed with {max_coord_value} coordinate tokens")

    def test_checkpoint_save_load_cycle_persistence(self):
        """Test coordinate token persistence through complete checkpoint save/load cycle."""
        logger.info("🔬 Testing checkpoint save/load cycle persistence...")
        
        # Create coordinate configuration
        max_coord_value = 50  # Smaller for faster testing
        config_path = self.config_factory.create_coordinate_enabled_config_with_value(
            Path(self.test_dir), "coordinate", max_coord_value
        )
        self.test_files_to_cleanup.append(config_path)
        
        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Create token manager and add coordinate tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Record state before saving
        original_vocab = tokenizer.get_vocab().copy()
        original_vocab_size = len(original_vocab)
        
        # Get embedding weights for coordinate tokens
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        embedding_layer = model.get_input_embeddings()
        original_coord_embeddings = embedding_layer.weight[coord_start_id:coord_end_id].detach().clone()
        
        # Save model and tokenizer
        save_dir = os.path.join(self.test_dir, "checkpoint_test")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save both model and tokenizer
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Load saved model and tokenizer
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            save_dir, trust_remote_code=True
        )
        loaded_model = AutoModel.from_pretrained(
            save_dir, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Verify vocabulary persistence
        loaded_vocab = loaded_tokenizer.get_vocab()
        loaded_vocab_size = len(loaded_vocab)
        
        self.assertEqual(
            loaded_vocab_size, original_vocab_size,
            f"Loaded vocabulary size should match original: "
            f"original={original_vocab_size}, loaded={loaded_vocab_size}"
        )
        
        # Verify all coordinate tokens persist
        missing_after_load = []
        for i in range(max_coord_value):
            coord_token = f"<|coord_{i}|>"
            if coord_token not in loaded_vocab:
                missing_after_load.append(coord_token)
        
        self.assertEqual(
            len(missing_after_load), 0,
            f"Coordinate tokens missing after load: {missing_after_load[:10]}..."
        )
        
        # Verify token ID consistency
        id_mismatches = []
        for i in range(min(10, max_coord_value)):  # Test first 10 tokens
            coord_token = f"<|coord_{i}|>"
            original_id = original_vocab[coord_token]
            loaded_id = loaded_vocab[coord_token]
            
            if original_id != loaded_id:
                id_mismatches.append((coord_token, original_id, loaded_id))
        
        self.assertEqual(
            len(id_mismatches), 0,
            f"Coordinate token ID mismatches after load: {id_mismatches}"
        )
        
        # Verify embedding weights persistence
        loaded_embedding_layer = loaded_model.get_input_embeddings()
        loaded_coord_embeddings = loaded_embedding_layer.weight[coord_start_id:coord_end_id].detach()
        
        embeddings_match = torch.allclose(
            original_coord_embeddings, loaded_coord_embeddings, rtol=1e-3, atol=1e-3
        )
        
        self.assertTrue(
            embeddings_match,
            "Coordinate token embeddings should be preserved after save/load cycle"
        )
        
        logger.info(f"✅ Checkpoint save/load persistence test passed with {max_coord_value} coordinate tokens")

    def test_coordinate_token_model_wrapper_integration(self):
        """Test coordinate token persistence with BBUModelWrapper integration."""
        logger.info("🔬 Testing coordinate token model wrapper integration...")
        
        # Create coordinate configuration
        max_coord_value = 75
        config_path = self.config_factory.create_coordinate_enabled_config_with_value(
            Path(self.test_dir), "coordinate", max_coord_value
        )
        self.test_files_to_cleanup.append(config_path)
        
        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)
        
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        base_model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Create token manager to add coordinate tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=base_model,
            max_coord_value=max_coord_value,
        )
        
        # Create wrapper model
        wrapper_model = BBUModelWrapper(
            base_model=base_model,
            tokenizer=tokenizer,
            config=config
        )
        
        # Verify wrapper model has correct vocabulary size
        wrapper_vocab_size = wrapper_model.get_input_embeddings().weight.shape[0]
        tokenizer_vocab_size = len(tokenizer.get_vocab())
        
        self.assertEqual(
            wrapper_vocab_size, tokenizer_vocab_size,
            f"Wrapper model embedding size should match tokenizer vocab size: "
            f"wrapper={wrapper_vocab_size}, tokenizer={tokenizer_vocab_size}"
        )
        
        # Test coordinate token detection through wrapper
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        
        # Verify coordinate tokens are accessible through wrapper
        for i in range(min(5, max_coord_value)):  # Test first 5 tokens
            coord_token_id = token_manager.get_coordinate_token_id(i)
            expected_id = coord_start_id + i
            
            self.assertEqual(
                coord_token_id, expected_id,
                f"Coordinate token ID should be sequential: coord_{i} = {coord_token_id}, expected = {expected_id}"
            )
        
        # Test forward pass with coordinate tokens (minimal test)
        if torch.cuda.is_available():
            wrapper_model = wrapper_model.cuda()
            
            # Create minimal input with coordinate tokens
            test_input_ids = torch.tensor([[
                coord_start_id,  # First coordinate token
                coord_start_id + 1,  # Second coordinate token
                tokenizer.eos_token_id  # End token
            ]], device='cuda')
            
            # Test forward pass
            with torch.no_grad():
                outputs = wrapper_model(input_ids=test_input_ids)
                
            # Verify outputs have correct shape
            expected_vocab_size = len(tokenizer.get_vocab())
            self.assertEqual(
                outputs.logits.shape[-1], expected_vocab_size,
                f"Output logits should have correct vocabulary size: "
                f"got {outputs.logits.shape[-1]}, expected {expected_vocab_size}"
            )
        
        logger.info(f"✅ Model wrapper integration test passed with {max_coord_value} coordinate tokens")

    def test_coordinate_token_conversion_standard_to_coordinate(self):
        """Test conversion from standard mode (no coordinate tokens) to coordinate mode."""
        logger.info("🔬 Testing standard to coordinate mode conversion...")
        
        # Step 1: Create model in standard mode (no coordinate tokens)
        tokenizer_std = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model_std = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Add only geometry tokens (standard mode)
        token_manager_std = UnifiedTokenManager(
            tokenizer=tokenizer_std,
            model=model_std,
            max_coord_value=0,  # No coordinate tokens
        )
        
        # Record standard mode state
        std_vocab = tokenizer_std.get_vocab().copy()
        std_vocab_size = len(std_vocab)
        
        # Step 2: Convert to coordinate mode
        max_coord_value = 50
        
        # Create new tokenizer and model for coordinate mode
        tokenizer_coord = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model_coord = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Add coordinate tokens
        token_manager_coord = UnifiedTokenManager(
            tokenizer=tokenizer_coord,
            model=model_coord,
            max_coord_value=max_coord_value,
        )
        
        # Record coordinate mode state
        coord_vocab = tokenizer_coord.get_vocab()
        coord_vocab_size = len(coord_vocab)
        
        # Verify vocabulary size increased correctly
        expected_increase = max_coord_value  # Only coordinate tokens added (geometry already present)
        expected_coord_vocab_size = std_vocab_size + expected_increase
        
        self.assertEqual(
            coord_vocab_size, expected_coord_vocab_size,
            f"Coordinate mode vocabulary should be larger: "
            f"std={std_vocab_size}, coord={coord_vocab_size}, expected={expected_coord_vocab_size}"
        )
        
        # Verify all standard tokens still exist in coordinate mode
        missing_std_tokens = []
        for token, token_id in std_vocab.items():
            if token not in coord_vocab:
                missing_std_tokens.append(token)
        
        self.assertEqual(
            len(missing_std_tokens), 0,
            f"Standard tokens missing in coordinate mode: {missing_std_tokens[:10]}..."
        )
        
        # Verify coordinate tokens were added
        missing_coord_tokens = []
        for i in range(max_coord_value):
            coord_token = f"<|coord_{i}|>"
            if coord_token not in coord_vocab:
                missing_coord_tokens.append(coord_token)
        
        self.assertEqual(
            len(missing_coord_tokens), 0,
            f"Coordinate tokens missing: {missing_coord_tokens[:10]}..."
        )
        
        # Verify coordinate token functionality
        coord_start_id, coord_end_id = token_manager_coord.get_coordinate_token_range()
        self.assertEqual(
            coord_end_id - coord_start_id, max_coord_value,
            f"Coordinate token range size should be {max_coord_value}"
        )
        
        logger.info(f"✅ Standard to coordinate conversion test passed")

    def test_coordinate_token_chat_processor_integration(self):
        """Test coordinate token integration with ChatProcessor."""
        logger.info("🔬 Testing coordinate token ChatProcessor integration...")
        
        # Load image processor
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        # Create test config
        max_coord_value = 100
        test_config = self.config_factory.create_test_config(
            coordinate_tokens_enabled=True,
            max_coord_value=max_coord_value,
            language="en"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        # Create chat processor with coordinate tokens
        chat_processor = ChatProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            config=test_config,
            coordinate_tokens_enabled=True,
            max_coord_value=max_coord_value,
            language="en"
        )
        
        # Initialize coordinate manager
        chat_processor._update_coordinate_token_ranges()
        
        # Verify coordinate manager initialization
        self.assertIsNotNone(
            chat_processor.coordinate_manager,
            "ChatProcessor coordinate manager should be initialized"
        )
        
        self.assertTrue(
            chat_processor.coordinate_manager.has_coordinate_tokens(),
            "ChatProcessor should have coordinate tokens available"
        )
        
        # Test object formatting with coordinate tokens
        test_objects = [
            {"bbox_2d": [10, 20, 30, 40], "desc": "Test bounding box"},
            {"line": [5, 15, 25, 35, 45, 55], "desc": "Test line"},
            {"square": [15, 25, 35, 25, 35, 45, 15, 45], "desc": "Test square"},
        ]
        
        # Format objects using coordinate tokens
        formatted_response = chat_processor._format_objects_response(test_objects)
        
        # Verify coordinate tokens are present in response
        import re
        coord_token_pattern = r'<\|coord_\d+\|>'
        coord_tokens = re.findall(coord_token_pattern, formatted_response)
        
        self.assertGreater(
            len(coord_tokens), 0,
            f"No coordinate tokens found in formatted response: {formatted_response[:200]}..."
        )
        
        # Verify geometry tokens are present
        geometry_patterns = [
            r'<\|box_start\|>', r'<\|box_end\|>',
            r'<\|line_start\|>', r'<\|line_end\|>',
            r'<\|square_start\|>', r'<\|square_end\|>'
        ]
        
        geometry_tokens_found = [
            pattern for pattern in geometry_patterns
            if re.search(pattern, formatted_response)
        ]
        
        self.assertGreater(
            len(geometry_tokens_found), 0,
            f"No geometry tokens found in response: {formatted_response[:200]}..."
        )
        
        # Verify object structure tokens
        self.assertIn(
            '<|object_ref_start|>', formatted_response,
            "Missing object reference start token"
        )
        self.assertIn(
            '<|object_ref_end|>', formatted_response,
            "Missing object reference end token"
        )
        
        logger.info(f"✅ ChatProcessor integration test passed with {len(coord_tokens)} coordinate tokens")

    def test_coordinate_token_error_handling_edge_cases(self):
        """Test coordinate token error handling and edge cases."""
        logger.info("🔬 Testing coordinate token error handling edge cases...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Test 1: Invalid max_coord_value
        with self.assertRaises(ValueError) as context:
            UnifiedTokenManager(
                tokenizer=tokenizer,
                model=model,
                max_coord_value=-1,  # Invalid negative value
            )
        
        self.assertIn("max_coord_value", str(context.exception).lower())
        
        # Test 2: Zero max_coord_value (should work - geometry tokens only)
        token_manager_zero = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=0,
        )
        
        self.assertFalse(
            token_manager_zero.has_coordinate_tokens(),
            "Should not have coordinate tokens with max_coord_value=0"
        )
        
        # Test 3: Very large max_coord_value (should raise warning or error)
        # Note: This might succeed but should be logged as potentially problematic
        try:
            token_manager_large = UnifiedTokenManager(
                tokenizer=tokenizer,
                model=model,
                max_coord_value=10000,  # Very large value
            )
            logger.warning("Large coordinate value test succeeded - consider adding limits")
        except (ValueError, RuntimeError) as e:
            logger.info(f"Large coordinate value properly rejected: {e}")
        
        # Test 4: Coordinate token range validation
        max_coord_value = 50
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Test valid coordinate access
        for i in [0, max_coord_value // 2, max_coord_value - 1]:
            try:
                token_id = token_manager.get_coordinate_token_id(i)
                self.assertIsInstance(token_id, int)
                self.assertGreater(token_id, 0)
            except ValueError:
                self.fail(f"Valid coordinate {i} should not raise ValueError")
        
        # Test invalid coordinate access
        invalid_coords = [-1, max_coord_value, max_coord_value + 1]
        for coord in invalid_coords:
            with self.assertRaises(ValueError) as context:
                token_manager.get_coordinate_token_id(coord)
            
            self.assertIn("coordinate", str(context.exception).lower())
        
        # Test 5: Static method validation
        # Test with tokenizer that has coordinate tokens
        has_tokens = has_coordinate_tokens_static(tokenizer)
        self.assertTrue(has_tokens, "Static method should detect coordinate tokens")
        
        # Test coordinate range detection
        start_id, end_id = get_coordinate_token_range_static(tokenizer, max_coord_value)
        self.assertIsNotNone(start_id, "Should find coordinate token start ID")
        self.assertIsNotNone(end_id, "Should find coordinate token end ID")
        self.assertEqual(
            end_id - start_id, max_coord_value,
            f"Coordinate range should be {max_coord_value}"
        )
        
        # Test with tokenizer that doesn't have coordinate tokens
        fresh_tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        has_tokens_fresh = has_coordinate_tokens_static(fresh_tokenizer)
        self.assertFalse(has_tokens_fresh, "Fresh tokenizer should not have coordinate tokens")
        
        logger.info("✅ Error handling edge cases test passed")

    def test_coordinate_token_loss_computation_validation(self):
        """Test coordinate token loss computation for training validation."""
        logger.info("🔬 Testing coordinate token loss computation validation...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Create token manager with coordinate tokens
        max_coord_value = 100
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create synthetic input with coordinate tokens
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        
        # Create input with mixed tokens: text + coordinate tokens
        text_tokens = tokenizer.encode("Object at", add_special_tokens=False)
        coordinate_tokens = [
            coord_start_id + 10,  # coord_10
            coord_start_id + 20,  # coord_20
            coord_start_id + 30,  # coord_30
            coord_start_id + 40,  # coord_40
        ]
        
        input_ids = text_tokens + coordinate_tokens + [tokenizer.eos_token_id]
        input_tensor = torch.tensor([input_ids], device=model.device)
        
        # Create labels (shifted input for next token prediction)
        labels_tensor = torch.roll(input_tensor, -1)
        labels_tensor[:, -1] = -100  # Mask last token
        
        # Create mock logits (simulate model output)
        vocab_size = len(tokenizer)
        seq_len = input_tensor.shape[1]
        mock_logits = torch.randn((1, seq_len, vocab_size), device=model.device)
        
        # Compute coordinate losses
        losses = token_manager.compute_coordinate_losses(mock_logits, labels_tensor)
        
        # Validate loss structure
        expected_keys = [
            "coordinate_loss", "total_coordinate_loss", "coordinate_l1_loss",
            "coordinate_spans_found", "total_coordinate_tokens"
        ]
        
        for key in expected_keys:
            self.assertIn(key, losses, f"Missing loss key: {key}")
        
        # Validate loss values
        coordinate_loss = losses["coordinate_loss"]
        self.assertIsInstance(coordinate_loss, torch.Tensor)
        self.assertTrue(coordinate_loss.requires_grad, "Coordinate loss should require gradients")
        
        # Check that we found coordinate tokens
        total_coord_tokens = losses["total_coordinate_tokens"]
        self.assertEqual(
            total_coord_tokens, 4,  # We have 4 coordinate tokens in our test input
            f"Should find 4 coordinate tokens, found {total_coord_tokens}"
        )
        
        # Check spans found
        spans_found = losses["coordinate_spans_found"]
        self.assertGreater(spans_found, 0, "Should find at least one coordinate span")
        
        # Test with no coordinate tokens
        text_only_input = torch.tensor([text_tokens + [tokenizer.eos_token_id]], device=model.device)
        text_only_labels = torch.roll(text_only_input, -1)
        text_only_labels[:, -1] = -100
        
        text_only_logits = torch.randn((1, text_only_input.shape[1], vocab_size), device=model.device)
        
        no_coord_losses = token_manager.compute_coordinate_losses(text_only_logits, text_only_labels)
        
        # Should return zero losses when no coordinate tokens
        self.assertEqual(
            no_coord_losses["total_coordinate_tokens"], 0,
            "Should find no coordinate tokens in text-only input"
        )
        
        self.assertEqual(
            no_coord_losses["coordinate_loss"].item(), 0.0,
            "Coordinate loss should be zero when no coordinate tokens present"
        )
        
        logger.info("✅ Coordinate token loss computation validation passed")

    def test_coordinate_token_batch_processing_consistency(self):
        """Test coordinate token processing consistency across batch sizes."""
        logger.info("🔬 Testing coordinate token batch processing consistency...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Create token manager
        max_coord_value = 50
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Test coordinate wrapping consistency across different inputs
        test_coordinates = [
            [10, 20, 30, 40],    # Standard bbox
            [0, 0, 49, 49],      # Edge coordinates
            [25, 25, 25, 25],    # Same coordinates
        ]
        
        for coords in test_coordinates:
            with self.subTest(coords=coords):
                # Test different geometry types
                for geometry_type in ["bbox", "line", "square"]:
                    if geometry_type == "line" and len(coords) < 6:
                        # Extend coordinates for line (needs 6 coordinates)
                        test_coords = coords + coords[:2]
                    elif geometry_type == "square" and len(coords) < 8:
                        # Extend coordinates for square (needs 8 coordinates)
                        test_coords = coords + coords
                    else:
                        test_coords = coords
                    
                    # Test coordinate wrapping
                    wrapped = token_manager.wrap_coordinates(test_coords, geometry_type)
                    
                    # Verify format consistency
                    geometry_name = geometry_type if geometry_type != "bbox" else "box"
                    self.assertIn(f"<|{geometry_name}_start|>", wrapped)
                    self.assertIn(f"<|{geometry_name}_end|>", wrapped)
                    
                    # Verify all coordinates are wrapped as tokens
                    for coord in test_coords:
                        coord_clamped = max(0, min(int(coord), max_coord_value - 1))
                        coord_token = f"<|coord_{coord_clamped}|>"
                        self.assertIn(coord_token, wrapped)
        
        # Test object formatting consistency
        test_objects = [
            {"bbox_2d": [10, 20, 30, 40], "desc": "Test object 1"},
            {"bbox_2d": [0, 0, 49, 49], "desc": "Edge case object"},
            {"bbox_2d": [25, 25, 25, 25], "desc": "Point object"},
        ]
        
        for obj in test_objects:
            with self.subTest(obj_desc=obj["desc"]):
                formatted = token_manager.format_object(obj)
                
                # Verify object structure
                self.assertIn("<|object_ref_start|>", formatted)
                self.assertIn("<|object_ref_end|>", formatted)
                self.assertIn("desc:", formatted)
                self.assertIn(obj["desc"], formatted)
                
                # Verify coordinates are properly formatted
                for coord in obj["bbox_2d"]:
                    coord_clamped = max(0, min(int(coord), max_coord_value - 1))
                    coord_token = f"<|coord_{coord_clamped}|>"
                    self.assertIn(coord_token, formatted)
        
        logger.info("✅ Batch processing consistency test passed")


if __name__ == "__main__":
    unittest.main()