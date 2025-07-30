"""
Comprehensive Token Management Tests

Tests the complete token management system using real implementations
from src/ with synthetic data only. Replaces redundant temporal tests.
"""

import sys
import unittest

import torch


# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

from transformers import AutoModel, AutoTokenizer

from src.logger_utils import configure_global_logging, get_logger
from src.utils.tokens.special_tokens import (
    SimpleCoordinateManager,
    UnifiedTokenManager,
    create_unified_token_manager,
)
from tests.fixtures import SyntheticDataGenerator
from tests.fixtures.gpu_test_base import GPUAwareTestCase


logger = get_logger("test_token_management")


class TestTokenManagement(GPUAwareTestCase):
    """Comprehensive tests for token management using real implementations."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        super().setUpClass()
        configure_global_logging(rank=0, world_size=1)

        # Load real tokenizer and model for testing
        cls.model_path = "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.model = AutoModel.from_pretrained(
            cls.model_path, trust_remote_code=True, torch_dtype=torch.float16
        )

        # Create synthetic data generator
        cls.data_generator = SyntheticDataGenerator(num_samples=5)

    def test_unified_token_manager_creation(self):
        """Test UnifiedTokenManager creation and basic functionality."""
        logger.info("Testing UnifiedTokenManager creation...")

        # Test with small coordinate range for faster testing
        token_manager = UnifiedTokenManager(
            self.tokenizer, self.model, max_coord_value=100
        )

        # Verify coordinate tokens were added
        self.assertTrue(token_manager.has_coordinate_tokens())

        # Verify coordinate token range
        start_id, end_id = token_manager.get_coordinate_token_range()
        self.assertEqual(end_id - start_id, 100)

        # Test token ID retrieval
        coord_0_id = token_manager.get_coordinate_token_id(0)
        coord_99_id = token_manager.get_coordinate_token_id(99)
        self.assertEqual(coord_99_id - coord_0_id, 99)

    def test_coordinate_wrapping_all_geometries(self):
        """Test coordinate wrapping for all geometry types."""
        logger.info("Testing coordinate wrapping for all geometries...")

        token_manager = UnifiedTokenManager(
            self.tokenizer, self.model, max_coord_value=100
        )

        # Test data for different geometries
        test_cases = {
            "bbox": [10, 20, 30, 40],
            "line": [10, 20, 15, 25, 20, 30],
            "square": [10, 20, 30, 20, 30, 40, 10, 40],
        }

        for geometry_type, coords in test_cases.items():
            with self.subTest(geometry=geometry_type):
                result = token_manager.wrap_coordinates(coords, geometry_type)

                # Verify format
                self.assertIn(
                    f"<|{geometry_type if geometry_type != 'bbox' else 'box'}_start|>",
                    result,
                )
                self.assertIn(
                    f"<|{geometry_type if geometry_type != 'bbox' else 'box'}_end|>",
                    result,
                )

                # Verify coordinate tokens are used
                for coord in coords:
                    coord_clamped = max(0, min(int(coord), 99))
                    self.assertIn(f"<|coord_{coord_clamped}|>", result)

    def test_object_formatting_with_synthetic_data(self):
        """Test object formatting using synthetic data."""
        logger.info("Testing object formatting with synthetic data...")

        token_manager = UnifiedTokenManager(
            self.tokenizer, self.model, max_coord_value=100
        )

        # Generate synthetic objects
        synthetic_samples = []
        for i in range(3):
            sample = self.data_generator.generate_flat_sample(i)
            synthetic_samples.append(sample)

        for sample in synthetic_samples:  # Test all samples
            for obj in sample["objects"][:2]:  # Test first 2 objects per sample
                with self.subTest(obj_desc=obj.get("desc", "unknown")[:20]):
                    # Test object formatting
                    formatted = token_manager.format_object(obj)

                    # Verify format structure
                    self.assertIn("<|object_ref_start|>", formatted)
                    self.assertIn("<|object_ref_end|>", formatted)
                    self.assertIn("desc:", formatted)

                    # Verify geometry tokens are present
                    has_geometry = any(
                        token in formatted
                        for token in [
                            "<|box_start|>",
                            "<|line_start|>",
                            "<|square_start|>",
                        ]
                    )
                    self.assertTrue(
                        has_geometry, f"No geometry tokens found in: {formatted}"
                    )

    def test_simple_coordinate_manager_compatibility(self):
        """Test SimpleCoordinateManager compatibility with UnifiedTokenManager."""
        logger.info("Testing SimpleCoordinateManager compatibility...")

        # First, add tokens using UnifiedTokenManager
        unified_manager = UnifiedTokenManager(
            self.tokenizer, self.model, max_coord_value=50
        )

        # Then create SimpleCoordinateManager with the same tokenizer
        simple_manager = SimpleCoordinateManager(self.tokenizer, max_coord_value=50)

        # Both should detect coordinate tokens
        self.assertTrue(unified_manager.has_coordinate_tokens())
        self.assertTrue(simple_manager.has_coordinate_tokens())

        # Test coordinate wrapping consistency
        test_coords = [10, 20, 30, 40]
        unified_result = unified_manager.wrap_coordinates(test_coords, "bbox")
        simple_result = simple_manager.wrap_coordinates(test_coords, "bbox")

        self.assertEqual(unified_result, simple_result)

    def test_coordinate_token_range_validation(self):
        """Test coordinate token range validation and error handling."""
        logger.info("Testing coordinate token range validation...")

        token_manager = UnifiedTokenManager(
            self.tokenizer, self.model, max_coord_value=50
        )

        # Test valid coordinates
        valid_coords = [0, 25, 49]
        for coord in valid_coords:
            with self.subTest(coord=coord):
                token_id = token_manager.get_coordinate_token_id(coord)
                self.assertIsInstance(token_id, int)
                self.assertGreater(token_id, 0)

        # Test invalid coordinates
        invalid_coords = [-1, 50, 100]
        for coord in invalid_coords:
            with self.subTest(coord=coord):
                with self.assertRaises(ValueError):
                    token_manager.get_coordinate_token_id(coord)

    def test_factory_function(self):
        """Test the factory function for creating token managers."""
        logger.info("Testing factory function...")

        token_manager = create_unified_token_manager(
            self.tokenizer, self.model, max_coord_value=25
        )

        self.assertIsInstance(token_manager, UnifiedTokenManager)
        self.assertTrue(token_manager.has_coordinate_tokens())

        start_id, end_id = token_manager.get_coordinate_token_range()
        self.assertEqual(end_id - start_id, 25)

    def test_backward_compatibility_methods(self):
        """Test backward compatibility static methods."""
        logger.info("Testing backward compatibility methods...")

        # Create token manager to add tokens
        token_manager = UnifiedTokenManager(
            self.tokenizer, self.model, max_coord_value=30
        )

        # Test static methods
        has_tokens = UnifiedTokenManager.has_coordinate_tokens_in_tokenizer(
            self.tokenizer
        )
        self.assertTrue(has_tokens)

        token_range = UnifiedTokenManager.get_coordinate_token_range_from_tokenizer(
            self.tokenizer, max_coord_value=30
        )
        self.assertEqual(len(token_range), 2)
        self.assertEqual(token_range[1] - token_range[0], 30)

    def test_coordinate_loss_computation(self):
        """Test coordinate loss computation is returning actual losses, not zeros."""
        logger.info("🧪 Testing coordinate loss computation")

        # Use existing model and tokenizer from setUp
        model = self.model.to("cuda")  # Ensure model is on GPU
        tokenizer = self.tokenizer

        # Create token manager
        max_coord_value = 100  # Use smaller value for faster testing
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )

        # Create synthetic input with coordinate tokens
        # Format: "Object <|box_start|><|coord_10|><|coord_20|><|coord_30|><|coord_40|><|box_end|>"

        # Get coordinate token IDs
        coord_10_id = token_manager.get_coordinate_token_id(10)
        coord_20_id = token_manager.get_coordinate_token_id(20)
        coord_30_id = token_manager.get_coordinate_token_id(30)
        coord_40_id = token_manager.get_coordinate_token_id(40)

        # Get box token IDs
        box_start_id = token_manager.get_token_id("box_start")
        box_end_id = token_manager.get_token_id("box_end")

        # Create text token IDs
        text_tokens = tokenizer.encode("Object", add_special_tokens=False)

        # Create full input IDs
        input_ids = text_tokens + [
            box_start_id,
            coord_10_id,
            coord_20_id,
            coord_30_id,
            coord_40_id,
            box_end_id,
        ]
        input_tensor = torch.tensor([input_ids], device=model.device)

        # Create labels (shift input_ids right)
        labels_tensor = torch.roll(input_tensor, -1)
        labels_tensor[:, -1] = -100  # Mask last token

        # Create mock logits (random)
        vocab_size = len(tokenizer)
        seq_len = input_tensor.shape[1]
        mock_logits = torch.randn((1, seq_len, vocab_size), device=model.device)

        # Compute coordinate losses
        losses = token_manager.compute_coordinate_losses(mock_logits, labels_tensor)

        # Check that losses aren't all zeros
        coordinate_loss = losses["coordinate_loss"]
        self.assertIsInstance(coordinate_loss, torch.Tensor)

        # IMPORTANT: This test is EXPECTED to fail with the placeholder implementation
        # This failure will confirm our diagnosis that the coordinate loss computation
        # is not implemented correctly
        self.assertNotEqual(
            coordinate_loss.item(),
            0.0,
            "Coordinate loss is zero - this indicates the placeholder implementation is being used",
        )

        logger.info(
            f"✅ Coordinate loss test completed: loss = {coordinate_loss.item()}"
        )

    def test_coordinate_token_persistence_in_vocabulary(self):
        """Test that coordinate tokens are properly persisted in tokenizer vocabulary."""
        logger.info("🧪 Testing coordinate token persistence in vocabulary...")
        
        # Create token manager to add coordinate tokens
        max_coord_value = 50  # Use smaller value for faster testing
        token_manager = UnifiedTokenManager(
            tokenizer=self.tokenizer,
            model=self.model,
            max_coord_value=max_coord_value,
        )
        
        # Verify coordinate tokens were added to vocabulary
        vocab = self.tokenizer.get_vocab()
        coord_0_token = "<|coord_0|>"
        
        self.assertIn(coord_0_token, vocab, "Coordinate token <|coord_0|> not found in vocabulary")
        
        # Test all coordinate tokens are present
        missing_tokens = []
        coord_token_ids = []
        
        for i in range(max_coord_value):
            coord_token = f"<|coord_{i}|>"
            if coord_token not in vocab:
                missing_tokens.append(coord_token)
            else:
                coord_token_ids.append(vocab[coord_token])
        
        self.assertEqual(
            len(missing_tokens), 0, 
            f"Missing coordinate tokens from vocabulary: {missing_tokens[:10]}..."
        )
        
        # Verify coordinate tokens have consecutive IDs (optimal implementation)
        coord_token_ids.sort()
        expected_consecutive = list(range(coord_token_ids[0], coord_token_ids[0] + max_coord_value))
        
        if coord_token_ids != expected_consecutive:
            logger.warning("⚠️ Coordinate tokens do not have consecutive IDs - this is acceptable but not optimal")
        
        # Test coordinate token ID retrieval consistency
        for i in range(min(10, max_coord_value)):  # Test first 10 tokens
            expected_id = vocab[f"<|coord_{i}|>"]
            retrieved_id = token_manager.get_coordinate_token_id(i)
            self.assertEqual(
                expected_id, retrieved_id,
                f"Coordinate token ID mismatch for coord_{i}: vocab={expected_id}, manager={retrieved_id}"
            )
        
        logger.info(f"✅ All {max_coord_value} coordinate tokens properly persisted in vocabulary")
    
    def test_coordinate_token_vocabulary_after_save_load_cycle(self):
        """Test coordinate token persistence through save/load cycle simulation."""
        logger.info("🧪 Testing coordinate token persistence through save/load simulation...")
        
        # Create temporary directory for testing
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create token manager and add coordinate tokens
            max_coord_value = 30  # Smaller for faster testing
            token_manager = UnifiedTokenManager(
                tokenizer=self.tokenizer,
                model=self.model,
                max_coord_value=max_coord_value,
            )
            
            # Get initial vocabulary state
            initial_vocab = self.tokenizer.get_vocab().copy()
            initial_vocab_size = len(initial_vocab)
            
            # Simulate saving tokenizer (key step in BBUTrainer._save)
            tokenizer_save_path = os.path.join(temp_dir, "test_tokenizer")
            os.makedirs(tokenizer_save_path, exist_ok=True)
            
            # Save tokenizer vocabulary
            self.tokenizer.save_pretrained(tokenizer_save_path)
            
            # Load tokenizer from saved files
            from transformers import AutoTokenizer
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_save_path, trust_remote_code=True
            )
            
            # Verify coordinate tokens persist in loaded tokenizer
            loaded_vocab = loaded_tokenizer.get_vocab()
            loaded_vocab_size = len(loaded_vocab)
            
            # Check vocabulary size consistency
            self.assertEqual(
                loaded_vocab_size, initial_vocab_size,
                f"Vocabulary size mismatch after save/load: initial={initial_vocab_size}, loaded={loaded_vocab_size}"
            )
            
            # Check all coordinate tokens are present in loaded tokenizer
            missing_after_load = []
            for i in range(max_coord_value):
                coord_token = f"<|coord_{i}|>"
                if coord_token not in loaded_vocab:
                    missing_after_load.append(coord_token)
            
            self.assertEqual(
                len(missing_after_load), 0,
                f"Coordinate tokens missing after save/load cycle: {missing_after_load[:10]}..."
            )
            
            # Verify token IDs are consistent
            id_mismatches = []
            for i in range(min(10, max_coord_value)):  # Test first 10 tokens
                coord_token = f"<|coord_{i}|>"
                initial_id = initial_vocab[coord_token]
                loaded_id = loaded_vocab[coord_token]
                
                if initial_id != loaded_id:
                    id_mismatches.append((coord_token, initial_id, loaded_id))
            
            self.assertEqual(
                len(id_mismatches), 0,
                f"Coordinate token ID mismatches after save/load: {id_mismatches}"
            )
            
            logger.info(f"✅ All {max_coord_value} coordinate tokens persisted correctly through save/load cycle")
    
    def test_chat_processor_coordinate_conversion_validation(self):
        """Test ChatProcessor coordinate token conversion with validation."""
        logger.info("🧪 Testing ChatProcessor coordinate token conversion validation...")
        
        # Import required modules
        from src.chat_processor import ChatProcessor
        from src.config.config import BBUConfig
        from transformers import AutoImageProcessor
        from pathlib import Path
        
        # Load image processor
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        # Use existing config factory for testing
        from tests.fixtures.config_factory import create_test_config
        test_config = create_test_config(
            coordinate_tokens_enabled=True,
            max_coord_value=100,
            language="en"
        )
        
        # Create chat processor with coordinate tokens enabled
        max_coord_value = 100
        chat_processor = ChatProcessor(
            tokenizer=self.tokenizer,
            image_processor=image_processor,
            config=test_config,  # Provide config directly
            coordinate_tokens_enabled=True,
            max_coord_value=max_coord_value,
            language="en"
        )
        
        # Initialize coordinate manager
        chat_processor._update_coordinate_token_ranges()
        
        # Verify coordinate manager was properly initialized
        self.assertIsNotNone(chat_processor.coordinate_manager, "Coordinate manager not initialized")
        self.assertTrue(
            chat_processor.coordinate_manager.has_coordinate_tokens(),
            "Coordinate tokens not available in chat processor"
        )
        
        # Test coordinate token conversion with valid objects
        test_objects = [
            {
                "bbox_2d": [10, 20, 30, 40],
                "desc": "Test object 1"
            },
            {
                "line": [5, 15, 25, 35, 45, 55],
                "desc": "Test line object"
            },
            {
                "square": [15, 25, 35, 25, 35, 45, 15, 45],
                "desc": "Test square object"
            }
        ]
        
        # Test coordinate conversion
        coordinate_response = chat_processor._format_objects_response(test_objects)
        
        # Validate coordinate token patterns are present
        import re
        coord_token_pattern = r'<\|coord_\d+\|>'
        coord_tokens = re.findall(coord_token_pattern, coordinate_response)
        
        self.assertGreater(
            len(coord_tokens), 0,
            f"No coordinate tokens found in response: {coordinate_response}"
        )
        
        # Validate geometry tokens are present
        geometry_patterns = [
            r'<\|box_start\|>', r'<\|box_end\|>',
            r'<\|line_start\|>', r'<\|line_end\|>',
            r'<\|square_start\|>', r'<\|square_end\|>'
        ]
        
        geometry_tokens_found = any(
            re.search(pattern, coordinate_response) for pattern in geometry_patterns
        )
        
        self.assertTrue(
            geometry_tokens_found,
            f"No geometry tokens found in response: {coordinate_response}"
        )
        
        # Validate object reference tokens are present
        self.assertIn(
            '<|object_ref_start|>', coordinate_response,
            "Missing object reference start token"
        )
        self.assertIn(
            '<|object_ref_end|>', coordinate_response,
            "Missing object reference end token"
        )
        
        logger.info(f"✅ ChatProcessor coordinate conversion validation passed")
        logger.info(f"   Generated {len(coord_tokens)} coordinate tokens")
        logger.info(f"   Response length: {len(coordinate_response)} characters")
    
    def test_coordinate_token_error_handling(self):
        """Test coordinate token conversion error handling and validation."""
        logger.info("🧪 Testing coordinate token error handling...")
        
        # Import required modules
        from src.chat_processor import ChatProcessor
        from transformers import AutoImageProcessor
        
        # Load image processor
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        # Create chat processor with coordinate tokens enabled
        max_coord_value = 50  # Use smaller range for testing
        chat_processor = ChatProcessor(
            tokenizer=self.tokenizer,
            image_processor=image_processor,
            coordinate_tokens_enabled=True,
            max_coord_value=max_coord_value,
            language="en"
        )
        
        # Initialize coordinate manager
        chat_processor._update_coordinate_token_ranges()
        
        # Test 1: Invalid object structure (missing geometry)
        invalid_objects_no_geometry = [
            {"desc": "Object without geometry"}
        ]
        
        with self.assertRaises(ValueError) as context:
            chat_processor._format_objects_response(invalid_objects_no_geometry)
        
        self.assertIn("geometry type", str(context.exception).lower())
        
        # Test 2: Invalid object structure (missing description)
        invalid_objects_no_desc = [
            {"bbox_2d": [10, 20, 30, 40]}  # Missing desc field
        ]
        
        with self.assertRaises(ValueError) as context:
            chat_processor._format_objects_response(invalid_objects_no_desc)
        
        self.assertIn("desc", str(context.exception).lower())
        
        # Test 3: Coordinate values out of range
        invalid_objects_coord_range = [
            {
                "bbox_2d": [10, 20, 60, 40],  # 60 exceeds max_coord_value=50
                "desc": "Object with invalid coordinate"
            }
        ]
        
        with self.assertRaises(RuntimeError) as context:
            chat_processor._format_objects_response(invalid_objects_coord_range)
        
        self.assertIn("coordinate", str(context.exception).lower())
        
        # Test 4: Negative coordinate values
        invalid_objects_negative = [
            {
                "bbox_2d": [-5, 20, 30, 40],  # Negative coordinate
                "desc": "Object with negative coordinate"
            }
        ]
        
        with self.assertRaises(RuntimeError) as context:
            chat_processor._format_objects_response(invalid_objects_negative)
        
        self.assertIn("negative", str(context.exception).lower())
        
        logger.info("✅ Coordinate token error handling validation passed")
    
    def test_coordinate_token_model_loading_validation(self):
        """Test coordinate token model loading validation (simulation)."""
        logger.info("🧪 Testing coordinate token model loading validation...")
        
        # Create a UnifiedTokenManager to set up coordinate tokens
        max_coord_value = 40
        token_manager = UnifiedTokenManager(
            tokenizer=self.tokenizer,
            model=self.model,
            max_coord_value=max_coord_value,
        )
        
        # Test coordinate token detection in vocabulary
        from src.utils.tokens.special_tokens import has_coordinate_tokens_static, get_coordinate_token_range_static
        
        # Test static detection methods
        has_tokens = has_coordinate_tokens_static(self.tokenizer)
        self.assertTrue(has_tokens, "Static method failed to detect coordinate tokens")
        
        # Test token range detection
        start_id, end_id = get_coordinate_token_range_static(self.tokenizer, max_coord_value)
        self.assertIsNotNone(start_id, "Failed to get coordinate token start ID")
        self.assertIsNotNone(end_id, "Failed to get coordinate token end ID")
        self.assertEqual(end_id - start_id, max_coord_value, "Incorrect coordinate token range size")
        
        # Test consistency with token manager
        manager_start, manager_end = token_manager.get_coordinate_token_range()
        self.assertEqual(start_id, manager_start, "Static method and token manager start ID mismatch")
        self.assertEqual(end_id, manager_end, "Static method and token manager end ID mismatch")
        
        # Test individual coordinate token ID retrieval
        for i in range(min(5, max_coord_value)):  # Test first 5 tokens
            expected_id = start_id + i
            token_id = token_manager.get_coordinate_token_id(i)
            self.assertEqual(
                expected_id, token_id,
                f"Coordinate token ID mismatch for coord_{i}: expected={expected_id}, got={token_id}"
            )
        
        logger.info("✅ Coordinate token model loading validation passed")
        logger.info(f"   Coordinate token range: [{start_id}, {end_id})")
        logger.info(f"   Range size: {end_id - start_id} tokens")

    def test_coordinate_token_embedding_consistency_validation(self):
        """Test coordinate token embedding consistency across operations."""
        logger.info("🔬 Testing coordinate token embedding consistency validation...")
        
        # Create token manager with coordinate tokens
        max_coord_value = 60
        token_manager = UnifiedTokenManager(
            tokenizer=self.tokenizer,
            model=self.model,
            max_coord_value=max_coord_value,
        )
        
        # Get coordinate token range
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        
        # Get original embedding weights
        embedding_layer = self.model.get_input_embeddings()
        original_embeddings = embedding_layer.weight[coord_start_id:coord_end_id].detach().clone()
        
        # Test 1: Verify embedding initialization is not random/zero
        # Check that embeddings are not all zeros
        is_all_zeros = torch.all(original_embeddings == 0.0)
        self.assertFalse(is_all_zeros, "Coordinate token embeddings should not be all zeros")
        
        # Check that embeddings have reasonable variance (not all same value)
        embedding_std = torch.std(original_embeddings)
        self.assertGreater(
            embedding_std.item(), 1e-6,
            f"Coordinate token embeddings should have reasonable variance, got std={embedding_std.item()}"
        )
        
        # Test 2: Verify embedding shape consistency
        expected_embedding_dim = embedding_layer.weight.shape[1]
        self.assertEqual(
            original_embeddings.shape[1], expected_embedding_dim,
            f"Coordinate token embeddings should have correct dimension: got {original_embeddings.shape[1]}, expected {expected_embedding_dim}"
        )
        
        # Test 3: Verify coordinate token IDs are correct
        for i in range(min(10, max_coord_value)):  # Test first 10 tokens
            token_id = token_manager.get_coordinate_token_id(i)
            expected_id = coord_start_id + i
            
            self.assertEqual(
                token_id, expected_id,
                f"Coordinate token ID should be sequential: coord_{i} = {token_id}, expected = {expected_id}"
            )
        
        # Test 4: Verify embedding retrieval consistency
        for i in range(min(5, max_coord_value)):  # Test first 5 tokens
            token_id = token_manager.get_coordinate_token_id(i)
            embedding_by_id = embedding_layer.weight[token_id].detach()
            embedding_by_offset = original_embeddings[i]
            
            embeddings_match = torch.allclose(embedding_by_id, embedding_by_offset, rtol=1e-6, atol=1e-6)
            self.assertTrue(
                embeddings_match,
                f"Embedding retrieval should be consistent for coord_{i}"
            )
        
        logger.info(f"✅ Coordinate token embedding consistency validation passed for {max_coord_value} tokens")
    
    def test_coordinate_token_vocabulary_expansion_validation(self):
        """Test coordinate token vocabulary expansion and validation."""
        logger.info("🔬 Testing coordinate token vocabulary expansion validation...")
        
        # Record original tokenizer state
        original_vocab = self.tokenizer.get_vocab().copy()
        original_vocab_size = len(original_vocab)
        original_model_size = self.model.get_input_embeddings().weight.shape[0]
        
        # Create token manager with specific coordinate tokens
        max_coord_value = 80
        token_manager = UnifiedTokenManager(
            tokenizer=self.tokenizer,
            model=self.model,
            max_coord_value=max_coord_value,
        )
        
        # Verify vocabulary expansion
        expanded_vocab = self.tokenizer.get_vocab()
        expanded_vocab_size = len(expanded_vocab)
        expanded_model_size = self.model.get_input_embeddings().weight.shape[0]
        
        # Calculate expected expansion (coordinate tokens + geometry tokens if not already present)
        expected_new_tokens = max_coord_value + 4  # coordinate tokens + 4 geometry tokens
        expected_vocab_size = original_vocab_size + expected_new_tokens
        
        # Check vocabulary size increase
        self.assertEqual(
            expanded_vocab_size, expected_vocab_size,
            f"Vocabulary should increase by {expected_new_tokens}: "
            f"original={original_vocab_size}, expanded={expanded_vocab_size}, expected={expected_vocab_size}"
        )
        
        # Check model embedding size matches vocabulary
        self.assertEqual(
            expanded_model_size, expanded_vocab_size,
            f"Model embedding size should match vocabulary size: "
            f"model={expanded_model_size}, vocab={expanded_vocab_size}"
        )
        
        # Verify all expected tokens were added
        expected_tokens = []
        
        # Add coordinate tokens
        for i in range(max_coord_value):
            expected_tokens.append(f"<|coord_{i}|>")
        
        # Add geometry tokens (if not already present)
        geometry_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
        for token in geometry_tokens:
            if token not in original_vocab:
                expected_tokens.append(token)
        
        # Check all expected tokens are present
        missing_tokens = []
        for token in expected_tokens:
            if token not in expanded_vocab:
                missing_tokens.append(token)
        
        self.assertEqual(
            len(missing_tokens), 0,
            f"Missing expected tokens after expansion: {missing_tokens[:10]}..."
        )
        
        # Verify token IDs are within valid range
        for token in expected_tokens[:10]:  # Test first 10 tokens
            token_id = expanded_vocab[token]
            self.assertGreaterEqual(
                token_id, 0,
                f"Token ID should be non-negative: {token} = {token_id}"
            )
            self.assertLess(
                token_id, expanded_vocab_size,
                f"Token ID should be within vocabulary range: {token} = {token_id}, vocab_size = {expanded_vocab_size}"
            )
        
        # Verify original tokens are preserved
        token_id_changes = []
        for token, original_id in original_vocab.items():
            if token in expanded_vocab:
                expanded_id = expanded_vocab[token]
                if original_id != expanded_id:
                    token_id_changes.append((token, original_id, expanded_id))
        
        self.assertEqual(
            len(token_id_changes), 0,
            f"Original token IDs should not change: {token_id_changes[:5]}..."
        )
        
        logger.info(f"✅ Vocabulary expansion validation passed: added {expected_new_tokens} tokens")
    
    def test_coordinate_token_training_integration_validation(self):
        """Test coordinate token training integration and gradient flow validation."""
        logger.info("🔬 Testing coordinate token training integration validation...")
        
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for training integration test")
        
        # Move model to GPU
        model = self.model.to("cuda")
        
        # Create token manager with coordinate tokens
        max_coord_value = 40  # Smaller for faster testing
        token_manager = UnifiedTokenManager(
            tokenizer=self.tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Create synthetic training batch with coordinate tokens
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        
        # Create input with mixed tokens: regular text + coordinate tokens
        text_tokens = self.tokenizer.encode("Object at coordinates", add_special_tokens=False)
        coordinate_tokens = [
            coord_start_id + 5,   # coord_5
            coord_start_id + 15,  # coord_15
            coord_start_id + 25,  # coord_25
            coord_start_id + 35,  # coord_35
        ]
        
        # Create batch input
        input_ids = text_tokens + coordinate_tokens
        input_tensor = torch.tensor([input_ids], device=model.device)
        
        # Create labels for next token prediction
        labels_tensor = torch.roll(input_tensor, -1)
        labels_tensor[:, -1] = -100  # Mask last token
        
        # Test forward pass
        model.train()  # Set to training mode
        
        with torch.enable_grad():
            # Get model outputs
            outputs = model(input_ids=input_tensor, labels=labels_tensor)
            
            # Verify outputs have correct structure
            self.assertIsNotNone(outputs.logits, "Model should return logits")
            
            # Verify logits have correct shape
            expected_shape = (1, len(input_ids), len(self.tokenizer))
            self.assertEqual(
                outputs.logits.shape, expected_shape,
                f"Logits should have shape {expected_shape}, got {outputs.logits.shape}"
            )
            
            # Test coordinate loss computation
            coord_losses = token_manager.compute_coordinate_losses(outputs.logits, labels_tensor)
            
            # Verify coordinate loss structure
            self.assertIn("coordinate_loss", coord_losses)
            self.assertIn("total_coordinate_tokens", coord_losses)
            
            coordinate_loss = coord_losses["coordinate_loss"]
            total_coord_tokens = coord_losses["total_coordinate_tokens"]
            
            # Verify we found coordinate tokens
            self.assertEqual(
                total_coord_tokens, 4,  # We have 4 coordinate tokens
                f"Should find 4 coordinate tokens, found {total_coord_tokens}"
            )
            
            # Verify coordinate loss is valid
            self.assertIsInstance(coordinate_loss, torch.Tensor)
            self.assertTrue(coordinate_loss.requires_grad, "Coordinate loss should require gradients")
            self.assertGreater(coordinate_loss.item(), 0.0, "Coordinate loss should be positive")
            
            # Test gradient computation
            coordinate_loss.backward(retain_graph=True)
            
            # Verify gradients are computed for coordinate token embeddings
            embedding_layer = model.get_input_embeddings()
            coord_embeddings_grad = embedding_layer.weight.grad[coord_start_id:coord_end_id]
            
            # Check that at least some coordinate token embeddings have gradients
            has_coord_gradients = torch.any(coord_embeddings_grad != 0.0)
            self.assertTrue(
                has_coord_gradients,
                "Coordinate token embeddings should have non-zero gradients during training"
            )
            
            # Clear gradients for cleanup
            model.zero_grad()
        
        logger.info("✅ Training integration validation passed")
    
    def test_coordinate_token_inference_validation(self):
        """Test coordinate token inference and generation validation."""
        logger.info("🔬 Testing coordinate token inference validation...")
        
        # Create token manager with coordinate tokens
        max_coord_value = 50
        token_manager = UnifiedTokenManager(
            tokenizer=self.tokenizer,
            model=self.model,
            max_coord_value=max_coord_value,
        )
        
        # Test coordinate token formatting for inference
        test_objects = [
            {"bbox_2d": [10, 20, 30, 40], "desc": "Test bounding box object"},
            {"line": [5, 15, 25, 35, 45, 49], "desc": "Test line object"},
        ]
        
        for obj in test_objects:
            with self.subTest(obj_type=list(obj.keys())[0]):
                # Format object using coordinate tokens
                formatted_obj = token_manager.format_object(obj)
                
                # Verify object structure
                self.assertIn("<|object_ref_start|>", formatted_obj)
                self.assertIn("<|object_ref_end|>", formatted_obj)
                self.assertIn("desc:", formatted_obj)
                
                # Test tokenization of formatted object
                tokenized = self.tokenizer.encode(formatted_obj, add_special_tokens=False)
                
                # Verify coordinate tokens are properly tokenized
                coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
                coord_tokens_in_sequence = [
                    token_id for token_id in tokenized
                    if coord_start_id <= token_id < coord_end_id
                ]
                
                # Should have coordinate tokens for each coordinate
                geometry_key = list(obj.keys())[0]  # Get geometry type (bbox_2d, line, etc.)
                if geometry_key == "bbox_2d":
                    expected_coord_count = 4
                elif geometry_key == "line":
                    expected_coord_count = 6
                else:
                    expected_coord_count = len(obj[geometry_key])
                
                self.assertEqual(
                    len(coord_tokens_in_sequence), expected_coord_count,
                    f"Should have {expected_coord_count} coordinate tokens for {geometry_key}, "
                    f"found {len(coord_tokens_in_sequence)}"
                )
                
                # Test coordinate token decoding
                for coord_token_id in coord_tokens_in_sequence:
                    coord_value = coord_token_id - coord_start_id
                    self.assertGreaterEqual(coord_value, 0, "Coordinate value should be non-negative")
                    self.assertLess(coord_value, max_coord_value, "Coordinate value should be within range")
                
                # Test roundtrip: tokenize then decode
                decoded = self.tokenizer.decode(tokenized, skip_special_tokens=False)
                
                # Should contain coordinate tokens
                self.assertIn("<|coord_", decoded, "Decoded text should contain coordinate tokens")
                
        logger.info("✅ Inference validation passed")


if __name__ == "__main__":
    unittest.main()
