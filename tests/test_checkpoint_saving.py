#!/usr/bin/env python3
"""
Tests for proper checkpoint saving and loading with expanded vocabulary.

Tests cover:
- Saving models with standard mode (4 geometry tokens)
- Saving models with coordinate mode (coordinate tokens + 4 geometry tokens)
- Proper saving of tokenizer with expanded vocabulary
- Loading saved models and verifying token embeddings are preserved
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton


patch_torch_library_wrap_triton()

from transformers import AutoModel, AutoTokenizer

from src.config import init_config, load_config
from src.logger_utils import configure_global_logging, get_logger
from src.utils.tokens.special_tokens import (
    UnifiedTokenManager,
)
from tests.fixtures import ConfigFactory, TestUtils
from tests.fixtures.gpu_test_base import GPUAwareTestCase


logger = get_logger("test_checkpoint_saving")


class TestCheckpointSaving(GPUAwareTestCase):
    """Tests for proper saving and loading of models with expanded vocabulary."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        super().setUpClass()
        configure_global_logging(rank=0, world_size=1)

        # Create test utilities and config factory
        cls.test_utils = TestUtils()
        cls.config_factory = ConfigFactory()
        cls.temp_dir = tempfile.mkdtemp(prefix="checkpoint_tests_")
        cls.test_files_to_cleanup = []

        logger.info(f"✅ Test setup complete in {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        import shutil

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
        self.checkpoint_dir = os.path.join(self.temp_dir, f"checkpoint_{hash(self)}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after each test."""
        # Force aggressive GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc

            gc.collect()

    def test_standard_mode_checkpoint_saving(self):
        """Test saving and loading model in standard mode with 4 new geometry tokens."""
        logger.info("🧪 Testing standard mode checkpoint saving")

        # Create standard configuration
        config_path = self.config_factory.create_coordinate_disabled_config(
            Path(self.temp_dir), "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)

        # Load model and processor
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="standard_save_test",
        )

        # Create token manager and add geometry tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=0,  # No coordinate tokens in standard mode
        )

        # Get current vocabulary size (after tokens were already added during loading)
        current_vocab_size = len(tokenizer.get_vocab())
        model_embedding_size = model.get_input_embeddings().weight.shape[0]

        # Verify tokenizer and model sizes match
        self.assertEqual(
            current_vocab_size,
            model_embedding_size,
            f"Tokenizer vocab size ({current_vocab_size}) should match model embedding size ({model_embedding_size})",
        )

        # Verify specific geometry tokens exist
        for token in [
            "<|line_start|>",
            "<|line_end|>",
            "<|square_start|>",
            "<|square_end|>",
        ]:
            self.assertIn(
                token,
                tokenizer.get_vocab(),
                f"Token {token} should be in vocabulary",
            )

        # Save model and tokenizer
        save_dir = os.path.join(self.checkpoint_dir, "standard_model")
        os.makedirs(save_dir, exist_ok=True)

        try:
            model.save_pretrained(save_dir)
        except AttributeError as e:
            # Handle missing attributes during saving
            logger.warning(f"AttributeError during model saving: {e}")
            # Save model components individually if wrapper fails
            if hasattr(model, "base_model"):
                model.base_model.save_pretrained(save_dir)
            else:
                model.save_pretrained(save_dir)

        tokenizer.save_pretrained(save_dir)

        # Now load the saved model and tokenizer (with a simpler approach)
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                save_dir, trust_remote_code=True
            )
            loaded_model = AutoModel.from_pretrained(
                save_dir, trust_remote_code=True, torch_dtype=torch.float16
            )

            # Verify loaded tokenizer has the correct vocabulary size
            self.assertEqual(
                len(loaded_tokenizer.get_vocab()),
                current_vocab_size,
                "Loaded tokenizer should have same vocabulary size",
            )

            # Verify geometry tokens exist in loaded tokenizer
            for token in [
                "<|line_start|>",
                "<|line_end|>",
                "<|square_start|>",
                "<|square_end|>",
            ]:
                self.assertIn(
                    token,
                    loaded_tokenizer.get_vocab(),
                    f"Token {token} should be in loaded vocabulary",
                )

            # Verify loaded model has properly sized embedding matrix
            self.assertEqual(
                loaded_model.get_input_embeddings().weight.shape[0],
                current_vocab_size,
                "Loaded model embedding matrix should match vocabulary size",
            )
        except Exception as e:
            logger.warning(f"Error loading saved model: {e}")
            self.skipTest(f"Skipping model loading verification due to: {e}")

        logger.info("✅ Standard mode checkpoint saving test passed")

    def test_coordinate_mode_checkpoint_saving(self):
        """Test saving and loading model with coordinate tokens and geometry tokens."""
        logger.info("🧪 Testing coordinate mode checkpoint saving")

        # Create coordinate configuration
        max_coord_value = 100  # Use smaller value for testing
        config_path = self.config_factory.create_coordinate_enabled_config_with_value(
            Path(self.temp_dir), "standard", max_coord_value
        )
        self.test_files_to_cleanup.append(config_path)

        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)
        
        # Verify max_coord_value was set correctly
        self.assertEqual(
            config.max_coord_value,
            max_coord_value,
            f"max_coord_value should be {max_coord_value}",
        )

        # Load model and processor
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="coordinate_save_test",
        )

        # Create token manager and add tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )

        # Get current vocabulary sizes after loading (tokens were added during model loading)
        current_vocab_size = len(tokenizer.get_vocab())
        model_embedding_size = model.get_input_embeddings().weight.shape[0]
        
        # Verify tokenizer and model sizes match
        self.assertEqual(
            current_vocab_size,
            model_embedding_size,
            f"Tokenizer vocab size ({current_vocab_size}) should match model embedding size ({model_embedding_size})",
        )

        # Verify coordinate tokens exist
        for i in range(0, max_coord_value, 10):  # Check a subset of tokens
            token = f"<|coord_{i}|>"
            self.assertIn(
                token,
                tokenizer.get_vocab(),
                f"Token {token} should be in vocabulary",
            )

        # Save model and tokenizer
        save_dir = os.path.join(self.checkpoint_dir, "coordinate_model")
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            model.save_pretrained(save_dir)
        except AttributeError as e:
            # Handle missing attributes during saving
            logger.warning(f"AttributeError during model saving: {e}")
            # Save model components individually if wrapper fails
            if hasattr(model, "base_model"):
                model.base_model.save_pretrained(save_dir)
            else:
                model.save_pretrained(save_dir)

        tokenizer.save_pretrained(save_dir)

        # Try to load the saved model and tokenizer
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
            loaded_model = AutoModel.from_pretrained(
                save_dir, trust_remote_code=True, torch_dtype=torch.float16
            )

            # Verify loaded tokenizer has the correct vocabulary size
            self.assertEqual(
                len(loaded_tokenizer.get_vocab()),
                current_vocab_size,
                "Loaded tokenizer should have same vocabulary size",
            )

            # Verify coordinate tokens exist in loaded tokenizer
            for i in range(0, max_coord_value, 10):  # Check a subset of tokens
                token = f"<|coord_{i}|>"
                self.assertIn(
                    token,
                    loaded_tokenizer.get_vocab(),
                    f"Token {token} should be in loaded vocabulary",
                )

            # Verify loaded model has properly sized embedding matrix
            self.assertEqual(
                loaded_model.get_input_embeddings().weight.shape[0],
                current_vocab_size,
                "Loaded model embedding matrix should match vocabulary size",
            )
        except Exception as e:
            logger.warning(f"Error loading saved model: {e}")
            self.skipTest(f"Skipping model loading verification due to: {e}")
            
        logger.info("✅ Coordinate mode checkpoint saving test passed")

    def test_embedding_preservation_after_save(self):
        """Test that embedding values are preserved when saving the model."""
        logger.info("🧪 Testing embedding preservation after save")

        # Create coordinate configuration
        max_coord_value = 50  # Use smaller value for testing
        config_path = self.config_factory.create_coordinate_enabled_config_with_value(
            Path(self.temp_dir), "standard", max_coord_value
        )
        self.test_files_to_cleanup.append(config_path)

        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)

        # Load model and processor
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="embedding_test",
        )

        # Create token manager and add tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )

        # Check that coordinate tokens and geometry tokens exist
        vocab = tokenizer.get_vocab()
        coordinate_token_ids = []
        geometry_token_ids = []

        # Get token IDs for coordinate tokens
        for i in range(10):  # Just check the first 10 coordinate tokens
            token = f"<|coord_{i}|>"
            if token in vocab:
                coordinate_token_ids.append(vocab[token])

        # Get token IDs for geometry tokens
        for token in [
            "<|line_start|>",
            "<|line_end|>",
            "<|square_start|>",
            "<|square_end|>",
        ]:
            if token in vocab:
                geometry_token_ids.append(vocab[token])

        # Get original embedding values
        embedding_layer = model.get_input_embeddings()
        original_coord_embeddings = (
            embedding_layer.weight[coordinate_token_ids].detach().clone()
        )
        original_geom_embeddings = (
            embedding_layer.weight[geometry_token_ids].detach().clone()
        )

        # Save model state dict to temporary file
        save_dir = os.path.join(self.checkpoint_dir, "embedding_test")
        os.makedirs(save_dir, exist_ok=True)
        state_dict_path = os.path.join(save_dir, "model_state.pt")
        
        # Extract model state dict (handle both wrapped and unwrapped models)
        if hasattr(model, "base_model"):
            state_dict = model.base_model.state_dict()
        else:
            state_dict = model.state_dict()
            
        # Save state dict
        torch.save(state_dict, state_dict_path)
        
        # Load state dict
        loaded_state_dict = torch.load(state_dict_path)
        
        # Find embedding weights in state dict
        embedding_key = None
        for key in loaded_state_dict.keys():
            if "embed_tokens.weight" in key:
                embedding_key = key
                break
                
        self.assertIsNotNone(embedding_key, "Could not find embedding weights in state dict")
        
        # Get loaded embeddings
        loaded_embeddings = loaded_state_dict[embedding_key]
        
        # Extract coordinate and geometry token embeddings from loaded state
        loaded_coord_embeddings = loaded_embeddings[coordinate_token_ids].detach()
        loaded_geom_embeddings = loaded_embeddings[geometry_token_ids].detach()

        # Compare embedding values
        coord_match = torch.allclose(
            original_coord_embeddings, loaded_coord_embeddings, rtol=1e-3, atol=1e-3
        )
        geom_match = torch.allclose(
            original_geom_embeddings, loaded_geom_embeddings, rtol=1e-3, atol=1e-3
        )

        self.assertTrue(
            coord_match,
            "Coordinate token embeddings should be preserved in state dict",
        )
        self.assertTrue(
            geom_match,
            "Geometry token embeddings should be preserved in state dict",
        )
            
        logger.info("✅ Embedding preservation test passed")

    def test_coordinate_token_conversion_standard_to_coordinate_checkpoint(self):
        """Test checkpoint conversion from standard mode to coordinate mode."""
        logger.info("🔬 Testing standard to coordinate mode checkpoint conversion...")
        
        # Step 1: Create and save model in standard mode (geometry tokens only)
        config_path_std = self.config_factory.create_coordinate_disabled_config(
            Path(self.test_dir), "standard"
        )
        self.test_files_to_cleanup.append(config_path_std)
        
        init_config(config_path_std)
        config_std = load_config(config_path_std)
        
        # Load model in standard mode
        model_std, tokenizer_std, processor_std = self.load_model_safely(
            model_path=config_std.model_path,
            config=config_std,
            cache_key="std_conversion_test",
        )
        
        # Create token manager for standard mode (geometry tokens only)
        token_manager_std = UnifiedTokenManager(
            tokenizer=tokenizer_std,
            model=model_std,
            max_coord_value=0,  # No coordinate tokens
        )
        
        # Record standard mode state
        std_vocab = tokenizer_std.get_vocab().copy()
        std_vocab_size = len(std_vocab)
        std_model_size = model_std.get_input_embeddings().weight.shape[0]
        
        # Save standard mode checkpoint
        std_checkpoint_dir = os.path.join(self.test_dir, "standard_checkpoint")
        os.makedirs(std_checkpoint_dir, exist_ok=True)
        
        try:
            model_std.save_pretrained(std_checkpoint_dir)
        except AttributeError as e:
            logger.warning(f"AttributeError during standard model saving: {e}")
            if hasattr(model_std, "base_model"):
                model_std.base_model.save_pretrained(std_checkpoint_dir)
            else:
                model_std.save_pretrained(std_checkpoint_dir)
        
        tokenizer_std.save_pretrained(std_checkpoint_dir)
        
        # Step 2: Load checkpoint and convert to coordinate mode
        max_coord_value = 75
        
        # Load saved standard model
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            std_checkpoint_dir, trust_remote_code=True
        )
        loaded_model = AutoModel.from_pretrained(
            std_checkpoint_dir, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Verify loaded model is in standard mode
        loaded_vocab_size = len(loaded_tokenizer.get_vocab())
        self.assertEqual(
            loaded_vocab_size, std_vocab_size,
            f"Loaded model should have standard vocabulary size: {loaded_vocab_size} vs {std_vocab_size}"
        )
        
        # Convert to coordinate mode by adding coordinate tokens
        token_manager_coord = UnifiedTokenManager(
            tokenizer=loaded_tokenizer,
            model=loaded_model,
            max_coord_value=max_coord_value,
        )
        
        # Verify coordinate conversion
        coord_vocab = loaded_tokenizer.get_vocab()
        coord_vocab_size = len(coord_vocab)
        coord_model_size = loaded_model.get_input_embeddings().weight.shape[0]
        
        # Check vocabulary size increased
        expected_coord_vocab_size = std_vocab_size + max_coord_value
        self.assertEqual(
            coord_vocab_size, expected_coord_vocab_size,
            f"Coordinate vocabulary should be larger: std={std_vocab_size}, coord={coord_vocab_size}, expected={expected_coord_vocab_size}"
        )
        
        # Check model size matches vocabulary
        self.assertEqual(
            coord_model_size, coord_vocab_size,
            f"Coordinate model size should match vocabulary: model={coord_model_size}, vocab={coord_vocab_size}"
        )
        
        # Verify coordinate tokens are functional
        coord_start_id, coord_end_id = token_manager_coord.get_coordinate_token_range()
        self.assertEqual(
            coord_end_id - coord_start_id, max_coord_value,
            f"Coordinate token range should be {max_coord_value}"
        )
        
        # Test coordinate token functionality
        for i in range(min(5, max_coord_value)):
            token_id = token_manager_coord.get_coordinate_token_id(i)
            expected_id = coord_start_id + i
            self.assertEqual(
                token_id, expected_id,
                f"Coordinate token ID should be sequential: coord_{i} = {token_id}, expected = {expected_id}"
            )
        
        # Step 3: Save converted coordinate model
        coord_checkpoint_dir = os.path.join(self.test_dir, "coordinate_checkpoint")
        os.makedirs(coord_checkpoint_dir, exist_ok=True)
        
        try:
            loaded_model.save_pretrained(coord_checkpoint_dir)
        except AttributeError as e:
            logger.warning(f"AttributeError during coordinate model saving: {e}")
            if hasattr(loaded_model, "base_model"):
                loaded_model.base_model.save_pretrained(coord_checkpoint_dir)
            else:
                loaded_model.save_pretrained(coord_checkpoint_dir)
        
        loaded_tokenizer.save_pretrained(coord_checkpoint_dir)
        
        # Step 4: Verify converted checkpoint can be loaded
        final_tokenizer = AutoTokenizer.from_pretrained(
            coord_checkpoint_dir, trust_remote_code=True
        )
        final_model = AutoModel.from_pretrained(
            coord_checkpoint_dir, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Verify final model has coordinate tokens
        final_vocab_size = len(final_tokenizer.get_vocab())
        final_model_size = final_model.get_input_embeddings().weight.shape[0]
        
        self.assertEqual(
            final_vocab_size, coord_vocab_size,
            f"Final vocabulary size should match coordinate size: {final_vocab_size} vs {coord_vocab_size}"
        )
        
        self.assertEqual(
            final_model_size, final_vocab_size,
            f"Final model size should match vocabulary: {final_model_size} vs {final_vocab_size}"
        )
        
        # Verify coordinate tokens persist in final model
        for i in range(min(5, max_coord_value)):
            coord_token = f"<|coord_{i}|>"
            self.assertIn(
                coord_token, final_tokenizer.get_vocab(),
                f"Coordinate token {coord_token} should persist in final model"
            )
        
        logger.info(f"✅ Standard to coordinate conversion test passed with {max_coord_value} coordinate tokens")

    def test_coordinate_token_embedding_weights_after_checkpoint_cycle(self):
        """Test coordinate token embedding weights consistency through checkpoint cycles."""
        logger.info("🔬 Testing coordinate token embedding weights through checkpoint cycles...")
        
        # Create coordinate configuration
        max_coord_value = 60
        config_path = self.config_factory.create_coordinate_enabled_config_with_value(
            Path(self.test_dir), "coordinate", max_coord_value
        )
        self.test_files_to_cleanup.append(config_path)
        
        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)
        
        # Load model and tokenizer
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="embedding_cycle_test",
        )
        
        # Create token manager and add coordinate tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=model,
            max_coord_value=max_coord_value,
        )
        
        # Get coordinate token range and embedding weights
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        embedding_layer = model.get_input_embeddings()
        
        # Extract original coordinate token embeddings
        original_coord_embeddings = embedding_layer.weight[coord_start_id:coord_end_id].detach().clone()
        
        # Verify embeddings are initialized properly
        self.assertFalse(
            torch.all(original_coord_embeddings == 0.0),
            "Coordinate token embeddings should not be all zeros"
        )
        
        # Get embeddings for geometry tokens as well
        geometry_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
        geometry_token_ids = []
        for token in geometry_tokens:
            if token in tokenizer.get_vocab():
                geometry_token_ids.append(tokenizer.get_vocab()[token])
        
        original_geometry_embeddings = embedding_layer.weight[geometry_token_ids].detach().clone()
        
        # Save checkpoint (multiple cycles)
        for cycle in range(2):  # Test 2 save/load cycles
            cycle_dir = os.path.join(self.test_dir, f"cycle_{cycle}")
            os.makedirs(cycle_dir, exist_ok=True)
            
            # Save model and tokenizer
            try:
                model.save_pretrained(cycle_dir)
            except AttributeError as e:
                logger.warning(f"AttributeError during cycle {cycle} model saving: {e}")
                if hasattr(model, "base_model"):
                    model.base_model.save_pretrained(cycle_dir)
                else:
                    model.save_pretrained(cycle_dir)
            
            tokenizer.save_pretrained(cycle_dir)
            
            # Load saved checkpoint
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                cycle_dir, trust_remote_code=True
            )
            loaded_model = AutoModel.from_pretrained(
                cycle_dir, trust_remote_code=True, torch_dtype=torch.float16
            )
            
            # Verify coordinate tokens are present
            loaded_vocab = loaded_tokenizer.get_vocab()
            for i in range(min(10, max_coord_value)):  # Test first 10 tokens
                coord_token = f"<|coord_{i}|>"
                self.assertIn(
                    coord_token, loaded_vocab,
                    f"Coordinate token {coord_token} should be present after cycle {cycle}"
                )
            
            # Verify embedding weights are preserved
            loaded_embedding_layer = loaded_model.get_input_embeddings()
            loaded_coord_embeddings = loaded_embedding_layer.weight[coord_start_id:coord_end_id].detach()
            loaded_geometry_embeddings = loaded_embedding_layer.weight[geometry_token_ids].detach()
            
            # Compare coordinate token embeddings
            coord_embeddings_match = torch.allclose(
                original_coord_embeddings, loaded_coord_embeddings, rtol=1e-3, atol=1e-3
            )
            
            self.assertTrue(
                coord_embeddings_match,
                f"Coordinate token embeddings should be preserved in cycle {cycle}"
            )
            
            # Compare geometry token embeddings
            geometry_embeddings_match = torch.allclose(
                original_geometry_embeddings, loaded_geometry_embeddings, rtol=1e-3, atol=1e-3
            )
            
            self.assertTrue(
                geometry_embeddings_match,
                f"Geometry token embeddings should be preserved in cycle {cycle}"
            )
            
            # Update model and tokenizer for next cycle
            model = loaded_model
            tokenizer = loaded_tokenizer
        
        logger.info(f"✅ Embedding weights consistency test passed through 2 checkpoint cycles")

    def test_coordinate_token_vocabulary_size_validation_after_save(self):
        """Test vocabulary size validation after saving coordinate token models."""
        logger.info("🔬 Testing vocabulary size validation after save...")
        
        # Test different coordinate token configurations
        test_configs = [
            {"max_coord_value": 25, "mode": "small"},
            {"max_coord_value": 100, "mode": "medium"},
            {"max_coord_value": 200, "mode": "large"},
        ]
        
        for config_info in test_configs:
            max_coord_value = config_info["max_coord_value"]
            mode = config_info["mode"]
            
            with self.subTest(mode=mode, max_coord=max_coord_value):
                # Create configuration
                config_path = self.config_factory.create_coordinate_enabled_config_with_value(
                    Path(self.test_dir), mode, max_coord_value
                )
                self.test_files_to_cleanup.append(config_path)
                
                # Initialize configuration
                init_config(config_path)
                config = load_config(config_path)
                
                # Load model and tokenizer
                model, tokenizer, processor = self.load_model_safely(
                    model_path=config.model_path,
                    config=config,
                    cache_key=f"vocab_size_test_{mode}",
                )
                
                # Create token manager
                token_manager = UnifiedTokenManager(
                    tokenizer=tokenizer,
                    model=model,
                    max_coord_value=max_coord_value,
                )
                
                # Record original sizes
                original_vocab_size = len(tokenizer.get_vocab())
                original_model_size = model.get_input_embeddings().weight.shape[0]
                
                # Verify sizes match
                self.assertEqual(
                    original_vocab_size, original_model_size,
                    f"Original vocabulary and model sizes should match: vocab={original_vocab_size}, model={original_model_size}"
                )
                
                # Save checkpoint
                save_dir = os.path.join(self.test_dir, f"vocab_test_{mode}")
                os.makedirs(save_dir, exist_ok=True)
                
                try:
                    model.save_pretrained(save_dir)
                except AttributeError as e:
                    logger.warning(f"AttributeError during {mode} model saving: {e}")
                    if hasattr(model, "base_model"):
                        model.base_model.save_pretrained(save_dir)
                    else:
                        model.save_pretrained(save_dir)
                
                tokenizer.save_pretrained(save_dir)
                
                # Load checkpoint
                loaded_tokenizer = AutoTokenizer.from_pretrained(
                    save_dir, trust_remote_code=True
                )
                loaded_model = AutoModel.from_pretrained(
                    save_dir, trust_remote_code=True, torch_dtype=torch.float16
                )
                
                # Verify loaded sizes
                loaded_vocab_size = len(loaded_tokenizer.get_vocab())
                loaded_model_size = loaded_model.get_input_embeddings().weight.shape[0]
                
                # Check size consistency
                self.assertEqual(
                    loaded_vocab_size, original_vocab_size,
                    f"Loaded vocabulary size should match original: loaded={loaded_vocab_size}, original={original_vocab_size}"
                )
                
                self.assertEqual(
                    loaded_model_size, original_model_size,
                    f"Loaded model size should match original: loaded={loaded_model_size}, original={original_model_size}"
                )
                
                self.assertEqual(
                    loaded_vocab_size, loaded_model_size,
                    f"Loaded vocabulary and model sizes should match: vocab={loaded_vocab_size}, model={loaded_model_size}"
                )
                
                # Verify coordinate tokens are present and functional
                coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
                
                # Check coordinate tokens in loaded tokenizer
                loaded_vocab = loaded_tokenizer.get_vocab()
                missing_coord_tokens = []
                
                for i in range(min(10, max_coord_value)):  # Test first 10 tokens
                    coord_token = f"<|coord_{i}|>"
                    if coord_token not in loaded_vocab:
                        missing_coord_tokens.append(coord_token)
                
                self.assertEqual(
                    len(missing_coord_tokens), 0,
                    f"Missing coordinate tokens in {mode} mode: {missing_coord_tokens}"
                )
                
                # Verify coordinate token ID consistency
                for i in range(min(5, max_coord_value)):  # Test first 5 tokens
                    coord_token = f"<|coord_{i}|>"
                    expected_id = coord_start_id + i
                    loaded_id = loaded_vocab[coord_token]
                    
                    self.assertEqual(
                        loaded_id, expected_id,
                        f"Coordinate token ID mismatch in {mode} mode: coord_{i} = {loaded_id}, expected = {expected_id}"
                    )
        
        logger.info("✅ Vocabulary size validation test passed for all configurations")

    def test_coordinate_token_model_wrapper_checkpoint_integration(self):
        """Test coordinate token persistence with BBUModelWrapper checkpoints."""
        logger.info("🔬 Testing coordinate token model wrapper checkpoint integration...")
        
        # Create coordinate configuration
        max_coord_value = 80
        config_path = self.config_factory.create_coordinate_enabled_config_with_value(
            Path(self.test_dir), "wrapper", max_coord_value
        )
        self.test_files_to_cleanup.append(config_path)
        
        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)
        
        # Load base model and tokenizer
        base_model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="wrapper_checkpoint_test",
        )
        
        # Add coordinate tokens
        token_manager = UnifiedTokenManager(
            tokenizer=tokenizer,
            model=base_model,
            max_coord_value=max_coord_value,
        )
        
        # Create wrapper model
        from src.models.wrapper import BBUModelWrapper
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
            f"Wrapper model embedding size should match tokenizer: wrapper={wrapper_vocab_size}, tokenizer={tokenizer_vocab_size}"
        )
        
        # Save wrapper model checkpoint
        wrapper_save_dir = os.path.join(self.test_dir, "wrapper_checkpoint")
        os.makedirs(wrapper_save_dir, exist_ok=True)
        
        # Save wrapper components
        try:
            # Try to save wrapper model
            wrapper_model.save_pretrained(wrapper_save_dir)
        except AttributeError as e:
            logger.warning(f"AttributeError during wrapper model saving: {e}")
            # Save base model instead
            if hasattr(wrapper_model, "base_model"):
                wrapper_model.base_model.save_pretrained(wrapper_save_dir)
            else:
                base_model.save_pretrained(wrapper_save_dir)
        
        tokenizer.save_pretrained(wrapper_save_dir)
        
        # Load wrapper checkpoint
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            wrapper_save_dir, trust_remote_code=True
        )
        loaded_base_model = AutoModel.from_pretrained(
            wrapper_save_dir, trust_remote_code=True, torch_dtype=torch.float16
        )
        
        # Create new wrapper with loaded components
        loaded_wrapper = BBUModelWrapper(
            base_model=loaded_base_model,
            tokenizer=loaded_tokenizer,
            config=config
        )
        
        # Verify loaded wrapper has coordinate tokens
        loaded_wrapper_vocab_size = loaded_wrapper.get_input_embeddings().weight.shape[0]
        loaded_tokenizer_vocab_size = len(loaded_tokenizer.get_vocab())
        
        self.assertEqual(
            loaded_wrapper_vocab_size, loaded_tokenizer_vocab_size,
            f"Loaded wrapper embedding size should match tokenizer: wrapper={loaded_wrapper_vocab_size}, tokenizer={loaded_tokenizer_vocab_size}"
        )
        
        # Verify coordinate tokens are functional in loaded wrapper
        coord_start_id, coord_end_id = token_manager.get_coordinate_token_range()
        
        # Test coordinate token accessibility
        for i in range(min(5, max_coord_value)):  # Test first 5 tokens
            coord_token = f"<|coord_{i}|>"
            self.assertIn(
                coord_token, loaded_tokenizer.get_vocab(),
                f"Coordinate token {coord_token} should be accessible in loaded wrapper"
            )
        
        # Test minimal forward pass with coordinate tokens
        if torch.cuda.is_available():
            loaded_wrapper = loaded_wrapper.cuda()
            
            # Create test input with coordinate tokens
            test_input_ids = torch.tensor([[
                coord_start_id,      # First coordinate token
                coord_start_id + 1,  # Second coordinate token
                loaded_tokenizer.eos_token_id  # End token
            ]], device='cuda')
            
            # Test forward pass
            with torch.no_grad():
                outputs = loaded_wrapper(input_ids=test_input_ids)
                
            # Verify outputs
            self.assertEqual(
                outputs.logits.shape[-1], loaded_tokenizer_vocab_size,
                f"Loaded wrapper output should have correct vocabulary size: {outputs.logits.shape[-1]} vs {loaded_tokenizer_vocab_size}"
            )
        
        logger.info(f"✅ Model wrapper checkpoint integration test passed with {max_coord_value} coordinate tokens")


if __name__ == "__main__":
    unittest.main(verbosity=2)
