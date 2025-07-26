"""
Model Loading Tests for BBU Training System

Comprehensive tests for model loading in different modes:
- Coordinate token enabled/disabled
- Different attention implementations
- Vocabulary validation and extension
- Flash attention compatibility
"""

import sys
import unittest

import torch


# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton


patch_torch_library_wrap_triton()

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from src.config import init_config, load_config
from src.logger_utils import configure_global_logging, get_logger
from src.models.wrapper import Qwen25VLWithDetection
from tests.fixtures import ConfigFactory, SyntheticDataGenerator, TestUtils
from tests.fixtures.gpu_test_base import GPUAwareTestCase


logger = get_logger("test_model_loading")


class TestModelLoading(GPUAwareTestCase):
    """Comprehensive tests for model loading in different modes."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Call parent setup for GPU management
        super().setUpClass()

        # Configure logging
        configure_global_logging(rank=0, world_size=1)
        logger.info("🧪 Starting Model Loading Tests")

        # Create test utilities
        cls.test_utils = TestUtils()

        # Create synthetic data for configuration
        cls.data_generator = SyntheticDataGenerator(num_samples=4)
        cls.train_path, cls.val_path, cls.teacher_path, cls.all_samples_path = (
            cls.data_generator.generate_complete_dataset()
        )
        cls.data_root = str(cls.data_generator.temp_dir)

        # Create configuration factory
        cls.config_factory = ConfigFactory()

        logger.info(f"✅ Model loading test setup complete")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.data_generator.cleanup()
        cls.config_factory.cleanup_test_configs()
        logger.info("🧹 Model loading test teardown complete")

        # Call parent cleanup for GPU management
        super().tearDownClass()

    def setUp(self):
        """Set up for each individual test."""
        # Call parent setup for GPU management
        super().setUp()

    def tearDown(self):
        """Clean up after each test."""
        # Call parent cleanup for GPU management
        super().tearDown()

    def test_standard_model_loading(self):
        """Test loading base model without coordinate tokens."""
        logger.info("🧪 Testing Standard Model Loading (No Coordinate Tokens)")

        # Create standard configuration
        config_path = self.config_factory.create_coordinate_disabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Validate configuration
        self.test_utils.validate_config_requirements(config)
        self.assertFalse(config.coordinate_tokens_enabled)
        self.assertEqual(
            config.max_coord_value, 2048
        )  # Should be 2048 as per bbu_v2.yaml reference

        # Load model (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path, config=config, cache_key="standard_model"
        )

        # Validate model type
        self.assertIsInstance(
            model,
            Qwen2_5_VLForConditionalGeneration,
            f"Expected base model, got {type(model)}",
        )

        # Validate vocabulary size (should be standard)
        vocab_size = len(tokenizer.get_vocab())
        expected_base_size = 151665  # Standard Qwen2.5-VL vocab

        # Allow for some geometry tokens but no coordinate tokens
        self.assertLessEqual(
            vocab_size,
            expected_base_size + 20,  # Small buffer for geometry tokens
            f"Vocabulary too large for standard mode: {vocab_size}",
        )

        # Validate tokenizer configuration
        self.assertEqual(
            tokenizer.padding_side, "left", "Tokenizer should use left padding"
        )
        self.assertIsNotNone(tokenizer.pad_token, "Pad token should be set")

        # Validate model device
        model_device = next(model.parameters()).device
        if torch.cuda.is_available():
            self.assertTrue(
                model_device.type == "cuda",
                f"Model should be on GPU, got {model_device}",
            )

        # Validate model dtype
        model_dtype = next(model.parameters()).dtype
        self.assertEqual(
            model_dtype, torch.bfloat16, f"Model should use bfloat16, got {model_dtype}"
        )

        logger.info(f"✅ Standard model loading test passed:")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Vocabulary size: {vocab_size}")
        logger.info(f"   Device: {model_device}")
        logger.info(f"   Dtype: {model_dtype}")

    def test_coordinate_model_loading(self):
        """Test loading model with coordinate tokens enabled."""
        logger.info("🧪 Testing Coordinate Model Loading (With Coordinate Tokens)")

        # Create coordinate configuration
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Validate configuration
        self.test_utils.validate_config_requirements(config)
        self.assertTrue(config.coordinate_tokens_enabled)
        self.assertGreater(config.max_coord_value, 0)

        # Load model (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path, config=config, cache_key="coordinate_model"
        )

        # Validate model type (should be detection wrapper)
        self.assertIsInstance(
            model, Qwen25VLWithDetection, f"Expected detection model, got {type(model)}"
        )

        # Validate coordinate token capability
        self.assertTrue(
            hasattr(model, "coordinate_tokens_enabled"),
            "Model should have coordinate_tokens_enabled attribute",
        )
        self.assertTrue(
            model.coordinate_tokens_enabled,
            "Model should have coordinate tokens enabled",
        )

        # Validate extended vocabulary
        vocab_size = len(tokenizer.get_vocab())
        expected_base_size = 151665  # Standard Qwen2.5-VL vocab
        expected_extensions = (
            config.max_coord_value + 8
        )  # coordinate tokens + geometry tokens
        expected_total = expected_base_size + expected_extensions

        self.assertGreaterEqual(
            vocab_size,
            expected_total - 10,  # Small tolerance
            f"Vocabulary too small for coordinate mode: {vocab_size} < {expected_total}",
        )
        self.assertLessEqual(
            vocab_size,
            expected_total + 10,  # Small tolerance
            f"Vocabulary too large for coordinate mode: {vocab_size} > {expected_total}",
        )

        # Validate model embedding size matches tokenizer
        model_embedding_size = model.get_input_embeddings().num_embeddings
        self.assertEqual(
            model_embedding_size,
            vocab_size,
            f"Model embedding size ({model_embedding_size}) != tokenizer vocab size ({vocab_size})",
        )

        logger.info(f"✅ Coordinate model loading test passed:")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Vocabulary size: {vocab_size}")
        logger.info(f"   Expected size: {expected_total}")
        logger.info(f"   Coordinate tokens: {config.max_coord_value}")

    def test_flash_attention_compatibility(self):
        """Test flash attention compatibility with both model types."""
        logger.info("🧪 Testing Flash Attention Compatibility")

        for coordinate_enabled in [False, True]:
            mode_name = "coordinate" if coordinate_enabled else "standard"
            logger.info(f"🔄 Testing flash attention - {mode_name} mode")

            # Create appropriate configuration
            if coordinate_enabled:
                config_path = self.config_factory.create_coordinate_enabled_config(
                    self.data_root, "standard"
                )
            else:
                config_path = self.config_factory.create_coordinate_disabled_config(
                    self.data_root, "standard"
                )

            self.test_files_to_cleanup.append(config_path)

            init_config(config_path)
            config = load_config(config_path)

            # Force flash attention
            config.attn_implementation = "flash_attention_2"

            # Load model (GPU-safe)
            cache_key = f"flash_attention_{mode_name}_model"
            model, tokenizer, processor = self.load_model_safely(
                model_path=config.model_path, config=config, cache_key=cache_key
            )

            # Verify flash attention is enabled
            if hasattr(model, "config") and hasattr(
                model.config, "_attn_implementation"
            ):
                self.assertEqual(
                    model.config._attn_implementation,
                    "flash_attention_2",
                    f"Flash attention not enabled for {mode_name} model",
                )

            # Test with a simple forward pass
            # Create minimal input
            input_text = "Test input for flash attention"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Create labels for the forward pass (required by wrapper)
            # For eval mode, labels are used for validation metrics
            labels = inputs["input_ids"].clone()

            # Create original_inputs dict that wrapper expects
            original_inputs = {"labels": labels, **inputs}

            # Set model to eval mode for compatibility testing
            model.eval()

            # Test forward pass
            with torch.no_grad():
                try:
                    outputs = model(**original_inputs)
                    self.assertIsNotNone(
                        outputs, f"Forward pass failed for {mode_name} model"
                    )

                    # Validate output structure
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                        self.assertIsInstance(
                            logits, torch.Tensor, "Logits should be tensor"
                        )
                        self.assertEqual(
                            logits.device, device, "Logits should be on correct device"
                        )

                    logger.info(f"✅ Flash attention {mode_name} mode test passed")

                except Exception as e:
                    self.fail(
                        f"Flash attention forward pass failed for {mode_name} model: {e}"
                    )

    def test_vocabulary_extension_validation(self):
        """Test vocabulary extension and token range validation."""
        logger.info("🧪 Testing Vocabulary Extension Validation")

        # Test coordinate model with specific coordinate value
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Load coordinate model (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="vocab_validation_model",
        )

        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)

        # Validate geometry tokens are present
        geometry_tokens = [
            "<|box_start|>",
            "<|box_end|>",
            "<|square_start|>",
            "<|square_end|>",
            "<|line_start|>",
            "<|line_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
        ]

        for token in geometry_tokens:
            self.assertIn(token, vocab, f"Missing geometry token: {token}")
            token_id = vocab[token]
            self.assertIsInstance(token_id, int, f"Invalid token ID for {token}")

        # Validate coordinate token range (if enabled)
        if config.coordinate_tokens_enabled:
            base_vocab_size = 151665
            coordinate_start = base_vocab_size + len(geometry_tokens)
            coordinate_end = coordinate_start + config.max_coord_value

            # Check that coordinate tokens are accessible
            for coord_id in [
                coordinate_start,
                coordinate_start + 100,
                coordinate_end - 1,
            ]:
                if coord_id < vocab_size:
                    # Token should be valid
                    try:
                        token_str = tokenizer.decode([coord_id])
                        self.assertIsInstance(
                            token_str,
                            str,
                            f"Failed to decode coordinate token {coord_id}",
                        )
                    except Exception as e:
                        self.fail(f"Failed to decode coordinate token {coord_id}: {e}")

        # Validate model embedding dimensions
        embedding_matrix = model.get_input_embeddings()
        embedding_size = embedding_matrix.num_embeddings
        embedding_dim = embedding_matrix.embedding_dim

        self.assertEqual(
            embedding_size,
            vocab_size,
            f"Embedding size ({embedding_size}) != vocab size ({vocab_size})",
        )

        # Expected embedding dimension for Qwen2.5-VL-3B (actual: 2048, not 3584)
        expected_dim = 2048
        self.assertEqual(
            embedding_dim,
            expected_dim,
            f"Unexpected embedding dimension: {embedding_dim} != {expected_dim}",
        )

        logger.info(f"✅ Vocabulary extension validation passed:")
        logger.info(f"   Total vocabulary size: {vocab_size}")
        logger.info(f"   Embedding dimensions: {embedding_size} x {embedding_dim}")
        logger.info(f"   Geometry tokens found: {len(geometry_tokens)}")

    def test_model_consistency_across_modes(self):
        """Test that model loading is consistent across different modes."""
        logger.info("🧪 Testing Model Loading Consistency")

        models_configs = []

        # Load both types of models
        for coordinate_enabled in [False, True]:
            mode_name = "coordinate" if coordinate_enabled else "standard"

            if coordinate_enabled:
                config_path = self.config_factory.create_coordinate_enabled_config(
                    self.data_root, "standard"
                )
            else:
                config_path = self.config_factory.create_coordinate_disabled_config(
                    self.data_root, "standard"
                )

            self.test_files_to_cleanup.append(config_path)

            init_config(config_path)
            config = load_config(config_path)

            # Load model (GPU-safe)
            cache_key = f"memory_comparison_{mode_name}_model"
            model, tokenizer, processor = self.load_model_safely(
                model_path=config.model_path, config=config, cache_key=cache_key
            )

            models_configs.append((model, tokenizer, processor, config, mode_name))

        standard_model, standard_tokenizer, _, standard_config, _ = models_configs[0]
        coord_model, coord_tokenizer, _, coord_config, _ = models_configs[1]

        # Compare base model parameters (should be identical for shared layers)
        if hasattr(coord_model, "base_model"):
            coord_base = coord_model.base_model
        else:
            coord_base = coord_model

        # Compare model configurations
        if hasattr(standard_model, "config") and hasattr(coord_base, "config"):
            std_config = standard_model.config
            coord_config_obj = coord_base.config

            # Core architecture should be identical
            self.assertEqual(
                std_config.hidden_size,
                coord_config_obj.hidden_size,
                "Hidden size should be consistent",
            )
            self.assertEqual(
                std_config.num_hidden_layers,
                coord_config_obj.num_hidden_layers,
                "Number of layers should be consistent",
            )
            self.assertEqual(
                std_config.num_attention_heads,
                coord_config_obj.num_attention_heads,
                "Number of attention heads should be consistent",
            )

        # Validate tokenizer consistency (base tokens should be identical)
        std_vocab = standard_tokenizer.get_vocab()
        coord_vocab = coord_tokenizer.get_vocab()

        # Check that all standard tokens exist in coordinate tokenizer
        base_vocab_size = 151665
        common_tokens = 0
        for token, token_id in std_vocab.items():
            if token_id < base_vocab_size:  # Only check base vocabulary
                self.assertIn(
                    token, coord_vocab, f"Missing token in coordinate vocab: {token}"
                )
                self.assertEqual(
                    coord_vocab[token],
                    token_id,
                    f"Token ID mismatch for {token}: {coord_vocab[token]} != {token_id}",
                )
                common_tokens += 1

        self.assertGreater(
            common_tokens, base_vocab_size * 0.99, "Too many missing common tokens"
        )

        logger.info(f"✅ Model consistency test passed:")
        logger.info(f"   Common vocabulary tokens validated: {common_tokens}")
        logger.info(f"   Standard vocab size: {len(std_vocab)}")
        logger.info(f"   Coordinate vocab size: {len(coord_vocab)}")

    def test_inference_mode_loading(self):
        """Test model loading in inference mode."""
        logger.info("🧪 Testing Inference Mode Loading")

        # Create configuration
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Load model in inference mode (GPU-safe)
        with TestUtils.gpu_memory_guard("Inference model loading", cleanup_after=False):
            from src.models.model_loader import load_model_and_processor_unified

            model, tokenizer, processor = load_model_and_processor_unified(
                model_path=config.model_path,
                for_inference=True,  # Inference mode
                attn_implementation=config.attn_implementation,
            )
            # Add to cleanup since this is not cached
            self.models_to_cleanup.append((model, tokenizer, processor))

        # Validate model is in eval mode
        self.assertFalse(model.training, "Model should be in eval mode for inference")

        # Validate cache is enabled for inference
        if hasattr(model, "config"):
            self.assertTrue(
                model.config.use_cache, "Cache should be enabled for inference"
            )

        # Test generation capability
        if hasattr(model, "generate"):
            test_input = "请描述这张图片中的设备。"
            inputs = tokenizer(test_input, return_tensors="pt")

            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Test generation (with very short output to save time)
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    self.assertIsInstance(
                        outputs, torch.Tensor, "Generation should return tensor"
                    )
                    self.assertGreater(
                        outputs.shape[-1],
                        inputs["input_ids"].shape[-1],
                        "Generated sequence should be longer than input",
                    )

                except Exception as e:
                    self.fail(f"Generation failed in inference mode: {e}")

        logger.info("✅ Inference mode loading test passed")


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
