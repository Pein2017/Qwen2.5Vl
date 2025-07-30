"""
Unified Data Processor Tests for BBU Training System

Tests the unified data processing implementation that eliminated
the ChatProcessorOutput intermediate schema and consolidated
data processing logic.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton

patch_torch_library_wrap_triton()

from src.config import BBUConfig
from src.data import BBUDataset, create_data_collator
from src.logger_utils import configure_global_logging, get_logger
from tests.fixtures import ConfigFactory, SyntheticDataGenerator, TestUtils
from tests.fixtures.gpu_test_base import GPUAwareTestCase

logger = get_logger("test_unified_data_processor")


class TestUnifiedDataProcessor(GPUAwareTestCase):
    """Tests for unified data processor implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        configure_global_logging(rank=0, world_size=1)
        logger.info("🧪 Starting Unified Data Processor Tests")

        # Create test utilities
        cls.test_utils = TestUtils()
        cls.config_factory = ConfigFactory()
        
        # Create minimal synthetic dataset
        cls.data_generator = SyntheticDataGenerator(num_samples=4)
        cls.train_path, cls.val_path, cls.teacher_path, cls.all_samples_path = (
            cls.data_generator.generate_complete_dataset()
        )
        cls.data_root = str(cls.data_generator.temp_dir)

        logger.info(f"✅ Unified data processor test setup complete: {cls.data_root}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.data_generator.cleanup()
        cls.config_factory.cleanup_test_configs()
        logger.info("🧹 Unified data processor test teardown complete")

    def setUp(self):
        """Set up for each individual test."""
        self.test_files_to_cleanup = []
        super().setUp()

    def tearDown(self):
        """Clean up after each test."""
        self.test_utils.cleanup_test_files(self.test_files_to_cleanup)
        super().tearDown()

    def test_unified_dataset_creation(self):
        """Test unified BBUDataset creation and basic functionality."""
        logger.info("🧪 Testing unified dataset creation")
        
        # Create test configuration
        config_path = self.config_factory.create_minimal_config(
            data_root=self.data_root,
            coordinate_enabled=True,
            collator_type="standard"
        )
        config = self.config_factory.load_test_config(config_path)
        
        # Create mock tokenizer for testing
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.padding_side = 'left'
        
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            formatted = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                formatted += f"<{role}>{content}</{role}>"
            return formatted
        
        def tokenize_fn(text, padding=False, truncation=False, add_special_tokens=True):
            # Mock tokenization - return character codes limited to reasonable range
            ids = [min(ord(c) % 1000, 999) for c in text[:100]]
            return {'input_ids': ids}
        
        mock_tokenizer.apply_chat_template = apply_chat_template
        mock_tokenizer.__call__ = tokenize_fn
        
        # Create unified dataset
        dataset = BBUDataset(
            data_path=self.train_path,
            tokenizer=mock_tokenizer,
            image_processor=None,  # Mock processor for testing
            teacher_pool_manager=None,  # No teachers for this test
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        self.assertGreater(len(dataset), 0)
        logger.info(f"✅ Created unified dataset with {len(dataset)} samples")

    def test_unified_sample_processing(self):
        """Test that unified sample processing produces correct output structure."""
        logger.info("🧪 Testing unified sample processing")
        
        # Create test configuration
        config_path = self.config_factory.create_minimal_config(
            data_root=self.data_root,
            coordinate_enabled=True,
            collator_type="standard"
        )
        config = self.config_factory.load_test_config(config_path)
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.padding_side = 'left'
        
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            return "mock formatted text"
        
        def tokenize_fn(text, padding=False, truncation=False, add_special_tokens=True):
            return {'input_ids': [1, 2, 3, 4, 5]}
        
        mock_tokenizer.apply_chat_template = apply_chat_template
        mock_tokenizer.__call__ = tokenize_fn
        
        # Create dataset
        dataset = BBUDataset(
            data_path=self.train_path,
            tokenizer=mock_tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        # Process a sample
        sample = dataset[0]
        
        # Verify output structure - should be plain dict, not intermediate schema
        self.assertIsInstance(sample, dict)
        
        # Check required keys
        required_keys = ['input_ids', 'labels', 'attention_mask']
        for key in required_keys:
            self.assertIn(key, sample, f"Missing required key: {key}")
            self.assertIsInstance(sample[key], torch.Tensor, f"Key {key} is not a tensor")
        
        logger.info("✅ Sample has all required keys as tensors")
        logger.info(f"   - input_ids shape: {sample['input_ids'].shape}")
        logger.info(f"   - labels shape: {sample['labels'].shape}")
        logger.info(f"   - attention_mask shape: {sample['attention_mask'].shape}")

    def test_collator_integration(self):
        """Test that data collator works with unified output."""
        logger.info("🧪 Testing collator integration with unified output")
        
        # Create test configuration
        config_path = self.config_factory.create_minimal_config(
            data_root=self.data_root,
            coordinate_enabled=True,
            collator_type="standard"
        )
        config = self.config_factory.load_test_config(config_path)
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.padding_side = 'left'
        
        def tokenize_fn(text, padding=False, truncation=False, add_special_tokens=True):
            return {'input_ids': [1, 2, 3, 4, 5]}
        
        mock_tokenizer.apply_chat_template = lambda messages, **kwargs: "mock text"
        mock_tokenizer.__call__ = tokenize_fn
        
        # Create dataset and collator
        dataset = BBUDataset(
            data_path=self.train_path,
            tokenizer=mock_tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        collator = create_data_collator("standard", mock_tokenizer)
        
        # Create batch from multiple samples
        batch = collator([dataset[0], dataset[1] if len(dataset) > 1 else dataset[0]])
        
        self.assertIsInstance(batch, dict)
        logger.info("✅ Successfully created batch with collator")
        logger.info(f"   - Batch keys: {list(batch.keys())}")
        
        if 'input_ids' in batch:
            logger.info(f"   - Batch input_ids shape: {batch['input_ids'].shape}")
        if 'labels' in batch:
            logger.info(f"   - Batch labels shape: {batch['labels'].shape}")

    def test_multi_geometry_support(self):
        """Test that unified processor supports multiple geometry types."""
        logger.info("🧪 Testing multi-geometry support")
        
        # Create test configuration
        config_path = self.config_factory.create_minimal_config(
            data_root=self.data_root,
            coordinate_enabled=True,
            collator_type="standard"
        )
        config = self.config_factory.load_test_config(config_path)
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.padding_side = 'left'
        
        mock_tokenizer.apply_chat_template = lambda messages, **kwargs: "mock text"
        mock_tokenizer.__call__ = lambda text, **kwargs: {'input_ids': [1, 2, 3, 4, 5]}
        
        # Create dataset
        dataset = BBUDataset(
            data_path=self.train_path,
            tokenizer=mock_tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        # Process all samples and check for geometry support
        geometry_types_found = set()
        for i in range(min(len(dataset), 3)):  # Test first 3 samples
            sample = dataset[i]
            gt_objects = sample.get('ground_truth_objects', [])
            for obj in gt_objects:
                # Check for different geometry types
                for key in obj.keys():
                    if key in ['bbox_2d', 'square', 'line', 'polygon']:
                        geometry_types_found.add(key)
        
        logger.info(f"✅ Found geometry types: {geometry_types_found}")
        # We should find at least one geometry type from synthetic data
        self.assertGreater(len(geometry_types_found), 0)

    def test_elimination_of_intermediate_schema(self):
        """Test that ChatProcessorOutput is no longer needed."""
        logger.info("🧪 Testing elimination of intermediate schema")
        
        # Verify that we can import and use data module without ChatProcessor
        from src.data import BBUDataset, StandardDataCollator, create_data_collator
        logger.info("✅ Successfully imported unified data components")
        
        # Create test configuration
        config_path = self.config_factory.create_minimal_config(
            data_root=self.data_root,
            coordinate_enabled=False,  # Simplified test
            collator_type="standard"
        )
        config = self.config_factory.load_test_config(config_path)
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.padding_side = 'left'
        
        mock_tokenizer.apply_chat_template = lambda messages, **kwargs: "mock text"
        mock_tokenizer.__call__ = lambda text, **kwargs: {'input_ids': [1, 2, 3, 4, 5]}
        
        # Create dataset without any reference to ChatProcessor
        dataset = BBUDataset(
            data_path=self.train_path,
            tokenizer=mock_tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        # Process a sample - this should work without ChatProcessor
        sample = dataset[0]
        
        # Verify the output is a plain dict (not ChatProcessorOutput)
        self.assertIsInstance(sample, dict, f"Expected dict, got {type(sample)}")
        self.assertIn('input_ids', sample, "Missing input_ids")
        self.assertIn('labels', sample, "Missing labels")
        
        logger.info("✅ Unified dataset works without ChatProcessor dependency")
        logger.info("✅ Output is plain dict (no intermediate schema)")


if __name__ == '__main__':
    unittest.main()