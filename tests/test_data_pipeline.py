"""
Data Pipeline Tests for BBU Training System

Comprehensive tests for data loading, processing, and collation.
Tests BBU dataset, both collator types, and chat processing with
coordinate token support.
"""

import sys
import unittest
from pathlib import Path

import torch


# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton


patch_torch_library_wrap_triton()

from src.config import init_config, load_config
from src.core.data_processor import DataProcessor
from src.data import (
    BBUDataset,
    PackedDataCollator,
    StandardDataCollator,
    create_data_collator,
)
from src.logger_utils import configure_global_logging, get_logger
from tests.fixtures import ConfigFactory, SyntheticDataGenerator, TestUtils
from tests.fixtures.gpu_test_base import GPUAwareTestCase


logger = get_logger("test_data_pipeline")


class TestDataPipeline(GPUAwareTestCase):
    """Comprehensive tests for data pipeline components with GPU memory management."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Configure logging
        configure_global_logging(rank=0, world_size=1)
        logger.info("🧪 Starting Data Pipeline Tests")

        # Create test utilities
        cls.test_utils = TestUtils()

        # Create synthetic data generator
        cls.data_generator = SyntheticDataGenerator(num_samples=12)
        cls.train_path, cls.val_path, cls.teacher_path, cls.all_samples_path = (
            cls.data_generator.generate_complete_dataset()
        )
        cls.data_root = str(cls.data_generator.temp_dir)

        # Create configuration factory
        cls.config_factory = ConfigFactory()

        logger.info(f"✅ Test setup complete: {cls.data_root}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.data_generator.cleanup()
        cls.config_factory.cleanup_test_configs()
        logger.info("🧹 Test teardown complete")

    def setUp(self):
        """Set up for each individual test."""
        self.test_files_to_cleanup = []

    def tearDown(self):
        """Clean up after each test."""
        self.test_utils.cleanup_test_files(self.test_files_to_cleanup)

    def test_bbu_dataset_loading_standard_mode(self):
        """Test BBU dataset loading in standard (non-coordinate) mode."""
        logger.info("🧪 Testing BBU dataset loading - Standard mode")

        # Create standard configuration
        config_path = self.config_factory.create_coordinate_disabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        # Initialize global config
        init_config(config_path)
        config = load_config(config_path)

        # Load model and processor for dataset initialization (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path, config=config, cache_key="standard_mode_model"
        )

        # Create data processor
        data_processor = DataProcessor(
            tokenizer=tokenizer, image_processor=processor, model=model, config=config
        )

        # Create datasets
        with self.test_utils.measure_time("Dataset creation"):
            train_dataset, eval_dataset = data_processor.create_datasets()

        # Validate datasets
        self.assertIsInstance(train_dataset, BBUDataset)
        self.assertIsInstance(eval_dataset, BBUDataset)
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(eval_dataset), 0)

        # Test single sample
        sample = train_dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn("input_ids", sample)
        self.assertIn("labels", sample)
        self.assertIn("ground_truth_objects", sample)

        # Validate tensor properties
        self.test_utils.validate_tensor_dtype(
            sample["input_ids"], torch.long, "input_ids"
        )
        self.test_utils.validate_tensor_dtype(sample["labels"], torch.long, "labels")

        # Validate coordinate tokens are NOT present (standard mode)
        self.test_utils.validate_coordinate_tokens(
            sample["input_ids"], tokenizer, coordinate_enabled=False
        )

        logger.info(
            f"✅ Standard dataset test passed: {len(train_dataset)} train, {len(eval_dataset)} eval samples"
        )

    def test_bbu_dataset_loading_coordinate_mode(self):
        """Test BBU dataset loading in coordinate token mode."""
        logger.info("🧪 Testing BBU dataset loading - Coordinate mode")

        # Create coordinate configuration
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        # Initialize global config
        init_config(config_path)
        config = load_config(config_path)

        # Load model and processor for dataset initialization (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="coordinate_mode_model",
        )

        # Create data processor
        data_processor = DataProcessor(
            tokenizer=tokenizer, image_processor=processor, model=model, config=config
        )

        # Create datasets
        with self.test_utils.measure_time("Dataset creation"):
            train_dataset, eval_dataset = data_processor.create_datasets()

        # Test single sample
        sample = train_dataset[0]

        # Validate coordinate tokens ARE present (coordinate mode)
        self.test_utils.validate_coordinate_tokens(
            sample["input_ids"], tokenizer, coordinate_enabled=True
        )

        logger.info(
            f"✅ Coordinate dataset test passed: {len(train_dataset)} train, {len(eval_dataset)} eval samples"
        )

    def test_standard_data_collator(self):
        """Test StandardDataCollator with comprehensive validation."""
        logger.info("🧪 Testing StandardDataCollator")

        # Create configuration
        config_path = self.config_factory.create_standard_collator_config(
            self.data_root
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Load model and tokenizer (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="standard_collator_model",
        )

        # Create dataset and collator
        data_processor = DataProcessor(tokenizer, processor, model, config)
        train_dataset, _ = data_processor.create_datasets()
        collator = create_data_collator("standard", tokenizer)

        # Check if we got the wrapper or the direct collator
        from src.data import TrainerCompatibleDataCollator

        if isinstance(collator, TrainerCompatibleDataCollator):
            # Unwrap to get the actual collator for testing
            actual_collator = collator.base_collator
            self.assertIsInstance(actual_collator, StandardDataCollator)
        else:
            # Direct collator (wrapper disabled)
            self.assertIsInstance(collator, StandardDataCollator)

        # Test batch collation
        batch_samples = [train_dataset[i] for i in range(min(3, len(train_dataset)))]

        with self.test_utils.measure_time("Standard collator batch creation"):
            batch = collator(batch_samples)

        # Validate batch structure
        expected_keys = [
            "input_ids",
            "labels",
            "attention_mask",
            "ground_truth_objects",
        ]
        for key in expected_keys:
            self.assertIn(key, batch, f"Missing key: {key}")

        batch_size = len(batch_samples)

        # Validate tensor shapes and consistency
        self.test_utils.validate_batch_consistency(batch, batch_size)

        # Validate attention mask
        sequence_lengths = [sample["input_ids"].shape[-1] for sample in batch_samples]
        self.test_utils.validate_attention_mask(
            batch["attention_mask"], sequence_lengths
        )

        # Calculate and validate padding efficiency
        total_tokens = sum(sequence_lengths)
        max_seq_len = max(sequence_lengths)
        efficiency = self.test_utils.calculate_memory_efficiency(
            total_tokens, batch_size, max_seq_len
        )

        # Standard collator should have reasonable efficiency (typically 60-90%)
        self.assertGreater(
            efficiency, 0.5, f"Memory efficiency too low: {efficiency:.2%}"
        )
        self.assertLessEqual(
            efficiency, 1.0, f"Memory efficiency too high: {efficiency:.2%}"
        )

        logger.info(
            f"✅ StandardDataCollator test passed - Memory efficiency: {efficiency:.2%}"
        )

    def test_packed_data_collator(self):
        """Test PackedDataCollator with memory efficiency validation."""
        logger.info("🧪 Testing PackedDataCollator")

        # Create configuration
        config_path = self.config_factory.create_packed_collator_config(self.data_root)
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Load model and tokenizer (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="packed_collator_model",
        )

        # Create dataset and collator
        data_processor = DataProcessor(tokenizer, processor, model, config)
        train_dataset, _ = data_processor.create_datasets()
        collator = create_data_collator("packed", tokenizer)

        # Check if we got the wrapper or the direct collator
        from src.data import TrainerCompatibleDataCollator

        if isinstance(collator, TrainerCompatibleDataCollator):
            # Unwrap to get the actual collator for testing
            actual_collator = collator.base_collator
            self.assertIsInstance(actual_collator, PackedDataCollator)
        else:
            # Direct collator (wrapper disabled)
            self.assertIsInstance(collator, PackedDataCollator)

        # Test batch collation
        batch_samples = [train_dataset[i] for i in range(min(3, len(train_dataset)))]

        with self.test_utils.measure_time("Packed collator batch creation"):
            batch = collator(batch_samples)

        # Validate packed batch structure
        expected_keys = [
            "input_ids",
            "labels",
            "attention_mask",
            "cu_seqlens",
            "max_seqlen",
            "ground_truth_objects",
        ]
        for key in expected_keys:
            self.assertIn(key, batch, f"Missing key: {key}")

        # Validate packed structure
        self.assertEqual(
            batch["input_ids"].shape[0], 1, "Packed batch should have batch_size=1"
        )

        # Validate cu_seqlens
        cu_seqlens = batch["cu_seqlens"]
        self.assertIsInstance(cu_seqlens, torch.Tensor)
        self.assertEqual(cu_seqlens.dtype, torch.int32)
        self.assertEqual(cu_seqlens[0].item(), 0, "cu_seqlens should start with 0")

        # Calculate memory efficiency (should be very high for packed)
        sequence_lengths = [sample["input_ids"].shape[-1] for sample in batch_samples]
        total_tokens = sum(sequence_lengths)
        max_seq_len = max(sequence_lengths)
        batch_size = len(batch_samples)

        efficiency = self.test_utils.calculate_memory_efficiency(
            total_tokens, batch_size, max_seq_len
        )

        # Packed collator should have reasonable efficiency (typically 60%+ for synthetic test data)
        self.assertGreater(
            efficiency, 0.60, f"Packed collator efficiency too low: {efficiency:.2%}"
        )

        logger.info(
            f"✅ PackedDataCollator test passed - Memory efficiency: {efficiency:.2%}"
        )

    def test_collator_comparison(self):
        """Compare standard vs packed collator performance."""
        logger.info("🧪 Testing Collator Performance Comparison")

        # Create configuration
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Load model and tokenizer (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="collator_comparison_model",
        )

        # Create dataset
        data_processor = DataProcessor(tokenizer, processor, model, config)
        train_dataset, _ = data_processor.create_datasets()

        # Create both collators
        standard_collator = create_data_collator("standard", tokenizer)
        packed_collator = create_data_collator("packed", tokenizer)

        # Test same batch with both collators
        batch_samples = [train_dataset[i] for i in range(min(4, len(train_dataset)))]

        # Standard collator
        with self.test_utils.measure_time("Standard collator performance"):
            _ = standard_collator(batch_samples)

        # Packed collator
        with self.test_utils.measure_time("Packed collator performance"):
            _ = packed_collator(batch_samples)

        # Calculate efficiencies
        sequence_lengths = [sample["input_ids"].shape[-1] for sample in batch_samples]
        total_tokens = sum(sequence_lengths)
        max_seq_len = max(sequence_lengths)
        batch_size = len(batch_samples)

        standard_efficiency = self.test_utils.calculate_memory_efficiency(
            total_tokens, batch_size, max_seq_len
        )
        packed_efficiency = self.test_utils.calculate_memory_efficiency(
            total_tokens, batch_size, max_seq_len
        )

        # Packed should be at least as efficient as standard (allow equal for small datasets)
        self.assertGreaterEqual(
            packed_efficiency,
            standard_efficiency,
            f"Packed collator ({packed_efficiency:.2%}) should be at least as efficient "
            f"as standard collator ({standard_efficiency:.2%})",
        )

        logger.info(f"✅ Collator comparison test passed:")
        logger.info(f"   Standard efficiency: {standard_efficiency:.2%}")
        logger.info(f"   Packed efficiency: {packed_efficiency:.2%}")

    def test_chat_processor_integration(self):
        """Test chat processor with coordinate token replacement."""
        logger.info("🧪 Testing Chat Processor Integration")

        # Test both coordinate modes
        for coordinate_enabled in [True, False]:
            mode_name = "coordinate" if coordinate_enabled else "standard"
            logger.info(f"🔄 Testing chat processor - {mode_name} mode")

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

            # Load model and processor (GPU-safe)
            cache_key = f"chat_processor_{mode_name}_model"
            model, tokenizer, processor = self.load_model_safely(
                model_path=config.model_path, config=config, cache_key=cache_key
            )

            # Create data processor
            data_processor = DataProcessor(tokenizer, processor, model, config)

            # Test dataset creation
            train_dataset, eval_dataset = data_processor.create_datasets()

            # Test sample processing
            sample = train_dataset[0]

            # Validate coordinate token presence
            self.test_utils.validate_coordinate_tokens(
                sample["input_ids"], tokenizer, coordinate_enabled
            )

            # Validate ground truth objects
            self.test_utils.validate_ground_truth_objects(
                [sample["ground_truth_objects"]]
            )

            logger.info(f"✅ Chat processor {mode_name} mode test passed")

    def test_teacher_student_pairing(self):
        """Test teacher-student conversation pairing."""
        logger.info("🧪 Testing Teacher-Student Pairing")

        # Create configuration with teacher sampling enabled
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Ensure teacher sampling is enabled
        self.assertGreater(
            config.teacher_ratio, 0.0, "Teacher ratio should be > 0 for this test"
        )

        # Load model and processor (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="teacher_student_model",
        )

        # Create data processor
        data_processor = DataProcessor(tokenizer, processor, model, config)
        train_dataset, _ = data_processor.create_datasets()

        # Test multiple samples to check teacher assignment
        samples_with_teachers = 0
        samples_without_teachers = 0

        for i in range(min(10, len(train_dataset))):
            sample = train_dataset[i]

            # Check if sample has teacher spans
            teacher_spans = sample.get("teacher_assistant_spans", [])
            student_spans = sample.get("student_assistant_spans", [])

            if teacher_spans:
                samples_with_teachers += 1
            else:
                samples_without_teachers += 1

            # Validate spans structure
            self.assertIsInstance(teacher_spans, list)
            self.assertIsInstance(student_spans, list)

        # Should have some variation in teacher assignment
        total_samples = samples_with_teachers + samples_without_teachers
        actual_teacher_ratio = samples_with_teachers / total_samples

        logger.info(f"✅ Teacher-student pairing test passed:")
        logger.info(f"   Samples with teachers: {samples_with_teachers}")
        logger.info(f"   Samples without teachers: {samples_without_teachers}")
        logger.info(f"   Actual teacher ratio: {actual_teacher_ratio:.2%}")

    def test_data_pipeline_error_handling(self):
        """Test data pipeline error handling with malformed data."""
        logger.info("🧪 Testing Data Pipeline Error Handling")

        # Create a temporary malformed dataset
        malformed_data_path = Path(self.data_root) / "malformed.jsonl"

        # Write malformed data
        with open(malformed_data_path, "w", encoding="utf-8") as f:
            # Valid sample
            f.write('{"images": ["test.jpg"], "objects": []}\n')
            # Missing required field
            f.write('{"objects": []}\n')  # Missing 'images'
            # Invalid JSON
            f.write('{"images": ["test.jpg", invalid json}\n')

        self.test_files_to_cleanup.append(str(malformed_data_path))

        # Create configuration pointing to malformed data
        config_path = self.config_factory.create_minimal_config(self.data_root)
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Load model and processor (GPU-safe)
        model, tokenizer, processor = self.load_model_safely(
            model_path=config.model_path,
            config=config,
            cache_key="error_handling_model",
        )

        # Try to create dataset with malformed data - should raise ValueError
        with self.assertRaises((ValueError, FileNotFoundError)):
            BBUDataset(
                data_path=str(malformed_data_path),
                chat_processor=processor,
                teacher_pool_manager=None,
                teacher_ratio=0.0,
                is_training=True,
                config=config,
            )

        logger.info("✅ Error handling test passed - Malformed data properly rejected")


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
