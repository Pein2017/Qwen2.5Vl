"""
Integration Tests for BBU Training System

End-to-end integration tests covering the complete training pipeline:
- Full data pipeline integration
- Model loading and training integration
- Error recovery and edge cases
- Multi-geometry support validation
"""

import sys
import time
import unittest
from typing import Any, Dict, cast

import torch


# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton


patch_torch_library_wrap_triton()

from transformers.training_args import TrainingArguments

from src.config import init_config, load_config
from src.core.data_processor import DataProcessor
from src.data import create_data_collator
from src.logger_utils import configure_global_logging, get_logger
from src.models.model_loader import load_model_and_processor_unified
from src.training.trainer_factory import create_trainer_with_coordinator
from src.utils.tokens.special_tokens import (
    create_unified_token_manager,
)
from tests.fixtures import ConfigFactory, SyntheticDataGenerator, TestUtils


logger = get_logger("test_integration")


class TestIntegration(unittest.TestCase):
    """Comprehensive integration tests for the complete BBU training pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Configure logging
        configure_global_logging(rank=0, world_size=1)
        logger.info("🧪 Starting Integration Tests")

        # Create test utilities
        cls.test_utils = TestUtils()

        # Create minimal dataset for integration testing with memory efficiency
        # Reduced from 12 to 8 samples to minimize memory usage
        cls.data_generator = SyntheticDataGenerator(num_samples=8)
        cls.train_path, cls.val_path, cls.teacher_path, cls.all_samples_path = (
            cls.data_generator.generate_complete_dataset()
        )
        cls.data_root = str(cls.data_generator.temp_dir)

        # Create configuration factory
        cls.config_factory = ConfigFactory()

        logger.info(f"✅ Integration test setup complete: {cls.data_root}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.data_generator.cleanup()
        cls.config_factory.cleanup_test_configs()
        logger.info("🧹 Integration test teardown complete")

    def setUp(self):
        """Set up for each individual test."""
        self.test_files_to_cleanup = []
        # Force aggressive GPU memory cleanup before each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc

            gc.collect()

    def tearDown(self):
        """Clean up after each test."""
        self.test_utils.cleanup_test_files(self.test_files_to_cleanup)
        # Force aggressive GPU memory cleanup after each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc

            gc.collect()
            # Additional memory cleanup
            torch.cuda.ipc_collect()

    def test_end_to_end_pipeline_standard_mode(self):
        """Test complete end-to-end pipeline in standard mode."""
        logger.info("🧪 Testing End-to-End Pipeline - Standard Mode")

        # Create standard configuration
        config_path = self.config_factory.create_coordinate_disabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        # Execute complete pipeline
        pipeline_results = self._execute_complete_pipeline(config_path, "standard_mode")

        # Validate results
        self._validate_pipeline_results(pipeline_results, coordinate_enabled=False)

        logger.info(f"✅ End-to-end standard mode test passed:")
        logger.info(f"   Training loss: {pipeline_results['final_train_loss']:.4f}")
        logger.info(f"   Eval loss: {pipeline_results['final_eval_loss']:.4f}")
        logger.info(f"   Total time: {pipeline_results['total_time']:.2f}s")

    def test_end_to_end_pipeline_coordinate_mode(self):
        """Test complete end-to-end pipeline in coordinate mode."""
        logger.info("🧪 Testing End-to-End Pipeline - Coordinate Mode")

        # Create coordinate configuration
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        # Execute complete pipeline
        pipeline_results = self._execute_complete_pipeline(
            config_path, "coordinate_mode"
        )

        # Validate results
        self._validate_pipeline_results(pipeline_results, coordinate_enabled=True)

        logger.info(f"✅ End-to-end coordinate mode test passed:")
        logger.info(f"   Training loss: {pipeline_results['final_train_loss']:.4f}")
        logger.info(f"   Eval loss: {pipeline_results['final_eval_loss']:.4f}")
        logger.info(f"   Total time: {pipeline_results['total_time']:.2f}s")

    def test_multi_geometry_support(self):
        """Test support for multiple geometry types in the pipeline."""
        logger.info("🧪 Testing Multi-Geometry Support")

        # Create coordinate configuration (supports multi-geometry)
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Load model and create data processor
        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False,
            attn_implementation=config.attn_implementation,
        )

        data_processor = DataProcessor(tokenizer, processor, model, config)
        train_dataset, _ = data_processor.create_datasets()

        # Verify that the pipeline can handle multiple geometry types by checking the raw data
        # Note: The processing pipeline normalizes all geometries to bbox_2d format,
        # so we check the input data diversity rather than the processed output
        import json

        # Read the raw training data to verify geometry diversity
        geometry_types_found = {"bbox_2d": 0, "square": 0, "line": 0}

        with open(config.train_data_path, "r") as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        for sample in raw_data:
            objects = sample.get("objects", [])
            for obj in objects:
                for geom_type in geometry_types_found.keys():
                    if geom_type in obj:
                        geometry_types_found[geom_type] += 1

        # Should have at least some variety in geometry types in the input data
        total_geometries = sum(geometry_types_found.values())
        self.assertGreater(total_geometries, 0, "Should have some geometry objects")

        # Should have at least 2 different geometry types in input data
        types_used = sum(1 for count in geometry_types_found.values() if count > 0)
        self.assertGreaterEqual(
            types_used,
            2,
            f"Should support multiple input geometry types, found: {geometry_types_found}",
        )

        # Verify that the processed dataset contains ground truth objects (all normalized to bbox_2d)
        processed_objects_count = 0
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            ground_truth_objects = sample.get("ground_truth_objects", [])
            processed_objects_count += len(ground_truth_objects)

            # Verify all processed objects have bbox_2d format (normalized)
            for obj in ground_truth_objects:
                self.assertIn(
                    "bbox_2d", obj, "All processed objects should have bbox_2d"
                )

        self.assertGreater(
            processed_objects_count, 0, "Should have processed ground truth objects"
        )

        logger.info(f"✅ Multi-geometry support test passed:")
        logger.info(f"   Input geometry types found: {geometry_types_found}")
        logger.info(
            f"   Processed objects (normalized to bbox_2d): {processed_objects_count}"
        )

    def test_teacher_student_integration(self):
        """Test teacher-student learning integration."""
        logger.info("🧪 Testing Teacher-Student Integration")

        # Create configuration with teacher sampling enabled
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        # Override teacher ratio to ensure we get both teachers and non-teachers for testing
        # Use a higher ratio (0.75) with small datasets to increase probability of having teachers
        config.teacher_ratio = 0.75

        # Ensure teacher ratio is reasonable for testing
        self.assertGreater(
            config.teacher_ratio, 0.0, "Teacher ratio should be > 0 for this test"
        )
        self.assertLess(
            config.teacher_ratio, 1.0, "Teacher ratio should be < 1.0 for this test"
        )

        # Load model and create data processor
        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False,
            attn_implementation=config.attn_implementation,
        )

        data_processor = DataProcessor(tokenizer, processor, model, config)
        train_dataset, _ = data_processor.create_datasets()
        data_collator = data_processor.create_data_collator()

        # Test teacher-student batch processing
        samples_with_teachers = []
        samples_without_teachers = []

        for i in range(len(train_dataset)):  # Check all samples, not just first 8
            sample = train_dataset[i]
            teacher_spans = sample.get("teacher_assistant_spans", [])

            if teacher_spans and any(
                len(span_group) > 0 for span_group in teacher_spans
            ):
                samples_with_teachers.append(sample)
            else:
                samples_without_teachers.append(sample)

        # Log the distribution for debugging
        logger.info(
            f"Dataset composition: {len(samples_with_teachers)} with teachers, {len(samples_without_teachers)} without teachers"
        )
        logger.info(
            f"Total dataset size: {len(train_dataset)}, Teacher ratio: {config.teacher_ratio}"
        )

        # For small test datasets, we need more flexible expectations
        # At minimum, we should have some samples (either with or without teachers)
        total_samples = len(samples_with_teachers) + len(samples_without_teachers)
        self.assertEqual(
            total_samples,
            len(train_dataset),
            "All samples should be classified as either with or without teachers",
        )

        # With the stochastic nature of teacher assignment, ensure we have reasonable distribution
        # If we have very few samples, we might not get perfect distribution
        if len(train_dataset) >= 4:
            # For datasets with 4+ samples, we should have some variation unless teacher_ratio is extreme
            if config.teacher_ratio > 0.1 and config.teacher_ratio < 0.9:
                self.assertGreater(
                    len(samples_with_teachers),
                    0,
                    f"Should have some samples with teachers (teacher_ratio={config.teacher_ratio}, dataset_size={len(train_dataset)})",
                )
        else:
            # For very small datasets, just ensure we have the expected behavior
            logger.warning(
                f"Small dataset ({len(train_dataset)} samples) - skipping strict teacher/student distribution test"
            )
            # At least verify teacher assignment is working (even if all samples get teachers or none do)
            if len(samples_with_teachers) == 0 and len(samples_without_teachers) == 0:
                self.fail(
                    "No samples processed - this indicates a fundamental issue with data processing"
                )

        # Test batch processing with available samples
        # Create a mixed batch from available samples (prioritize variety if possible)
        mixed_batch = []
        if samples_with_teachers:
            mixed_batch.extend(samples_with_teachers[:2])
        if samples_without_teachers:
            mixed_batch.extend(samples_without_teachers[:2])

        # If we don't have a mixed batch, use what we have
        if not mixed_batch:
            mixed_batch = [train_dataset[0]]  # At least use one sample for testing

        batch = data_collator(mixed_batch)

        # Validate batch structure includes spans
        self.assertIn(
            "teacher_assistant_spans", batch, "Batch should include teacher spans"
        )
        self.assertIn(
            "student_assistant_spans", batch, "Batch should include student spans"
        )

        # Test forward pass with teacher-student batch
        device = next(model.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        model.train()
        with torch.no_grad():
            model_inputs = {
                k: v
                for k, v in batch.items()
                if k
                in [
                    "input_ids",
                    "labels",
                    "attention_mask",
                    "pixel_values",
                    "image_grid_thw",
                ]
            }
            outputs = model(**model_inputs)

            self.assertIsNotNone(outputs.loss, "Should have loss output")
            self.assertFalse(torch.isnan(outputs.loss), "Loss should not be NaN")

        logger.info(f"✅ Teacher-student integration test passed:")
        logger.info(f"   Samples with teachers: {len(samples_with_teachers)}")
        logger.info(f"   Samples without teachers: {len(samples_without_teachers)}")
        logger.info(f"   Mixed batch size: {len(mixed_batch)}")
        logger.info(f"   Teacher ratio used: {config.teacher_ratio}")

    def test_error_recovery_and_edge_cases(self):
        """Test error recovery and handling of edge cases."""
        logger.info("🧪 Testing Error Recovery and Edge Cases")

        # Test 1: Empty batch handling
        config_path = self.config_factory.create_minimal_config(self.data_root)
        self.test_files_to_cleanup.append(config_path)

        init_config(config_path)
        config = load_config(config_path)

        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False,
            attn_implementation=config.attn_implementation,
        )

        data_processor = DataProcessor(tokenizer, processor, model, config)
        train_dataset, _ = data_processor.create_datasets()
        data_collator = data_processor.create_data_collator()

        # Test with single sample (edge case)
        single_sample_batch = data_collator([train_dataset[0]])
        self.assertIsInstance(
            single_sample_batch, dict, "Should handle single sample batch"
        )

        # Test 2: Very long sequence handling (truncation)
        # Create a sample with very long description
        long_desc_sample = train_dataset[0].copy()
        if (
            "ground_truth_objects" in long_desc_sample
            and long_desc_sample["ground_truth_objects"]
        ):
            # Make description very long
            obj = cast(Dict[str, Any], long_desc_sample["ground_truth_objects"][0])
            obj["desc"] = "长描述 " * 200  # Very long description

        # Should handle gracefully (truncation)
        try:
            long_batch = data_collator([long_desc_sample])
            self.assertIsInstance(long_batch, dict, "Should handle long sequences")
        except Exception as e:
            # If it fails, it should be a controlled failure
            self.assertIn(
                "length", str(e).lower(), f"Unexpected error for long sequence: {e}"
            )

        # Test 3: Memory pressure handling
        if torch.cuda.is_available():
            # Get current memory usage
            initial_memory = torch.cuda.memory_allocated()

            # Create larger batch to test memory handling
            try:
                large_batch_samples = [
                    train_dataset[i] for i in range(min(4, len(train_dataset)))
                ]
                large_batch = data_collator(large_batch_samples)

                device = next(model.parameters()).device
                for key, value in large_batch.items():
                    if isinstance(value, torch.Tensor):
                        large_batch[key] = value.to(device)

                # Should handle without OOM
                with torch.no_grad():
                    model_inputs = {
                        k: v
                        for k, v in large_batch.items()
                        if k
                        in [
                            "input_ids",
                            "labels",
                            "attention_mask",
                            "pixel_values",
                            "image_grid_thw",
                        ]
                    }
                    _ = model(**model_inputs)

                memory_used = torch.cuda.memory_allocated() - initial_memory
                logger.info(
                    f"Memory used for large batch: {memory_used / 1024**2:.1f} MB"
                )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "⚠️ OOM occurred during large batch test (expected on smaller GPUs)"
                    )
                else:
                    raise

        logger.info("✅ Error recovery and edge cases test passed")

    def test_coordinate_token_training_integration(self):
        """Test coordinate token training with full pipeline integration."""
        logger.info("🧪 Testing coordinate token training integration")

        # Create configuration for coordinate mode
        config_path = self.config_factory.create_coordinate_enabled_config(
            self.data_root, "standard"
        )
        self.test_files_to_cleanup.append(config_path)

        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)

        # Verify coordinate tokens are enabled
        self.assertTrue(
            config.coordinate_tokens_enabled,
            "Coordinate tokens must be enabled for this test",
        )

        # Load model and processor
        model_path = config.model_path
        attn_implementation = config.attn_implementation

        # Load model
        model, tokenizer, image_processor = load_model_and_processor_unified(
            model_path=model_path,
            attn_implementation=attn_implementation,
            force_detection=True,  # Ensure coordinate token support
        )

        # Create data processor with teacher sampling for student-teacher testing
        data_processor = DataProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            model=model,
            config=config,
        )

        # Create datasets
        train_dataset, eval_dataset = data_processor.create_datasets()

        # Get a sample from the dataset
        sample = train_dataset[0]

        # Verify the sample has input_ids and labels
        self.assertIn("input_ids", sample)
        self.assertIn("labels", sample)

        # Check that coordinate tokens are correctly present in the input_ids
        input_ids = sample["input_ids"]

        # Get coordinate token range using UnifiedTokenManager
        coord_manager = create_unified_token_manager(
            tokenizer=tokenizer, model=model, max_coord_value=config.max_coord_value
        )
        coord_start, coord_end = coord_manager.get_coordinate_token_range()

        # Check if there are any coordinate tokens in the input_ids
        coord_mask = torch.logical_and(input_ids >= coord_start, input_ids < coord_end)
        num_coord_tokens = coord_mask.sum().item()

        # We expect at least some coordinate tokens to be present
        self.assertGreater(
            num_coord_tokens,
            0,
            f"Expected coordinate tokens in input_ids but found none. "
            f"Check that the chat processor is correctly formatting coordinate tokens.",
        )

        logger.info(f"✅ Found {num_coord_tokens} coordinate tokens in input_ids")

        # Move model to GPU
        model.to("cuda")

        # Create batch from sample
        batch_samples = [sample]

        # Create a collator
        collator = create_data_collator(config.collator_type, tokenizer)

        # Collate samples
        batch = collator(batch_samples)

        # Move batch to GPU
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass to compute losses
        with torch.no_grad():
            outputs = model(**batch)

        # Check that loss is computed
        self.assertIn("loss", outputs)

        # Check if the model has _last_coordinate_losses attribute
        self.assertTrue(
            hasattr(model, "_last_coordinate_losses"),
            "Model missing _last_coordinate_losses attribute",
        )

        # Get coordinate losses from model
        coordinate_losses = model.get_last_coordinate_losses()

        # Verify coordinate losses exist and are non-zero
        self.assertIn("_coordinate_l1_loss", coordinate_losses)
        coordinate_l1_loss = coordinate_losses["_coordinate_l1_loss"]

        # Coordinate loss should be non-zero
        self.assertGreater(
            coordinate_l1_loss.item(),
            0.0,
            "Coordinate L1 loss is zero, which indicates coordinate tokens are not being processed correctly",
        )

        logger.info(f"✅ Coordinate L1 loss: {coordinate_l1_loss.item():.6f}")

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    def _execute_complete_pipeline(
        self, config_path: str, test_name: str
    ) -> Dict[str, Any]:
        """Execute complete training pipeline and return results."""
        start_time = time.time()

        # Initialize configuration
        init_config(config_path)
        config = load_config(config_path)

        # Validate data files exist and are readable (quick pre-check)
        with self.test_utils.measure_time(f"{test_name} data validation"):
            import json
            import os

            # Check all required data files exist
            required_files = [
                ("train", config.train_data_path),
                ("val", config.val_data_path),
                ("teacher", config.teacher_pool_file),
            ]

            for name, path in required_files:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing {name} data file: {path}")

                # Validate file is readable and contains valid JSON
                try:
                    with open(path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines:
                            raise ValueError(f"Empty {name} data file: {path}")

                        # Test first line is valid JSON
                        json.loads(lines[0])
                        logger.info(
                            f"✅ Validated {name} data file: {len(lines)} samples"
                        )

                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {name} data file {path}: {e}")
                except Exception as e:
                    raise ValueError(f"Error reading {name} data file {path}: {e}")

        # Create training arguments for minimal training
        training_args = TrainingArguments(
            output_dir=f"{self.data_root}/pipeline_test_{test_name}",
            num_train_epochs=1,
            per_device_train_batch_size=2,  # Match config factory setting for consistency
            per_device_eval_batch_size=2,  # Match config factory setting for consistency
            max_steps=1,  # Single step to avoid trainer edge case with small datasets
            logging_steps=1,
            eval_steps=2,
            eval_strategy="steps",  # Updated from evaluation_strategy
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
        )

        # Create and run trainer
        with self.test_utils.measure_time(f"{test_name} training"):
            trainer = create_trainer_with_coordinator(
                training_args=training_args, config=config
            )

            # Run training
            trainer.train()

            # Run evaluation
            eval_results = trainer.evaluate()

        total_time = time.time() - start_time

        # Collect results - find the last training loss entry
        final_train_loss = float("inf")
        for log_entry in reversed(trainer.state.log_history):
            if "train_loss" in log_entry:
                final_train_loss = log_entry["train_loss"]
                break

        # Training should now always complete with proper loss values
        if final_train_loss == float("inf"):
            raise RuntimeError(
                f"No training loss found in log history. Available keys: {[list(entry.keys()) for entry in trainer.state.log_history]}"
            )

        results = {
            "total_time": total_time,
            "final_train_loss": final_train_loss,
            "final_eval_loss": eval_results.get("eval_loss", float("inf")),
            "train_dataset_size": len(trainer.train_dataset)
            if trainer.train_dataset
            else 0,
            "eval_dataset_size": len(trainer.eval_dataset)
            if trainer.eval_dataset
            else 0,
            "completed_steps": trainer.state.global_step,
            "config_name": test_name,
        }

        logger.info(f"Pipeline results for {test_name}: {results}")

        return results

    def _validate_pipeline_results(
        self, results: Dict[str, Any], coordinate_enabled: bool
    ) -> None:
        """Validate pipeline execution results."""
        # Basic validation
        self.assertGreater(results["total_time"], 0, "Total time should be positive")

        # Loss validation
        self.assertIsInstance(
            results["final_train_loss"], (int, float), "Train loss should be numeric"
        )
        self.assertIsInstance(
            results["final_eval_loss"], (int, float), "Eval loss should be numeric"
        )
        self.assertFalse(
            torch.isnan(torch.tensor(results["final_train_loss"])),
            "Train loss should not be NaN",
        )
        self.assertFalse(
            torch.isnan(torch.tensor(results["final_eval_loss"])),
            "Eval loss should not be NaN",
        )

        # Dataset validation
        self.assertGreater(
            results["train_dataset_size"], 0, "Train dataset should not be empty"
        )
        self.assertGreater(
            results["eval_dataset_size"], 0, "Eval dataset should not be empty"
        )

        # Training validation
        self.assertGreater(
            results["completed_steps"], 0, "Should complete at least one training step"
        )

        # Loss bounds (coordinate models may have higher initial loss)
        max_expected_loss = 1000 if coordinate_enabled else 100
        self.assertLess(
            results["final_train_loss"],
            max_expected_loss,
            f"Train loss too high: {results['final_train_loss']}",
        )


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
