"""
Test Utilities for BBU Training Pipeline

Common utilities and helper functions for comprehensive testing.
"""

import gc
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.logger_utils import get_logger


logger = get_logger("test_utils")


class TestUtils:
    """Professional test utilities for BBU training pipeline testing."""

    @staticmethod
    def validate_tensor_shape(
        tensor: torch.Tensor,
        expected_shape: Tuple[int, ...],
        tensor_name: str = "tensor",
    ) -> None:
        """
        Validate tensor shape with descriptive error messages.

        Args:
            tensor: Tensor to validate
            expected_shape: Expected tensor shape
            tensor_name: Name of tensor for error messages

        Raises:
            AssertionError: If tensor shape doesn't match expected shape
        """
        actual_shape = tensor.shape
        assert actual_shape == expected_shape, (
            f"{tensor_name} shape mismatch: expected {expected_shape}, "
            f"got {actual_shape}"
        )

    @staticmethod
    def validate_tensor_dtype(
        tensor: torch.Tensor, expected_dtype: torch.dtype, tensor_name: str = "tensor"
    ) -> None:
        """
        Validate tensor data type.

        Args:
            tensor: Tensor to validate
            expected_dtype: Expected tensor dtype
            tensor_name: Name of tensor for error messages

        Raises:
            AssertionError: If tensor dtype doesn't match expected dtype
        """
        actual_dtype = tensor.dtype
        assert actual_dtype == expected_dtype, (
            f"{tensor_name} dtype mismatch: expected {expected_dtype}, "
            f"got {actual_dtype}"
        )

    @staticmethod
    def validate_batch_consistency(batch: Dict[str, Any], batch_size: int) -> None:
        """
        Validate batch consistency across all tensors.

        Args:
            batch: Batch dictionary containing tensors
            batch_size: Expected batch size

        Raises:
            AssertionError: If batch is inconsistent
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Special handling for pixel_values and image_grid_thw which are concatenated
                if key in ["pixel_values", "image_grid_thw"]:
                    # These tensors are concatenated across all images in the batch
                    # so they don't follow the standard batch_size pattern
                    continue

                actual_batch_size = value.shape[0]
                assert actual_batch_size == batch_size, (
                    f"Batch size mismatch in {key}: expected {batch_size}, "
                    f"got {actual_batch_size}"
                )

    @staticmethod
    def calculate_memory_efficiency(
        total_tokens: int, batch_size: int, max_seq_len: int
    ) -> float:
        """
        Calculate memory efficiency for packed sequences.

        Args:
            total_tokens: Total number of tokens in batch
            batch_size: Batch size
            max_seq_len: Maximum sequence length in batch

        Returns:
            Memory efficiency as a float between 0 and 1
        """
        padded_tokens = batch_size * max_seq_len
        if padded_tokens == 0:
            return 0.0
        return total_tokens / padded_tokens

    @staticmethod
    def validate_attention_mask(
        attention_mask: torch.Tensor, sequence_lengths: List[int]
    ) -> None:
        """
        Validate attention mask consistency with sequence lengths.

        Args:
            attention_mask: Attention mask tensor
            sequence_lengths: List of sequence lengths for each sample

        Raises:
            AssertionError: If attention mask is inconsistent
        """
        batch_size = len(sequence_lengths)
        assert attention_mask.shape[0] == batch_size, (
            f"Attention mask batch size {attention_mask.shape[0]} doesn't match "
            f"sequence lengths batch size {batch_size}"
        )

        for i, seq_len in enumerate(sequence_lengths):
            actual_len = attention_mask[i].sum().item()
            assert actual_len == seq_len, (
                f"Attention mask row {i} has {actual_len} true values, "
                f"expected {seq_len}"
            )

    @staticmethod
    def validate_coordinate_tokens(
        input_ids: torch.Tensor, tokenizer: Any, coordinate_enabled: bool
    ) -> None:
        """
        Validate coordinate token setup and usage in input_ids.

        Args:
            input_ids: Input token IDs
            tokenizer: Tokenizer instance
            coordinate_enabled: Whether coordinate tokens should be enabled

        Raises:
            AssertionError: If coordinate token setup doesn't match expectation
        """
        vocab_size = len(tokenizer.get_vocab())
        base_vocab_size = 151665  # Standard Qwen2.5-VL vocab size

        if coordinate_enabled:
            # Check that vocabulary has been extended beyond base size
            assert vocab_size > base_vocab_size, (
                f"Coordinate tokens enabled but vocabulary not extended: "
                f"size={vocab_size}, expected > {base_vocab_size}"
            )

            # Verify extended vocabulary includes geometry tokens
            vocab = tokenizer.get_vocab()
            geometry_tokens = [
                "<|box_start|>",
                "<|box_end|>",
                "<|square_start|>",
                "<|square_end|>",
                "<|line_start|>",
                "<|line_end|>",
            ]
            for token in geometry_tokens:
                assert token in vocab, (
                    f"Required geometry token {token} not found in vocabulary"
                )

            # Check that input tokens are within vocabulary bounds
            max_token_id = torch.max(input_ids).item()
            assert max_token_id < vocab_size, (
                f"Found token ID {max_token_id} exceeding vocabulary size {vocab_size}"
            )
        else:
            # Check that vocabulary is close to standard size (minimal extension)
            # Allow some extension for basic geometry tokens but not coordinate tokens
            assert vocab_size <= base_vocab_size + 10, (
                f"Coordinate tokens disabled but vocabulary heavily extended: "
                f"size={vocab_size}, expected <= {base_vocab_size + 10}"
            )

            # Check that no tokens exceed vocabulary
            max_token_id = torch.max(input_ids).item()
            assert max_token_id < vocab_size, (
                f"Found token ID {max_token_id} exceeding vocabulary size {vocab_size}"
            )

    @staticmethod
    def validate_ground_truth_objects(
        ground_truth_objects: List[List[Dict[str, Any]]],
    ) -> None:
        """
        Validate ground truth objects structure.

        Args:
            ground_truth_objects: List of ground truth objects per sample

        Raises:
            AssertionError: If ground truth objects are malformed
        """
        assert isinstance(ground_truth_objects, list), (
            f"Ground truth objects must be a list, got {type(ground_truth_objects)}"
        )

        for sample_idx, sample_objects in enumerate(ground_truth_objects):
            assert isinstance(sample_objects, list), (
                f"Sample {sample_idx} ground truth objects must be a list, "
                f"got {type(sample_objects)}"
            )

            for obj_idx, obj in enumerate(sample_objects):
                assert isinstance(obj, dict), (
                    f"Sample {sample_idx}, object {obj_idx} must be a dict, "
                    f"got {type(obj)}"
                )
                assert "desc" in obj, (
                    f"Sample {sample_idx}, object {obj_idx} missing 'desc' field"
                )

    @staticmethod
    @contextmanager
    def measure_time(operation_name: str):
        """
        Context manager for measuring execution time.

        Args:
            operation_name: Name of operation being measured

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ {operation_name} took {elapsed_time:.2f} seconds")

    @staticmethod
    @contextmanager
    def measure_memory(operation_name: str):
        """
        Context manager for measuring GPU memory usage.

        Args:
            operation_name: Name of operation being measured

        Yields:
            None
        """
        if not torch.cuda.is_available():
            logger.warning(
                f"⚠️ CUDA not available, skipping memory measurement for {operation_name}"
            )
            yield
            return

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        try:
            yield
        finally:
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            logger.info(f"🧠 {operation_name} memory usage:")
            logger.info(f"   Initial: {initial_memory / 1024**2:.1f} MB")
            logger.info(f"   Final: {final_memory / 1024**2:.1f} MB")
            logger.info(f"   Peak: {peak_memory / 1024**2:.1f} MB")
            logger.info(f"   Delta: {(final_memory - initial_memory) / 1024**2:.1f} MB")

    @staticmethod
    def validate_model_forward_output(outputs: Any, coordinate_enabled: bool) -> None:
        """
        Validate model forward pass output.

        Args:
            outputs: Model forward pass outputs
            coordinate_enabled: Whether coordinate tokens are enabled

        Raises:
            AssertionError: If outputs are invalid
        """
        assert hasattr(outputs, "loss"), "Model outputs missing 'loss' attribute"
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Loss must be a tensor, got {type(outputs.loss)}"
        )
        assert outputs.loss.dim() == 0, (
            f"Loss must be scalar, got shape {outputs.loss.shape}"
        )
        assert not torch.isnan(outputs.loss), "Loss is NaN"
        assert not torch.isinf(outputs.loss), "Loss is infinite"

        if coordinate_enabled:
            # For coordinate models, loss should be reasonable (not too large)
            assert outputs.loss.item() < 1000, (
                f"Loss too large for coordinate model: {outputs.loss.item()}"
            )

    @staticmethod
    def create_mock_sample() -> Dict[str, Any]:
        """
        Create a mock sample for unit testing.

        Returns:
            Mock sample dictionary
        """
        return {
            "images": ["images/mock_sample.jpg"],
            "objects": [
                {"desc": "测试设备/显示完整/模拟数据", "bbox_2d": [100, 50, 200, 150]}
            ],
            "width": 420,
            "height": 924,
        }

    @staticmethod
    def validate_config_requirements(config: Any) -> None:
        """
        Validate that config has all required attributes for testing.

        Args:
            config: Configuration object to validate

        Raises:
            AssertionError: If required attributes are missing
        """
        required_attrs = [
            "coordinate_tokens_enabled",
            "max_coord_value",
            "collator_type",
            "model_path",
            "max_total_length",
            "per_device_train_batch_size",
        ]

        for attr in required_attrs:
            assert hasattr(config, attr), f"Config missing required attribute: {attr}"

    @staticmethod
    def cleanup_test_files(file_paths: List[str]) -> None:
        """
        Clean up test files safely.

        Args:
            file_paths: List of file paths to remove
        """
        for file_path in file_paths:
            try:
                path_obj = Path(file_path)
                if path_obj.exists():
                    if path_obj.is_file():
                        path_obj.unlink()
                    elif path_obj.is_dir():
                        import shutil

                        shutil.rmtree(path_obj)
                    logger.debug(f"🧹 Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup {file_path}: {e}")

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """
        Get current GPU memory information.

        Returns:
            Dictionary with memory information in MB
        """
        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

    @staticmethod
    def force_gpu_memory_cleanup(verbose: bool = True) -> None:
        """
        Force comprehensive GPU memory cleanup.

        Args:
            verbose: Whether to log memory cleanup details
        """
        if not torch.cuda.is_available():
            if verbose:
                logger.info("🔧 CUDA not available, skipping GPU memory cleanup")
            return

        # Get initial memory state
        initial_allocated = torch.cuda.memory_allocated() / 1024**2
        initial_reserved = torch.cuda.memory_reserved() / 1024**2

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache multiple times for thorough cleanup
        for i in range(3):
            torch.cuda.empty_cache()
            if i < 2:  # Small delay between cache clears
                import time

                time.sleep(0.1)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Get final memory state
        final_allocated = torch.cuda.memory_allocated() / 1024**2
        final_reserved = torch.cuda.memory_reserved() / 1024**2

        if verbose:
            logger.info(f"🧹 GPU Memory Cleanup Complete:")
            logger.info(
                f"   Allocated: {initial_allocated:.1f} MB → {final_allocated:.1f} MB (-{initial_allocated - final_allocated:.1f} MB)"
            )
            logger.info(
                f"   Reserved: {initial_reserved:.1f} MB → {final_reserved:.1f} MB (-{initial_reserved - final_reserved:.1f} MB)"
            )

    @staticmethod
    @contextmanager
    def gpu_memory_guard(operation_name: str, cleanup_after: bool = True):
        """
        Context manager that ensures GPU memory is cleaned up after operation.

        Args:
            operation_name: Name of operation for logging
            cleanup_after: Whether to force cleanup after operation

        Yields:
            None
        """
        if not torch.cuda.is_available():
            logger.info(f"🔧 {operation_name} - CUDA not available")
            yield
            return

        # Initial memory state
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        logger.info(
            f"🚀 {operation_name} - Initial GPU memory: {initial_memory:.1f} MB"
        )

        try:
            yield
        finally:
            if cleanup_after:
                TestUtils.force_gpu_memory_cleanup(verbose=True)
            else:
                final_memory = torch.cuda.memory_allocated() / 1024**2
                logger.info(
                    f"✅ {operation_name} - Final GPU memory: {final_memory:.1f} MB (+{final_memory - initial_memory:.1f} MB)"
                )

    @staticmethod
    def cleanup_model_references(
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
    ) -> None:
        """
        Clean up model references and move them off GPU.

        Args:
            model: Model to cleanup (optional)
            tokenizer: Tokenizer to cleanup (optional)
            processor: Processor to cleanup (optional)
        """
        objects_cleaned = []

        if model is not None:
            try:
                # Move model to CPU if it has parameters
                if hasattr(model, "cpu"):
                    model.cpu()
                    objects_cleaned.append("model")
                # Delete model reference
                del model
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up model: {e}")

        if tokenizer is not None:
            try:
                del tokenizer
                objects_cleaned.append("tokenizer")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up tokenizer: {e}")

        if processor is not None:
            try:
                del processor
                objects_cleaned.append("processor")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up processor: {e}")

        if objects_cleaned:
            logger.info(f"🧹 Cleaned up: {', '.join(objects_cleaned)}")

        # Force garbage collection
        gc.collect()

    @staticmethod
    def check_gpu_memory_available(required_mb: float = 1000.0) -> bool:
        """
        Check if enough GPU memory is available for operation.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if enough memory is available
        """
        if not torch.cuda.is_available():
            return False

        # Get total and allocated memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
        available_memory = total_memory - allocated_memory

        logger.info(
            f"💾 GPU Memory Check: {available_memory:.1f} MB available, {required_mb:.1f} MB required"
        )

        return available_memory >= required_mb
