"""
GPU-Aware Test Base Class

Provides base functionality for tests that require GPU memory management.
Ensures proper cleanup between tests to prevent CUDA OOM errors.
"""

import sys
import unittest
from typing import Any, Optional, Tuple

import torch


# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

from src.logger_utils import get_logger
from tests.fixtures.test_utils import TestUtils


logger = get_logger("gpu_test_base")


class GPUAwareTestCase(unittest.TestCase):
    """
    Base test case with GPU memory management capabilities.

    Features:
    - Automatic GPU memory cleanup between tests
    - Memory usage monitoring
    - Safe model loading/unloading
    - GPU availability checks
    """

    # Class-level model cache to avoid reloading
    _cached_models = {}
    _model_load_count = 0
    MAX_MODEL_LOADS_PER_TEST = 1  # Limit model loads per test method

    @classmethod
    def setUpClass(cls):
        """Set up test class with GPU memory management."""
        super().setUpClass()

        # Log GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU Available: {gpu_name} ({total_memory:.1f} GB)")
        else:
            logger.warning("⚠️ No GPU available - tests will run on CPU")

        # Initial memory cleanup
        TestUtils.force_gpu_memory_cleanup(verbose=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level resources."""
        super().tearDownClass()

        # Clear cached models
        for model_key in list(cls._cached_models.keys()):
            model, tokenizer, processor = cls._cached_models.pop(model_key)
            TestUtils.cleanup_model_references(model, tokenizer, processor)

        # Final cleanup
        TestUtils.force_gpu_memory_cleanup(verbose=True)

    def setUp(self):
        """Set up for each test with memory monitoring."""
        super().setUp()

        # Reset test-level model load counter
        self._model_load_count = 0

        # Check initial GPU state
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            logger.info(
                f"🧪 Test {self._testMethodName} - Initial GPU memory: {initial_memory:.1f} MB"
            )

        # Track cleanup items
        self.test_files_to_cleanup = []
        self.models_to_cleanup = []

    def tearDown(self):
        """Clean up after each test."""
        super().tearDown()

        # Clean up models loaded during test
        for model_data in self.models_to_cleanup:
            if isinstance(model_data, tuple) and len(model_data) == 3:
                model, tokenizer, processor = model_data
                TestUtils.cleanup_model_references(model, tokenizer, processor)

        # Clean up test files
        if hasattr(self, "test_utils"):
            self.test_utils.cleanup_test_files(self.test_files_to_cleanup)  # type: ignore

        # Force GPU memory cleanup after each test
        TestUtils.force_gpu_memory_cleanup(verbose=True)

    def load_model_safely(
        self, model_path: str, config: Any, cache_key: Optional[str] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Load model with GPU memory management and caching.

        Args:
            model_path: Path to model
            config: Configuration object
            cache_key: Optional cache key for model reuse

        Returns:
            Tuple of (model, tokenizer, processor)
        """
        # Limit model loads per test
        if self._model_load_count >= self.MAX_MODEL_LOADS_PER_TEST:
            # Force cleanup before loading new model
            TestUtils.force_gpu_memory_cleanup(verbose=True)
            logger.warning(
                f"⚠️ Test {self._testMethodName} exceeded model load limit, forced cleanup"
            )

        # Check if model is cached
        if cache_key and cache_key in self._cached_models:
            logger.info(f"🔄 Using cached model: {cache_key}")
            model, tokenizer, processor = self._cached_models[cache_key]
            self._model_load_count += 1
            return model, tokenizer, processor

        # Check GPU memory availability
        if not TestUtils.check_gpu_memory_available(required_mb=2000.0):
            TestUtils.force_gpu_memory_cleanup(verbose=True)

        # Load model with memory guard
        with TestUtils.gpu_memory_guard(
            f"Model loading ({model_path})", cleanup_after=False
        ):
            from src.models.model_loader import load_model_and_processor_unified

            model, tokenizer, processor = load_model_and_processor_unified(
                model_path=model_path,
                for_inference=False,
                attn_implementation=config.attn_implementation,
            )

            self._model_load_count += 1

            # Cache model if cache key provided
            if cache_key:
                self._cached_models[cache_key] = (model, tokenizer, processor)
                logger.info(f"💾 Cached model: {cache_key}")
            else:
                # Add to cleanup list if not cached
                self.models_to_cleanup.append((model, tokenizer, processor))

            return model, tokenizer, processor

    def run_with_memory_guard(
        self, operation_name: str, operation_func, *args, **kwargs
    ):
        """
        Run operation with GPU memory monitoring.

        Args:
            operation_name: Name for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments for function

        Returns:
            Function result
        """
        with TestUtils.gpu_memory_guard(operation_name, cleanup_after=False):
            return operation_func(*args, **kwargs)

    def skip_if_no_gpu(self):
        """Skip test if GPU is not available."""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")

    def skip_if_insufficient_memory(self, required_mb: float = 2000.0):
        """Skip test if insufficient GPU memory available."""
        if not TestUtils.check_gpu_memory_available(required_mb):
            self.skipTest(f"Insufficient GPU memory (required: {required_mb:.1f} MB)")
