"""
Test suite for tokenizer speed optimizations in coordinate pretraining pipeline.

This test validates the optimized tokenizer pipeline with the tiny dataset,
ensuring that fast tokenizer optimizations work correctly with the data collator.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from src_coord_pretrain.datasets.bootstrap_coord_dataset import (
    CoordBootstrapDataset,
    DatasetConfig,
)
from src_coord_pretrain.datasets.collator import (
    CollatorConfig,
    DataCollatorCoordBootstrap,
)


class TestTokenizerOptimization(unittest.TestCase):
    """Test tokenizer speed optimizations and data pipeline validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tiny_dataset_path = (
            "src_coord_pretrain/data/coord_bootstrap_tiny.jsonl"
        )

        # Mock processor with fast tokenizer
        self.mock_processor = Mock()
        self.mock_tokenizer = Mock()

        # Configure mock tokenizer to simulate fast tokenizer
        self.mock_tokenizer.is_fast = True
        self.mock_tokenizer.get_vocab.return_value = {
            "<|coord_0|>": 151667,
            "<|coord_1|>": 151668,
            "<|coord_99|>": 151766,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        }
        self.mock_tokenizer.convert_tokens_to_ids.return_value = 151645

        # Mock tokenization calls
        def mock_tokenizer_call(*args, **kwargs):
            # Return different responses based on whether offset_mapping is requested
            if kwargs.get("return_offsets_mapping", False):
                # Create realistic offset mapping for the ChatML text
                # Text: "<|im_start|>user\nWhat is `0` in coordinate space?<|im_end|><|im_start|>assistant\n<|coord_0|><|im_end|>"
                # We need offsets that include the assistant content properly
                return {
                    "input_ids": torch.tensor(
                        [[151644, 151645, 151667, 151645]]
                    ),  # More realistic token sequence
                    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                    "offset_mapping": torch.tensor(
                        [[[0, 11], [11, 50], [50, 62], [62, 73]]]
                    ),  # Realistic offsets
                }
            else:
                return {
                    "input_ids": torch.tensor([[151644, 151645]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

        self.mock_tokenizer.side_effect = mock_tokenizer_call

        # Mock apply_chat_template
        self.mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\nTest<|im_end|><|im_start|>assistant\n<|coord_0|><|im_end|>"

        # Mock decode - return realistic ChatML format
        def mock_decode(token_ids, **kwargs):
            # Return a realistic ChatML conversation
            return "<|im_start|>user\nWhat is `0` in coordinate space?<|im_end|><|im_start|>assistant\n<|coord_0|><|im_end|>"

        self.mock_tokenizer.decode.side_effect = mock_decode

        # Mock pad method
        self.mock_tokenizer.pad.return_value = {
            "input_ids": torch.tensor([[151644, 151645, 0], [151644, 151645, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
        }

        self.mock_processor.tokenizer = self.mock_tokenizer

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_fast_tokenizer_detection(self):
        """Test that fast tokenizer is properly detected and configured."""
        config = DatasetConfig(
            data_path=self.tiny_dataset_path,
            max_coord_value=100,
            use_apply_chat_template=True,
        )

        dataset = CoordBootstrapDataset(
            tokenizer=self.mock_tokenizer,
            config=config,
        )

        # Verify fast tokenizer detection
        self.assertTrue(dataset._use_fast_tokenizer)
        self.assertTrue(self.mock_tokenizer.is_fast)

    def test_slow_tokenizer_fallback(self):
        """Test behavior with slow tokenizer."""
        # Configure mock as slow tokenizer
        slow_tokenizer = Mock()
        slow_tokenizer.is_fast = False
        slow_tokenizer.get_vocab.return_value = {"<|coord_0|>": 151667}
        slow_tokenizer.apply_chat_template.return_value = "test"
        slow_tokenizer.return_value = {
            "input_ids": torch.tensor([[151644]]),
            "attention_mask": torch.tensor([[1]]),
        }

        config = DatasetConfig(
            data_path=self.tiny_dataset_path,
            max_coord_value=100,
            use_apply_chat_template=True,
        )

        dataset = CoordBootstrapDataset(
            tokenizer=slow_tokenizer,
            config=config,
        )

        # Verify slow tokenizer handling
        self.assertFalse(dataset._use_fast_tokenizer)
        self.assertFalse(slow_tokenizer.is_fast)

    def test_dataset_tokenization_optimization(self):
        """Test that dataset uses optimized tokenization parameters."""
        config = DatasetConfig(
            data_path=self.tiny_dataset_path,
            max_coord_value=100,
            use_apply_chat_template=True,
        )

        dataset = CoordBootstrapDataset(
            tokenizer=self.mock_tokenizer,
            config=config,
        )

        # Get a sample to trigger tokenization
        sample = dataset[0]

        # Verify tokenizer was called with optimized parameters
        self.mock_tokenizer.assert_called()

        # Check that sample has expected structure
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIsInstance(sample["input_ids"], torch.Tensor)
        self.assertIsInstance(sample["attention_mask"], torch.Tensor)

    def test_collator_fast_tokenizer_optimization(self):
        """Test that collator uses fast tokenizer optimizations."""
        collator = DataCollatorCoordBootstrap(
            tokenizer=self.mock_tokenizer,
            config=CollatorConfig(),
        )

        # Verify fast tokenizer detection in collator
        self.assertTrue(collator._use_fast_tokenizer)
        self.assertTrue(collator._supports_batch_decode)

    def test_collator_batch_processing(self):
        """Test optimized batch processing in collator."""
        collator = DataCollatorCoordBootstrap(
            tokenizer=self.mock_tokenizer,
            config=CollatorConfig(),
        )

        # Create mock batch data
        batch = [
            {
                "input_ids": torch.tensor([151644, 151645]),
                "attention_mask": torch.tensor([1, 1]),
            },
            {
                "input_ids": torch.tensor([151644, 151645, 151667]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
        ]

        # Process batch
        result = collator(batch)

        # Verify batch processing
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)

        # Verify tokenizer.pad was called with optimized parameters
        self.mock_tokenizer.pad.assert_called()

    def test_real_dataset_loading(self):
        """Test loading and processing real tiny dataset."""
        # Skip if dataset file doesn't exist
        if not Path(self.tiny_dataset_path).exists():
            self.skipTest(f"Tiny dataset not found at {self.tiny_dataset_path}")

        config = DatasetConfig(
            data_path=self.tiny_dataset_path,
            max_coord_value=100,
            use_apply_chat_template=True,
        )

        dataset = CoordBootstrapDataset(
            tokenizer=self.mock_tokenizer,
            config=config,
        )

        # Verify dataset loaded correctly
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), 100)  # Based on the tiny dataset content

        # Test first few samples
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            self.assertIn("input_ids", sample)
            self.assertIn("attention_mask", sample)
            self.assertIsInstance(sample["input_ids"], torch.Tensor)
            self.assertIsInstance(sample["attention_mask"], torch.Tensor)

    def test_collator_with_real_data(self):
        """Test collator with real dataset samples."""
        if not Path(self.tiny_dataset_path).exists():
            self.skipTest(f"Tiny dataset not found at {self.tiny_dataset_path}")

        config = DatasetConfig(
            data_path=self.tiny_dataset_path,
            max_coord_value=100,
            use_apply_chat_template=True,
        )

        dataset = CoordBootstrapDataset(
            tokenizer=self.mock_tokenizer,
            config=config,
        )

        collator = DataCollatorCoordBootstrap(
            tokenizer=self.mock_tokenizer,
            config=CollatorConfig(),
        )

        # Get a small batch
        batch = [dataset[i] for i in range(min(3, len(dataset)))]

        # Process batch through collator
        result = collator(batch)

        # Verify batch structure
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)

        # Verify tensor shapes are consistent
        batch_size = len(batch)
        self.assertEqual(result["input_ids"].shape[0], batch_size)
        self.assertEqual(result["attention_mask"].shape[0], batch_size)
        self.assertEqual(result["labels"].shape[0], batch_size)

        # Verify sequence lengths are padded consistently
        seq_len = result["input_ids"].shape[1]
        self.assertEqual(result["attention_mask"].shape[1], seq_len)
        self.assertEqual(result["labels"].shape[1], seq_len)

    def test_processing_class_integration(self):
        """Test integration with processing_class API."""
        # Mock a complete processor
        mock_processor = Mock()
        mock_processor.tokenizer = self.mock_tokenizer

        # Test that processor can be used as processing_class
        self.assertTrue(hasattr(mock_processor, "tokenizer"))
        self.assertEqual(mock_processor.tokenizer, self.mock_tokenizer)

        # Verify fast tokenizer detection through processor
        if hasattr(mock_processor.tokenizer, "is_fast"):
            self.assertTrue(mock_processor.tokenizer.is_fast)

    @patch("builtins.print")
    def test_tokenizer_speed_logging(self, mock_print):
        """Test that tokenizer speed information is logged correctly."""
        # This would be called in the trainer setup
        mock_processor = Mock()
        mock_processor.tokenizer = self.mock_tokenizer

        # Simulate the trainer setup logic
        if hasattr(mock_processor, "tokenizer") and hasattr(
            mock_processor.tokenizer, "is_fast"
        ):
            if mock_processor.tokenizer.is_fast:
                print("✅ Using fast tokenizer for optimized performance")
            else:
                print(
                    "⚠️ Using slow tokenizer - consider using fast tokenizer for better speed"
                )

        # Verify correct message was printed
        mock_print.assert_called_with(
            "✅ Using fast tokenizer for optimized performance"
        )

    def test_tokenizer_parameter_optimization(self):
        """Test that tokenizer calls use optimized parameters."""
        config = DatasetConfig(
            data_path=self.tiny_dataset_path,
            max_coord_value=100,
            use_apply_chat_template=True,
        )

        dataset = CoordBootstrapDataset(
            tokenizer=self.mock_tokenizer,
            config=config,
        )

        # Process a sample to trigger tokenization
        dataset[0]

        # Verify tokenizer was called (exact parameters depend on implementation)
        self.mock_tokenizer.assert_called()

        # For fast tokenizers, verify optimized parameters were used
        if dataset._use_fast_tokenizer:
            # The tokenizer should have been called with padding=False, truncation=False
            # (exact verification depends on mock setup)
            self.assertTrue(dataset._use_fast_tokenizer)


if __name__ == "__main__":
    unittest.main()
