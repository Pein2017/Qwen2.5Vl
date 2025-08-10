#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the refactored HuggingFace-first dataset.

Tests the clean dataset implementation that uses ConversationProcessor
instead of the old complex custom processing logic.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from src_new.data.dataset import Dataset, read_jsonl


class TestDatasetRefactored(unittest.TestCase):
    """Test suite for refactored HuggingFace-first dataset."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.data_root = "test_data"
        self.mock_config.teacher_ratio = 0.5
        self.mock_config.num_teacher_samples = 1
        self.mock_config.max_coord_value = 2048
        self.mock_config.max_dataset_size = None  # No limit by default

        # Create mock tokenizer and image processor
        self.mock_tokenizer = Mock()
        self.mock_image_processor = Mock()
        self.mock_teacher_pool_manager = Mock()
        self.mock_teacher_pool_manager.teacher_pool = []

        # Create temporary JSONL file with test data
        self.test_data = [
            {
                "images": ["test1.jpg"],
                "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "test_device"}],
            },
            {
                "images": ["test2.jpg"],
                "objects": [
                    {"quad": [50, 60, 70, 80, 90, 100, 110, 120], "desc": "test_label"}
                ],
            },
        ]

        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for item in self.test_data:
            json.dump(item, self.temp_file)
            self.temp_file.write("\n")
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_read_jsonl_valid_file(self):
        """Test reading valid JSONL file."""
        result = read_jsonl(self.temp_file.name)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["objects"][0]["desc"], "test_device")

    def test_read_jsonl_nonexistent_file(self):
        """Test reading nonexistent JSONL file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_jsonl("nonexistent.jsonl")

    def test_dataset_initialization_valid(self):
        """Test valid dataset initialization."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        self.assertEqual(len(dataset), 2)
        self.assertIsNone(dataset.conversation_processor)  # Not set yet

    def test_dataset_initialization_empty_data_path(self):
        """Test dataset initialization with empty data path raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Dataset(
                data_path="",
                tokenizer=self.mock_tokenizer,
                image_processor=self.mock_image_processor,
                teacher_pool_manager=self.mock_teacher_pool_manager,
                config=self.mock_config,
            )
        self.assertIn("data_path cannot be empty", str(cm.exception))

    def test_dataset_initialization_none_config(self):
        """Test dataset initialization with None config raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Dataset(
                data_path=self.temp_file.name,
                tokenizer=self.mock_tokenizer,
                image_processor=self.mock_image_processor,
                teacher_pool_manager=self.mock_teacher_pool_manager,
                config=None,
            )
        self.assertIn("config cannot be None", str(cm.exception))

    def test_set_processor_valid(self):
        """Test setting valid HuggingFace processor."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        mock_hf_processor = Mock()

        with patch(
            "src_new.processing.conversation_processor.ConversationProcessor"
        ) as mock_conv_processor:
            dataset.set_processor(mock_hf_processor)

            self.assertEqual(dataset.hf_processor, mock_hf_processor)
            mock_conv_processor.assert_called_once_with(
                processor=mock_hf_processor, max_coord_value=2048
            )

    def test_set_processor_none(self):
        """Test setting None processor raises ValueError."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        with self.assertRaises(ValueError) as cm:
            dataset.set_processor(None)
        self.assertIn("hf_processor cannot be None", str(cm.exception))

    def test_getitem_without_processor_raises_error(self):
        """Test __getitem__ without processor raises ValueError."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        with self.assertRaises(ValueError) as cm:
            dataset[0]
        self.assertIn("Conversation processor not initialized", str(cm.exception))

    @patch("src_new.data.dataset.Image")
    @patch("src_new.data.dataset.Path")
    def test_getitem_with_processor_simple_conversation(self, mock_path, mock_image):
        """Test __getitem__ with processor for simple conversation."""
        # Setup mocks for Path - mock the / operator properly
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True

        # Mock the Path constructor and / operator
        def mock_path_constructor(path):
            mock_obj = Mock()
            mock_obj.__truediv__ = Mock(return_value=mock_path_instance)
            return mock_obj

        mock_path.side_effect = mock_path_constructor

        mock_image_instance = Mock()
        mock_image.open.return_value.convert.return_value = mock_image_instance

        # Create dataset
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        # Set up conversation processor
        mock_conversation_processor = Mock()
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(4, 1024),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        mock_conversation_processor.create_simple_conversation.return_value = (
            mock_inputs
        )
        dataset.conversation_processor = mock_conversation_processor

        # Test __getitem__
        result = dataset[0]

        # Verify conversation processor was called
        mock_conversation_processor.create_simple_conversation.assert_called_once()

        # Verify result structure
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)
        self.assertIn("pixel_values", result)

        # Verify labels have the same shape as input_ids (but may have different values due to masking)
        self.assertEqual(result["labels"].shape, result["input_ids"].shape)
        # Verify labels are a tensor
        self.assertIsInstance(result["labels"], torch.Tensor)

    def test_is_valid_sample_valid(self):
        """Test _is_valid_sample with valid sample."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        valid_sample = {
            "images": ["test.jpg"],
            "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "test_device"}],
        }

        self.assertTrue(dataset._is_valid_sample(valid_sample))

    def test_is_valid_sample_no_objects(self):
        """Test _is_valid_sample with no objects."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        invalid_sample = {"images": ["test.jpg"], "objects": []}

        self.assertFalse(dataset._is_valid_sample(invalid_sample))

    def test_is_valid_sample_no_images(self):
        """Test _is_valid_sample with no images."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        invalid_sample = {
            "images": [],
            "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "test_device"}],
        }

        self.assertFalse(dataset._is_valid_sample(invalid_sample))

    def test_is_valid_sample_no_geometry(self):
        """Test _is_valid_sample with object missing geometry."""
        dataset = Dataset(
            data_path=self.temp_file.name,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            teacher_pool_manager=self.mock_teacher_pool_manager,
            config=self.mock_config,
        )

        invalid_sample = {
            "images": ["test.jpg"],
            "objects": [
                {"desc": "test_device"}  # Missing geometry
            ],
        }

        self.assertFalse(dataset._is_valid_sample(invalid_sample))


if __name__ == "__main__":
    unittest.main()
