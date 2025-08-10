#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Multi-Modal Data Processing Tests

This module tests the complete multi-modal data processing pipeline including
image processing, conversation handling, coordinate token integration, and
teacher-student data preparation.

Key Features:
- Tests image data processing and validation
- Tests conversation format handling
- Tests coordinate token integration in conversations
- Tests teacher-student conversation preparation
- Tests data collation and batching
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from src_new.data.collator import StandardDataCollator
from src_new.data.dataset import Dataset
from src_new.processing.coordinate_converter import CoordinateTokenConverter
from src_new.processing.token_processor import TokenConfig, TokenProcessor


class TestMultiModalProcessing:
    """Test suite for multi-modal data processing."""

    @pytest.fixture
    def sample_multimodal_data(self):
        """Create sample multi-modal data for testing."""
        return [
            {
                "images": ["test_image_1.jpg"],
                "conversations": [
                    {"from": "user", "value": "请识别图中的设备位置。"},
                    {
                        "from": "assistant",
                        "value": "图中有一个BBU设备<|obj_ref_start|>BBU设备<|obj_ref_end|><|box_start|>[<|coord_100|>, <|coord_150|>, <|coord_200|>, <|coord_250|>]<|box_end|>。",
                    },
                ],
                "objects": [{"bbox_2d": [100, 150, 200, 250], "desc": "BBU设备"}],
                "width": 800,
                "height": 600,
            },
            {
                "images": ["test_image_2.jpg"],
                "conversations": [
                    {"from": "user", "value": "这个设备是什么类型？"},
                    {
                        "from": "assistant",
                        "value": "这是一个光纤设备<|obj_ref_start|>光纤设备<|obj_ref_end|><|quad_start|>[<|coord_50|>, <|coord_60|>, <|coord_70|>, <|coord_80|>, <|coord_90|>, <|coord_100|>, <|coord_110|>, <|coord_120|>]<|quad_end|>。",
                    },
                ],
                "objects": [
                    {"quad": [50, 60, 70, 80, 90, 100, 110, 120], "desc": "光纤设备"}
                ],
                "width": 640,
                "height": 480,
            },
            {
                "images": ["test_image_3.jpg"],
                "conversations": [
                    {"from": "user", "value": "线缆连接情况如何？"},
                    {
                        "from": "assistant",
                        "value": "线缆连接<|obj_ref_start|>光纤线缆<|obj_ref_end|><|line_start|>[<|coord_10|>, <|coord_20|>, <|coord_30|>, <|coord_40|>, <|coord_50|>, <|coord_60|>]<|line_end|>正常。",
                    },
                ],
                "objects": [{"line": [10, 20, 30, 40, 50, 60], "desc": "光纤线缆"}],
                "width": 1024,
                "height": 768,
            },
        ]

    @pytest.fixture
    def sample_teacher_student_data(self):
        """Create sample teacher-student data for testing."""
        return [
            {
                "images": ["teacher_image_1.jpg"],
                "conversations": [
                    {"from": "user", "value": "识别这个设备。"},
                    {
                        "from": "teacher",
                        "value": "这是一个专业的BBU基带处理单元，用于无线通信基站。",
                    },
                    {
                        "from": "assistant",
                        "value": "设备位于<|obj_ref_start|>BBU设备<|obj_ref_end|><|box_start|>[<|coord_300|>, <|coord_400|>, <|coord_500|>, <|coord_600|>]<|box_end|>。",
                    },
                ],
                "objects": [{"bbox_2d": [300, 400, 500, 600], "desc": "BBU设备"}],
                "width": 1024,
                "height": 768,
            },
            {
                "images": ["teacher_image_2.jpg"],
                "conversations": [
                    {"from": "user", "value": "这些线缆的作用是什么？"},
                    {
                        "from": "teacher",
                        "value": "这些是光纤跳线，用于连接不同的网络设备。",
                    },
                    {
                        "from": "assistant",
                        "value": "光纤线缆<|obj_ref_start|>光纤跳线<|obj_ref_end|><|line_start|>[<|coord_100|>, <|coord_200|>, <|coord_300|>, <|coord_400|>, <|coord_500|>, <|coord_600|>]<|line_end|>连接正常。",
                    },
                ],
                "objects": [
                    {"line": [100, 200, 300, 400, 500, 600], "desc": "光纤跳线"}
                ],
                "width": 800,
                "height": 600,
            },
        ]

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock()
        config.coordinate_tokens_enabled = True
        config.max_coord_value = 1024
        config.max_total_length = 12000
        config.language = "chinese"
        config.max_pixels = 401408
        config.teacher_ratio = 0.5
        config.num_teacher_samples = 1
        return config

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.vocab_size = 151665 + 1027
        tokenizer.model_max_length = 32000
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Create vocabulary with coordinate tokens
        vocab = {"<|pad|>": 0, "<|eos|>": 1, "<|im_start|>": 2, "<|im_end|>": 3}
        for i in range(1025):
            vocab[f"<|coord_{i}|>"] = 151667 + i

        tokenizer.get_vocab.return_value = vocab

        # Mock encoding
        def mock_encode(text, **kwargs):
            return [1, 2, 3, 4, 5] * (len(text) // 10 + 1)

        tokenizer.encode = mock_encode
        tokenizer.decode.return_value = "mock decoded text"
        tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        return tokenizer

    @pytest.fixture
    def coordinate_converter(self):
        """Create coordinate token converter."""
        return CoordinateTokenConverter(max_coord_value=1024)

    @pytest.fixture
    def token_processor(self, mock_config):
        """Create token processor."""
        token_config = TokenConfig(
            coordinate_tokens_enabled=mock_config.coordinate_tokens_enabled,
            max_coord_value=mock_config.max_coord_value,
        )
        return TokenProcessor(token_config)

    def test_coordinate_token_conversion(
        self, coordinate_converter, sample_multimodal_data
    ):
        """Test coordinate token conversion for different geometry types."""
        # Test bbox_2d conversion
        bbox_objects = [sample_multimodal_data[0]["objects"][0]]
        bbox_result = coordinate_converter.convert_objects_to_tokens(bbox_objects)

        assert "<|obj_ref_start|>BBU设备<|obj_ref_end|>" in bbox_result
        assert "<|box_start|>" in bbox_result
        assert "<|box_end|>" in bbox_result
        assert "<|coord_100|>" in bbox_result
        assert "<|coord_150|>" in bbox_result

        # Test quad conversion
        quad_objects = [sample_multimodal_data[1]["objects"][0]]
        quad_result = coordinate_converter.convert_objects_to_tokens(quad_objects)

        assert "<|obj_ref_start|>光纤设备<|obj_ref_end|>" in quad_result
        assert "<|quad_start|>" in quad_result
        assert "<|quad_end|>" in quad_result
        assert "<|coord_50|>" in quad_result

        # Test line conversion
        line_objects = [sample_multimodal_data[2]["objects"][0]]
        line_result = coordinate_converter.convert_objects_to_tokens(line_objects)

        assert "<|obj_ref_start|>光纤线缆<|obj_ref_end|>" in line_result
        assert "<|line_start|>" in line_result
        assert "<|line_end|>" in line_result
        assert "<|coord_10|>" in line_result

    def test_conversation_processing(self, sample_multimodal_data):
        """Test conversation format processing."""
        for item in sample_multimodal_data:
            conversations = item["conversations"]

            # Should have user and assistant messages
            assert len(conversations) >= 2
            assert conversations[0]["from"] == "user"
            assert conversations[-1]["from"] == "assistant"

            # Assistant message should contain coordinate tokens
            assistant_msg = conversations[-1]["value"]
            assert any(
                token in assistant_msg
                for token in ["<|coord_", "<|obj_ref_", "<|box_", "<|quad_", "<|line_"]
            )

    def test_teacher_student_conversation_processing(self, sample_teacher_student_data):
        """Test teacher-student conversation format processing."""
        for item in sample_teacher_student_data:
            conversations = item["conversations"]

            # Should have user, teacher, and assistant messages
            assert len(conversations) == 3
            assert conversations[0]["from"] == "user"
            assert conversations[1]["from"] == "teacher"
            assert conversations[2]["from"] == "assistant"

            # Teacher message should provide additional context
            teacher_msg = conversations[1]["value"]
            assert len(teacher_msg) > 0

            # Assistant message should contain coordinate tokens
            assistant_msg = conversations[2]["value"]
            assert any(token in assistant_msg for token in ["<|coord_", "<|obj_ref_"])

    def test_image_data_validation(self, sample_multimodal_data):
        """Test image data validation and processing."""
        for item in sample_multimodal_data:
            # Should have image information
            assert "images" in item
            assert len(item["images"]) > 0
            assert "width" in item
            assert "height" in item

            # Dimensions should be positive
            assert item["width"] > 0
            assert item["height"] > 0

            # Should have objects with coordinates
            assert "objects" in item
            assert len(item["objects"]) > 0

            for obj in item["objects"]:
                assert "desc" in obj
                # Should have one of the geometry types
                geometry_types = ["bbox_2d", "quad", "line"]
                assert any(geom_type in obj for geom_type in geometry_types)

    def test_dataset_item_processing(
        self, sample_multimodal_data, real_test_config, real_extended_tokenizer
    ):
        """Test dataset item processing with coordinate tokens."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in sample_multimodal_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            temp_file = f.name

        try:
            # Create dataset with new constructor signature
            dataset = Dataset(
                data_path=temp_file,
                tokenizer=real_extended_tokenizer,
                image_processor=None,  # Not needed for this test
                teacher_pool_manager=None,  # Not needed for this test
                config=real_test_config,
            )

            # Set up a mock processor for the dataset
            mock_processor = Mock()
            mock_processor.tokenizer = real_extended_tokenizer

            # Mock the conversation processor to return proper data
            def mock_create_simple_conversation(sample, images):
                grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
                hidden = 1024
                return {
                    "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    "pixel_values": torch.randn(4, hidden),
                    "image_grid_thw": grid,
                }

            dataset.set_processor(mock_processor)
            dataset.conversation_processor.create_simple_conversation = (
                mock_create_simple_conversation
            )

            # Test dataset length
            assert len(dataset) == len(sample_multimodal_data)

            # Test getting items
            for i in range(len(dataset)):
                item = dataset[i]

                # Should have required fields
                assert "input_ids" in item
                assert "labels" in item
                assert isinstance(item["input_ids"], torch.Tensor)
                assert isinstance(item["labels"], torch.Tensor)

                # Should have same length
                assert len(item["input_ids"]) == len(item["labels"])

                # May have image data
                if "pixel_values" in item:
                    assert isinstance(item["pixel_values"], torch.Tensor)
                    # image_grid_thw must be present and 2D [num_images, 3]
                    assert "image_grid_thw" in item
                    assert (
                        item["image_grid_thw"].ndim == 2
                        and item["image_grid_thw"].shape[-1] == 3
                    )

        finally:
            Path(temp_file).unlink()

    def test_teacher_student_dataset_processing(
        self, sample_teacher_student_data, real_test_config, real_extended_tokenizer
    ):
        """Test teacher-student dataset processing."""
        # Create temporary data files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in sample_teacher_student_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            teacher_file = f.name

        try:
            # Update config for teacher-student training
            real_test_config.teacher_pool_file = teacher_file
            real_test_config.teacher_ratio = 1.0  # Use all teacher data for testing

            # Create dataset with new constructor signature
            dataset = Dataset(
                data_path=teacher_file,
                tokenizer=real_extended_tokenizer,
                image_processor=None,  # Not needed for this test
                teacher_pool_manager=None,  # Not needed for this test
                config=real_test_config,
            )

            # Set up a mock processor for the dataset
            mock_processor = Mock()
            mock_processor.tokenizer = real_extended_tokenizer

            # Mock the conversation processor to return proper data
            def mock_create_simple_conversation(sample, images):
                grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
                hidden = 1024
                return {
                    "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    "pixel_values": torch.randn(4, hidden),
                    "image_grid_thw": grid,
                }

            dataset.set_processor(mock_processor)
            dataset.conversation_processor.create_simple_conversation = (
                mock_create_simple_conversation
            )

            # Test dataset processing
            assert len(dataset) > 0

            # Test getting items
            item = dataset[0]
            assert "input_ids" in item
            assert "labels" in item

            # May have teacher/student spans
            if hasattr(dataset, "_create_masked_labels_with_spans"):
                # Test span creation
                input_ids = item["input_ids"]
                labels, teacher_spans, student_spans = (
                    dataset._create_masked_labels_with_spans(
                        input_ids, real_extended_tokenizer, has_teachers=True
                    )
                )

                assert isinstance(labels, torch.Tensor)
                assert isinstance(teacher_spans, list)
                assert isinstance(student_spans, list)

        finally:
            Path(teacher_file).unlink()

    def test_data_collation(
        self, sample_multimodal_data, real_test_config, real_extended_tokenizer
    ):
        """Test data collation for batching."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in sample_multimodal_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            temp_file = f.name

        try:
            # Create dataset with new constructor signature
            dataset = Dataset(
                data_path=temp_file,
                tokenizer=real_extended_tokenizer,
                image_processor=None,  # Not needed for this test
                teacher_pool_manager=None,  # Not needed for this test
                config=real_test_config,
            )

            # Set up a mock processor for the dataset
            mock_processor = Mock()
            mock_processor.tokenizer = real_extended_tokenizer

            # Mock the conversation processor to return proper data
            def mock_create_simple_conversation(sample, images):
                grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
                hidden = 1024
                return {
                    "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    "pixel_values": torch.randn(4, hidden),
                    "image_grid_thw": grid,
                }

            dataset.set_processor(mock_processor)
            dataset.conversation_processor.create_simple_conversation = (
                mock_create_simple_conversation
            )

            # Create collator
            collator = StandardDataCollator(
                tokenizer=real_extended_tokenizer,
                config=real_test_config,
            )

            # Get sample items
            items = [dataset[i] for i in range(min(2, len(dataset)))]

            # Test collation
            batch = collator(items)

            # Should have batched tensors
            assert "input_ids" in batch
            assert "labels" in batch
            assert isinstance(batch["input_ids"], torch.Tensor)
            assert isinstance(batch["labels"], torch.Tensor)

            # Batch dimension should match number of items
            assert batch["input_ids"].shape[0] == len(items)
            assert batch["labels"].shape[0] == len(items)

        finally:
            Path(temp_file).unlink()

    def test_coordinate_token_extraction(
        self, real_token_processor, real_extended_tokenizer
    ):
        """Test coordinate token extraction from processed data."""
        # Create input with coordinate tokens
        coord_start = 151667
        input_ids = torch.tensor(
            [
                100,
                200,  # Regular tokens
                coord_start + 100,  # <|coord_100|>
                coord_start + 200,  # <|coord_200|>
                coord_start + 300,  # <|coord_300|>
                coord_start + 400,  # <|coord_400|>
                300,
                400,  # Regular tokens
            ]
        )

        # Extract coordinates using real tokenizer
        sequences = real_token_processor.extract_coordinates_from_tokens(
            input_ids, real_extended_tokenizer
        )

        # Should find coordinate sequences
        assert len(sequences) > 0

        # Should extract correct coordinate values
        for seq in sequences:
            assert "coordinates" in seq
            assert len(seq["coordinates"]) > 0

    def test_processor_like_image_token_expansion(self):
        """Validate that expected image tokens equals sum(prod(thw)//merge_size**2)."""
        import torch

        merge_size = 2
        grids = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long)
        expected_tokens = int(
            (torch.prod(grids, dim=1) // (merge_size * merge_size)).sum().item()
        )
        # Simulate chat template inserting one <|image_pad|> per image and processor expansion
        # which replicates tokens based on grid and merge_size.
        assert expected_tokens == 5
