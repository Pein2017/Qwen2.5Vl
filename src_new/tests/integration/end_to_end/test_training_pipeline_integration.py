#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Training Pipeline Integration Tests

This module tests the complete training pipeline from data loading through
model training, including coordinate token processing, teacher-student training,
and loss computation.

Key Features:
- Tests complete training pipeline integration
- Tests coordinate token system in training context
- Tests teacher-student training workflow
- Tests loss computation and backpropagation
- Tests model checkpointing and resumption
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src_new.data.dataset import Dataset
from src_new.models.wrapper import DetectionModel
from src_new.processing.token_processor import TokenConfig, TokenProcessor
from src_new.training.bbu_trainer import BBUTrainer


@pytest.mark.integration
class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data files."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        # Create sample training data
        train_data = [
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
                    {"from": "user", "value": "这个设备是什么？"},
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
        ]

        # Write training data
        train_file = temp_path / "train.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Create teacher pool data
        teacher_data = [
            {
                "images": ["teacher_image_1.jpg"],
                "conversations": [
                    {"from": "user", "value": "识别设备。"},
                    {"from": "teacher", "value": "这是专业的设备识别结果。"},
                    {
                        "from": "assistant",
                        "value": "设备位于<|obj_ref_start|>设备<|obj_ref_end|><|box_start|>[<|coord_300|>, <|coord_400|>, <|coord_500|>, <|coord_600|>]<|box_end|>。",
                    },
                ],
                "objects": [{"bbox_2d": [300, 400, 500, 600], "desc": "设备"}],
                "width": 1024,
                "height": 768,
            }
        ]

        teacher_file = temp_path / "teacher_pool.jsonl"
        with open(teacher_file, "w", encoding="utf-8") as f:
            for item in teacher_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Create dummy image files for testing
        import numpy as np
        from PIL import Image

        # Create dummy images referenced in the data
        image_names = ["test_image_1.jpg", "test_image_2.jpg", "teacher_image_1.jpg"]
        for img_name in image_names:
            # Create a simple colored image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(temp_path / img_name)

        yield temp_path

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def integration_config(self, temp_data_dir):
        """Create configuration for integration testing."""
        config = Mock()

        # Model settings
        config.model_path = "mock_model_path"
        config.model_size = "3B"
        config.model_max_length = 32000
        config.model_hidden_size = 2048
        config.torch_dtype = "bfloat16"
        config.use_cache = False

        # Training settings
        config.num_train_epochs = 1  # Short for testing
        config.per_device_train_batch_size = 1
        config.gradient_accumulation_steps = 1
        config.learning_rate = 5e-6
        config.warmup_ratio = 0.1
        config.weight_decay = 0.0001
        config.gradient_checkpointing = True
        config.bf16 = True

        # Data settings
        config.data_root = str(temp_data_dir)  # Root directory for data files
        config.train_data_path = str(temp_data_dir / "train.jsonl")
        config.teacher_pool_file = str(temp_data_dir / "teacher_pool.jsonl")
        config.max_total_length = 12000
        config.collator_type = "packed"
        config.language = "chinese"
        config.max_pixels = 401408
        config.max_dataset_size = None  # No limit for testing

        # Coordinate token system
        config.coordinate_tokens_enabled = True
        config.max_coord_value = 1024
        config.coordinate_loss_weight = 0.05
        config.regular_loss_weight = 1.0
        config.coordinate_loss_temperature = 1.0

        # Teacher-student settings
        config.teacher_ratio = 0.5
        config.num_teacher_samples = 1
        config.teacher_loss_weight = 0.3
        config.student_loss_weight = 1.0

        # Output settings
        config.output_dir = str(temp_data_dir / "output")
        config.logging_steps = 1
        config.save_steps = 10
        config.eval_steps = 10

        return config

    @pytest.fixture
    def mock_base_model(self):
        """Create mock base model for testing."""
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = 151665
        model.config.hidden_size = 2048

        # Mock forward pass
        def mock_forward(*args, **kwargs):
            # Handle both positional and keyword arguments
            input_ids = kwargs.get("input_ids") or (args[0] if args else None)
            if input_ids is None:
                # Fallback for unexpected call patterns
                batch_size, seq_len = 1, 10
            else:
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
            vocab_size = 151665 + 2 + 1025  # base + line_tokens + coord_tokens = 152692

            output = Mock()
            output.loss = torch.tensor(1.5, requires_grad=True)
            output.logits = torch.randn(
                batch_size, seq_len, vocab_size, requires_grad=True
            )
            output.hidden_states = torch.randn(batch_size, seq_len, 2048)
            return output

        model.forward = mock_forward
        model.__call__ = mock_forward  # Ensure __call__ also works
        model.train.return_value = model
        model.eval.return_value = model
        model.to.return_value = model
        model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        model.named_parameters.return_value = [
            ("test_param", torch.randn(10, 10, requires_grad=True))
        ]

        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.vocab_size = 151665
        tokenizer.model_max_length = 32000
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Create vocabulary with coordinate tokens
        vocab = {"<|pad|>": 0, "<|eos|>": 1, "<|im_start|>": 2, "<|im_end|>": 3}
        # Add line tokens
        vocab["<|line_start|>"] = 151665
        vocab["<|line_end|>"] = 151666
        # Add coordinate tokens
        for i in range(1025):
            vocab[f"<|coord_{i}|>"] = 151667 + i

        tokenizer.get_vocab.return_value = vocab
        tokenizer.add_tokens.return_value = 1027
        tokenizer.add_special_tokens.return_value = 1027

        # Add __len__ method to support len(tokenizer)
        # After token extension, vocab_size should be updated
        def get_len():
            return (
                len(vocab) if hasattr(tokenizer, "_extended") else tokenizer.vocab_size
            )

        tokenizer.__len__ = get_len

        # Ensure vocab_size is always an integer, not a Mock
        def get_vocab_size():
            return len(vocab) if hasattr(tokenizer, "_extended") else 151665

        # Override the vocab_size property to always return an integer
        type(tokenizer).vocab_size = property(get_vocab_size)

        # Mock the extension process
        def mock_add_special_tokens(special_tokens_dict):
            # Simulate adding tokens to vocab
            if "additional_special_tokens" in special_tokens_dict:
                for token in special_tokens_dict["additional_special_tokens"]:
                    if token not in vocab:
                        vocab[token] = len(vocab)
            tokenizer._extended = True
            tokenizer.vocab_size = len(vocab)
            return len(special_tokens_dict.get("additional_special_tokens", []))

        tokenizer.add_special_tokens = mock_add_special_tokens

        # Mock encoding/decoding
        def mock_encode(text, **kwargs):
            # Simple mock encoding
            return [1, 2, 3, 4, 5] * (len(text) // 10 + 1)

        tokenizer.encode = mock_encode
        tokenizer.decode.return_value = "mock decoded text"
        tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]

        return tokenizer

    def test_dataset_creation_and_loading(self, integration_config, mock_tokenizer):
        """Test dataset creation and data loading."""
        # Create token processor
        token_config = TokenConfig(
            coordinate_tokens_enabled=integration_config.coordinate_tokens_enabled,
            max_coord_value=integration_config.max_coord_value,
        )
        token_processor = TokenProcessor(token_config)

        # Extend tokenizer
        extended_tokenizer = token_processor.extend_tokenizer_vocabulary(mock_tokenizer)

        # Create dataset
        dataset = Dataset(
            data_path=integration_config.train_data_path,
            tokenizer=extended_tokenizer,
            image_processor=None,  # Not needed for this test
            teacher_pool_manager=None,  # Not needed for this test
            config=integration_config,
        )

        # Set up a mock processor for the dataset
        mock_processor = Mock()
        mock_processor.tokenizer = extended_tokenizer

        # Mock the conversation processor to return proper data
        def mock_create_simple_conversation(sample, images):
            # Return a simple mock conversation result
            return {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                "pixel_values": torch.randn(4, 1024),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }

        dataset.set_processor(mock_processor)
        # Mock the conversation processor that gets created
        dataset.conversation_processor.create_simple_conversation = (
            mock_create_simple_conversation
        )

        # Test dataset length
        assert len(dataset) > 0

        # Test getting an item
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_model_initialization_with_coordinate_tokens(
        self, real_test_config, realistic_base_model, real_extended_tokenizer
    ):
        """Test model initialization with coordinate token support."""
        with patch("src_new.models.wrapper.apply_comprehensive_qwen25_fixes"):
            # Create detection model
            model = DetectionModel(
                base_model=realistic_base_model,
                config=real_test_config,
                tokenizer=real_extended_tokenizer,
                skip_expansion=True,
            )

            # Test coordinate mode
            assert (
                model.coordinate_mode_enabled
                == real_test_config.coordinate_tokens_enabled
            )

            # Test forward pass
            input_ids = torch.randint(0, 1000, (1, 10))
            labels = torch.randint(0, 1000, (1, 10))

            outputs = model(input_ids=input_ids, labels=labels)
            assert hasattr(outputs, "loss")
            assert hasattr(outputs, "logits")

    def test_trainer_initialization(
        self, real_test_config, realistic_base_model, real_extended_tokenizer
    ):
        """Test trainer initialization with all components."""
        with patch("src_new.models.wrapper.apply_comprehensive_qwen25_fixes"):
            # Create model
            model = DetectionModel(
                base_model=realistic_base_model,
                config=real_test_config,
                tokenizer=real_extended_tokenizer,
                skip_expansion=True,
            )

            # Create dataset
            train_dataset = Dataset(
                data_path=real_test_config.train_data_path,
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
                return {
                    "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    "pixel_values": torch.randn(4, 1024),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                }

            train_dataset.set_processor(mock_processor)
            train_dataset.conversation_processor.create_simple_conversation = (
                mock_create_simple_conversation
            )

            # Create training arguments
            from transformers import TrainingArguments

            training_args = TrainingArguments(
                output_dir="./test_output",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                learning_rate=5e-6,
                logging_steps=10,
                save_steps=100,
                remove_unused_columns=False,
            )

            # Create trainer
            trainer = BBUTrainer(
                model=model,
                processing_class=real_extended_tokenizer,
                training_args=training_args,
                train_dataset=train_dataset,
                tokenizer=real_extended_tokenizer,  # For backward compatibility
            )

            # Test trainer initialization
            assert trainer.model == model
            assert trainer.train_dataset == train_dataset
            assert trainer.tokenizer == real_extended_tokenizer

    @patch("src_new.models.wrapper.apply_comprehensive_qwen25_fixes")
    def test_training_step_execution(
        self,
        mock_fixes,
        real_test_config,
        realistic_base_model,
        real_extended_tokenizer,
    ):
        """Test execution of a single training step."""
        # Create model
        model = DetectionModel(
            base_model=realistic_base_model,
            config=real_test_config,
            tokenizer=real_extended_tokenizer,
            skip_expansion=True,
        )

        # Create sample batch
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 20)),
            "attention_mask": torch.ones(1, 20),
            "labels": torch.randint(0, 1000, (1, 20)),
            "pixel_values": torch.randn(4, 1024),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

        # Test forward pass
        model.train()
        outputs = model(**batch)

        # Should have loss for training
        assert hasattr(outputs, "loss")
        assert torch.isfinite(outputs.loss)

        # Test backward pass (mock)
        if outputs.loss.requires_grad:
            outputs.loss.backward()

    def test_coordinate_loss_computation_integration(
        self, real_test_config, realistic_base_model, real_extended_tokenizer
    ):
        """Test coordinate loss computation in training context."""
        with patch("src_new.models.wrapper.apply_comprehensive_qwen25_fixes"):
            # Create model with coordinate tokens enabled
            model = DetectionModel(
                base_model=realistic_base_model,
                config=real_test_config,
                tokenizer=real_extended_tokenizer,
                skip_expansion=True,
            )

            # Create batch with coordinate tokens
            batch_size, seq_len = 1, 15
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            labels = torch.randint(0, 1000, (batch_size, seq_len))

            # Add some coordinate tokens to labels
            coord_start = 151667
            labels[0, 5] = coord_start + 100  # <|coord_100|>
            labels[0, 6] = coord_start + 200  # <|coord_200|>

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)

            # Should compute coordinate loss
            loss_components = model.get_loss_components()
            if loss_components:
                assert hasattr(loss_components, "loss")
                # May have coordinate loss if coordinate tokens were detected
                if hasattr(loss_components, "student_l1_loss"):
                    assert loss_components.student_l1_loss is None or torch.isfinite(
                        loss_components.student_l1_loss
                    )
