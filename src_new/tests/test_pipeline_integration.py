#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick integration test script for teacher-student training pipeline.

This script tests the key components without requiring pytest or complex setup.
"""

import json
import tempfile
from pathlib import Path

import torch


def create_mock_tokenizer():
    """Create a simple mock tokenizer for testing."""

    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                "<|im_start|>": 151644,
                "<|im_end|>": 151645,
                "system": 1587,
                "user": 882,
                "assistant": 77091,
                "<|coord_0|>": 151667,
                "<|coord_1|>": 151668,
                "<|coord_2048|>": 153715,
            }
            self.model_max_length = 2048

        def get_vocab(self):
            return self.vocab

        def apply_chat_template(
            self, conversation, tokenize=False, add_generation_prompt=False
        ):
            return "mocked chat template"

        def encode(self, text, add_special_tokens=False):
            if text == "system":
                return [1587]
            elif text == "user":
                return [882]
            elif text == "assistant":
                return [77091]
            else:
                return [1, 2, 3, 4, 5]

        def decode(self, token_ids):
            return "mocked decode"

        def __call__(self, text, **kwargs):
            # Return longer sequence with proper assistant tokens for testing
            return {
                "input_ids": torch.tensor(
                    [[151644, 1587, 151645, 151644, 77091, 1, 2, 3, 4, 5, 151645]]
                ),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
            }

    return MockTokenizer()


def test_teacher_pool_loading():
    """Test teacher pool loading."""
    print("🧪 Testing teacher pool loading...")

    # Create sample teacher pool data
    teacher_pool_data = [
        {
            "images": ["teacher1.jpg"],
            "objects": [
                {"bbox_2d": [100, 100, 200, 200], "desc": "BBU设备1"},
                {"bbox_2d": [300, 300, 400, 400], "desc": "螺丝连接点1"},
            ],
        },
        {
            "images": ["teacher2.jpg"],
            "objects": [{"bbox_2d": [150, 150, 250, 250], "desc": "BBU设备2"}],
        },
    ]

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for teacher in teacher_pool_data:
            f.write(json.dumps(teacher, ensure_ascii=False) + "\n")
        temp_path = f.name

    try:
        from src_new.data.teacher_pool import TeacherPoolManager

        teacher_pool_manager = TeacherPoolManager(temp_path)

        assert len(teacher_pool_manager.teacher_pool) == 2
        assert teacher_pool_manager.teacher_pool[0]["objects"][0]["desc"] == "BBU设备1"

        print("✅ Teacher pool loading test passed")

    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_chat_processor():
    """Test chat processor functionality."""
    print("🧪 Testing chat processor...")

    from src_new.processing.chat_processor import ChatConfig, ChatProcessor
    from src_new.processing.token_processor import TokenConfig, TokenProcessor

    # Create configurations
    token_config = TokenConfig(
        coordinate_tokens_enabled=True,
        max_coord_value=2048,
        new_geometry_tokens=["<|line_start|>", "<|line_end|>"],
    )

    chat_config = ChatConfig(
        language="chinese",
        use_training_prompts=True,
        teacher_ratio=0.7,
        max_teacher_examples=2,
        coordinate_tokens_enabled=True,
    )

    # Create components
    tokenizer = create_mock_tokenizer()
    token_processor = TokenProcessor(token_config)
    chat_processor = ChatProcessor(tokenizer, token_processor, chat_config)

    # Test standalone conversation
    sample_student = {
        "images": ["student.jpg"],
        "objects": [{"bbox_2d": [120, 120, 220, 220], "desc": "BBU设备学生"}],
    }

    conversation = chat_processor.build_conversation(sample_student)
    assert len(conversation) == 3  # system + user + assistant
    assert conversation[0]["role"] == "system"

    print("✅ Chat processor test passed")


def test_loss_masking():
    """Test loss masking with span tracking."""
    print("🧪 Testing loss masking...")

    from src_new.processing.chat_processor import ChatConfig, ChatProcessor
    from src_new.processing.token_processor import TokenConfig, TokenProcessor

    # Create configurations
    token_config = TokenConfig(coordinate_tokens_enabled=True, max_coord_value=2048)
    chat_config = ChatConfig(coordinate_tokens_enabled=True)

    # Create components
    tokenizer = create_mock_tokenizer()
    token_processor = TokenProcessor(token_config)
    chat_processor = ChatProcessor(tokenizer, token_processor, chat_config)

    # Test loss masking
    input_ids = torch.tensor([151644, 1587, 151645, 151644, 77091, 1, 2, 3, 151645])
    conversation = [
        {"role": "system", "content": "System"},
        {"role": "assistant", "content": "Response"},
    ]

    labels, teacher_spans, student_spans = chat_processor.create_loss_mask_with_spans(
        input_ids, conversation
    )

    # Check masking
    assert labels[0].item() == -100  # <|im_start|> should be masked
    assert labels[1].item() == -100  # system should be masked

    print("✅ Loss masking test passed")


def test_coordinate_loss_integration():
    """Test coordinate loss integration."""
    print("🧪 Testing coordinate loss integration...")

    from src_new.models.loss_manager import LossManager
    from src_new.processing.token_processor import TokenConfig, TokenProcessor

    # Create token processor
    token_config = TokenConfig(coordinate_tokens_enabled=True, max_coord_value=2048)
    token_processor = TokenProcessor(token_config)

    # Mock get_coordinate_token_range
    def mock_get_range(tokenizer):
        return (151667, 153715)

    token_processor.get_coordinate_token_range = mock_get_range

    # Create mock config and tokenizer
    class MockConfig:
        def __init__(self):
            self.coordinate_loss_weight = 0.05
            self.regular_loss_weight = 1.0
            self.teacher_loss_weight = 0.3
            self.student_loss_weight = 1.0

    config = MockConfig()
    tokenizer = create_mock_tokenizer()
    loss_manager = LossManager(
        config=config,
        token_processor=token_processor,
        tokenizer=tokenizer,
    )

    # Verify coordinate loss function was created with correct token IDs
    assert loss_manager.coordinate_loss_fn.coord_start_id == 151667
    assert (
        loss_manager.coordinate_loss_fn.coord_end_id == 153716
    )  # +1 for exclusive end

    print("✅ Coordinate loss integration test passed")


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("🧪 Testing end-to-end pipeline...")

    from src_new.data.teacher_pool import TeacherPoolManager
    from src_new.processing.chat_processor import ChatConfig, ChatProcessor
    from src_new.processing.token_processor import TokenConfig, TokenProcessor

    # Create sample data
    teacher_pool_data = [
        {
            "images": ["teacher1.jpg"],
            "objects": [{"bbox_2d": [100, 100, 200, 200], "desc": "BBU设备1"}],
        }
    ]

    sample_student = {
        "images": ["student.jpg"],
        "objects": [{"bbox_2d": [120, 120, 220, 220], "desc": "BBU设备学生"}],
    }

    # Create temporary teacher pool file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for teacher in teacher_pool_data:
            f.write(json.dumps(teacher, ensure_ascii=False) + "\n")
        temp_path = f.name

    try:
        # Initialize components
        token_config = TokenConfig(coordinate_tokens_enabled=True, max_coord_value=2048)
        chat_config = ChatConfig(
            coordinate_tokens_enabled=True, teacher_ratio=1.0
        )  # Always use teachers

        tokenizer = create_mock_tokenizer()
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(tokenizer, token_processor, chat_config)
        teacher_pool_manager = TeacherPoolManager(temp_path)

        # Process sample with teacher pool
        processed_sample = chat_processor.process_sample_with_teacher_pool(
            sample_student, teacher_pool_manager.teacher_pool
        )

        # Verify processed sample
        assert processed_sample.input_ids is not None
        assert processed_sample.attention_mask is not None
        assert processed_sample.labels is not None
        assert processed_sample.teacher_spans is not None
        assert processed_sample.student_spans is not None

        # Verify sample validation
        assert chat_processor.validate_processed_sample(processed_sample) == True

        print("✅ End-to-end pipeline test passed")

    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    """Run all integration tests."""
    print("🚀 Running Teacher-Student Pipeline Integration Tests")
    print("=" * 60)

    try:
        test_teacher_pool_loading()
        test_chat_processor()
        test_loss_masking()
        test_coordinate_loss_integration()
        test_end_to_end_pipeline()

        print("=" * 60)
        print("🎉 All integration tests passed!")
        print("✅ Teacher-student training pipeline is ready for use")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
