"""
Tests for image processing pipeline in src_new.

This module tests the critical image processing functionality that was causing
the "Image features and image tokens do not match" error in training.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch

from src_new.config.config import Config
from src_new.tests.fixtures.mock_objects import MockTokenizer, MockImageProcessor


class TestImageProcessing:
    """Test image processing pipeline components."""

    def test_vision_token_calculation(self):
        """Test that vision token calculation works correctly."""
        # Create mock config
        config = Config(
            model_path="/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
            model_size="3B",
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
            coordinate_tokens_enabled=True,
            max_coord_value=1000,
            train_data_path="/data3/Qwen2.5-VL-main/data/ds_v2_full/train.jsonl",
            val_data_path="/data3/Qwen2.5-VL-main/data/ds_v2_full/val.jsonl",
            teacher_pool_file="/data3/Qwen2.5-VL-main/data/ds_v2_full/teacher.jsonl",
            teacher_ratio=0.5,
            model_max_length=2048,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=5e-6,
            vision_lr=1e-5,
            merger_lr=1e-5,
            llm_lr=5e-6,
            data_root="/data3/Qwen2.5-VL-main/data/ds_v2_full",
            output_dir="/fake/output",
            run_name="test_run",
        )

        # Create synthetic image
        synthetic_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        # Create mock image processor
        image_processor = MockImageProcessor()

        # Test vision token calculation
        from src_new.data.dataset import Dataset
        
        # Create dataset instance (without loading data)
        dataset = Dataset.__new__(Dataset)  # Create without calling __init__
        dataset.image_processor = image_processor
        dataset.config = config

        # Test the vision token calculation method
        num_tokens = dataset._calculate_image_tokens(synthetic_image)
        
        # Should return a positive integer
        assert isinstance(num_tokens, int), f"Expected int, got {type(num_tokens)}"
        assert num_tokens > 0, f"Expected positive tokens, got {num_tokens}"
        
        print(f"✅ Vision token calculation: {num_tokens} tokens for 224x224 image")

    def test_vision_token_formatting(self):
        """Test that vision tokens are formatted correctly."""
        from src_new.data.dataset import Dataset
        
        # Create dataset instance (without loading data)
        dataset = Dataset.__new__(Dataset)  # Create without calling __init__
        
        # Test formatting with different token counts
        test_cases = [
            (1, "<|image_pad|>"),
            (3, "<|image_pad|> <|image_pad|> <|image_pad|>"),
            (0, "<|image_pad|>"),  # Should handle edge case
        ]
        
        for num_tokens, expected_pattern in test_cases:
            result = dataset._format_vision_tokens(num_tokens)
            
            # Check that result contains the right number of tokens
            token_count = result.count("<|image_pad|>")
            expected_count = max(1, num_tokens)  # At least 1 token
            
            assert token_count == expected_count, (
                f"Expected {expected_count} tokens, got {token_count} in: {result}"
            )
            
        print(f"✅ Vision token formatting test passed")

    def test_conversation_expansion(self):
        """Test that <image> placeholders are expanded correctly."""
        from src_new.data.dataset import Dataset
        
        # Create dataset instance (without loading data)
        dataset = Dataset.__new__(Dataset)  # Create without calling __init__
        
        # Mock the image processor and token calculation
        dataset.image_processor = MockImageProcessor()
        dataset._calculate_image_tokens = Mock(return_value=5)  # Mock 5 tokens per image
        
        # Test conversation with single image
        conversation = "User: Look at this <image> and tell me what you see."
        images = [Mock()]  # Single mock image
        
        expanded = dataset._expand_vision_tokens(conversation, images)
        
        # Should replace <image> with 5 image_pad tokens
        assert "<image>" not in expanded, "Should have replaced <image> placeholder"
        assert expanded.count("<|image_pad|>") == 5, f"Expected 5 image_pad tokens, got {expanded.count('<|image_pad|>')}"
        
        # Test conversation with multiple images
        conversation_multi = "First: <image> Second: <image> Third: <image>"
        images_multi = [Mock(), Mock(), Mock()]  # Three mock images
        
        expanded_multi = dataset._expand_vision_tokens(conversation_multi, images_multi)
        
        # Should replace all <image> placeholders
        assert "<image>" not in expanded_multi, "Should have replaced all <image> placeholders"
        assert expanded_multi.count("<|image_pad|>") == 15, f"Expected 15 image_pad tokens (3x5), got {expanded_multi.count('<|image_pad|>')}"
        
        print(f"✅ Conversation expansion test passed")

    def test_image_loading_from_paths(self):
        """Test image loading from file paths."""
        from src_new.data.dataset import Dataset
        
        # Create dataset instance (without loading data)
        dataset = Dataset.__new__(Dataset)  # Create without calling __init__
        dataset.data_root = Path("/fake/data")
        
        # Test with empty paths
        result = dataset._load_images_from_paths([])
        assert result == [], "Should return empty list for empty paths"
        
        # Test with non-existent paths (should handle gracefully)
        result = dataset._load_images_from_paths(["nonexistent.jpg"])
        assert result == [], "Should return empty list for non-existent files"
        
        print(f"✅ Image loading test passed")

    def test_pixel_values_processing(self):
        """Test pixel values processing with mock image processor."""
        from src_new.data.dataset import Dataset
        
        # Create dataset instance (without loading data)
        dataset = Dataset.__new__(Dataset)  # Create without calling __init__
        dataset.image_processor = MockImageProcessor()
        
        # Create mock images
        mock_images = [Mock(), Mock()]
        
        # Test processing
        pixel_values, image_grid_thw = dataset._process_images_from_list(mock_images)
        
        # Should return tensors
        assert pixel_values is not None, "Should return pixel_values tensor"
        assert image_grid_thw is not None, "Should return image_grid_thw tensor"
        assert isinstance(pixel_values, torch.Tensor), f"Expected tensor, got {type(pixel_values)}"
        assert isinstance(image_grid_thw, torch.Tensor), f"Expected tensor, got {type(image_grid_thw)}"
        
        # Check shapes
        assert pixel_values.dim() == 4, f"Expected 4D pixel_values, got {pixel_values.dim()}D"
        assert image_grid_thw.dim() == 2, f"Expected 2D image_grid_thw, got {image_grid_thw.dim()}D"
        
        print(f"✅ Pixel values processing test passed")
        print(f"   - Pixel values shape: {pixel_values.shape}")
        print(f"   - Image grid THW shape: {image_grid_thw.shape}")

    def test_tokenization_with_vision_tokens(self):
        """Test that tokenization works with vision tokens."""
        from src_new.data.dataset import Dataset
        
        # Create dataset instance (without loading data)
        dataset = Dataset.__new__(Dataset)  # Create without calling __init__
        dataset.tokenizer = MockTokenizer()
        dataset.image_processor = MockImageProcessor()
        dataset.config = Mock()
        dataset.config.model_max_length = 2048
        
        # Test conversation with vision tokens
        conversation = "User: <|image_pad|> <|image_pad|> <|image_pad|> What do you see?"
        mock_sample = {"student": {"objects": []}}
        mock_images = [Mock()]
        
        # Test tokenization
        result = dataset._tokenize_simple_conversation(conversation, mock_sample, mock_images)
        
        # Should return required fields
        required_fields = ["input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
            assert isinstance(result[field], torch.Tensor), f"Field {field} should be tensor"
        
        # Check that input_ids contains image_pad tokens
        image_pad_token_id = dataset.tokenizer.special_tokens.get("<|image_pad|>", -1)
        if image_pad_token_id != -1:
            token_count = (result["input_ids"] == image_pad_token_id).sum().item()
            assert token_count > 0, "Should have image_pad tokens in input_ids"
        
        print(f"✅ Tokenization with vision tokens test passed")
        print(f"   - Input IDs shape: {result['input_ids'].shape}")
        print(f"   - Pixel values shape: {result['pixel_values'].shape}")
        print(f"   - Image grid THW shape: {result['image_grid_thw'].shape}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestImageProcessing()
    test_instance.test_vision_token_calculation()
    test_instance.test_vision_token_formatting()
    test_instance.test_conversation_expansion()
    test_instance.test_image_loading_from_paths()
    test_instance.test_pixel_values_processing()
    test_instance.test_tokenization_with_vision_tokens()
    print("🎉 All image processing tests passed!")
