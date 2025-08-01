#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge case tests for image tensor handling in the collator.

This module tests various image tensor dimension scenarios to validate
proper handling and identify potential sources of the teacher-student bug.

Focus Areas:
1. Different pixel_values tensor dimensions (2D, 3D, 4D)
2. Qwen2.5-VL patch format vs standard image format
3. Batch size variations and concatenation behavior
4. Error handling for invalid tensor shapes
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock

from src_new.data.collator import StandardDataCollator
from src_new.tests.fixtures.mock_objects import MockTokenizer
from src_new.tests.fixtures.test_utils import assert_tensor_shapes


class TestImageTensorEdgeCases:
    """Test edge cases for image tensor handling in collator."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        return MockTokenizer(vocab_size=151665)

    @pytest.fixture
    def mock_config(self):
        """Create mock config for collator."""
        config = Mock()
        config.max_total_length = 4096
        return config

    @pytest.fixture
    def standard_collator(self, mock_tokenizer, mock_config):
        """Create standard data collator for testing."""
        return StandardDataCollator(
            tokenizer=mock_tokenizer,
            config=mock_config,
            pad_token_id=0,
            max_length=4096,
            label_pad_token_id=-100
        )

    def test_qwen_patch_format_variations(self, standard_collator):
        """
        Test various Qwen2.5-VL patch format scenarios.
        
        Qwen2.5-VL uses [num_patches, patch_features] format where:
        - num_patches varies based on image size and patch size
        - patch_features is typically 1176 for Qwen2.5-VL-3B
        """
        test_cases = [
            {
                "name": "Small image patches",
                "pixel_values": torch.randn(256, 1176),  # 16x16 patches
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
            },
            {
                "name": "Medium image patches", 
                "pixel_values": torch.randn(1024, 1176),  # 32x32 patches
                "image_grid_thw": torch.tensor([[1, 32, 32]]),
            },
            {
                "name": "Large image patches",
                "pixel_values": torch.randn(4096, 1176),  # 64x64 patches
                "image_grid_thw": torch.tensor([[1, 64, 64]]),
            },
            {
                "name": "Non-square patches",
                "pixel_values": torch.randn(1200, 1176),  # 30x40 patches
                "image_grid_thw": torch.tensor([[1, 30, 40]]),
            },
        ]
        
        for case in test_cases:
            print(f"🔍 Testing {case['name']}: {case['pixel_values'].shape}")
            
            sample = {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "pixel_values": case["pixel_values"],
                "image_grid_thw": case["image_grid_thw"],
            }
            
            batch = standard_collator([sample])
            
            # For single sample, output should match input for patch format
            assert batch["pixel_values"].shape == case["pixel_values"].shape
            assert batch["image_grid_thw"].shape == (1, 3)  # [batch_size, 3]
            
            print(f"   ✅ Handled correctly: {batch['pixel_values'].shape}")

    def test_standard_image_format_variations(self, standard_collator):
        """
        Test various standard image format scenarios.
        
        Standard format is [C, H, W] for single images or [N, C, H, W] for batches.
        """
        test_cases = [
            {
                "name": "Standard RGB image",
                "pixel_values": torch.randn(3, 224, 224),
                "expected_output_shape": (1, 3, 224, 224),
            },
            {
                "name": "High resolution image",
                "pixel_values": torch.randn(3, 512, 512),
                "expected_output_shape": (1, 3, 512, 512),
            },
            {
                "name": "Grayscale image",
                "pixel_values": torch.randn(1, 224, 224),
                "expected_output_shape": (1, 1, 224, 224),
            },
            {
                "name": "Multi-channel image",
                "pixel_values": torch.randn(4, 224, 224),  # RGBA
                "expected_output_shape": (1, 4, 224, 224),
            },
        ]
        
        for case in test_cases:
            print(f"🔍 Testing {case['name']}: {case['pixel_values'].shape}")
            
            sample = {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "pixel_values": case["pixel_values"],
                "image_grid_thw": torch.tensor([[1, case["pixel_values"].shape[1], case["pixel_values"].shape[2]]]),
            }
            
            batch = standard_collator([sample])
            
            # Standard format should be stacked to 4D
            assert batch["pixel_values"].shape == case["expected_output_shape"]
            
            print(f"   ✅ Stacked to 4D: {batch['pixel_values'].shape}")

    def test_multi_image_4d_format(self, standard_collator):
        """
        Test 4D input format [N, C, H, W] representing multiple images.
        
        This tests how the collator handles pre-batched images.
        """
        test_cases = [
            {
                "name": "Two RGB images",
                "pixel_values": torch.randn(2, 3, 224, 224),
                "expected_behavior": "flatten_to_individual_images",
            },
            {
                "name": "Three high-res images",
                "pixel_values": torch.randn(3, 3, 512, 512),
                "expected_behavior": "flatten_to_individual_images",
            },
        ]
        
        for case in test_cases:
            print(f"🔍 Testing {case['name']}: {case['pixel_values'].shape}")
            
            sample = {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "pixel_values": case["pixel_values"],
                "image_grid_thw": torch.tensor([[1, 224, 224]]),
            }
            
            batch = standard_collator([sample])
            
            # 4D input should be flattened to individual images and then stacked
            num_images = case["pixel_values"].shape[0]
            expected_shape = (num_images, case["pixel_values"].shape[1], 
                            case["pixel_values"].shape[2], case["pixel_values"].shape[3])
            
            assert batch["pixel_values"].shape == expected_shape
            
            print(f"   ✅ Flattened and stacked: {batch['pixel_values'].shape}")

    def test_batch_concatenation_behavior(self, standard_collator):
        """
        Test how different batch sizes affect tensor concatenation.
        
        This is critical for understanding the teacher-student bug.
        """
        # Test with increasing batch sizes
        for batch_size in [1, 2, 3, 5, 10]:
            print(f"🔍 Testing batch size {batch_size}")
            
            features = []
            total_expected_patches = 0
            
            for i in range(batch_size):
                # Vary patch count per sample
                num_patches = 1000 + i * 100
                total_expected_patches += num_patches
                
                sample = {
                    "input_ids": torch.tensor([1, 2, 3, 4]),
                    "attention_mask": torch.tensor([1, 1, 1, 1]),
                    "labels": torch.tensor([1, 2, 3, 4]),
                    "pixel_values": torch.randn(num_patches, 1176),
                    "image_grid_thw": torch.tensor([[1, 25 + i, 25 + i]]),
                }
                features.append(sample)
            
            batch = standard_collator(features)
            
            # Check concatenation behavior
            assert batch["pixel_values"].dim() == 2, f"Expected 2D, got {batch['pixel_values'].dim()}D"
            assert batch["pixel_values"].shape[0] == total_expected_patches
            assert batch["pixel_values"].shape[1] == 1176
            assert batch["image_grid_thw"].shape == (batch_size, 3)
            
            print(f"   ✅ Concatenated {total_expected_patches} patches from {batch_size} samples")

    def test_zero_and_minimal_patches(self, standard_collator):
        """
        Test edge cases with zero or minimal patch counts.
        """
        test_cases = [
            {
                "name": "Single patch",
                "pixel_values": torch.randn(1, 1176),
                "should_succeed": True,
            },
            {
                "name": "Two patches",
                "pixel_values": torch.randn(2, 1176),
                "should_succeed": True,
            },
        ]
        
        for case in test_cases:
            print(f"🔍 Testing {case['name']}: {case['pixel_values'].shape}")
            
            sample = {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "pixel_values": case["pixel_values"],
                "image_grid_thw": torch.tensor([[1, 1, case["pixel_values"].shape[0]]]),
            }
            
            try:
                batch = standard_collator([sample])
                assert case["should_succeed"], f"Expected failure but succeeded: {case['name']}"
                print(f"   ✅ Handled successfully: {batch['pixel_values'].shape}")
                
            except Exception as e:
                assert not case["should_succeed"], f"Expected success but failed: {case['name']}: {e}"
                print(f"   💥 Failed as expected: {e}")

    def test_dimension_validation_comprehensive(self, standard_collator):
        """
        Comprehensive test of dimension validation logic.
        """
        test_cases = [
            {
                "name": "Valid 2D patch format",
                "pixel_values": torch.randn(100, 1176),
                "should_succeed": True,
                "expected_dim": 2,
            },
            {
                "name": "Valid 3D standard format",
                "pixel_values": torch.randn(3, 224, 224),
                "should_succeed": True,
                "expected_dim": 4,  # Gets stacked to 4D
            },
            {
                "name": "Valid 4D multi-image format",
                "pixel_values": torch.randn(2, 3, 224, 224),
                "should_succeed": True,
                "expected_dim": 4,  # Stays 4D but flattened
            },
            {
                "name": "Invalid 1D format",
                "pixel_values": torch.randn(1176),
                "should_succeed": False,
                "expected_error": "Invalid pixel_values dimensions 1",
            },
            {
                "name": "Invalid 5D format",
                "pixel_values": torch.randn(1, 1, 3, 224, 224),
                "should_succeed": False,
                "expected_error": "Invalid pixel_values dimensions 5",
            },
        ]
        
        for case in test_cases:
            print(f"🔍 Testing {case['name']}: {case['pixel_values'].shape}")
            
            sample = {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "pixel_values": case["pixel_values"],
                "image_grid_thw": torch.tensor([[1, 10, 10]]),
            }
            
            try:
                batch = standard_collator([sample])
                assert case["should_succeed"], f"Expected failure but succeeded: {case['name']}"
                assert batch["pixel_values"].dim() == case["expected_dim"]
                print(f"   ✅ Validated successfully: {batch['pixel_values'].shape}")
                
            except Exception as e:
                assert not case["should_succeed"], f"Expected success but failed: {case['name']}: {e}"
                if "expected_error" in case:
                    assert case["expected_error"] in str(e), f"Wrong error message: {e}"
                print(f"   💥 Correctly rejected: {e}")


if __name__ == "__main__":
    # Run tests directly for debugging
    import sys
    sys.path.append("/data3/Qwen2.5-VL-main")
    
    test_instance = TestImageTensorEdgeCases()
    
    # Create fixtures manually
    mock_tokenizer = MockTokenizer(vocab_size=151665)
    mock_config = Mock()
    mock_config.max_total_length = 4096
    
    collator = StandardDataCollator(
        tokenizer=mock_tokenizer,
        config=mock_config,
        pad_token_id=0,
        max_length=4096,
        label_pad_token_id=-100
    )
    
    print("🚀 Running image tensor edge case tests...")
    
    try:
        test_instance.test_qwen_patch_format_variations(collator)
        test_instance.test_standard_image_format_variations(collator)
        test_instance.test_multi_image_4d_format(collator)
        test_instance.test_batch_concatenation_behavior(collator)
        test_instance.test_zero_and_minimal_patches(collator)
        test_instance.test_dimension_validation_comprehensive(collator)
        
        print("🎉 All image tensor edge case tests completed!")
        
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
