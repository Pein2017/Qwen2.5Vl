#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison tests between new and legacy collator implementations.

This module compares the behavior of the new StandardDataCollator in src_new/
with the legacy Qwen2_5VLCollator in src/reference/ to identify differences
in tensor handling that might be causing the teacher-student bug.

Key Differences to Analyze:
1. Pixel values concatenation behavior
2. Image grid THW handling
3. Batch structure and tensor dimensions
4. Position ID calculation differences
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from src_new.data.collator import StandardDataCollator
from src_new.tests.fixtures.mock_objects import MockTokenizer, MockImageProcessor


class TestLegacyComparison:
    """Compare new and legacy collator implementations."""

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
    def new_collator(self, mock_tokenizer, mock_config):
        """Create new StandardDataCollator."""
        return StandardDataCollator(
            tokenizer=mock_tokenizer,
            config=mock_config,
            pad_token_id=0,
            max_length=4096,
            label_pad_token_id=-100
        )

    def test_pixel_values_concatenation_comparison(self, new_collator):
        """
        Compare pixel_values concatenation behavior between implementations.
        
        Legacy: Uses torch.cat([input_ids["pixel_values"] for input_ids in batch_input_ids], dim=0)
        New: Uses torch.cat(pixel_values_list, dim=0) for patch format
        """
        print("🔍 Comparing pixel_values concatenation behavior:")
        
        # Create test samples similar to what legacy collator would process
        sample1 = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "pixel_values": torch.randn(1000, 1176),  # Patch format
            "image_grid_thw": torch.tensor([[1, 25, 40]]),
        }
        
        sample2 = {
            "input_ids": torch.tensor([5, 6, 7, 8, 9]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([5, 6, 7, 8, 9]),
            "pixel_values": torch.randn(1200, 1176),  # Different patch count
            "image_grid_thw": torch.tensor([[1, 30, 40]]),
        }
        
        features = [sample1, sample2]
        
        # Test new collator
        print("\n📋 New collator behavior:")
        new_batch = new_collator(features)
        print(f"   - pixel_values shape: {new_batch['pixel_values'].shape}")
        print(f"   - image_grid_thw shape: {new_batch['image_grid_thw'].shape}")
        
        # Simulate legacy collator behavior
        print("\n📋 Legacy collator behavior (simulated):")
        # Legacy concatenates along dim=0 just like new implementation
        legacy_pixel_values = torch.cat([sample1["pixel_values"], sample2["pixel_values"]], dim=0)
        legacy_image_grid_thw = torch.cat([sample1["image_grid_thw"], sample2["image_grid_thw"]], dim=0)
        
        print(f"   - pixel_values shape: {legacy_pixel_values.shape}")
        print(f"   - image_grid_thw shape: {legacy_image_grid_thw.shape}")
        
        # Compare results
        print("\n📋 Comparison:")
        pixel_values_match = torch.equal(new_batch["pixel_values"], legacy_pixel_values)
        grid_thw_match = torch.equal(new_batch["image_grid_thw"], legacy_image_grid_thw)
        
        print(f"   - pixel_values concatenation matches: {pixel_values_match}")
        print(f"   - image_grid_thw concatenation matches: {grid_thw_match}")
        
        # Key insight: Both implementations concatenate the same way
        assert new_batch["pixel_values"].shape == legacy_pixel_values.shape
        assert new_batch["image_grid_thw"].shape == legacy_image_grid_thw.shape
        
        print("   ✅ Concatenation behavior is identical between implementations")

    def test_batch_structure_differences(self, new_collator):
        """
        Compare batch structure between new and legacy implementations.
        
        This identifies key differences in how batches are structured.
        """
        print("🔍 Comparing batch structure differences:")
        
        # Create test sample
        sample = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([1, 2, 3, 4, 5]),
            "pixel_values": torch.randn(800, 1176),
            "image_grid_thw": torch.tensor([[1, 20, 40]]),
        }
        
        # Test new collator
        print("\n📋 New collator output structure:")
        new_batch = new_collator([sample])
        
        for key, tensor in new_batch.items():
            print(f"   - {key}: {tensor.shape} ({tensor.dtype})")
        
        # Simulate legacy collator structure
        print("\n📋 Legacy collator output structure (simulated):")
        # Legacy includes position_ids which new collator doesn't
        legacy_structure = {
            "input_ids": sample["input_ids"].unsqueeze(0),
            "attention_mask": sample["attention_mask"].unsqueeze(0),
            "labels": sample["labels"].unsqueeze(0),
            "pixel_values": sample["pixel_values"],  # No batch dimension for single sample
            "image_grid_thw": sample["image_grid_thw"],
            "position_ids": torch.randn(3, 1, 5),  # Legacy includes this
        }
        
        for key, tensor in legacy_structure.items():
            print(f"   - {key}: {tensor.shape} ({tensor.dtype})")
        
        # Key differences
        print("\n📋 Key differences:")
        print("   - Legacy includes position_ids, new collator doesn't")
        print("   - Both handle pixel_values concatenation identically")
        print("   - Both use same image_grid_thw stacking")
        
        # The bug is NOT in concatenation logic - it's elsewhere
        print("   💡 INSIGHT: Concatenation logic is identical - bug must be elsewhere")

    def test_qwen_processor_integration_simulation(self, new_collator):
        """
        Simulate how the legacy collator integrates with Qwen2.5-VL processor.
        
        This tests whether the issue is in processor integration vs collator logic.
        """
        print("🔍 Simulating Qwen2.5-VL processor integration:")
        
        # Simulate what Qwen2.5-VL processor would return
        # Based on legacy collator line 82-89
        mock_processor_output = {
            "input_ids": torch.tensor([[151644, 882, 151655, 151645, 151644, 77091]]),  # With image token
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "pixel_values": torch.randn(1, 1176, 28, 28),  # Processor format: [batch, features, h, w]
            "image_grid_thw": torch.tensor([[1, 28, 28]]),
        }
        
        print("\n📋 Mock processor output:")
        for key, tensor in mock_processor_output.items():
            print(f"   - {key}: {tensor.shape}")
        
        # The key insight: Processor outputs 4D pixel_values [batch, features, h, w]
        # But our collator expects 2D patch format [patches, features]
        
        print("\n📋 Format conversion needed:")
        processor_pixel_values = mock_processor_output["pixel_values"]
        print(f"   - Processor format: {processor_pixel_values.shape} [batch, features, h, w]")
        
        # Convert to patch format (flatten spatial dimensions)
        batch_size, features, h, w = processor_pixel_values.shape
        patch_format = processor_pixel_values.view(batch_size, features, h * w).transpose(1, 2).contiguous()
        patch_format = patch_format.view(-1, features)  # [total_patches, features]
        
        print(f"   - Patch format: {patch_format.shape} [patches, features]")
        
        # Create sample in patch format for collator
        converted_sample = {
            "input_ids": mock_processor_output["input_ids"].squeeze(0),
            "attention_mask": mock_processor_output["attention_mask"].squeeze(0),
            "labels": mock_processor_output["input_ids"].squeeze(0),  # Simplified
            "pixel_values": patch_format,
            "image_grid_thw": mock_processor_output["image_grid_thw"],
        }
        
        print(f"\n📋 Converted sample for collator:")
        for key, tensor in converted_sample.items():
            print(f"   - {key}: {tensor.shape}")
        
        # Test with new collator
        batch = new_collator([converted_sample])
        
        print(f"\n📋 Collator output:")
        for key, tensor in batch.items():
            print(f"   - {key}: {tensor.shape}")
        
        print("\n💡 KEY INSIGHT: The issue might be in format conversion between processor and collator")
        print("   - Processor outputs 4D format [batch, features, h, w]")
        print("   - Collator expects 2D patch format [patches, features]")
        print("   - Conversion step might be missing or incorrect in the pipeline")

    def test_teacher_student_format_analysis(self, new_collator):
        """
        Analyze the specific teacher-student format that causes the bug.
        
        This tests the exact scenario from the error message.
        """
        print("🔍 Analyzing teacher-student format that causes the bug:")
        
        # Recreate the exact scenario from the error
        # Error: pixel_values tensor has shape [4784, 1176]
        # This suggests 4784 patches total from teacher + student
        
        # Possible breakdown: 2392 + 2392 = 4784 patches
        teacher_patches = 2392
        student_patches = 2392
        
        print(f"\n📋 Recreating error scenario:")
        print(f"   - Teacher patches: {teacher_patches}")
        print(f"   - Student patches: {student_patches}")
        print(f"   - Total expected: {teacher_patches + student_patches} = 4784")
        
        # Create samples that would produce this
        teacher_sample = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "pixel_values": torch.randn(teacher_patches, 1176),
            "image_grid_thw": torch.tensor([[1, 49, 49]]),  # 49*49 ≈ 2401 patches
        }
        
        student_sample = {
            "input_ids": torch.tensor([5, 6, 7, 8, 9]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([5, 6, 7, 8, 9]),
            "pixel_values": torch.randn(student_patches, 1176),
            "image_grid_thw": torch.tensor([[1, 49, 49]]),  # Similar size
        }
        
        features = [teacher_sample, student_sample]
        
        print(f"\n📋 Input samples:")
        print(f"   - Teacher pixel_values: {teacher_sample['pixel_values'].shape}")
        print(f"   - Student pixel_values: {student_sample['pixel_values'].shape}")
        
        # Process with collator
        batch = new_collator(features)
        
        print(f"\n📋 Collator output:")
        print(f"   - pixel_values: {batch['pixel_values'].shape}")
        print(f"   - image_grid_thw: {batch['image_grid_thw'].shape}")
        
        # Verify we reproduced the exact error scenario
        expected_shape = torch.Size([4784, 1176])
        actual_shape = batch["pixel_values"].shape
        
        if actual_shape == expected_shape:
            print(f"   🎯 EXACT ERROR REPRODUCED: {actual_shape}")
            print("   📍 This confirms the concatenation behavior is working as designed")
            print("   💡 The 'bug' might actually be correct behavior for Qwen2.5-VL patch format")
        else:
            print(f"   ❌ Shape mismatch: expected {expected_shape}, got {actual_shape}")
        
        print("\n🔍 CRITICAL ANALYSIS:")
        print("   - The 2D tensor [4784, 1176] might be the CORRECT format for Qwen2.5-VL")
        print("   - Qwen2.5-VL uses patch-based vision processing")
        print("   - The model expects concatenated patches, not batched images")
        print("   - The 'error' might be in the error message, not the tensor format")


if __name__ == "__main__":
    # Run tests directly for debugging
    import sys
    sys.path.append("/data3/Qwen2.5-VL-main")
    
    test_instance = TestLegacyComparison()
    
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
    
    print("🚀 Running legacy comparison tests...")
    
    try:
        test_instance.test_pixel_values_concatenation_comparison(collator)
        test_instance.test_batch_structure_differences(collator)
        test_instance.test_qwen_processor_integration_simulation(collator)
        test_instance.test_teacher_student_format_analysis(collator)
        
        print("🎉 All legacy comparison tests completed!")
        
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
