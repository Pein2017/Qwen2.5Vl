#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical tests for teacher-student image processing bug in data collator.

This module reproduces and analyzes the specific ValueError in _collate_with_images
where pixel_values tensor has incorrect 2D shape [4784, 1176] instead of expected
3D [C,H,W] or 4D [N,C,H,W] format.

Bug Context:
- Training pipeline fails with ValueError in src_new/data/collator.py line 306
- Problem: pixel_values tensor has incorrect 2D shape [4784, 1176]
- Expected: 3D [C,H,W] or 4D [N,C,H,W] tensor format
- Error location: Image token calculations involving grid_thw values and merge_size

Test Scenario: 1 teacher sample + 1 student sample combined as single input batch
"""

from unittest.mock import Mock

import pytest
import torch

from src_new.data.collator import StandardDataCollator
from src_new.tests.fixtures.mock_objects import MockTokenizer


class TestTeacherStudentCollatorBug:
    """Test suite for reproducing and analyzing the teacher-student collator bug."""

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
            label_pad_token_id=-100,
        )

    def test_reproduce_teacher_student_pixel_values_bug(self, standard_collator):
        """
        CRITICAL TEST: Reproduce the exact teacher-student pixel_values bug.

        This test reproduces the scenario where:
        - 1 teacher sample + 1 student sample are combined
        - pixel_values tensor becomes 2D [4784, 1176] instead of proper 3D/4D
        - Error occurs in _collate_with_images method
        """
        # Create teacher-student sample that reproduces the bug
        # Based on the error, we expect pixel_values to be incorrectly flattened

        # Teacher sample with image data
        teacher_sample = {
            "input_ids": torch.tensor(
                [151644, 1587, 151645, 151644, 882, 100, 101, 151645]
            ),  # system + user
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, -100, -100, -100, 100, 101, -100]),
            # CRITICAL: This pixel_values shape might be causing the issue
            # Qwen2.5-VL uses patch-based format [num_patches, patch_features]
            "pixel_values": torch.randn(
                2392, 1176
            ),  # Half of the problematic 4784 patches
            "image_grid_thw": torch.tensor([[1, 28, 28]]),  # Single image grid info
        }

        # Student sample with image data
        student_sample = {
            "input_ids": torch.tensor(
                [151644, 882, 102, 103, 151645, 151644, 77091, 104, 105, 151645]
            ),  # user + assistant
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor(
                [-100, -100, -100, -100, -100, -100, 104, 105, -100, -100]
            ),
            # CRITICAL: Another patch-based pixel_values
            "pixel_values": torch.randn(
                2392, 1176
            ),  # Other half of the problematic 4784 patches
            "image_grid_thw": torch.tensor([[1, 28, 28]]),  # Single image grid info
        }

        # Combine into batch - this should trigger the bug
        features = [teacher_sample, student_sample]

        print(f"🔍 Testing teacher-student collation with:")
        print(
            f"   - Teacher pixel_values shape: {teacher_sample['pixel_values'].shape}"
        )
        print(
            f"   - Student pixel_values shape: {student_sample['pixel_values'].shape}"
        )
        print(f"   - Expected combined patches: {2392 + 2392} = 4784")

        # This should reproduce the bug where pixel_values becomes [4784, 1176]
        try:
            batch = standard_collator(features)

            print(f"✅ Collation succeeded!")
            print(f"   - Final pixel_values shape: {batch['pixel_values'].shape}")
            print(f"   - Final image_grid_thw shape: {batch['image_grid_thw'].shape}")

            # Analyze the result
            pixel_values = batch["pixel_values"]

            # Check if we reproduced the bug (2D tensor when expecting 3D/4D)
            if pixel_values.dim() == 2:
                print(f"🐛 BUG REPRODUCED: pixel_values is 2D {pixel_values.shape}")
                print(f"   - This matches the error: [4784, 1176]")
                print(f"   - Expected: 3D [C,H,W] or 4D [N,C,H,W]")

                # This is the bug - pixel_values should not be 2D for standard image processing
                assert pixel_values.shape == torch.Size([4784, 1176]), (
                    f"Expected bug shape [4784, 1176], got {pixel_values.shape}"
                )

            elif pixel_values.dim() == 3:
                print(f"✅ Correct 3D format: {pixel_values.shape}")

            elif pixel_values.dim() == 4:
                print(f"✅ Correct 4D format: {pixel_values.shape}")

            else:
                print(
                    f"❌ Unexpected dimension: {pixel_values.dim()}D - {pixel_values.shape}"
                )

        except ValueError as e:
            print(f"💥 ValueError caught (expected): {e}")
            # This is the actual error we're trying to reproduce
            assert "Invalid pixel_values dimensions" in str(
                e
            ) or "Image features and image tokens do not match" in str(e)
            print(f"🎯 Successfully reproduced the teacher-student collator bug!")

    def test_qwen_patch_format_handling(self, standard_collator):
        """
        Test how the collator handles Qwen2.5-VL patch-based pixel_values format.

        Qwen2.5-VL uses flattened patches [num_patches, patch_features] instead of
        standard image tensors [C, H, W].
        """
        # Create samples with Qwen2.5-VL patch format
        sample_with_patches = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([1, 2, 3, 4, 5]),
            # Qwen2.5-VL patch format: [num_patches, patch_features]
            "pixel_values": torch.randn(1024, 1176),  # Typical patch dimensions
            "image_grid_thw": torch.tensor([[1, 32, 32]]),  # Grid dimensions
        }

        features = [sample_with_patches]

        print(f"🔍 Testing Qwen2.5-VL patch format:")
        print(
            f"   - Input pixel_values shape: {sample_with_patches['pixel_values'].shape}"
        )
        print(
            f"   - Input image_grid_thw shape: {sample_with_patches['image_grid_thw'].shape}"
        )

        batch = standard_collator(features)

        print(f"✅ Patch format handling:")
        print(f"   - Output pixel_values shape: {batch['pixel_values'].shape}")
        print(f"   - Output image_grid_thw shape: {batch['image_grid_thw'].shape}")

        # For Qwen2.5-VL, 2D patch format is actually expected
        pixel_values = batch["pixel_values"]
        if pixel_values.dim() == 2:
            print(f"✅ Correct Qwen2.5-VL patch format: {pixel_values.shape}")
        else:
            print(f"❌ Unexpected format for Qwen2.5-VL: {pixel_values.shape}")

    def test_standard_image_format_handling(self, standard_collator):
        """
        Test how the collator handles standard image format [C, H, W].

        This tests the fallback to standard image processing when not using
        Qwen2.5-VL patch format.
        """
        # Create samples with standard image format
        sample_with_standard_images = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([1, 2, 3, 4, 5]),
            # Standard image format: [C, H, W]
            "pixel_values": torch.randn(3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 224, 224]]),
        }

        features = [sample_with_standard_images]

        print(f"🔍 Testing standard image format:")
        print(
            f"   - Input pixel_values shape: {sample_with_standard_images['pixel_values'].shape}"
        )

        batch = standard_collator(features)

        print(f"✅ Standard format handling:")
        print(f"   - Output pixel_values shape: {batch['pixel_values'].shape}")

        # For standard images, should be stacked to 4D [N, C, H, W]
        pixel_values = batch["pixel_values"]
        expected_shape = (1, 3, 224, 224)  # [batch_size, channels, height, width]
        assert pixel_values.shape == expected_shape, (
            f"Expected {expected_shape}, got {pixel_values.shape}"
        )

    def test_mixed_format_handling(self, standard_collator):
        """
        Test how the collator handles mixed image formats in a single batch.

        This could be a source of the bug if different samples have different formats.
        """
        # Create samples with mixed formats
        patch_sample = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
            "pixel_values": torch.randn(512, 1176),  # Patch format
            "image_grid_thw": torch.tensor([[1, 16, 16]]),
        }

        standard_sample = {
            "input_ids": torch.tensor([4, 5, 6, 7]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([4, 5, 6, 7]),
            "pixel_values": torch.randn(3, 224, 224),  # Standard format
            "image_grid_thw": torch.tensor([[1, 224, 224]]),
        }

        features = [patch_sample, standard_sample]

        print(f"🔍 Testing mixed image formats:")
        print(f"   - Sample 1 (patch): {patch_sample['pixel_values'].shape}")
        print(f"   - Sample 2 (standard): {standard_sample['pixel_values'].shape}")

        try:
            batch = standard_collator(features)
            print(f"✅ Mixed format handling succeeded:")
            print(f"   - Output pixel_values shape: {batch['pixel_values'].shape}")

        except Exception as e:
            print(f"💥 Mixed format error: {e}")
            # This might be where the bug occurs - inconsistent formats
            print(f"🎯 Potential bug source: Mixed image formats in batch")

    def test_teacher_student_image_grid_thw_handling(self, standard_collator):
        """
        Test image_grid_thw handling in teacher-student scenarios.

        The bug might be related to how image_grid_thw is processed when
        combining teacher and student samples.
        """
        # Teacher-student sample with specific image_grid_thw patterns
        teacher_sample = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
            "pixel_values": torch.randn(1000, 1176),
            # Teacher might have different grid dimensions
            "image_grid_thw": torch.tensor([[2, 20, 25]]),  # Different aspect ratio
        }

        student_sample = {
            "input_ids": torch.tensor([4, 5, 6, 7]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([4, 5, 6, 7]),
            "pixel_values": torch.randn(1200, 1176),
            # Student might have different grid dimensions
            "image_grid_thw": torch.tensor([[1, 30, 40]]),  # Different dimensions
        }

        features = [teacher_sample, student_sample]

        print(f"🔍 Testing teacher-student image_grid_thw:")
        print(f"   - Teacher grid_thw: {teacher_sample['image_grid_thw']}")
        print(f"   - Student grid_thw: {student_sample['image_grid_thw']}")

        batch = standard_collator(features)

        print(f"✅ Grid THW handling:")
        print(f"   - Output image_grid_thw shape: {batch['image_grid_thw'].shape}")
        print(f"   - Output image_grid_thw values: {batch['image_grid_thw']}")

        # Check that grid_thw is properly stacked
        expected_grid_shape = (2, 3)  # [batch_size, 3] for (t, h, w)
        assert batch["image_grid_thw"].shape == expected_grid_shape, (
            f"Expected {expected_grid_shape}, got {batch['image_grid_thw'].shape}"
        )

    def test_large_batch_teacher_student_scenario(self, standard_collator):
        """
        Test larger batch with multiple teacher-student pairs.

        This tests if the bug scales with batch size and helps identify
        if the issue is in concatenation logic.
        """
        features = []

        # Create multiple teacher-student pairs
        for i in range(3):  # 3 teacher-student pairs = 6 samples total
            # Teacher sample
            teacher = {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 3, 4]),
                "pixel_values": torch.randn(
                    800 + i * 100, 1176
                ),  # Varying patch counts
                "image_grid_thw": torch.tensor([[1, 20 + i * 2, 20 + i * 2]]),
            }

            # Student sample
            student = {
                "input_ids": torch.tensor([5, 6, 7, 8, 9]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([5, 6, 7, 8, 9]),
                "pixel_values": torch.randn(
                    900 + i * 100, 1176
                ),  # Varying patch counts
                "image_grid_thw": torch.tensor([[1, 22 + i * 2, 22 + i * 2]]),
            }

            features.extend([teacher, student])

        print(f"🔍 Testing large batch (6 samples):")
        total_patches = sum(f["pixel_values"].shape[0] for f in features)
        print(f"   - Total patches across all samples: {total_patches}")

        batch = standard_collator(features)

        print(f"✅ Large batch handling:")
        print(f"   - Output pixel_values shape: {batch['pixel_values'].shape}")
        print(f"   - Output image_grid_thw shape: {batch['image_grid_thw'].shape}")

        # Verify concatenation worked correctly
        if batch["pixel_values"].dim() == 2:
            assert batch["pixel_values"].shape[0] == total_patches, (
                f"Expected {total_patches} total patches, got {batch['pixel_values'].shape[0]}"
            )

    def test_edge_case_empty_pixel_values(self, standard_collator):
        """
        Test edge case with empty or minimal pixel_values.

        This tests robustness of the collator with edge cases.
        """
        # Sample with minimal pixel_values
        minimal_sample = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
            "pixel_values": torch.randn(1, 1176),  # Minimal patches
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }

        features = [minimal_sample]

        print(f"🔍 Testing minimal pixel_values:")
        print(f"   - Input shape: {minimal_sample['pixel_values'].shape}")

        batch = standard_collator(features)

        print(f"✅ Minimal pixel_values handling:")
        print(f"   - Output shape: {batch['pixel_values'].shape}")

    def test_dimension_mismatch_detection(self, standard_collator):
        """
        Test detection of dimension mismatches that could cause the bug.

        This tests various invalid dimension scenarios to understand
        how the collator handles them.
        """
        # Test cases with invalid dimensions
        invalid_cases = [
            {
                "name": "1D pixel_values",
                "pixel_values": torch.randn(1176),  # 1D - invalid
                "should_fail": True,
            },
            {
                "name": "5D pixel_values",
                "pixel_values": torch.randn(1, 1, 3, 224, 224),  # 5D - invalid
                "should_fail": True,
            },
            {
                "name": "Wrong 2D dimensions",
                "pixel_values": torch.randn(1176, 100),  # Wrong feature size
                "should_fail": False,  # Might be handled as patch format
            },
        ]

        for case in invalid_cases:
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
                print(f"   ✅ Handled successfully: {batch['pixel_values'].shape}")

            except Exception as e:
                print(f"   💥 Failed as expected: {e}")
                if case["should_fail"]:
                    print(f"   ✅ Correctly rejected invalid input")
                else:
                    print(f"   ❌ Unexpected failure")

    def test_coordinate_span_integration(self, standard_collator):
        """
        Test teacher-student collation with coordinate spans.

        This tests if the bug is related to coordinate token processing
        in teacher-student scenarios.
        """
        # Teacher sample with coordinate spans
        teacher_sample = {
            "input_ids": torch.tensor(
                [1, 2, 151667, 151668, 3, 4]
            ),  # With coord tokens
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor([1, 2, 151667, 151668, 3, 4]),
            "pixel_values": torch.randn(1000, 1176),
            "image_grid_thw": torch.tensor([[1, 25, 25]]),
            "teacher_assistant_spans": [(2, 4)],  # Coordinate token span
        }

        # Student sample with coordinate spans
        student_sample = {
            "input_ids": torch.tensor(
                [5, 6, 151669, 151670, 7, 8, 9]
            ),  # With coord tokens
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor([5, 6, 151669, 151670, 7, 8, 9]),
            "pixel_values": torch.randn(1200, 1176),
            "image_grid_thw": torch.tensor([[1, 30, 30]]),
            "student_assistant_spans": [(2, 4)],  # Coordinate token span
        }

        features = [teacher_sample, student_sample]

        print(f"🔍 Testing coordinate span integration:")
        print(
            f"   - Teacher has coordinate tokens at positions: {teacher_sample['teacher_assistant_spans']}"
        )
        print(
            f"   - Student has coordinate tokens at positions: {student_sample['student_assistant_spans']}"
        )

        batch = standard_collator(features)

        print(f"✅ Coordinate span handling:")
        print(f"   - Pixel values shape: {batch['pixel_values'].shape}")
        print(f"   - Teacher spans preserved: {'teacher_assistant_spans' in batch}")
        print(f"   - Student spans preserved: {'student_assistant_spans' in batch}")

        # Verify spans are preserved in batch
        if "teacher_assistant_spans" in batch:
            print(f"   - Teacher spans: {batch['teacher_assistant_spans']}")
        if "student_assistant_spans" in batch:
            print(f"   - Student spans: {batch['student_assistant_spans']}")


if __name__ == "__main__":
    # Run tests directly for debugging
    import sys

    sys.path.append("/data3/Qwen2.5-VL-main")

    test_instance = TestTeacherStudentCollatorBug()

    # Create fixtures manually
    mock_tokenizer = MockTokenizer(vocab_size=151665)
    mock_config = Mock()
    mock_config.max_total_length = 4096

    collator = StandardDataCollator(
        tokenizer=mock_tokenizer,
        config=mock_config,
        pad_token_id=0,
        max_length=4096,
        label_pad_token_id=-100,
    )

    print("🚀 Running teacher-student collator bug tests...")

    try:
        test_instance.test_reproduce_teacher_student_pixel_values_bug(collator)
        test_instance.test_qwen_patch_format_handling(collator)
        test_instance.test_standard_image_format_handling(collator)
        test_instance.test_mixed_format_handling(collator)
        test_instance.test_teacher_student_image_grid_thw_handling(collator)
        test_instance.test_large_batch_teacher_student_scenario(collator)
        test_instance.test_edge_case_empty_pixel_values(collator)
        test_instance.test_dimension_mismatch_detection(collator)
        test_instance.test_coordinate_span_integration(collator)

        print("🎉 All teacher-student collator tests completed!")

    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        import traceback

        traceback.print_exc()
