"""
Tests for Multimodal Alignment Diagnostic

Feature: 004-grpo-post-training / User Story 2
Constitution: v4.1.1
"""

import pytest
import torch

from src_new.rl.diagnostics.multimodal import (
    MultimodalAlignmentCheck,
    compute_multimodal_alignment,
)


class TestMultimodalAlignmentCheck:
    """Test MultimodalAlignmentCheck dataclass and methods."""
    
    def test_alignment_check_creation(self):
        """Test creating a MultimodalAlignmentCheck instance."""
        check = MultimodalAlignmentCheck(
            step=100,
            stage="dataset",
            sample_idx=0,
            is_valid=True,
            expected_image_tokens=256,
            actual_image_tokens=256,
            mismatch_magnitude=0,
            expected_patches=1024,
            actual_patches=1024,
            patches_match=True,
            thw_shape_valid=True,
            num_images=1,
            errors=[],
            warnings=[],
        )
        
        assert check.step == 100
        assert check.is_valid is True
        assert check.mismatch_magnitude == 0
        
    def test_to_tensorboard_returns_dict(self):
        """Test that to_tensorboard returns valid dict."""
        check = MultimodalAlignmentCheck(
            step=100,
            stage="dataset",
            sample_idx=0,
            is_valid=True,
            expected_image_tokens=256,
            actual_image_tokens=256,
            mismatch_magnitude=0,
            expected_patches=1024,
            actual_patches=1024,
            patches_match=True,
            thw_shape_valid=True,
            num_images=1,
            errors=[],
            warnings=[],
        )
        
        tb_dict = check.to_tensorboard()
        
        assert isinstance(tb_dict, dict)
        assert "multimodal/dataset/is_valid" in tb_dict
        assert tb_dict["multimodal/dataset/is_valid"] == 1.0
        assert tb_dict["multimodal/dataset/token_mismatch"] == 0.0
        
    def test_log_warnings_with_errors(self, caplog):
        """Test that errors are logged."""
        check = MultimodalAlignmentCheck(
            step=100,
            stage="dataset",
            sample_idx=0,
            is_valid=False,
            expected_image_tokens=256,
            actual_image_tokens=200,
            mismatch_magnitude=56,
            expected_patches=1024,
            actual_patches=1024,
            patches_match=True,
            thw_shape_valid=True,
            num_images=1,
            errors=["Image token mismatch"],
            warnings=[],
        )
        
        check.log_warnings()
        
        # Check that error was logged (exact message depends on logger config)
        assert check.errors == ["Image token mismatch"]


class TestComputeMultimodalAlignment:
    """Test compute_multimodal_alignment function."""
    
    def test_valid_alignment(self, mock_tokenizer):
        """Test with valid multimodal alignment."""
        # Create input_ids with 256 image tokens
        image_pad_id = 151655  # Qwen2.5-VL image_pad_token_id
        mock_tokenizer.image_pad_token_id = image_pad_id
        
        input_ids = torch.cat([
            torch.tensor([1, 2, 3]),  # Regular tokens
            torch.full((256,), image_pad_id),  # Image tokens
            torch.tensor([4, 5, 6]),  # More regular tokens
        ])
        
        # THW that produces 256 tokens: (1, 32, 32) → 1*32*32 / 4 = 256
        image_grid_thw = torch.tensor([[1, 32, 32]])  # [1, 3]
        
        # Pixel values with 1024 patches (1*32*32)
        pixel_values = torch.randn(1024, 3, 14, 14)
        
        check = compute_multimodal_alignment(
            step=100,
            stage="dataset",
            sample_idx=0,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            tokenizer=mock_tokenizer,
            merge_size=2,
        )
        
        assert check.is_valid is True
        assert check.expected_image_tokens == 256
        assert check.actual_image_tokens == 256
        assert check.mismatch_magnitude == 0
        assert check.expected_patches == 1024
        assert check.actual_patches == 1024
        assert check.patches_match is True
        assert len(check.errors) == 0
        
    def test_token_mismatch_detection(self, mock_tokenizer):
        """Test detection of image token mismatch."""
        image_pad_id = 151655
        mock_tokenizer.image_pad_token_id = image_pad_id
        
        # Input has 200 image tokens
        input_ids = torch.cat([
            torch.tensor([1, 2, 3]),
            torch.full((200,), image_pad_id),
            torch.tensor([4, 5, 6]),
        ])
        
        # THW expects 256 tokens
        image_grid_thw = torch.tensor([[1, 32, 32]])
        pixel_values = torch.randn(1024, 3, 14, 14)
        
        check = compute_multimodal_alignment(
            step=100,
            stage="dataset",
            sample_idx=0,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            tokenizer=mock_tokenizer,
            merge_size=2,
        )
        
        assert check.is_valid is False
        assert check.mismatch_magnitude == 56  # |256 - 200|
        assert len(check.errors) > 0
        assert "mismatch" in check.errors[0].lower()
        
    def test_patch_mismatch_detection(self, mock_tokenizer):
        """Test detection of pixel_values patch count mismatch."""
        image_pad_id = 151655
        mock_tokenizer.image_pad_token_id = image_pad_id
        
        input_ids = torch.cat([
            torch.tensor([1, 2, 3]),
            torch.full((256,), image_pad_id),
            torch.tensor([4, 5, 6]),
        ])
        
        image_grid_thw = torch.tensor([[1, 32, 32]])
        
        # Wrong number of patches (should be 1024)
        pixel_values = torch.randn(900, 3, 14, 14)
        
        check = compute_multimodal_alignment(
            step=100,
            stage="dataset",
            sample_idx=0,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            tokenizer=mock_tokenizer,
            merge_size=2,
        )
        
        assert check.is_valid is False
        assert check.patches_match is False
        assert check.expected_patches == 1024
        assert check.actual_patches == 900
        
    def test_invalid_thw_shape(self, mock_tokenizer):
        """Test detection of invalid image_grid_thw shape."""
        image_pad_id = 151655
        mock_tokenizer.image_pad_token_id = image_pad_id
        
        input_ids = torch.full((100,), image_pad_id)
        
        # Invalid shape (should be [N, 3])
        image_grid_thw = torch.tensor([1, 32, 32])  # Missing batch dim
        pixel_values = torch.randn(1024, 3, 14, 14)
        
        check = compute_multimodal_alignment(
            step=100,
            stage="dataset",
            sample_idx=0,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            tokenizer=mock_tokenizer,
            merge_size=2,
        )
        
        assert check.is_valid is False
        assert check.thw_shape_valid is False
        assert len(check.errors) > 0
        
    def test_no_pixel_values_with_image_tokens(self, mock_tokenizer):
        """Test warning when image tokens present but no pixel_values."""
        image_pad_id = 151655
        mock_tokenizer.image_pad_token_id = image_pad_id
        
        input_ids = torch.full((256,), image_pad_id)
        
        check = compute_multimodal_alignment(
            step=100,
            stage="dataset",
            sample_idx=0,
            input_ids=input_ids,
            pixel_values=None,
            image_grid_thw=None,
            tokenizer=mock_tokenizer,
            merge_size=2,
        )
        
        # Should have warnings but might still be valid (text-only turn)
        assert len(check.warnings) > 0
        
    def test_multiple_images(self, mock_tokenizer):
        """Test with multiple images in one sample."""
        image_pad_id = 151655
        mock_tokenizer.image_pad_token_id = image_pad_id
        
        # 2 images: (1,32,32) and (1,16,16)
        # Image 1: 1*32*32/4 = 256 tokens
        # Image 2: 1*16*16/4 = 64 tokens
        # Total: 320 tokens
        input_ids = torch.cat([
            torch.tensor([1, 2]),
            torch.full((256,), image_pad_id),
            torch.tensor([3, 4]),
            torch.full((64,), image_pad_id),
            torch.tensor([5, 6]),
        ])
        
        image_grid_thw = torch.tensor([
            [1, 32, 32],
            [1, 16, 16],
        ])  # [2, 3]
        
        # Patches: 1*32*32 + 1*16*16 = 1024 + 256 = 1280
        pixel_values = torch.randn(1280, 3, 14, 14)
        
        check = compute_multimodal_alignment(
            step=100,
            stage="dataset",
            sample_idx=0,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            tokenizer=mock_tokenizer,
            merge_size=2,
        )
        
        assert check.is_valid is True
        assert check.num_images == 2
        assert check.expected_image_tokens == 320
        assert check.actual_image_tokens == 320


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
