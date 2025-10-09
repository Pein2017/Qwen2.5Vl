"""
Multimodal Alignment Diagnostic

Feature: 004-grpo-post-training / User Story 2
Constitution: v4.1.1

Validates image token consistency across pipeline stages:
1. Dataset output
2. Buffer generation
3. Loss computation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer

from src_new.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MultimodalAlignmentCheck:
    """Results from multimodal alignment validation."""
    
    step: int
    stage: str
    sample_idx: int
    is_valid: bool
    
    # Token counts
    expected_image_tokens: int
    actual_image_tokens: int
    mismatch_magnitude: int
    
    # Pixel validation
    expected_patches: int
    actual_patches: int
    patches_match: bool
    
    # THW validation
    thw_shape_valid: bool
    num_images: int
    
    # Lists
    errors: List[str]
    warnings: List[str]
    
    def to_tensorboard(self) -> Dict[str, float]:
        """Convert to TensorBoard scalar dict."""
        return {
            f"multimodal/{self.stage}/is_valid": float(self.is_valid),
            f"multimodal/{self.stage}/token_mismatch": float(self.mismatch_magnitude),
            f"multimodal/{self.stage}/patches_match": float(self.patches_match),
            f"multimodal/{self.stage}/thw_valid": float(self.thw_shape_valid),
            f"multimodal/{self.stage}/num_images": float(self.num_images),
        }
    
    def log_warnings(self) -> None:
        """Log warnings and errors if present."""
        if self.errors:
            logger.error(
                f"[Multimodal {self.stage}] Step {self.step}, Sample {self.sample_idx}: "
                f"FAILED with {len(self.errors)} errors"
            )
            for err in self.errors:
                logger.error(f"  ❌ {err}")
        
        if self.warnings:
            logger.warning(
                f"[Multimodal {self.stage}] Step {self.step}, Sample {self.sample_idx}: "
                f"{len(self.warnings)} warnings"
            )
            for warn in self.warnings:
                logger.warning(f"  ⚠️  {warn}")


def compute_multimodal_alignment(
    step: int,
    stage: str,
    sample_idx: int,
    input_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    merge_size: int = 2,
) -> MultimodalAlignmentCheck:
    """
    Validate multimodal alignment at a pipeline stage.
    
    Args:
        step: Current training step
        stage: Pipeline stage name
        sample_idx: Sample index in batch
        input_ids: Token IDs [seq_len] or [1, seq_len]
        pixel_values: Vision tensor [num_patches, C, H, W] or None
        image_grid_thw: Grid metadata [num_images, 3] or None
        tokenizer: Tokenizer with image_pad_token_id
        merge_size: Vision patch merge factor (default 2)
        
    Returns:
        MultimodalAlignmentCheck with validation results
    """
    errors = []
    warnings = []
    
    # Flatten input_ids if needed
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    
    # Count actual image tokens
    image_pad_id = getattr(tokenizer, "image_pad_token_id", None)
    if image_pad_id is None:
        errors.append("Tokenizer missing image_pad_token_id")
        actual_image_tokens = 0
    else:
        actual_image_tokens = (input_ids == image_pad_id).sum().item()
    
    # Validate THW shape
    thw_shape_valid = True
    num_images = 0
    if image_grid_thw is not None:
        if image_grid_thw.dim() != 2 or image_grid_thw.size(1) != 3:
            errors.append(
                f"image_grid_thw shape {tuple(image_grid_thw.shape)} != [num_images, 3]"
            )
            thw_shape_valid = False
        else:
            num_images = image_grid_thw.size(0)
    
    # Compute expected image tokens
    expected_image_tokens = 0
    if image_grid_thw is not None and thw_shape_valid:
        # Formula: sum(t*h*w) // merge_size^2
        thw_prod = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        expected_image_tokens = int(thw_prod.sum().item() // (merge_size ** 2))
    
    # Check token mismatch
    mismatch_magnitude = abs(expected_image_tokens - actual_image_tokens)
    if pixel_values is not None and mismatch_magnitude > 0:
        errors.append(
            f"Image token mismatch: expected {expected_image_tokens} from THW, "
            f"found {actual_image_tokens} in input_ids (diff={mismatch_magnitude})"
        )
    elif pixel_values is None and actual_image_tokens > 0:
        warnings.append(
            f"Found {actual_image_tokens} image tokens but pixel_values is None"
        )
    
    # Validate pixel_values vs THW
    patches_match = True
    expected_patches = 0
    actual_patches = 0
    
    if pixel_values is not None and image_grid_thw is not None and thw_shape_valid:
        # Expected patches: sum(t*h*w)
        expected_patches = int(
            (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            .sum()
            .item()
        )
        actual_patches = pixel_values.size(0)
        
        if expected_patches != actual_patches:
            errors.append(
                f"Pixel patches mismatch: expected {expected_patches} from THW, "
                f"got {actual_patches} rows in pixel_values"
            )
            patches_match = False
    
    # Final validity
    is_valid = len(errors) == 0
    
    return MultimodalAlignmentCheck(
        step=step,
        stage=stage,
        sample_idx=sample_idx,
        is_valid=is_valid,
        expected_image_tokens=expected_image_tokens,
        actual_image_tokens=actual_image_tokens,
        mismatch_magnitude=mismatch_magnitude,
        expected_patches=expected_patches,
        actual_patches=actual_patches,
        patches_match=patches_match,
        thw_shape_valid=thw_shape_valid,
        num_images=num_images,
        errors=errors,
        warnings=warnings,
    )
