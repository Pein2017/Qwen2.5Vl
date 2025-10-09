"""
Validation Checkpoint Interface

Defines the contract for pipeline-stage validation in GRPO diagnostics.
Each stage (dataset, buffer, generation, reward, loss) must implement this interface.
"""

from typing import Any, Dict, Protocol

import torch


class ValidationCheckpoint(Protocol):
    """Interface for multimodal validation at pipeline stages."""

    def validate_at_stage(
        self,
        stage: str,
        sample_idx: int,
        **tensors: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Validate tensor consistency at a specific pipeline stage.

        Args:
            stage: One of {"dataset", "buffer_generation", "loss_computation"}
            sample_idx: Index of the sample being validated
            **tensors: Stage-specific tensors to validate:
                - dataset: input_ids, attention_mask, pixel_values, image_grid_thw, labels
                - buffer_generation: input_ids, pixel_values, image_grid_thw, generation_logps
                - loss_computation: logits, labels, pixel_values, image_grid_thw, old_logps

        Returns:
            Dict with validation results:
            {
                "is_valid": bool,
                "errors": List[str],  # Empty if valid
                "warnings": List[str],
                "metrics": Dict[str, float],  # Stage-specific metrics
            }

        Raises:
            ImageTokenMismatchError: If fail-fast validation detects critical mismatch
            ValueError: If stage is invalid or required tensors missing
        """
        ...

    def get_expected_image_tokens(
        self, image_grid_thw: torch.Tensor, merge_size: int = 2
    ) -> int:
        """
        Compute expected <|image_pad|> count from THW metadata.

        Formula: (∑ t*h*w for (t,h,w) in image_grid_thw) // (merge_size ** 2)

        Args:
            image_grid_thw: Tensor of shape [num_images, 3] with (temporal, height, width)
            merge_size: Vision patch merge factor (default 2 for Qwen2.5-VL)

        Returns:
            Expected number of image_pad tokens in input_ids
        """
        ...

    def assert_patches_match_thw(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        """
        Assert that packed pixel_values rows match THW-derived patch count.

        Validates:
        - pixel_values.shape[0] == sum(t*h*w for t,h,w in image_grid_thw)
        - image_grid_thw.shape == [num_images, 3]

        Args:
            pixel_values: Packed vision tensor [∑(t*h*w), C, H, W]
            image_grid_thw: Grid metadata [num_images, 3]

        Raises:
            AssertionError: If patch count mismatch detected
        """
        ...


class MultimodalAlignmentValidator:
    """
    Concrete implementation of ValidationCheckpoint for multimodal diagnostics.

    Usage:
        validator = MultimodalAlignmentValidator(merge_size=2, tokenizer=tokenizer)
        result = validator.validate_at_stage(
            stage="buffer_generation",
            sample_idx=0,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if not result["is_valid"]:
            logger.error(f"Validation failed: {result['errors']}")
    """

    def __init__(self, merge_size: int, tokenizer):
        self.merge_size = merge_size
        self.tokenizer = tokenizer
        self.image_pad_token = "<|image_pad|>"

    def validate_at_stage(
        self,
        stage: str,
        sample_idx: int,
        **tensors: torch.Tensor,
    ) -> Dict[str, Any]:
        """Implement validation checkpoint interface."""

        valid_stages = {"dataset", "buffer_generation", "loss_computation"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}', must be one of {valid_stages}")

        errors = []
        warnings = []
        metrics = {}

        # Extract tensors
        input_ids = tensors.get("input_ids")
        pixel_values = tensors.get("pixel_values")
        image_grid_thw = tensors.get("image_grid_thw")

        if input_ids is None or pixel_values is None or image_grid_thw is None:
            errors.append(f"Missing required tensors for stage '{stage}'")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "metrics": metrics,
            }

        # Validation 1: Image token count
        expected_tokens = self.get_expected_image_tokens(
            image_grid_thw, self.merge_size
        )
        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        actual_tokens = decoded_text.count(self.image_pad_token)

        if expected_tokens != actual_tokens:
            errors.append(
                f"Image token mismatch: expected {expected_tokens}, got {actual_tokens}"
            )

        metrics["expected_image_tokens"] = expected_tokens
        metrics["actual_image_tokens"] = actual_tokens

        # Validation 2: Pixel values row count
        expected_rows = sum(t * h * w for t, h, w in image_grid_thw.tolist())
        actual_rows = pixel_values.shape[0]

        if expected_rows != actual_rows:
            errors.append(
                f"Pixel values row mismatch: expected {expected_rows}, got {actual_rows}"
            )

        metrics["expected_pixel_rows"] = expected_rows
        metrics["actual_pixel_rows"] = actual_rows

        # Validation 3: THW shape
        num_images = image_grid_thw.shape[0]
        if image_grid_thw.shape != (num_images, 3):
            errors.append(
                f"Invalid image_grid_thw shape: expected [{num_images}, 3], got {list(image_grid_thw.shape)}"
            )

        metrics["num_images"] = num_images
        metrics["thw_shape_valid"] = image_grid_thw.shape == (num_images, 3)

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    def get_expected_image_tokens(
        self, image_grid_thw: torch.Tensor, merge_size: int = 2
    ) -> int:
        """Compute expected image_pad count from THW."""
        total_patches = sum(t * h * w for t, h, w in image_grid_thw.tolist())
        return total_patches // (merge_size**2)

    def assert_patches_match_thw(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        """Assert pixel_values rows match THW patch count."""
        expected_rows = sum(t * h * w for t, h, w in image_grid_thw.tolist())
        actual_rows = pixel_values.shape[0]

        assert actual_rows == expected_rows, (
            f"Pixel values row count mismatch: expected {expected_rows}, got {actual_rows}. "
            f"image_grid_thw shape: {image_grid_thw.shape}, pixel_values shape: {pixel_values.shape}"
        )

        num_images = image_grid_thw.shape[0]
        assert image_grid_thw.shape == (num_images, 3), (
            f"Invalid image_grid_thw shape: expected [{num_images}, 3], got {image_grid_thw.shape}"
        )
