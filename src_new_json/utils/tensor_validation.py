#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor validation utilities for Qwen2.5-VL project.

This module provides consistent tensor shape and dimension validation patterns
used throughout the codebase, eliminating duplicate tensor validation logic
and ensuring consistent error handling for tensor operations.

Key Features:
- Centralized tensor shape validation
- Dimension consistency checks
- Batch size validation
- Multimodal tensor validation (vision + text)
- Clear error messages with tensor context
- Support for both PyTorch and NumPy tensors

Usage:
    from src_new_json.utils.tensor_validation import TensorValidator, TensorValidationError

    # Validate tensor shape
    TensorValidator.validate_shape(
        tensor=pixel_values,
        expected_shape=(batch_size, num_patches, patch_dim),
        tensor_name="pixel_values"
    )

    # Validate multimodal consistency
    TensorValidator.validate_multimodal_consistency(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        tensor_names=("pixel_values", "image_grid_thw")
    )
"""

from typing import Any, Dict, Optional, Tuple

from .common_imports import NUMPY_AVAILABLE, TORCH_AVAILABLE, np, torch
from .error_formatting import ErrorMessageBuilder
from .rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger("tensor_validation")


class TensorValidationError(Exception):
    """Exception raised for tensor validation errors."""

    def __init__(self, message: str, tensor_info: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.tensor_info = tensor_info or {}


class TensorValidator:
    """
    Centralized tensor validation utilities.

    Provides consistent tensor validation patterns used throughout the codebase,
    eliminating duplicate validation logic and ensuring fail-fast behavior for
    tensor operations.
    """

    @staticmethod
    def validate_tensor_type(
        tensor: Any, tensor_name: str = "tensor", allow_none: bool = False
    ) -> bool:
        """
        Validate that an object is a valid tensor type.

        Args:
            tensor: Object to validate
            tensor_name: Name for error messages
            allow_none: Whether to allow None values

        Returns:
            True if valid tensor type

        Raises:
            TensorValidationError: If not a valid tensor type
        """
        if tensor is None:
            if allow_none:
                return True
            else:
                raise TensorValidationError(
                    f"{tensor_name} cannot be None",
                    tensor_info={"tensor_name": tensor_name, "value": None},
                )

        valid_types = []
        if TORCH_AVAILABLE:
            valid_types.append(torch.Tensor)
        if NUMPY_AVAILABLE:
            valid_types.append(np.ndarray)

        if not valid_types:
            raise TensorValidationError(
                "No tensor libraries available (PyTorch or NumPy required)",
                tensor_info={"tensor_name": tensor_name},
            )

        if not isinstance(tensor, tuple(valid_types)):
            type_names = [t.__name__ for t in valid_types]
            raise TensorValidationError(
                f"{tensor_name} must be one of {type_names}, got {type(tensor)}",
                tensor_info={
                    "tensor_name": tensor_name,
                    "expected_types": type_names,
                    "actual_type": type(tensor).__name__,
                },
            )

        return True

    @staticmethod
    def validate_shape(
        tensor: Any,
        expected_shape: Tuple[int, ...],
        tensor_name: str = "tensor",
        allow_batch_dim: bool = False,
    ) -> bool:
        """
        Validate tensor shape matches expected shape.

        Args:
            tensor: Tensor to validate
            expected_shape: Expected shape tuple
            tensor_name: Name for error messages
            allow_batch_dim: Whether to allow flexible batch dimension

        Returns:
            True if shape is valid

        Raises:
            TensorValidationError: If shape doesn't match
        """
        TensorValidator.validate_tensor_type(tensor, tensor_name)

        actual_shape = tuple(tensor.shape)

        if allow_batch_dim and len(actual_shape) == len(expected_shape):
            # Compare all dimensions except the first (batch dimension)
            if actual_shape[1:] != expected_shape[1:]:
                error_msg = ErrorMessageBuilder.build_tensor_shape_error(
                    tensor_name=tensor_name,
                    expected_shape=expected_shape,
                    actual_shape=actual_shape,
                    context="batch dimension allowed to vary",
                )
                raise TensorValidationError(
                    error_msg,
                    tensor_info={
                        "tensor_name": tensor_name,
                        "expected_shape": expected_shape,
                        "actual_shape": actual_shape,
                        "allow_batch_dim": allow_batch_dim,
                    },
                )
        elif actual_shape != expected_shape:
            error_msg = ErrorMessageBuilder.build_tensor_shape_error(
                tensor_name=tensor_name,
                expected_shape=expected_shape,
                actual_shape=actual_shape,
            )
            raise TensorValidationError(
                error_msg,
                tensor_info={
                    "tensor_name": tensor_name,
                    "expected_shape": expected_shape,
                    "actual_shape": actual_shape,
                    "allow_batch_dim": allow_batch_dim,
                },
            )

        logger.debug(f"✅ Validated {tensor_name} shape: {actual_shape}")
        return True

    @staticmethod
    def validate_dimension_count(
        tensor: Any, expected_dims: int, tensor_name: str = "tensor"
    ) -> bool:
        """
        Validate tensor has expected number of dimensions.

        Args:
            tensor: Tensor to validate
            expected_dims: Expected number of dimensions
            tensor_name: Name for error messages

        Returns:
            True if dimension count is valid

        Raises:
            TensorValidationError: If dimension count doesn't match
        """
        TensorValidator.validate_tensor_type(tensor, tensor_name)

        actual_dims = len(tensor.shape)

        if actual_dims != expected_dims:
            raise TensorValidationError(
                f"{tensor_name} expected {expected_dims} dimensions, got {actual_dims} "
                f"(shape: {tuple(tensor.shape)})",
                tensor_info={
                    "tensor_name": tensor_name,
                    "expected_dims": expected_dims,
                    "actual_dims": actual_dims,
                    "shape": tuple(tensor.shape),
                },
            )

        logger.debug(f"✅ Validated {tensor_name} dimensions: {actual_dims}")
        return True

    @staticmethod
    def validate_batch_consistency(tensors: Dict[str, Any], batch_dim: int = 0) -> bool:
        """
        Validate that multiple tensors have consistent batch sizes.

        Args:
            tensors: Dictionary of tensor_name -> tensor
            batch_dim: Which dimension represents the batch size

        Returns:
            True if batch sizes are consistent

        Raises:
            TensorValidationError: If batch sizes are inconsistent
        """
        if not tensors:
            return True

        batch_sizes = {}
        for name, tensor in tensors.items():
            TensorValidator.validate_tensor_type(tensor, name)

            if len(tensor.shape) <= batch_dim:
                raise TensorValidationError(
                    f"{name} has insufficient dimensions for batch_dim={batch_dim} "
                    f"(shape: {tuple(tensor.shape)})",
                    tensor_info={
                        "tensor_name": name,
                        "shape": tuple(tensor.shape),
                        "batch_dim": batch_dim,
                    },
                )

            batch_sizes[name] = tensor.shape[batch_dim]

        # Check all batch sizes are the same
        unique_batch_sizes = set(batch_sizes.values())
        if len(unique_batch_sizes) > 1:
            raise TensorValidationError(
                f"Inconsistent batch sizes: {batch_sizes}",
                tensor_info={"batch_sizes": batch_sizes, "batch_dim": batch_dim},
            )

        batch_size = list(unique_batch_sizes)[0]
        logger.debug(
            f"✅ Validated batch consistency: {batch_size} across {list(tensors.keys())}"
        )
        return True

    @staticmethod
    def validate_multimodal_consistency(
        pixel_values: Any,
        image_grid_thw: Any,
        tensor_names: Tuple[str, str] = ("pixel_values", "image_grid_thw"),
    ) -> bool:
        """
        Validate consistency between pixel values and image grid tensors.

        Args:
            pixel_values: Flattened pixel values tensor [num_patches, patch_features]
            image_grid_thw: Image grid tensor [num_images, 3] (time, height, width)
            tensor_names: Names for error messages

        Returns:
            True if tensors are consistent

        Raises:
            TensorValidationError: If tensors are inconsistent
        """
        pixel_name, grid_name = tensor_names

        # Validate tensor types
        TensorValidator.validate_tensor_type(pixel_values, pixel_name)
        TensorValidator.validate_tensor_type(image_grid_thw, grid_name)

        # Validate pixel_values is 2D flattened patches
        TensorValidator.validate_dimension_count(pixel_values, 2, pixel_name)

        # Validate image_grid_thw is 2D with 3 columns (t, h, w)
        TensorValidator.validate_dimension_count(image_grid_thw, 2, grid_name)

        if image_grid_thw.shape[1] != 3:
            raise TensorValidationError(
                f"{grid_name} must have 3 columns (t, h, w), got {image_grid_thw.shape[1]}",
                tensor_info={
                    "tensor_name": grid_name,
                    "shape": tuple(image_grid_thw.shape),
                    "expected_columns": 3,
                },
            )

        # Calculate expected patches from grid
        if TORCH_AVAILABLE and isinstance(image_grid_thw, torch.Tensor):
            grid = image_grid_thw.to(dtype=torch.long)
            expected_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
        else:
            grid = image_grid_thw.astype(int)
            expected_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())

        actual_patches = pixel_values.shape[0]

        if actual_patches != expected_patches:
            error_msg = ErrorMessageBuilder.build_validation_error(
                item="multimodal tensor consistency",
                expected=f"{expected_patches} patches from grid",
                actual=f"{actual_patches} patches in pixel_values",
                context="pixel_values rows must equal sum(t*h*w) from image_grid_thw",
                suggestions=[
                    "Check image preprocessing pipeline",
                    "Verify image grid calculation",
                    "Ensure consistent patch extraction",
                ],
            )
            raise TensorValidationError(
                error_msg,
                tensor_info={
                    "pixel_values_shape": tuple(pixel_values.shape),
                    "image_grid_thw_shape": tuple(image_grid_thw.shape),
                    "expected_patches": expected_patches,
                    "actual_patches": actual_patches,
                    "grid_values": image_grid_thw.tolist()
                    if hasattr(image_grid_thw, "tolist")
                    else str(image_grid_thw),
                },
            )

        logger.debug(f"✅ Validated multimodal consistency: {actual_patches} patches")
        return True

    @staticmethod
    def validate_finite_values(
        tensor: Any,
        tensor_name: str = "tensor",
        check_nan: bool = True,
        check_inf: bool = True,
    ) -> bool:
        """
        Validate tensor contains only finite values (no NaN or Inf).

        Args:
            tensor: Tensor to validate
            tensor_name: Name for error messages
            check_nan: Whether to check for NaN values
            check_inf: Whether to check for infinite values

        Returns:
            True if all values are finite

        Raises:
            TensorValidationError: If non-finite values found
        """
        TensorValidator.validate_tensor_type(tensor, tensor_name)

        issues = []

        if check_nan:
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                has_nan = torch.isnan(tensor).any().item()
            elif NUMPY_AVAILABLE and isinstance(tensor, np.ndarray):
                has_nan = np.isnan(tensor).any()
            else:
                has_nan = False

            if has_nan:
                issues.append("NaN values")

        if check_inf:
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                has_inf = torch.isinf(tensor).any().item()
            elif NUMPY_AVAILABLE and isinstance(tensor, np.ndarray):
                has_inf = np.isinf(tensor).any()
            else:
                has_inf = False

            if has_inf:
                issues.append("infinite values")

        if issues:
            raise TensorValidationError(
                f"{tensor_name} contains {' and '.join(issues)}",
                tensor_info={
                    "tensor_name": tensor_name,
                    "shape": tuple(tensor.shape),
                    "issues": issues,
                },
            )

        logger.debug(f"✅ Validated {tensor_name} finite values")
        return True


# Convenience functions for common validation patterns
def validate_tensor_shape(
    tensor: Any, expected_shape: Tuple[int, ...], name: str = "tensor"
) -> bool:
    """Convenience function for tensor shape validation."""
    return TensorValidator.validate_shape(tensor, expected_shape, name)


def validate_batch_size(tensors: Dict[str, Any]) -> bool:
    """Convenience function for batch size validation."""
    return TensorValidator.validate_batch_consistency(tensors)


def validate_multimodal_tensors(pixel_values: Any, image_grid_thw: Any) -> bool:
    """Convenience function for multimodal tensor validation."""
    return TensorValidator.validate_multimodal_consistency(pixel_values, image_grid_thw)


# Export public API
__all__ = [
    "TensorValidationError",
    "TensorValidator",
    "validate_tensor_shape",
    "validate_batch_size",
    "validate_multimodal_tensors",
]
