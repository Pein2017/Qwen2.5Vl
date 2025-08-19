#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized tensor and token ID shape/type contracts for Qwen2.5-VL.

This module defines typed constants and helper validation functions for
commonly referenced shapes and index semantics used across the codebase.

- Image grid THW tensor shape: [num_images, 3]
- Pixel values tensor shape (packed): [total_patches, patch_features]
- Pixel values tensor shape (standard, stacked): [num_images, ...] or [num_patches, patch_features]

Note: Coordinate token range type lives in `src_new.types.coords.CoordTokenRange`.
"""

# Shapes as symbolic constants (for logging/messages)
IMAGE_GRID_THW_SHAPE_DESC: str = "[num_images, 3]"
PIXEL_VALUES_PACKED_SHAPE_DESC: str = "[total_patches, patch_features]"
PIXEL_VALUES_STANDARD_SHAPE_DESC: str = "[num_patches, patch_features]"


__all__ = [
    "IMAGE_GRID_THW_SHAPE_DESC",
    "PIXEL_VALUES_PACKED_SHAPE_DESC",
    "PIXEL_VALUES_STANDARD_SHAPE_DESC",
]
