#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typed tensor shape aliases and decorators for runtime shape enforcement.

We use jaxtyping + beartype to enforce shapes for non-trivial tensors at
module boundaries (collators, dataset processing, model wrapper). This is
intended to be always-on for strict fail-fast behavior.

Common aliases:
- InputIds, AttentionMask, Labels: [batch, seq]
- PixelValuesPacked: [total_patches, patch_features]
- ImageGridTHW: [num_images, 3]
"""

from typing import Callable

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


# Public aliases for shapes used across the codebase
InputIds = Int[Tensor, "batch seq"]
AttentionMask = Int[Tensor, "batch seq"]
Labels = Int[Tensor, "batch seq"]

PixelValuesPacked = Float[Tensor, "patches features"]
ImageGridTHW = Int[Tensor, "images 3"]


def jaxtyped_beartype(func: Callable) -> Callable:
    """Decorator applying jaxtyping with beartype as the runtime checker."""
    return jaxtyped(typechecker=beartype)(func)  # type: ignore[misc]


__all__ = [
    "InputIds",
    "AttentionMask",
    "Labels",
    "PixelValuesPacked",
    "ImageGridTHW",
    "jaxtyped_beartype",
]
