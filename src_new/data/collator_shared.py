#!/usr/bin/env python3
"""
Shared padding and multimodal collation utilities.

These helpers centralize the packing logic that was duplicated between the
standard SFT collator and the new RL prompt-only collator so both paths enforce
the same invariants.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from src_new.types.shapes import (
    IMAGE_GRID_THW_SHAPE_DESC,
    PIXEL_VALUES_STANDARD_SHAPE_DESC,
)


def pad_sequence(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Right-pad variable length tensors to the longest length in the batch."""
    if not sequences:
        raise ValueError("pad_sequence requires at least one tensor")
    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_value, dtype=seq.dtype)
            padded.append(torch.cat([seq, padding], dim=0))
        else:
            padded.append(seq)
    return torch.stack(padded)


def extract_pixel_values(features: Iterable[Dict[str, Any]]) -> Optional[torch.Tensor]:
    """Flatten packed pixel values across the batch if present."""
    values: List[torch.Tensor] = []
    for feat in features:
        pv = feat.get("pixel_values")
        if pv is None:
            continue
        if pv.dim() == 2:
            values.append(pv)
        elif pv.dim() == 4:
            values.extend(pv.unbind(dim=0))
        elif pv.dim() == 3:
            values.append(pv)
        else:
            raise ValueError(
                f"Invalid pixel_values dims={pv.dim()} shape={pv.shape} "
                f"(expected {PIXEL_VALUES_STANDARD_SHAPE_DESC})"
            )
    if not values:
        return None
    if values[0].dim() != 2:
        return torch.stack(values)
    return torch.cat(values, dim=0)


def extract_image_grid_thw(
    features: Iterable[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    """Flatten image_grid_thw entries across the batch if present."""
    grids: List[torch.Tensor] = []
    for feat in features:
        grid = feat.get("image_grid_thw")
        if grid is None:
            continue
        if grid.dim() == 2:
            if grid.shape[0] == 1 and grid.shape[1] == 3:
                grids.append(grid.squeeze(0))
            elif grid.shape[1] == 3:
                grids.extend(grid.unbind(dim=0))
            else:
                raise ValueError(
                    f"Unsupported image_grid_thw shape {grid.shape} "
                    f"(expected {IMAGE_GRID_THW_SHAPE_DESC})"
                )
        elif grid.dim() == 1:
            if grid.shape[0] == 3:
                grids.append(grid)
            elif grid.shape[0] == 2:
                h, w = grid
                grids.append(
                    torch.tensor([1, h, w], dtype=grid.dtype, device=grid.device)
                )
            else:
                raise ValueError(
                    f"Invalid 1D image_grid_thw shape {grid.shape} "
                    f"(expected 3 elements for THW)"
                )
        else:
            raise ValueError(
                f"Invalid image_grid_thw dims={grid.dim()} shape={grid.shape} "
                f"(expected {IMAGE_GRID_THW_SHAPE_DESC})"
            )
    if not grids:
        return None
    return torch.stack(grids)


def validate_multimodal_batch(
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
) -> None:
    """Ensure the packed pixel tensors and THW triplets stay aligned."""
    if pixel_values is None:
        if image_grid_thw is not None:
            raise ValueError(
                "image_grid_thw provided without matching pixel_values tensor"
            )
        return
    if image_grid_thw is None:
        raise ValueError(
            "pixel_values present but image_grid_thw missing; cannot build multimodal batch."
        )
    if pixel_values.dim() != 2:
        raise ValueError(
            f"pixel_values must be {PIXEL_VALUES_STANDARD_SHAPE_DESC}, "
            f"got shape={tuple(pixel_values.shape)}"
        )
    if image_grid_thw.dim() != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(
            f"image_grid_thw must be {IMAGE_GRID_THW_SHAPE_DESC}, "
            f"got shape={tuple(image_grid_thw.shape)}"
        )
    expected = int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item())
    actual = int(pixel_values.shape[0])
    if expected != actual:
        raise ValueError(
            f"pixel_values rows ({actual}) != sum(t*h*w) ({expected}) from image_grid_thw"
        )


__all__ = [
    "pad_sequence",
    "extract_pixel_values",
    "extract_image_grid_thw",
    "validate_multimodal_batch",
]
