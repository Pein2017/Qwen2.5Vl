"""Validation helpers shared across the manual RL trainer."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Tuple

import torch

from src_new.processing.special_tokens import IMAGE_PAD
from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.validators")


def debug_validate_image_alignment(
    tokenizer: any,
    prompt_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
) -> None:
    """Decode prompts and compare ``<|image_pad|>`` occurrences against THW metadata."""

    if image_grid_thw is None or tokenizer is None:
        return
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    num_samples = int(prompt_ids.size(0))
    for idx in range(num_samples):
        ids_row = prompt_ids[idx]
        try:
            decoded = tokenizer.decode(ids_row, skip_special_tokens=False)
        except Exception as exc:  # pragma: no cover - diagnostic path
            _LOGGER.debug("Failed to decode prompt %d for alignment check: %s", idx, exc)
            continue

        image_token_count = decoded.count(IMAGE_PAD)
        if image_grid_thw.dim() == 3:
            # [B, num_images, 3]
            grid_row = image_grid_thw[idx]
        elif image_grid_thw.dim() == 2:
            grid_row = image_grid_thw
        else:
            grid_row = None

        num_images = int(grid_row.size(0)) if grid_row is not None else 0
        if image_token_count == 0 and num_images > 0:
            _LOGGER.error(
                "Image token mismatch: no %s tokens but %d image grids present (sample %d)",
                IMAGE_PAD,
                num_images,
                idx,
            )
        elif image_token_count > 0 and num_images == 0:
            _LOGGER.error(
                "Image token mismatch: found %d %s tokens but no image grids (sample %d)",
                image_token_count,
                IMAGE_PAD,
                idx,
            )
        elif num_images > 0:
            tokens_per_image = float(image_token_count) / float(max(num_images, 1))
            if tokens_per_image < 50.0:
                _LOGGER.warning(
                    "Very low image tokens per image: %.1f (sample %d) — check processor/template",
                    tokens_per_image,
                    idx,
                )
            elif tokens_per_image > 2000.0:
                _LOGGER.warning(
                    "Very high image tokens per image: %.1f (sample %d) — check processor/template",
                    tokens_per_image,
                    idx,
                )


def assert_patches_match_thw(
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    images_per_sample: Optional[torch.Tensor],
    chunk_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    """Ensure packed ``pixel_values`` align with THW metadata for every chunk."""

    if pixel_values is None or image_grid_thw is None or images_per_sample is None:
        return

    total_images = int(images_per_sample.sum().item())
    if total_images != int(image_grid_thw.size(0)):
        raise ValueError(
            f"images_per_sample sum {total_images} does not match image_grid_thw rows {image_grid_thw.size(0)}"
        )

    patches_per_image = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
    if int(patches_per_image.sum().item()) != int(pixel_values.size(0)):
        raise ValueError(
            "Packed pixel_values row count does not match THW-derived patch total"
        )

    if not chunk_ranges:
        return

    image_cumsum = images_per_sample.cumsum(0)
    patch_cumsum = torch.cumsum(patches_per_image, dim=0)

    for idx, (start, end) in enumerate(chunk_ranges):
        if start < 0 or end > len(images_per_sample) or start >= end:
            raise ValueError(f"Invalid chunk range {idx}: ({start}, {end})")

        img_start = image_cumsum[start - 1] if start > 0 else torch.tensor(0, device=image_cumsum.device)
        img_end = image_cumsum[end - 1]

        patch_start_idx = int(img_start.item()) - 1 if img_start.item() > 0 else -1
        patch_end_idx = int(img_end.item()) - 1

        patch_start = patch_cumsum[patch_start_idx] if patch_start_idx >= 0 else torch.tensor(0, device=patch_cumsum.device)
        patch_end = patch_cumsum[patch_end_idx]
        expected = int(patch_end.item()) - int(patch_start.item())
        if expected < 0:
            raise ValueError(
                f"Negative expected patch count for chunk {idx}: start={start}, end={end}"
            )


def check_completion_masks(mask: torch.Tensor) -> None:
    """Basic sanity checks for completion masks before loss computation."""

    if not torch.is_tensor(mask):
        raise ValueError("completion_mask must be a tensor")
    if mask.dim() != 2:
        raise ValueError("completion_mask must be 2D")
    if mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("completion_mask must be an integer or boolean tensor")
    if mask.numel() == 0:
        raise ValueError("Empty completion_mask provided")

    row_sums = mask.sum(dim=1)
    if torch.any(row_sums == 0):
        logging.getLogger(__name__).warning(
            "At least one completion has zero valid tokens after masking"
        )


__all__ = [
    "debug_validate_image_alignment",
    "assert_patches_match_thw",
    "check_completion_masks",
]
