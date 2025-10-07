"""Log-probability helpers for multimodal GRPO."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn


def clear_gpu_memory() -> None:
    """Comprehensive GPU memory cleanup with error handling.

    Safe to call multiple times. Clears cache and synchronizes to ensure
    cleanup completes before continuing.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        # Silently ignore cleanup errors to avoid disrupting training
        pass


def slice_packed_vision_for_sample(
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    images_cumsum: Optional[torch.Tensor],
    patch_offset: int,
    sample_idx: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Slice packed vision tensors for a single sample.

    ``pixel_values`` is expected to be packed along the first dimension where
    each image contributes ``t*h*w`` rows. ``image_grid_thw`` stores the ``(t,h,w)``
    triplets for every image in the batch. ``images_cumsum`` denotes the cumulative
    image counts per sample. The helper returns the per-sample ``pixel_values``
    slice, the matching ``image_grid_thw`` slice, and the updated ``patch_offset``.
    """

    if (
        pixel_values is None
        or image_grid_thw is None
        or images_cumsum is None
        or images_cumsum.numel() == 0
    ):
        return None, None, patch_offset

    start = int(images_cumsum[sample_idx - 1].item()) if sample_idx > 0 else 0
    end = int(images_cumsum[sample_idx].item())
    if end <= start:
        return None, None, patch_offset

    grid_slice = image_grid_thw[start:end]
    if grid_slice.numel() == 0:
        return None, grid_slice, patch_offset

    patch_counts = (grid_slice[:, 0] * grid_slice[:, 1] * grid_slice[:, 2]).long()
    num_patches = int(patch_counts.sum().item())
    if num_patches <= 0:
        return None, grid_slice, patch_offset

    next_offset = patch_offset + num_patches
    pv_slice = pixel_values[patch_offset:next_offset]
    return pv_slice, grid_slice, next_offset


def get_per_token_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    *,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    images_per_sample: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    detach: bool = False,
    extra_model_kwargs: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Compute per-token log probabilities for multimodal prompts.

    The function mirrors the sequential evaluation performed in the legacy
    TRL trainer (and now `BBUGRPOTrainer`) to limit memory usage. Each sample is processed
    independently so packed vision tensors must be sliced manually.
    """

    if input_ids.dim() != 2 or attention_mask.dim() != 2:
        raise ValueError("input_ids and attention_mask must be 2D tensors")
    if input_ids.size(0) != attention_mask.size(0):
        raise ValueError(
            "Batch dimension mismatch between input_ids and attention_mask"
        )

    total = int(input_ids.size(0))
    if total == 0:
        raise ValueError("Empty batch provided to get_per_token_logps")
    if logits_to_keep <= 0:
        raise ValueError("logits_to_keep must be positive")

    inputs_device = input_ids.device
    images_cumsum = (
        images_per_sample.cumsum(0) if images_per_sample is not None else None
    )

    all_logps = []
    patch_offset = 0
    for sample_idx in range(total):
        input_ids_batch = input_ids[sample_idx : sample_idx + 1]
        attention_mask_batch = attention_mask[sample_idx : sample_idx + 1]

        model_kwargs: dict[str, Any] = {}
        if extra_model_kwargs:
            model_kwargs.update(extra_model_kwargs)

        pv_slice, grid_slice, patch_offset = slice_packed_vision_for_sample(
            pixel_values,
            image_grid_thw,
            images_cumsum,
            patch_offset,
            sample_idx,
        )
        if pv_slice is not None:
            model_kwargs["pixel_values"] = pv_slice.to(inputs_device)
        if grid_slice is not None:
            model_kwargs["image_grid_thw"] = grid_slice.to(inputs_device)

        outputs = model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            logits_to_keep=logits_to_keep + 1,
            **model_kwargs,
        )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model forward pass did not return logits")

        logits = logits[:, :-1, :]
        tail_ids = input_ids_batch[:, -logits_to_keep:]
        scaled_logits = logits / float(temperature)
        logps = (
            torch.log_softmax(scaled_logits, dim=-1)
            .gather(-1, tail_ids.unsqueeze(-1))
            .squeeze(-1)
        )
        if detach:
            logps = logps.detach()
        all_logps.append(logps)

        clear_gpu_memory()

    return torch.cat(all_logps, dim=0)


__all__ = [
    "slice_packed_vision_for_sample",
    "get_per_token_logps",
    "clear_gpu_memory",
]
