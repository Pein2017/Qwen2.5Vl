#!/usr/bin/env python3
"""
RL dataset: JSONL reader yielding preprocessed tensors for GRPO.

- Emits dicts with input_ids, attention_mask, pixel_values, image_grid_thw
- Attaches meta (width, height, object_count if present)
- Batch size expected to be 1; no teacher pairing or augmentation here
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import torch

from src_new.data.dataset import read_jsonl
from src_new.rl.prompting.conversation import (
    RLConversationContext,
    build_simple_generation_inputs,
)


class RLDenseJSONLDataset:
    def __init__(self, jsonl_path: str, ctx: RLConversationContext) -> None:
        self.path = str(jsonl_path)
        self.ctx = ctx
        records = read_jsonl(self.path)
        if not records:
            raise ValueError(f"No valid JSON lines found in: {self.path}")
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            sample = self._records[int(idx)]
        except Exception as exc:
            if isinstance(exc, IndexError):
                raise IndexError(f"Index out of range: {idx}") from exc
            raise IndexError(f"Index out of range: {idx}")
        tensors = build_simple_generation_inputs(sample, self.ctx)

        input_ids = tensors.get("input_ids")
        attention_mask = tensors.get("attention_mask")
        pixel_values = tensors.get("pixel_values")
        image_grid_thw = tensors.get("image_grid_thw")

        if input_ids is None or attention_mask is None:
            raise ValueError(
                "Missing input tensors (input_ids/attention_mask) from conversation builder"
            )

        prompt_text = tensors.get("conversation_text")
        if not isinstance(prompt_text, str):
            raise ValueError(
                "Conversation builder did not return conversation_text string"
            )

        # Remove trivial batch dimension added by processor for consistency with Trainer expectations
        if (
            isinstance(input_ids, torch.Tensor)
            and input_ids.dim() == 2
            and input_ids.size(0) == 1
        ):
            input_ids = input_ids[0]
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.dim() == 2
            and attention_mask.size(0) == 1
        ):
            attention_mask = attention_mask[0]
        if (
            isinstance(pixel_values, torch.Tensor)
            and pixel_values.dim() >= 1
            and pixel_values.size(0) == 1
        ):
            pixel_values = pixel_values[0]
        if (
            isinstance(image_grid_thw, torch.Tensor)
            and image_grid_thw.dim() >= 1
            and image_grid_thw.size(0) == 1
        ):
            image_grid_thw = image_grid_thw[0]

        objects = sample.get("objects")
        meta = {
            "width": sample.get("width"),
            "height": sample.get("height"),
            "objects": objects if isinstance(objects, list) else [],
            "object_count": len(objects) if isinstance(objects, list) else 0,
        }

        return {
            "prompt": prompt_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "conversation_text": tensors.get("conversation_text"),
            "meta": meta,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]
