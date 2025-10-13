#!/usr/bin/env python3
"""Prompt-only collator for TRL-style RL runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from src_new.data.collator_shared import (
    extract_image_grid_thw,
    extract_pixel_values,
    pad_sequence,
    validate_multimodal_batch,
)


@dataclass
class PromptOnlyCollator:
    """Collate prompt tensors without teacher/student labels.

    Expected feature keys:
        - input_ids: 1D tensor
        - attention_mask: 1D tensor
        - pixel_values: optional packed tensor produced by the processor
        - image_grid_thw: optional THW triplets aligned with pixel_values
        - conversation_text: optional str for logging
        - meta: optional dict carried through to rewards
    """

    tokenizer: PreTrainedTokenizerBase
    pad_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.pad_token_id is None:
            pad = getattr(self.tokenizer, "pad_token_id", None)
            if pad is None or pad < 0:
                raise ValueError("PromptOnlyCollator requires a valid pad_token_id")
            self.pad_token_id = int(pad)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("PromptOnlyCollator received an empty batch")

        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]

        pixel_values = extract_pixel_values(features)
        image_grid_thw = extract_image_grid_thw(features)

        batch: Dict[str, Any] = {
            "input_ids": pad_sequence(input_ids, int(self.pad_token_id)),
            "attention_mask": pad_sequence(attention_masks, 0),
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw

        validate_multimodal_batch(pixel_values, image_grid_thw)

        # Carry-through optional fields for logging/rewards
        if "conversation_text" in features[0]:
            batch["conversation_text"] = [f.get("conversation_text", "") for f in features]
        if "meta" in features[0]:
            batch["meta"] = [f.get("meta") for f in features]

        return batch


__all__ = ["PromptOnlyCollator"]
