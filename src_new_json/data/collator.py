"""
Data collator factory

This module provides a thin factory that creates collators from split modules:
- StandardDataCollator (src_new_json/data/collator_standard.py)
- TrainerCompatibleDataCollator (src_new_json/data/collator_utils.py)
"""
from __future__ import annotations

from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .collator_standard import StandardDataCollator
from .collator_utils import TrainerCompatibleDataCollator


def create_data_collator(
    collator_type: str,
    tokenizer: PreTrainedTokenizerBase,
    config: Any | None = None,
) -> TrainerCompatibleDataCollator:
    if collator_type == "standard":
        base_collator = StandardDataCollator(tokenizer=tokenizer, config=config)
    else:
        raise ValueError(f"Unsupported collator_type: {collator_type}")

    return TrainerCompatibleDataCollator(base_collator=base_collator)


__all__ = [
    "create_data_collator",
    "StandardDataCollator",
    "TrainerCompatibleDataCollator",
]
