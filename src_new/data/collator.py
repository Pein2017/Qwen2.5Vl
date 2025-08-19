"""
Data collation factory for Qwen2.5-VL training.

This module provides a thin factory that creates collators from split modules:
- StandardDataCollator (src_new/data/collator_standard.py)
- PackedDataCollator (src_new/data/collator_packed.py)
- TrainerCompatibleDataCollator (src_new/data/collator_utils.py)
"""

from typing import Optional

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .collator_packed import PackedDataCollator
from .collator_standard import StandardDataCollator
from .collator_utils import TrainerCompatibleDataCollator


def create_data_collator(
    collator_type: str,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[object] = None,
) -> TrainerCompatibleDataCollator:
    if collator_type == "standard":
        base_collator = StandardDataCollator(tokenizer=tokenizer, config=config)
    elif collator_type == "packed":
        base_collator = PackedDataCollator(tokenizer=tokenizer, config=config)
    else:
        raise ValueError(f"Unsupported collator_type: {collator_type}")
    return TrainerCompatibleDataCollator(base_collator=base_collator)


__all__ = [
    "create_data_collator",
    "StandardDataCollator",
    "PackedDataCollator",
    "TrainerCompatibleDataCollator",
]
