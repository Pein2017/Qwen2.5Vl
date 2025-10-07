#!/usr/bin/env python3
"""Small RL utilities shared across runner/eval.

Also provides shared helpers for token resolution to avoid duplication.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src_new.processing.conversation.builder import ConversationBuilder
from src_new.processing.special_tokens import IM_END


def resolve_im_end_id(tokenizer: Any) -> int | None:
    """Resolve the <|im_end|> token id with fallback to tokenizer.eos_token_id.

    Returns None if neither resolution succeeds.
    """
    if tokenizer is None:
        return None
    try:
        token_id = tokenizer.convert_tokens_to_ids(IM_END)
        if token_id is not None and int(token_id) >= 0:
            return int(token_id)
    except Exception:
        pass
    try:
        token_id = getattr(tokenizer, "eos_token_id", None)
        if token_id is not None and int(token_id) >= 0:
            return int(token_id)
    except Exception:
        pass
    return None


def create_builder(processor: Any) -> ConversationBuilder:
    """Factory for ConversationBuilder used in RL modules.

    Keeping this centralized avoids duplication across runner/eval.
    """
    return ConversationBuilder(processor=processor)


def get_model_device(model: nn.Module) -> torch.device:
    """Return the device of the model's parameters; CPU if no parameters.

    Kept here for reuse across RL helpers (generation/eval/etc.).
    """
    try:
        param = next(model.parameters())
        return param.device
    except StopIteration:
        return torch.device("cpu")
