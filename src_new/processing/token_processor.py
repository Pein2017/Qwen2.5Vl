#!/usr/bin/env python3
"""Minimal tokenizer extension stub.

Coordinate-token specific functionality was removed from the src_new pipeline.
This module now provides a tiny compatibility layer for legacy call sites that
previously interacted with ``TokenProcessor``. All methods are implemented as
no-ops so the rest of the training and inference stack can rely on standard
HuggingFace behaviour without additional configuration flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class TokenConfig:
    """Placeholder config kept for backwards compatibility."""

    # The config no longer carries coordinate-specific toggles; it only exists so
    # callers that instantiated ``TokenConfig`` continue to function.
    pass


class TokenProcessor:
    """No-op token processor kept for interface stability.

    All methods deliberately avoid mutating the tokenizer or model. They return
    the inputs unchanged while providing the helpers that existing call sites
    expect. This allows the rest of the pipeline to operate purely with numeric
    coordinates rendered via regular geometry tokens.
    """

    def __init__(self, config: Optional[TokenConfig] = None) -> None:
        self.config = config or TokenConfig()

    # ------------------------------------------------------------------
    # Legacy public helpers (now no-ops)
    # ------------------------------------------------------------------
    def extend_tokenizer_vocabulary(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> PreTrainedTokenizerBase:
        """Return the tokenizer unchanged.

        The tokenizer already contains every required geometry wrapper token.
        """
        return tokenizer

    def extend_model_embeddings(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Qwen2_5_VLForConditionalGeneration:
        """Return the model unchanged.

        Embedding matrices no longer need to be expanded for coordinate tokens.
        """
        return model

    def create_coordinate_mask(
        self, input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase
    ) -> torch.Tensor:
        """Return an all-zero mask with the same shape as ``input_ids``."""
        return torch.zeros_like(input_ids, dtype=torch.bool)

    def get_coordinate_token_range(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[int, int]:
        """Coordinate tokens are no longer present; report an empty range."""
        return (0, 0)

    def coordinates_to_tokens(self, coordinates: Iterable[int]) -> List[str]:
        """Render coordinates as plain integers (no special tokens)."""
        return [str(int(value)) for value in coordinates]

    def tokens_to_coordinates(self, tokens: Iterable[str]) -> List[int]:
        """Parse integer strings back into coordinates."""
        parsed: List[int] = []
        for token in tokens:
            parsed.append(int(token))
        return parsed


__all__ = ["TokenConfig", "TokenProcessor"]
