#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import List
import re

from src_new_json.types.coords import CoordTokenRange


# Typed special token constants (single source of truth)
IM_START: str = "<|im_start|>"
IM_END: str = "<|im_end|>"
IMAGE_PAD: str = "<|image_pad|>"
END_OF_TEXT: str = "<|endoftext|>"
ASSISTANT_HEADER: str = f"{IM_START}assistant\n"
# Regex pattern for assistant content spans (non-greedy)
ASSISTANT_SPAN_PATTERN: str = r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>"
ASSISTANT_SPAN_RE = re.compile(ASSISTANT_SPAN_PATTERN, re.DOTALL)


# NOTE: Wrapper/geometry token system removed in JSON-first pipeline.
# The following coord helper is retained as a compatibility no-op.

def _collect_coordinate_token_ids(tokenizer) -> List[int]:
    """Compatibility stub: returns empty list (no coordinate tokens)."""
    return []


def get_coord_token_range(tokenizer) -> CoordTokenRange:
    """Return (0, 0) to indicate no coordinate token range in JSON mode."""
    return CoordTokenRange(0, 0)


def validate_geometry_tokens(tokenizer) -> None:
    """No-op in JSON mode (legacy wrapper-token validation removed)."""
    return None


__all__ = [
    "CoordTokenRange",
    "get_coord_token_range",
    "validate_geometry_tokens",
    "IM_START",
    "IM_END",
    "IMAGE_PAD",
    "END_OF_TEXT",
    "ASSISTANT_HEADER",
    "ASSISTANT_SPAN_PATTERN",
    "ASSISTANT_SPAN_RE",
]
