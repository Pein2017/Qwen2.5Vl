#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Tuple, Optional, Iterable
import re

from src_new.types.coords import CoordTokenRange


# Typed special token constants (single source of truth)
IM_START: str = "<|im_start|>"
IM_END: str = "<|im_end|>"
IMAGE_PAD: str = "<|image_pad|>"
END_OF_TEXT: str = "<|endoftext|>"
ASSISTANT_HEADER: str = f"{IM_START}assistant\n"
# Regex pattern for assistant content spans (non-greedy)
ASSISTANT_SPAN_PATTERN: str = r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>"
ASSISTANT_SPAN_RE = re.compile(ASSISTANT_SPAN_PATTERN, re.DOTALL)


# Canonical geometry tokens used across the stack
GEOMETRY_TOKENS: Dict[str, Tuple[str, str, str, str]] = {
    "bbox_2d": (
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
    ),
    "quad": (
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
    ),
    "line": (
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|line_start|>",
        "<|line_end|>",
    ),
}

# Synonyms accepted at inference for object ref wrappers
OBJECT_REF_SYNONYMS: Tuple[str, str] = ("<|object_ref_start|>", "<|obj_ref_start|>")
OBJECT_REF_SYNONYMS_END: Tuple[str, str] = ("<|object_ref_end|>", "<|obj_ref_end|>")


def _collect_coordinate_token_ids(tokenizer) -> List[int]:
    """Deprecated: coordinate tokens not supported; return empty.
    """
    return []


def get_coord_token_range(tokenizer) -> CoordTokenRange:
    """Deprecated: coordinate tokens not supported; return empty range (0, 0)."""
    return CoordTokenRange(0, 0)


def validate_geometry_tokens(tokenizer) -> None:
    """Ensure required geometry tokens exist in the tokenizer vocabulary.

    Emits a warning listing any missing tokens but does not raise.
    This allows running models trained/inferred with a subset of shapes
    (e.g., bbox and quad without line).
    """
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    missing_by_type: Dict[str, List[str]] = {}
    for geom_type, (ref_s, ref_e, geo_s, geo_e) in GEOMETRY_TOKENS.items():
        missing: List[str] = []
        if ref_s not in vocab:
            missing.append(ref_s)
        if ref_e not in vocab:
            missing.append(ref_e)
        if geo_s not in vocab:
            missing.append(geo_s)
        if geo_e not in vocab:
            missing.append(geo_e)
        if missing:
            missing_by_type[geom_type] = missing

    if missing_by_type:
        # Flatten for concise message
        all_missing = sorted({tok for toks in missing_by_type.values() for tok in toks})
        logging.getLogger(__name__).warning(
            "Geometry tokens missing from vocabulary (non-fatal): %s. "
            "Continuing with available tokens; shapes with missing tokens will be disabled.",
            all_missing,
        )


# ---- NEW: strict validators that raise on missing tokens (for training fail-fast) ----

def _require_tokens(tokenizer, tokens: Iterable[str], what: str) -> None:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    missing = [t for t in tokens if t not in vocab]
    if missing:
        raise ValueError(
            f"Missing required {what} in tokenizer vocabulary: {missing}. "
            f"Load a checkpoint that already contains these tokens, or extend the tokenizer before training."
        )


def require_core_special_tokens(tokenizer) -> None:
    """Fail fast if core chat/image tokens are missing."""
    _require_tokens(
        tokenizer,
        tokens=(IM_START, IM_END, IMAGE_PAD, END_OF_TEXT),
        what="core special tokens (<|im_*|>, <|image_pad|>, <|endoftext|>)",
    )


def require_geometry_tokens(tokenizer, *, require_line: bool = True) -> None:
    """Fail fast if geometry/object-ref wrapper tokens are missing.

    Args:
        tokenizer: tokenizer instance
        require_line: whether to require <|line_start|>/<|line_end|>; set False if line is not used
    """
    required: List[str] = []
    for geom_type, (ref_s, ref_e, geo_s, geo_e) in GEOMETRY_TOKENS.items():
        if geom_type == "line" and not require_line:
            # still require object_ref wrappers
            required.extend([ref_s, ref_e])
            continue
        required.extend([ref_s, ref_e, geo_s, geo_e])
    _require_tokens(tokenizer, tokens=tuple(sorted(set(required))), what="geometry/object-ref wrapper tokens")


def require_coordinate_token_range(tokenizer, *, min_count: Optional[int] = None) -> None:
    """Fail fast if coordinate token range is absent or too small.

    Args:
        min_count: when provided, require at least this many coord tokens present (e.g., max_coord_value+1)
    """
    rng = get_coord_token_range(tokenizer)
    if rng.end_exclusive <= rng.start_id:
        raise ValueError("Coordinate tokens not found in tokenizer (<|coord_*|> range is empty)")
    if isinstance(min_count, int) and min_count > 0:
        count = int(rng.end_exclusive - rng.start_id)
        if count < min_count:
            raise ValueError(
                f"Coordinate token range too small: found {count}, required >= {min_count}. "
                f"Ensure the checkpoint includes <|coord_0|>.. tokens for your configured max_coord_value."
            )


__all__ = [
    "CoordTokenRange",
    "GEOMETRY_TOKENS",
    "OBJECT_REF_SYNONYMS",
    "OBJECT_REF_SYNONYMS_END",
    "get_coord_token_range",
    "validate_geometry_tokens",
    # core and strict validators
    "IM_START",
    "IM_END",
    "IMAGE_PAD",
    "END_OF_TEXT",
    "ASSISTANT_HEADER",
    "ASSISTANT_SPAN_PATTERN",
    "ASSISTANT_SPAN_RE",
    "require_core_special_tokens",
    "require_geometry_tokens",
    "require_coordinate_token_range",
]
