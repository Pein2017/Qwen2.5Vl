#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Tuple

from src_new.types.coords import CoordTokenRange


# Typed special token constants (single source of truth)
IM_START: str = "<|im_start|>"
IM_END: str = "<|im_end|>"
IMAGE_PAD: str = "<|image_pad|>"
END_OF_TEXT: str = "<|endoftext|>"
ASSISTANT_HEADER: str = f"{IM_START}assistant\n"
# Regex pattern for assistant content spans (non-greedy)
ASSISTANT_SPAN_PATTERN: str = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"


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
    """Collect all tokenizer IDs that correspond to <|coord_N|> tokens.

    Returns an empty list if none are present.
    """
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    coord_ids: List[int] = []
    for tok, idx in vocab.items():
        if not isinstance(tok, str):
            continue
        if tok.startswith("<|coord_") and tok.endswith("|>"):
            # Extract integer part; skip if it is not a valid integer
            inner = tok[len("<|coord_") : -2]
            try:
                int(inner)
            except Exception:
                continue
            try:
                coord_ids.append(int(idx))
            except Exception:
                continue
    return coord_ids


def get_coord_token_range(tokenizer) -> CoordTokenRange:
    """Derive coordinate token ID range from tokenizer.

    Returns:
            CoordTokenRange with start-inclusive and end-exclusive semantics. If no
            coordinate tokens are found, returns (0, 0).
    """
    ids = _collect_coordinate_token_ids(tokenizer)
    if not ids:
        return CoordTokenRange(0, 0)
    start_id = min(ids)
    end_exclusive = max(ids) + 1
    return CoordTokenRange(start_id, end_exclusive)


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


__all__ = [
    "CoordTokenRange",
    "GEOMETRY_TOKENS",
    "OBJECT_REF_SYNONYMS",
    "OBJECT_REF_SYNONYMS_END",
    "get_coord_token_range",
    "validate_geometry_tokens",
    "IM_START",
    "IM_END",
    "IMAGE_PAD",
    "END_OF_TEXT",
    "ASSISTANT_HEADER",
    "ASSISTANT_SPAN_PATTERN",
]
