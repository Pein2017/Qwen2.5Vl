#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assistant span extraction utilities.

Centralizes assistant span detection with offset mapping and optional EOS inclusion.
"""
from typing import List, Optional, Tuple

import torch

from src_new_json.processing.special_tokens import ASSISTANT_SPAN_RE, IM_END


def _char_to_token_position(char_pos: int, offset_mapping: torch.Tensor) -> Optional[int]:
    """Convert character position to token index using offset mapping."""
    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        if start_char <= char_pos < end_char:
            return token_idx
        elif char_pos == end_char and token_idx < len(offset_mapping) - 1:
            return token_idx + 1
    for token_idx, (start_char, _end_char) in enumerate(offset_mapping):
        if start_char >= char_pos:
            return token_idx
    return len(offset_mapping)


def find_assistant_spans(
    full_text: str,
    offset_mapping: torch.Tensor,
    tokenizer,
    include_eos: bool = True,
    has_teachers: bool = False,
    input_ids_1d: Optional[torch.Tensor] = None,
    num_teachers: int = 0,
) -> List[Tuple[int, int, bool]]:
    """
    Find assistant content spans as token index intervals.

    Args:
        full_text: Decoded conversation text with special tokens
        offset_mapping: Token-to-char offsets of full_text (shape [seq_len, 2])
        tokenizer: Tokenizer to resolve special token ids
        include_eos: If True, extend span end to include immediate <|im_end|>
        has_teachers: If True, mark the first N assistant spans as teachers
        input_ids_1d: Optional input ids aligned to full_text for EOS extension
        num_teachers: Number of teacher assistant turns (remaining are student)

    Returns:
        List of (start_token, end_token, is_teacher) with end exclusive
    """
    spans: List[Tuple[int, int, bool]] = []
    assistant_index = 0

    for match in ASSISTANT_SPAN_RE.finditer(full_text):
        content_start_char = match.start(1)
        content_end_char = match.end(1)

        start_token = _char_to_token_position(content_start_char, offset_mapping)
        end_token = _char_to_token_position(content_end_char, offset_mapping)

        # Optional EOS inclusion (<|im_end|>)
        if include_eos and input_ids_1d is not None:
            try:
                im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
            except Exception:
                im_end_id = None
            if isinstance(im_end_id, int) and im_end_id >= 0:
                for pos in range(end_token, min(len(input_ids_1d), end_token + 5)):
                    if int(input_ids_1d[pos].item()) == int(im_end_id):
                        end_token = pos + 1
                        break

        # Mark as teacher if within the first num_teachers spans
        is_teacher = bool(has_teachers and assistant_index < num_teachers)
        spans.append((start_token, end_token, is_teacher))
        assistant_index += 1

    return spans
