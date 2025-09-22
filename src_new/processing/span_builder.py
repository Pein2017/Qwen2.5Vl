#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple, Optional

import torch

from .special_tokens import IMAGE_PAD, IM_END
from .span_extraction import find_assistant_spans
from .span_mapping import map_spans_unexpanded_to_expanded


def _build_unexpanded_to_expanded_index_map(
    ids_unexpanded: torch.Tensor,
    ids_expanded: torch.Tensor,
    image_pad_id: int,
) -> List[int]:
    """Build a position map from unexpanded token indices -> expanded token indices.

    Strategy: align in-order while skipping IMAGE_PAD tokens in the expanded stream.
    """
    if ids_unexpanded.dim() != 1:
        ids_unexpanded = ids_unexpanded.view(-1)
    if ids_expanded.dim() != 1:
        ids_expanded = ids_expanded.view(-1)

    mapping: List[int] = [0] * int(ids_unexpanded.shape[0])
    j = 0  # pointer over unexpanded
    for i in range(int(ids_expanded.shape[0])):
        token_id = int(ids_expanded[i].item())
        if token_id == int(image_pad_id):
            continue
        if j >= int(ids_unexpanded.shape[0]):
            break
        mapping[j] = i
        j += 1
    if j != int(ids_unexpanded.shape[0]):
        # In rare cases, template or processor may introduce extra specials; fallback to best-effort size clamp
        for k in range(j, int(ids_unexpanded.shape[0])):
            mapping[k] = int(ids_expanded.shape[0]) - 1
    return mapping


def build_assistant_spans_with_token_offsets(
    *,
    conversation_text: str,
    offset_mapping: torch.Tensor,
    tokenizer,
    input_ids_expanded: torch.Tensor,
    has_teachers: bool = False,
    num_teachers: int = 0,
    include_eos: bool = True,
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[List[Tuple[int, Tuple[int, int]]]],
    List[List[Tuple[int, Tuple[int, int]]]],
]:
    """Like build_assistant_spans_token_aligned, but also returns per-token offsets.

    Returns:
      - teacher_spans, student_spans: [(start_idx_exp, end_idx_exp))
      - teacher_token_offsets, student_token_offsets: per-span lists of (expanded_token_idx, (char_start, char_end))
    """
    # Tokenize unexpanded to get baseline ids and offsets.
    tokenized = tokenizer(
        conversation_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )
    offset_map = offset_mapping
    if not isinstance(offset_map, torch.Tensor):
        offset_map = tokenized["offset_mapping"][0]
    input_ids_unexpanded = tokenized["input_ids"][0]

    # Normalize expanded ids to 1D
    if isinstance(input_ids_expanded, torch.Tensor) and input_ids_expanded.dim() == 2 and int(input_ids_expanded.shape[0]) == 1:
        input_ids_expanded = input_ids_expanded[0]

    # Find assistant spans on unexpanded indices
    assistant_spans = find_assistant_spans(
        full_text=conversation_text,
        offset_mapping=offset_map,
        tokenizer=tokenizer,
        include_eos=include_eos,
        has_teachers=has_teachers,
        input_ids_1d=input_ids_unexpanded,
        num_teachers=num_teachers,
    )

    # Build index map unexpanded -> expanded (skip image_pad tokens)
    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    idx_map = _build_unexpanded_to_expanded_index_map(
        ids_unexpanded=input_ids_unexpanded,
        ids_expanded=input_ids_expanded,
        image_pad_id=image_pad_id,
    )

    # Prepare outputs
    teacher_spans: List[Tuple[int, int]] = []
    student_spans: List[Tuple[int, int]] = []
    teacher_token_offsets: List[List[Tuple[int, Tuple[int, int]]]] = []
    student_token_offsets: List[List[Tuple[int, Tuple[int, int]]]] = []

    # Detect IM_END id for span-end adjustment
    try:
        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    except Exception:
        im_end_id = None

    for st_unexp, ed_unexp, is_teacher in assistant_spans:
        # Trim IM_END from content-only window for accurate substring retokenization
        content_end_unexp = int(ed_unexp)
        if (
            include_eos
            and im_end_id is not None
            and content_end_unexp - 1 >= int(st_unexp)
        ):
            last_id = int(input_ids_unexpanded[content_end_unexp - 1].item())
            if last_id == int(im_end_id):
                content_end_unexp -= 1

        # Char window for assistant content
        base_char_start = int(offset_map[int(st_unexp)][0].item())
        base_char_end = int(offset_map[int(content_end_unexp - 1)][1].item()) if content_end_unexp > int(st_unexp) else base_char_start
        assistant_subtext = conversation_text[base_char_start:base_char_end]

        # Retokenize the assistant content only (clean offsets for specials)
        local = tokenizer(
            assistant_subtext,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors=None,
        )
        local_ids = local["input_ids"] if isinstance(local["input_ids"], list) else local["input_ids"][0]
        local_offsets = local["offset_mapping"] if isinstance(local["offset_mapping"], list) else local["offset_mapping"][0]

        # Map each local token to global expanded index and char offsets
        per_token: List[Tuple[int, Tuple[int, int]]] = []
        for j, _ in enumerate(local_ids):
            unexp_idx = int(st_unexp) + j
            if unexp_idx < 0 or unexp_idx >= len(idx_map):
                continue
            exp_idx = int(idx_map[unexp_idx])
            ls, le = local_offsets[j]
            # Normalize to Python ints and shift by base_char_start
            cs = int(ls)
            ce = int(le)
            per_token.append((exp_idx, (base_char_start + cs, base_char_start + ce)))

        # Map span to expanded indices (include optional IM_END per original behavior)
        span_st_exp = int(idx_map[int(st_unexp)])
        span_ed_exp = int(idx_map[int(ed_unexp) - 1]) + 1 if int(ed_unexp) > int(st_unexp) else span_st_exp
        if include_eos and im_end_id is not None:
            # Extend to include the next IM_END if present shortly after
            for pos in range(span_ed_exp, min(int(input_ids_expanded.shape[0]), span_ed_exp + 5)):
                if int(input_ids_expanded[pos].item()) == int(im_end_id):
                    span_ed_exp = pos + 1
                    break

        if is_teacher:
            teacher_spans.append((span_st_exp, span_ed_exp))
            teacher_token_offsets.append(per_token)
        else:
            student_spans.append((span_st_exp, span_ed_exp))
            student_token_offsets.append(per_token)

    return teacher_spans, student_spans, teacher_token_offsets, student_token_offsets


def build_assistant_spans_token_aligned(
    *,
    conversation_text: str,
    offset_mapping: torch.Tensor,
    tokenizer,
    input_ids_expanded: torch.Tensor,
    has_teachers: bool = False,
    num_teachers: int = 0,
    include_eos: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Build teacher and student assistant spans in token indices aligned to expanded input_ids.

    Returns (teacher_spans, student_spans), each a list of (start, end) with end exclusive.
    """
    # Compute unexpanded tokenization to obtain ids/offsets baseline
    tokenized = tokenizer(
        conversation_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )
    offset_map = offset_mapping
    if not isinstance(offset_map, torch.Tensor):
        offset_map = tokenized["offset_mapping"][0]
    input_ids_unexpanded = tokenized["input_ids"][0]

    # Normalize expanded ids to 1D when provided as [1, S]
    if isinstance(input_ids_expanded, torch.Tensor) and input_ids_expanded.dim() == 2 and int(input_ids_expanded.shape[0]) == 1:
        input_ids_expanded = input_ids_expanded[0]

    # Find spans at char-level, map to unexpanded token indices
    assistant_spans = find_assistant_spans(
        full_text=conversation_text,
        offset_mapping=offset_map,
        tokenizer=tokenizer,
        include_eos=include_eos,
        has_teachers=has_teachers,
        input_ids_1d=input_ids_unexpanded,
        num_teachers=num_teachers,
    )

    # Map to expanded token indices when needed (image_pad runs)
    if int(input_ids_unexpanded.shape[0]) != int(input_ids_expanded.shape[0]):
        image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        assistant_spans = map_spans_unexpanded_to_expanded(
            ids_unexpanded=input_ids_unexpanded,
            ids_expanded=input_ids_expanded,
            image_pad_id=image_pad_id,
            spans_unexpanded=assistant_spans,
        )

    # Split into teacher/student and include immediate IM_END when present
    teacher_spans: List[Tuple[int, int]] = []
    student_spans: List[Tuple[int, int]] = []

    im_end_id: Optional[int]
    try:
        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    except Exception:
        im_end_id = None

    for st, ed, is_teacher in assistant_spans:
        final_end = int(ed)
        if (
            include_eos
            and im_end_id is not None
            and im_end_id >= 0
            and final_end <= int(input_ids_expanded.shape[0])
        ):
            for pos in range(final_end, min(int(input_ids_expanded.shape[0]), final_end + 5)):
                if int(input_ids_expanded[pos].item()) == int(im_end_id):
                    final_end = pos + 1
                    break
        span = (int(st), int(final_end))
        if is_teacher:
            teacher_spans.append(span)
        else:
            student_spans.append(span)

    return teacher_spans, student_spans

__all__ = [
    "build_assistant_spans_token_aligned",
    "build_assistant_spans_with_token_offsets",
]
