#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple, Optional

import torch

from .special_tokens import IMAGE_PAD, IM_END
from .span_extraction import find_assistant_spans
from .span_mapping import map_spans_unexpanded_to_expanded


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
    )
    offset_map = offset_mapping
    if not isinstance(offset_map, torch.Tensor):
        offset_map = tokenized["offset_mapping"][0]
    input_ids_unexpanded = tokenized["input_ids"][0]

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
