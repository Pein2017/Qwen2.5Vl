#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token grouping plugin for grouped LLM loss (caption, grounding, formatting).

JSON-first grouping rules:
- caption: tokens inside assistant spans that are NOT JSON punctuation/keys and NOT numeric-like
- grounding: tokens inside assistant spans that are numeric-like (contain digits and no letters)
- formatting: JSON punctuation ({}[],: ") and JSON key tokens (label/box_points/quadrilateral_points/line_points)

The plugin operates on token IDs, input_ids and spans. It builds per-token class masks,
intersects with teacher/student spans and aligns to the shifted CE convention.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from src_new_json.utils.rank_aware_logging import get_rank_aware_logger

from .grouping_core import assign_residual_to_formatting


_dbg_logger = get_rank_aware_logger(__name__)
_DEBUG_GROUPING: bool = os.getenv("BBU_DEBUG_GROUPING", "0").strip() not in ("", "0", "false", "False")
_DEBUG_DUMPED: bool = False


@dataclass(frozen=True)
class GroupMasks:
    """Shifted masks for CE:

    All masks are [batch, seq_len-1] booleans aligned to per-token CE tensors.
    """

    teacher_caption: torch.Tensor
    teacher_grounding: torch.Tensor
    teacher_formatting: torch.Tensor
    student_caption: torch.Tensor
    student_grounding: torch.Tensor
    student_formatting: torch.Tensor


class TokenGroupingPlugin:
    """Builds category masks from token IDs and assistant spans (JSON-first)."""

    def __init__(self, tokenizer) -> None:
        if tokenizer is None:
            raise ValueError("TokenGroupingPlugin requires a non-None tokenizer")
        self._tok = tokenizer
        # Precompile regexes
        self._re_has_digit = re.compile(r".*\d.*")
        self._re_has_alpha = re.compile(r".*[A-Za-z].*")
        # JSON punctuation characters
        self._json_punct_chars = set(['[', ']', '{', '}', ',', ':', '"'])
        # JSON key substrings to detect (lowercased)
        self._json_key_substrings = (
            "label",
            "box_points",
            "quadrilateral_points",
            "line_points",
        )

    def build_group_masks(
        self,
        labels: torch.Tensor,
        teacher_spans: Optional[List[List[Tuple[int, int]]]],
        student_spans: Optional[List[List[Tuple[int, int]]]],
        input_ids: Optional[torch.Tensor] = None,
        variant_key: Optional[str] = None,
    ) -> GroupMasks:
        if labels is None or labels.dim() != 2:
            raise ValueError(
                f"labels must be 2D [batch, seq_len], got {None if labels is None else tuple(labels.shape)}"
            )
        if input_ids is None:
            raise ValueError("input_ids are required for JSON-based token grouping")
        if input_ids.shape != labels.shape:
            raise ValueError(
                f"input_ids and labels must have same shape; got {tuple(input_ids.shape)} vs {tuple(labels.shape)}"
            )

        batch, seq_len = labels.shape

        # Build assistant masks from spans (unshifted)
        teacher_mask = torch.zeros_like(labels, dtype=torch.bool)
        student_mask = torch.zeros_like(labels, dtype=torch.bool)
        if teacher_spans:
            for i in range(min(batch, len(teacher_spans))):
                for start, end in teacher_spans[i]:
                    if not (0 <= start < end <= seq_len):
                        raise ValueError(
                            f"Invalid teacher span bounds: {(start, end)} for seq_len={seq_len}"
                        )
                    teacher_mask[i, start:end] = True
        if student_spans:
            for i in range(min(batch, len(student_spans))):
                for start, end in student_spans[i]:
                    if not (0 <= start < end <= seq_len):
                        raise ValueError(
                            f"Invalid student span bounds: {(start, end)} for seq_len={seq_len}"
                        )
                    student_mask[i, start:end] = True

        # Build per-token classification masks (unshifted)
        is_punct = torch.zeros_like(labels, dtype=torch.bool)
        is_key = torch.zeros_like(labels, dtype=torch.bool)
        is_digit_like = torch.zeros_like(labels, dtype=torch.bool)

        # Process each sample to account for tokenizer differences
        for b in range(batch):
            ids_row = input_ids[b].tolist()
            toks = self._tok.convert_ids_to_tokens(ids_row)
            for j, t in enumerate(toks):
                s = t if isinstance(t, str) else str(t)
                sl = s.lower()
                # JSON punctuation detection (if token contains any punctuation char)
                if any(ch in s for ch in self._json_punct_chars):
                    is_punct[b, j] = True
                # Key substring detection (subword-friendly)
                if any(sub in sl for sub in self._json_key_substrings):
                    is_key[b, j] = True
                # Numeric-like tokens: contain a digit and no alphabetic characters
                if self._re_has_digit.match(s) and not self._re_has_alpha.match(s):
                    is_digit_like[b, j] = True

        # Base group definitions (JSON-first)
        caption_all = (~is_punct) & (~is_key) & (~is_digit_like)
        grounding_all = is_digit_like
        formatting_all = is_punct | is_key

        # Variant-aware adjustments
        if isinstance(variant_key, str):
            vk = variant_key.strip().lower()
            if vk == "summary":
                # Summary: no grounding
                grounding_all = torch.zeros_like(labels, dtype=torch.bool)
            elif vk == "desc_to_coords":
                # Desc→Coords: focus on geometry; disable caption
                caption_all = torch.zeros_like(labels, dtype=torch.bool)
            elif vk == "coords_to_desc":
                # Coords→Desc: focus on description; disable grounding
                grounding_all = torch.zeros_like(labels, dtype=torch.bool)

        # Intersect with assistant masks and shift by one for CE alignment
        def _shift_intersect(cat_mask: torch.Tensor, who_mask: torch.Tensor) -> torch.Tensor:
            if cat_mask.shape != who_mask.shape:
                raise ValueError(
                    f"Mask shape mismatch: cat={tuple(cat_mask.shape)} vs who={tuple(who_mask.shape)}"
                )
            unshifted = cat_mask & who_mask
            return unshifted[:, 1:]

        t_caption = _shift_intersect(caption_all, teacher_mask)
        t_ground = _shift_intersect(grounding_all, teacher_mask)
        t_format = _shift_intersect(formatting_all, teacher_mask)
        s_caption = _shift_intersect(caption_all, student_mask)
        s_ground = _shift_intersect(grounding_all, student_mask)
        s_format = _shift_intersect(formatting_all, student_mask)

        # Optional one-time debug dump
        global _DEBUG_DUMPED
        if _DEBUG_GROUPING and not _DEBUG_DUMPED:
            try:
                t_assist = teacher_mask[:, 1:]
                s_assist = student_mask[:, 1:]
                def _cnt(x: torch.Tensor) -> int:
                    return int(x.sum().item()) if isinstance(x, torch.Tensor) else 0
                _dbg_logger.debug(
                    "[GroupingDebug] teacher: assist=%d cap=%d grd=%d fmt=%d | student: assist=%d cap=%d grd=%d fmt=%d",
                    _cnt(t_assist), _cnt(t_caption), _cnt(t_ground), _cnt(t_format),
                    _cnt(s_assist), _cnt(s_caption), _cnt(s_ground), _cnt(s_format),
                )
            finally:
                _DEBUG_DUMPED = True

        # Ensure full coverage equals assistant masks (shifted). Assign residual to formatting.
        t_assist = teacher_mask[:, 1:]
        s_assist = student_mask[:, 1:]

        t_caption, t_ground, t_format = assign_residual_to_formatting(
            t_caption, t_ground, t_format, t_assist
        )
        s_caption, s_ground, s_format = assign_residual_to_formatting(
            s_caption, s_ground, s_format, s_assist
        )

        return GroupMasks(
            teacher_caption=t_caption,
            teacher_grounding=t_ground,
            teacher_formatting=t_format,
            student_caption=s_caption,
            student_grounding=s_ground,
            student_formatting=s_format,
        )


__all__ = ["TokenGroupingPlugin", "GroupMasks"]
