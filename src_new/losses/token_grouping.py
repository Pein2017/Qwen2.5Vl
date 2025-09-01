#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token grouping plugin for grouped LLM loss (caption, grounding, formatting).

This module constructs boolean masks for category-specific cross-entropy
computation aligned to the existing single-pass CE path in LossManager.

- caption: natural-language description tokens strictly inside
  <|object_ref_start|> ... <|object_ref_end|>, excluding formatting punctuation.
- grounding: geometry grounding tokens, i.e., coordinate value tokens
  (<|coord_N|>) and geometry wrapper tokens (<|box_*|>, <|quad_*|>, <|line_*|>).
  Object-ref wrappers are not grounding; they are formatting tokens.
- formatting: structural glue: punctuation/brackets, object-ref wrappers,
  and explicit terminators like <|im_end|> if present within assistant spans.

The plugin operates purely on label IDs and spans, without text decode, and
returns masks already intersected with teacher/student assistant spans and
aligned to the shifted CE convention (drop the first position).

This module exposes no external configuration. Category definitions and
punctuation sets are internal and conservative by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src_new.processing.special_tokens import (
    IM_END,
    get_coord_token_range,
)


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
    """Builds category masks from label IDs and assistant spans.

    Strictly fails on shape/type mismatches; missing special tokens are handled
    gracefully by producing empty masks for that token class.
    """

    def __init__(self, tokenizer) -> None:
        if tokenizer is None:
            raise ValueError("TokenGroupingPlugin requires a non-None tokenizer")
        self._tok = tokenizer

        # Pre-compute special token ids (missing -> None)
        self._ids: Dict[str, Optional[int]] = {}
        self._ids["im_end"] = self._safe_id(IM_END)
        # Object-ref wrappers are treated as formatting
        self._ids["obj_ref_start"] = self._safe_id("<|object_ref_start|>")
        self._ids["obj_ref_end"] = self._safe_id("<|object_ref_end|>")
        # Geometry wrappers are treated as grounding
        self._ids["box_start"] = self._safe_id("<|box_start|>")
        self._ids["box_end"] = self._safe_id("<|box_end|>")
        self._ids["quad_start"] = self._safe_id("<|quad_start|>")
        self._ids["quad_end"] = self._safe_id("<|quad_end|>")
        self._ids["line_start"] = self._safe_id("<|line_start|>")
        self._ids["line_end"] = self._safe_id("<|line_end|>")

        # Coordinate token range
        coord_rng = get_coord_token_range(self._tok)
        self._coord_start: int = int(coord_rng.start_id)
        self._coord_end_exclusive: int = int(coord_rng.end_exclusive)

        # General punctuation set (formatting)
        punctuation_chars: Sequence[str] = (
            "[",
            "]",
            "{",
            "}",
            "(",
            ")",
            ",",
            ":",
            '"',
            "/",
        )
        self._punctuation_ids: Dict[int, bool] = {}
        for ch in punctuation_chars:
            ids = self._tok.encode(ch, add_special_tokens=False)
            if not isinstance(ids, list) or len(ids) == 0:
                raise ValueError(
                    f"Tokenizer failed to encode punctuation character: {ch!r}"
                )
            for tid in ids:
                if not isinstance(tid, int):
                    raise ValueError(
                        f"Tokenizer returned non-int id for punctuation {ch!r}: {tid!r}"
                    )
                self._punctuation_ids[int(tid)] = True

        # Geometry separators used inside geometry blocks: only '[', ']', ','
        geom_sep_chars: Sequence[str] = ("[", "]", ",")
        self._geom_sep_ids: Dict[int, bool] = {}
        for ch in geom_sep_chars:
            ids = self._tok.encode(ch, add_special_tokens=False)
            if not isinstance(ids, list) or len(ids) == 0:
                raise ValueError(
                    f"Tokenizer failed to encode geometry separator: {ch!r}"
                )
            for tid in ids:
                if not isinstance(tid, int):
                    raise ValueError(
                        f"Tokenizer returned non-int id for geometry separator {ch!r}: {tid!r}"
                    )
                self._geom_sep_ids[int(tid)] = True

        # Geometry wrapper id set
        self._geom_wrapper_ids: Dict[int, bool] = {}
        for key in (
            "box_start",
            "box_end",
            "quad_start",
            "quad_end",
            "line_start",
            "line_end",
        ):
            tid = self._ids.get(key)
            if isinstance(tid, int) and tid >= 0:
                self._geom_wrapper_ids[int(tid)] = True

        # Object-ref wrapper id set (formatting)
        self._objref_wrapper_ids: Dict[int, bool] = {}
        for key in ("obj_ref_start", "obj_ref_end"):
            tid = self._ids.get(key)
            if isinstance(tid, int) and tid >= 0:
                self._objref_wrapper_ids[int(tid)] = True

    def _safe_id(self, token: str) -> Optional[int]:
        tid = self._tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            return int(tid)
        return None

    def build_group_masks(
        self,
        labels: torch.Tensor,
        teacher_spans: Optional[List[List[Tuple[int, int]]]],
        student_spans: Optional[List[List[Tuple[int, int]]]],
    ) -> GroupMasks:
        """Construct shifted masks for caption/grounding/formatting per group.

        Args:
            labels: [batch, seq_len] int64 token ids
            teacher_spans: list of lists of (start, end) token indices per batch item
            student_spans: list of lists of (start, end) token indices per batch item

        Returns:
            GroupMasks with [batch, seq_len-1] boolean tensors.

        Raises:
            ValueError: on invalid shapes or inconsistent span bounds.
        """
        if labels is None or labels.dim() != 2:
            raise ValueError(
                f"labels must be 2D [batch, seq_len], got {None if labels is None else tuple(labels.shape)}"
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

        # Base element-wise predicates from labels
        is_coord = (labels >= self._coord_start) & (labels < self._coord_end_exclusive)
        is_geom_wrapper = torch.zeros_like(labels, dtype=torch.bool)
        is_objref_wrapper = torch.zeros_like(labels, dtype=torch.bool)
        is_punct = torch.zeros_like(labels, dtype=torch.bool)
        is_geom_sep = torch.zeros_like(labels, dtype=torch.bool)

        if len(self._geom_wrapper_ids) > 0:
            for tid in self._geom_wrapper_ids.keys():
                is_geom_wrapper |= labels.eq(int(tid))
        if len(self._objref_wrapper_ids) > 0:
            for tid in self._objref_wrapper_ids.keys():
                is_objref_wrapper |= labels.eq(int(tid))
        if len(self._punctuation_ids) > 0:
            for tid in self._punctuation_ids.keys():
                is_punct |= labels.eq(int(tid))
        if len(self._geom_sep_ids) > 0:
            for tid in self._geom_sep_ids.keys():
                is_geom_sep |= labels.eq(int(tid))

        # Caption scopes: strictly inside object_ref content intervals
        inside_desc = self._compute_inside_ranges(
            labels,
            start_id=self._ids.get("obj_ref_start"),
            end_id=self._ids.get("obj_ref_end"),
        )

        # Geometry scopes: strictly inside each geometry wrapper pair
        inside_box = self._compute_inside_ranges(
            labels,
            start_id=self._ids.get("box_start"),
            end_id=self._ids.get("box_end"),
        )
        inside_quad = self._compute_inside_ranges(
            labels,
            start_id=self._ids.get("quad_start"),
            end_id=self._ids.get("quad_end"),
        )
        inside_line = self._compute_inside_ranges(
            labels,
            start_id=self._ids.get("line_start"),
            end_id=self._ids.get("line_end"),
        )
        inside_any_geom = inside_box | inside_quad | inside_line

        # Category unshifted masks (global, not yet intersected with assistant spans)
        caption_all = inside_desc & ~is_punct & ~is_geom_wrapper & ~is_coord
        # Grounding: ALL content inside geometry spans, regardless of coord-mode, minus separators
        grounding_all = is_coord | is_geom_wrapper | (inside_any_geom & ~is_geom_sep)
        # Formatting: object-ref wrappers and separators (punctuation is included; geom seps explicitly too)
        formatting_all = is_objref_wrapper | is_geom_sep | (is_punct & ~is_geom_sep)

        # Intersect with assistant masks and shift by one for CE alignment
        def _shift_intersect(
            cat_mask: torch.Tensor, who_mask: torch.Tensor
        ) -> torch.Tensor:
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

        # Ensure full coverage equals assistant masks (shifted). Assign residual to formatting.
        t_assist = teacher_mask[:, 1:]
        s_assist = student_mask[:, 1:]

        def _assign_residual_to_formatting(
            m_cap: torch.Tensor,
            m_grd: torch.Tensor,
            m_fmt: torch.Tensor,
            assist: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            union = m_cap | m_grd | m_fmt
            residual = assist & (~union)
            if residual.any():
                m_fmt = m_fmt | residual
            return m_cap, m_grd, m_fmt

        t_caption, t_ground, t_format = _assign_residual_to_formatting(
            t_caption, t_ground, t_format, t_assist
        )
        s_caption, s_ground, s_format = _assign_residual_to_formatting(
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

    @staticmethod
    def _compute_inside_ranges(
        labels: torch.Tensor, start_id: Optional[int], end_id: Optional[int]
    ) -> torch.Tensor:
        """Compute positions strictly inside [start_token, end_token) intervals.

        Returns a boolean mask with True for tokens that lie between a start and the
        next end token (exclusive of the wrapper tokens themselves). If start/end
        are missing, returns an all-False mask.
        """
        if start_id is None or end_id is None:
            return torch.zeros_like(labels, dtype=torch.bool)
        batch, seq_len = labels.shape
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for b in range(batch):
            open_flag = False
            for pos in range(seq_len):
                val = int(labels[b, pos].item())
                if not open_flag and val == start_id:
                    open_flag = True
                    continue
                if open_flag and val == end_id:
                    open_flag = False
                    continue
                if open_flag:
                    mask[b, pos] = True
        return mask


__all__ = ["TokenGroupingPlugin", "GroupMasks"]
