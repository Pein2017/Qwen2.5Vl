#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token grouping plugin for grouped LLM loss (caption, grounding, formatting).

This module constructs boolean masks for category-specific cross-entropy
computation aligned to the existing single-pass CE path in LossManager.

- caption: natural-language description tokens strictly inside
  <|object_ref_start|> ... <|object_ref_end|>, excluding formatting punctuation.
- grounding: numeric coordinate value tokens strictly inside geometry wrappers
  (line/box/quad). Geometry wrapper tokens themselves are not grounding.
- formatting: structural glue: punctuation/brackets/separators, object-ref and
  geometry wrapper tokens, and explicit terminators like <|im_end|> if present
  within assistant spans.

The plugin operates purely on label IDs and spans, without text decode, and
returns masks already intersected with teacher/student assistant spans and
aligned to the shifted CE convention (drop the first position).

This module exposes no external configuration. Category definitions and
punctuation sets are internal and conservative by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import re
import os
import logging

import torch

from src_new.processing.special_tokens import IM_END
from .grouping_core import (
    build_id_sets,
    build_base_predicates,
    assign_residual_to_formatting,
)
from src_new.utils.rank_aware_logging import get_rank_aware_logger

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
    """Builds category masks from label IDs and assistant spans.

    Strictly fails on shape/type mismatches; missing special tokens are handled
    gracefully by producing empty masks for that token class.
    """

    def __init__(self, tokenizer) -> None:
        if tokenizer is None:
            raise ValueError("TokenGroupingPlugin requires a non-None tokenizer")
        self._tok = tokenizer

        # Centralized ID sets for grouping
        self._ids_core = build_id_sets(self._tok)

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
        input_ids: Optional[torch.Tensor] = None,
        variant_key: Optional[str] = None,
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
        global _DEBUG_DUMPED
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

        # Base predicates & wrapper presence flag from core
        is_coord, is_geom_wrapper, is_objref_wrapper, is_punct, is_geom_sep, wrappers_present = build_base_predicates(
            labels=labels, ids=self._ids_core
        )

        # Caption scopes: strictly inside object_ref content intervals
        inside_desc = self._compute_inside_ranges(
            labels,
            start_id=self._safe_id("<|object_ref_start|>"),
            end_id=self._safe_id("<|object_ref_end|>"),
        )

        # Geometry scopes: strictly inside each geometry wrapper pair
        inside_box = self._compute_inside_ranges(
            labels,
            start_id=self._safe_id("<|box_start|>"),
            end_id=self._safe_id("<|box_end|>"),
        )
        inside_quad = self._compute_inside_ranges(
            labels,
            start_id=self._safe_id("<|quad_start|>"),
            end_id=self._safe_id("<|quad_end|>"),
        )
        inside_line = self._compute_inside_ranges(
            labels,
            start_id=self._safe_id("<|line_start|>"),
            end_id=self._safe_id("<|line_end|>"),
        )
        inside_any_geom = inside_box | inside_quad | inside_line

        # New: numeric-only grounding inside geometry; separators/wrappers go to formatting.
        # Numeric token IDs have been precomputed in grouping_core.build_id_sets().
        is_numeric = torch.zeros_like(labels, dtype=torch.bool)
        try:
            for tid in self._ids_core.numeric_ids:
                is_numeric |= labels.eq(int(tid))
        except Exception:
            # Fallback: no numeric set available -> rely on coord tokens if present
            is_numeric = torch.zeros_like(labels, dtype=torch.bool)

        # Category unshifted masks (global, not yet intersected with assistant spans)
        if isinstance(variant_key, str) and variant_key.strip().lower() == "summary":
            # Summary variant: treat all assistant tokens as caption except punctuation-only which is formatting.
            caption_all = (~is_punct) & (~is_geom_wrapper) & (~is_coord)
            grounding_all = torch.zeros_like(labels, dtype=torch.bool)
            formatting_all = is_punct
        else:
            caption_all = inside_desc & ~is_punct & ~is_geom_wrapper & ~is_coord
            # Grounding: numeric tokens OR coordinate tokens inside geometry spans
            grounding_all = inside_any_geom & (is_numeric | is_coord)
            # Formatting: wrappers + separators + residual non-numeric/non-coord inside geometry
            formatting_all = (
                is_objref_wrapper
                | is_geom_sep
                | is_geom_wrapper
                | (inside_any_geom & ~(is_numeric | is_coord))
                | (is_punct & ~is_geom_sep)
            )

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

        # Optional one-time debug dump (train/eval) controlled by env BBU_DEBUG_GROUPING=1
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
            except Exception:
                pass
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

        # Optional debug: verify disjointness and full coverage of assistant masks
        try:
            debug_alignment = bool(getattr(getattr(self._tok, "_config", None), "debug_alignment", False))
        except Exception:
            debug_alignment = False
        if debug_alignment:
            def _check(m_cap, m_grd, m_fmt, assist, who):
                # pairwise disjoint
                if (m_cap & m_grd).any() or (m_cap & m_fmt).any() or (m_grd & m_fmt).any():
                    raise RuntimeError(f"debug_alignment: overlapping group masks for {who}")
                union = m_cap | m_grd | m_fmt
                if (union != assist).any():
                    raise RuntimeError(f"debug_alignment: union of group masks does not equal assistant mask for {who}")
            _check(t_caption, t_ground, t_format, t_assist, "teacher")
            _check(s_caption, s_ground, s_format, s_assist, "student")

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
