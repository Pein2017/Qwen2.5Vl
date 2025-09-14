#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundational utilities for grouped LLM loss (caption/grounding/formatting).

Centralizes:
- Special/token ID set construction (wrappers, punctuation, separators, coord range)
- Base boolean predicates built from label IDs
- Residual assignment to formatting to ensure full assistant coverage

This module reduces duplication in token_grouping and makes future extensions simpler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import torch

from src_new_json.processing.special_tokens import ASSISTANT_SPAN_RE, get_coord_token_range


@dataclass(frozen=True)
class GroupingIdSets:
	coord_start: int
	coord_end_exclusive: int
	geom_wrapper_ids: Tuple[int, ...]
	objref_wrapper_ids: Tuple[int, ...]
	punctuation_ids: Tuple[int, ...]
	geom_sep_ids: Tuple[int, ...]


def _safe_id(tok, token: str) -> Optional[int]:
	tid = tok.convert_tokens_to_ids(token)
	if isinstance(tid, int) and tid >= 0:
		return int(tid)
	return None


def _encode_chars_to_ids(tok, chars: Sequence[str]) -> Tuple[int, ...]:
	ids_set: Dict[int, bool] = {}
	for ch in chars:
		ids = tok.encode(ch, add_special_tokens=False)
		if not isinstance(ids, list) or len(ids) == 0:
			raise ValueError(f"Tokenizer failed to encode character: {ch!r}")
		for tid in ids:
			if not isinstance(tid, int):
				raise ValueError(f"Tokenizer returned non-int id for character {ch!r}: {tid!r}")
			ids_set[int(tid)] = True
	return tuple(sorted(ids_set.keys()))


def build_id_sets(tok) -> GroupingIdSets:
	"""Build stable ID sets for grouping predicates from tokenizer."""
	coord_rng = get_coord_token_range(tok)
	coord_start = int(coord_rng.start_id)
	coord_end_exclusive = int(coord_rng.end_exclusive)

	# Geometry wrappers
	geom_wrapper_tokens = (
		"<|box_start|>",
		"<|box_end|>",
		"<|quad_start|>",
		"<|quad_end|>",
		"<|line_start|>",
		"<|line_end|>",
	)
	geom_wrapper_ids: List[int] = []
	for t in geom_wrapper_tokens:
		tid = _safe_id(tok, t)
		if isinstance(tid, int) and tid >= 0:
			geom_wrapper_ids.append(int(tid))

	# Object-ref wrappers
	objref_tokens = ("<|object_ref_start|>", "<|object_ref_end|>")
	objref_wrapper_ids: List[int] = []
	for t in objref_tokens:
		tid = _safe_id(tok, t)
		if isinstance(tid, int) and tid >= 0:
			objref_wrapper_ids.append(int(tid))

	# General punctuation and geometry separators
	punctuation_chars: Sequence[str] = ("[", "]", "{", "}", "(", ")", ",", ":", '"', "/")
	geom_sep_chars: Sequence[str] = ("[", "]", ",")
	punctuation_ids = _encode_chars_to_ids(tok, punctuation_chars)
	geom_sep_ids = _encode_chars_to_ids(tok, geom_sep_chars)

	return GroupingIdSets(
		coord_start=coord_start,
		coord_end_exclusive=coord_end_exclusive,
		geom_wrapper_ids=tuple(sorted(set(geom_wrapper_ids))),
		objref_wrapper_ids=tuple(sorted(set(objref_wrapper_ids))),
		punctuation_ids=tuple(sorted(set(punctuation_ids))),
		geom_sep_ids=tuple(sorted(set(geom_sep_ids))),
	)


def build_base_predicates(
	*,
	labels: torch.Tensor,
	ids: GroupingIdSets,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
	"""Build base boolean predicates and wrapper presence flag from labels.

	Returns:
	- is_coord, is_geom_wrapper, is_objref_wrapper, is_punct, is_geom_sep, wrappers_present
	"""
	if labels is None or labels.dim() != 2:
		raise ValueError(
			f"labels must be 2D [batch, seq_len], got {None if labels is None else tuple(labels.shape)}"
		)
	is_coord = (labels >= ids.coord_start) & (labels < ids.coord_end_exclusive)
	is_geom_wrapper = torch.zeros_like(labels, dtype=torch.bool)
	is_objref_wrapper = torch.zeros_like(labels, dtype=torch.bool)
	is_punct = torch.zeros_like(labels, dtype=torch.bool)
	is_geom_sep = torch.zeros_like(labels, dtype=torch.bool)

	for tid in ids.geom_wrapper_ids:
		is_geom_wrapper |= labels.eq(int(tid))
	for tid in ids.objref_wrapper_ids:
		is_objref_wrapper |= labels.eq(int(tid))
	for tid in ids.punctuation_ids:
		is_punct |= labels.eq(int(tid))
	for tid in ids.geom_sep_ids:
		is_geom_sep |= labels.eq(int(tid))

	wrappers_present = is_geom_wrapper.any() or is_objref_wrapper.any()
	return is_coord, is_geom_wrapper, is_objref_wrapper, is_punct, is_geom_sep, bool(wrappers_present)




def assign_residual_to_formatting(
	m_cap: torch.Tensor,
	m_grd: torch.Tensor,
	m_fmt: torch.Tensor,
	assist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Ensure full coverage by assigning residual assistant tokens to formatting."""
	union = m_cap | m_grd | m_fmt
	residual = assist & (~union)
	if residual.any():
		m_fmt = m_fmt | residual
	return m_cap, m_grd, m_fmt
