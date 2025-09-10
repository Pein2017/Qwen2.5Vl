#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundational utilities for grouped LLM loss (caption/grounding/formatting).

Centralizes:
- Special/token ID set construction (wrappers, punctuation, separators, coord range)
- Base boolean predicates built from label IDs
- Plain-text JSON mode caption/grounding extraction via char spans
- Residual assignment to formatting to ensure full assistant coverage

This module reduces duplication in token_grouping and makes future extensions simpler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import torch

from src_new.processing.special_tokens import ASSISTANT_SPAN_RE, get_coord_token_range


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


def extract_plain_mode_masks(
	*,
	tok,
	input_ids: torch.Tensor,
	labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Extract caption and grounding masks in plain JSON mode via char-span mapping.

	Returns: caption_all, grounding_all (both unshifted, [B,S] booleans)
	"""
	if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
		raise ValueError("Plain-mode grouping requires valid 2D input_ids for decoding and offset mapping.")
	batch, _ = labels.shape
	caption_all = torch.zeros_like(labels, dtype=torch.bool)
	grounding_all = torch.zeros_like(labels, dtype=torch.bool)

	for b in range(batch):
		row_ids = input_ids[b]
		# Decode conversation and build char offsets
		row_list = [int(x) for x in row_ids.tolist()]
		full_text = tok.decode(row_list, skip_special_tokens=False)
		tokenized = tok(
			full_text,
			return_offsets_mapping=True,
			add_special_tokens=False,
			return_tensors="pt",
		)
		offsets = tokenized["offset_mapping"][0].tolist()

		# Find assistant content spans (char positions)
		content_spans: List[Tuple[int, int]] = [(m.start(1), m.end(1)) for m in ASSISTANT_SPAN_RE.finditer(full_text)]
		if not content_spans:
			continue

		def _mark_char_range(c0: int, c1: int, dst_mask: torch.Tensor) -> None:
			start_tok = None
			end_tok = None
			for idx, (s, e) in enumerate(offsets):
				if start_tok is None and e > c0:
					start_tok = idx
				if end_tok is None and s >= c1:
					end_tok = idx
					break
			if start_tok is None:
				start_tok = len(offsets) - 1
			if end_tok is None:
				end_tok = len(offsets)
			max_len = int(labels.size(1))
			start_tok = max(0, min(int(start_tok), max_len))
			end_tok = max(int(start_tok), min(int(end_tok), max_len))
			if end_tok > start_tok:
				dst_mask[b, start_tok:end_tok] = True

		# Scan assistant content block line by line
		for c_start, c_end in content_spans:
			sub = full_text[c_start:c_end]
			base = c_start
			for ln in sub.split("\n"):
				ls = ln.strip()
				if not ls:
					base += len(ln) + 1
					continue
				# desc value
				m_desc = re.search(r"\"desc\"\s*:\s*\"", ls)
				if m_desc:
					v0 = m_desc.end()
					j = v0
					while j < len(ls):
						if ls[j] == '"' and ls[j - 1] != "\\":
							break
						j += 1
					v1 = j
					if v1 > v0:
						_mark_char_range(base + v0, base + v1, caption_all)
				# geometry key and array
				for gk in ('"bbox_2d"', '"quad"', '"line"'):
					kpos = ls.find(gk)
					if kpos >= 0:
						_mark_char_range(base + kpos, base + kpos + len(gk), grounding_all)
						sb = ls.find("[", kpos)
						eb = ls.find("]", sb + 1) if sb >= 0 else -1
						if sb >= 0 and eb > sb:
							_mark_char_range(base + sb + 1, base + eb, grounding_all)
						break
				base += len(ln) + 1

	return caption_all, grounding_all


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
