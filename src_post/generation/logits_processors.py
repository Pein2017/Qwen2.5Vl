#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List, Optional, Set

import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src_new.processing.special_tokens import GEOMETRY_TOKENS, get_coord_token_range


def _lookup_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> Optional[int]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if token in vocab:
        try:
            return int(vocab[token])
        except Exception:
            return None
    return None


def _collect_geometry_token_ids(tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    ids: Set[int] = set()
    for (_k, (ref_s, ref_e, geo_s, geo_e)) in GEOMETRY_TOKENS.items():
        for tok in (ref_s, ref_e, geo_s, geo_e):
            tid = _lookup_token_id(tokenizer, tok)
            if tid is not None:
                ids.add(tid)
    return ids


def _collect_coordinate_token_ids(tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    rng = get_coord_token_range(tokenizer)
    if rng.end_exclusive <= rng.start_id:
        return set()
    return set(range(int(rng.start_id), int(rng.end_exclusive)))


class GeometryCoordMaskLogitsProcessor(LogitsProcessor):
    """Masks geometry and/or coordinate tokens during decoding.

    This processor is intended for RL Stage-A/Stage-B where outputs must be pure text
    (no geometry wrappers or coordinate tokens). It only affects generation time.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_geometry_tokens: bool = True,
        mask_coordinate_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_geometry_tokens = bool(mask_geometry_tokens)
        self.mask_coordinate_tokens = bool(mask_coordinate_tokens)

        banned_ids: Set[int] = set()
        if self.mask_geometry_tokens:
            banned_ids |= _collect_geometry_token_ids(tokenizer)
        if self.mask_coordinate_tokens:
            banned_ids |= _collect_coordinate_token_ids(tokenizer)

        self._banned_ids: List[int] = sorted(banned_ids)
        self._banned_ids_tensor: Optional[torch.Tensor] = None

    def _ensure_ids_tensor(self, device: torch.device) -> torch.Tensor:
        if self._banned_ids_tensor is None or self._banned_ids_tensor.device != device:
            if len(self._banned_ids) == 0:
                self._banned_ids_tensor = torch.empty(0, dtype=torch.long, device=device)
            else:
                self._banned_ids_tensor = torch.tensor(self._banned_ids, dtype=torch.long, device=device)
        return self._banned_ids_tensor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(self._banned_ids) == 0:
            return scores
        banned_ids_tensor = self._ensure_ids_tensor(scores.device)
        if banned_ids_tensor.numel() == 0:
            return scores
        # scores shape: [batch, vocab]
        # Set banned token logits to a very low value to avoid selection
        min_val = torch.finfo(scores.dtype).min
        scores.index_fill_(dim=1, index=banned_ids_tensor, value=min_val)
        return scores


class _Removed: pass
    """Adds a positive bias to token ids derived from a set of mission tokens.

    Use this at training time to make Stage-A more likely to emit mission fail tokens
    for GT=fail groups. The bias is additive on logits.
    """

    pass
