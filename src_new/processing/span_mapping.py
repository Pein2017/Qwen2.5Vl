#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def _to_list(x: Iterable[int] | Sequence[int]) -> List[int]:
    try:
        return list(x)  # works for torch.Tensor as well
    except Exception:
        return [int(v) for v in x]


def map_spans_unexpanded_to_expanded(
    ids_unexpanded: Iterable[int] | Sequence[int],
    ids_expanded: Iterable[int] | Sequence[int],
    image_pad_id: int | None,
    spans_unexpanded: List[Tuple[int, int, bool]],
) -> List[Tuple[int, int, bool]]:
    """
    Map token spans computed on unexpanded tokenization to indices aligned with the
    expanded input_ids sequence produced by the processor (with <|image_pad|> expansion).

    The primary path is simple and robust:
    - Collapse each contiguous run of `image_pad_id` in `ids_expanded` into a single
      token and remember (start_index, run_length) per collapsed position.
    - Assert the collapsed expanded sequence equals `ids_unexpanded`.
    - Map each unexpanded position k → the corresponding expanded start index.
      For end positions, advance by the collapsed run_length.

    If sequences differ beyond image token runs (rare), fall back to a minimal
    greedy local alignment that advances over image_pad runs and searches forward
    for the next matching token within a small window.
    """
    ids_u = _to_list(ids_unexpanded)
    ids_e = _to_list(ids_expanded)

    if not isinstance(image_pad_id, int) or image_pad_id < 0:
        return spans_unexpanded[:]  # No image expansion: identity mapping

    # Build collapsed expanded sequence + mapping to expanded indices
    collapsed: List[int] = []
    col_to_exp_start: List[int] = []
    col_run_len: List[int] = []

    i = 0
    n = len(ids_e)
    while i < n:
        tok = ids_e[i]
        if tok == image_pad_id:
            # Run of image_pad
            j = i
            while j < n and ids_e[j] == image_pad_id:
                j += 1
            collapsed.append(image_pad_id)
            col_to_exp_start.append(i)
            col_run_len.append(j - i)
            i = j
        else:
            collapsed.append(tok)
            col_to_exp_start.append(i)
            col_run_len.append(1)
            i += 1

    mapped: List[Tuple[int, int, bool]] = []

    if collapsed == ids_u:
        # Fast path: 1-1 alignment after collapsing
        def _map_pos(pos_u: int) -> int:
            if pos_u <= 0:
                return 0
            if pos_u >= len(col_to_exp_start):
                return len(ids_e)
            return col_to_exp_start[pos_u]

        for st_u, ed_u, is_teacher in spans_unexpanded:
            st_u_i = int(max(0, st_u))
            ed_u_i = int(max(st_u_i, ed_u))
            st_e = _map_pos(st_u_i)
            # End maps to the start of the end-token, then extend by that token's run length
            if ed_u_i >= len(col_to_exp_start):
                ed_e = len(ids_e)
            else:
                ed_e = col_to_exp_start[ed_u_i]  # start of end token
            mapped.append((st_e, ed_e, bool(is_teacher)))
        return mapped

    # Fallback: minimal greedy alignment tolerant to minor template/tokenization drift
    exp_len = len(ids_e)

    # Precompute indices of each token (excluding image_pad) to accelerate small-window search
    from collections import defaultdict

    idx_by_token: dict[int, List[int]] = defaultdict(list)
    for idx, t in enumerate(ids_e):
        if t != image_pad_id:
            idx_by_token[t].append(idx)

    def _greedy_map_pos(pos_u: int) -> int:
        if pos_u <= 0:
            return 0
        if pos_u >= len(ids_u):
            return exp_len
        target = ids_u[pos_u]
        # Direct match near previous mapped position is unknown here; search from beginning with small stride
        # Use the first occurrence not earlier than the previous mapped one could be improved, but keep simple.
        cand = idx_by_token.get(target)
        if not cand:
            return min(exp_len, pos_u)  # fallback monotone
        # Choose the smallest index that is >= pos_u if possible; else the last occurrence
        import bisect

        j = bisect.bisect_left(cand, pos_u)
        if j < len(cand):
            return cand[j]
        return cand[-1]

    for st_u, ed_u, is_teacher in spans_unexpanded:
        st_u_i = int(max(0, st_u))
        ed_u_i = int(max(st_u_i, ed_u))
        st_e = _greedy_map_pos(st_u_i)
        ed_e = _greedy_map_pos(ed_u_i)
        if ed_e < st_e:
            ed_e = st_e
        mapped.append((st_e, ed_e, bool(is_teacher)))

    return mapped


__all__ = ["map_spans_unexpanded_to_expanded"]
