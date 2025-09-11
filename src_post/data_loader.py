#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List


def build_epoch_indices(
    dataset_len: int,
    world_size: int,
    rank: int,
    limit_groups: int,
    seed: int,
    epoch: int,
) -> List[int]:
    import random as _random
    if dataset_len <= 0:
        return []
    base = list(range(dataset_len))
    rng = _random.Random(int(seed) + int(epoch))
    rng.shuffle(base)
    if int(limit_groups) > 0:
        base = base[: int(limit_groups)]
    if int(world_size) > 1:
        if len(base) >= int(world_size):
            effective = (len(base) // int(world_size)) * int(world_size)
            base = base[: effective]
        else:
            base = []
        base = [base[i] for i in range(len(base)) if (i % int(world_size)) == int(rank)]
    return base


def iter_batches(indices: List[int], batch_size: int, drop_last: bool) -> Iterable[List[int]]:
    ptr = 0
    N = len(indices)
    while ptr < N:
        remaining = N - ptr
        cur_bs = min(int(batch_size), remaining)
        if drop_last and cur_bs < int(batch_size):
            break
        yield indices[ptr : ptr + cur_bs]
        ptr += cur_bs
