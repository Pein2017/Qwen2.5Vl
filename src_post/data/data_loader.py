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


def build_balanced_indices(
    pass_indices: List[int],
    fail_indices: List[int],
    world_size: int,
    rank: int,
    limit_groups: int,
    seed: int,
    epoch: int,
) -> List[int]:
    import random as _random
    rng = _random.Random(int(seed) * 7919 + int(epoch) * 104729)
    p = list(pass_indices)
    f = list(fail_indices)
    rng.shuffle(p)
    rng.shuffle(f)
    # Pair up to the smaller class (or limit if set)
    n = min(len(p), len(f))
    if int(limit_groups) > 0:
        n = min(n, int(limit_groups) // 2)
    paired: List[int] = []
    for i in range(n):
        paired.append(p[i])
        paired.append(f[i])
    # DDP shard by position
    if int(world_size) > 1:
        if len(paired) >= int(world_size):
            effective = (len(paired) // int(world_size)) * int(world_size)
            paired = paired[: effective]
        else:
            paired = []
        paired = [paired[i] for i in range(len(paired)) if (i % int(world_size)) == int(rank)]
    return paired


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
