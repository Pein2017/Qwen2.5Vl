#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimizer and scheduler helpers for src_post GRPO training."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src_post.models import apply_freeze_and_param_groups


def build_optimizer(
    policy,
    tokenizer,
    cfg,
) -> Tuple[Optimizer, List[Dict[str, float]]]:
    """Create the AdamW optimizer with fused fallback and capture per-group LRs."""

    param_groups = apply_freeze_and_param_groups(policy, tokenizer, cfg)
    lr = float(cfg.learning_rate)
    weight_decay = float(cfg.weight_decay)
    group_infos: List[Dict[str, float]] = []
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )  # type: ignore[arg-type]
        fused = True
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
        )
        fused = False

    for idx, group in enumerate(param_groups):
        name = str(group.get("name", f"group{idx}"))
        group_lr = float(group.get("lr", lr))
        group_infos.append({"name": name, "lr": group_lr})

    if fused:
        logging.getLogger("src_post.training.optim").info(
            "Using fused AdamW optimizer (fused=True)"
        )
    return optimizer, group_infos


def build_scheduler(
    cfg,
    optimizer: Optimizer,
    dataset_len: int,
    world_size: int,
    rank: int,
    limit_groups: int,
) -> Tuple[Optional[LRScheduler], int]:
    """Create LR scheduler based on cfg (cosine/warmup) and return steps per epoch."""

    try:
        from transformers import get_scheduler
    except Exception:
        return None, 0

    from src_post.data.data_loader import build_epoch_indices, iter_batches

    per_rank_indices = build_epoch_indices(
        dataset_len=dataset_len,
        world_size=world_size,
        rank=rank,
        limit_groups=int(limit_groups),
        seed=int(cfg.seed),
        epoch=0,
    )
    batches = list(
        iter_batches(
            per_rank_indices,
            int(cfg.batch_size),
            bool(cfg.drop_last),
        )
    )
    updates_per_epoch = int(len(batches))
    total_steps = max(1, int(cfg.epochs) * max(1, updates_per_epoch))
    warmup_steps = max(
        0, int(getattr(cfg, "warmup_ratio", 0.0) * float(total_steps))
    )
    try:
        scheduler = get_scheduler(
            name=str(getattr(cfg, "lr_scheduler_type", "cosine")),
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return scheduler, updates_per_epoch
    except Exception:
        return None, updates_per_epoch
