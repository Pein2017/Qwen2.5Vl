#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import torch.distributed as dist
from transformers import Qwen2VLProcessor

from src_new.models.wrapper import DetectionModel


def save_if_rank0(
    policy: DetectionModel,
    processor: Qwen2VLProcessor,
    output_dir: str,
    tag: str,
    skip_save: bool,
) -> None:
    save_dir = Path(output_dir) / "checkpoints" / str(tag)
    if skip_save:
        # Still touch barriers for multi-GPU safety
        if dist.is_initialized():
            dist.barrier()
        return
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create checkpoint directory: {save_dir}. Error: {e}"
        )
    # Rank detection (env-level)
    try:
        int(
            Path("/proc/self/stat").read_text().split()[0]
        )  # fallback dummy; not reliable
    except Exception:
        pass
    # We still guard with distributed state when available
    is_rank0 = True
    try:
        if dist.is_available() and dist.is_initialized():
            is_rank0 = dist.get_rank() == 0
    except Exception:
        is_rank0 = True
    if is_rank0:
        policy.base_model.save_pretrained(str(save_dir))
        try:
            (save_dir / "processor").mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(str(save_dir / "processor"))
        except Exception:
            pass
    if dist.is_initialized():
        dist.barrier()
