#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities to translate legacy src_post config keys."""
from __future__ import annotations

from typing import Dict, List


TRANSLATION_MAP = {
    "processor_path": "processor",
    "lr_lm_head": "llm_lr",
    "lr_last_layers": "llm_lr",
    "merger_lr": "aligner_lr",
    "lr_aligner": "aligner_lr",
}


def translate_legacy_config(cfg: Dict[str, any]) -> List[str]:
    warnings: List[str] = []
    for legacy_key, new_key in TRANSLATION_MAP.items():
        if legacy_key in cfg and new_key not in cfg:
            cfg[new_key] = cfg.pop(legacy_key)
            warnings.append(f"Config key '{legacy_key}' is deprecated; mapped to '{new_key}'.")
    return warnings
