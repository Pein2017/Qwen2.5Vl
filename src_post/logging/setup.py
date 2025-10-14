#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank-aware logging configuration helpers for src_post."""
from __future__ import annotations

import logging
from typing import Optional

from .repeat_filter import RepeatedMessageFilter


def configure_logging(
    rank: int,
    *,
    debug: bool = False,
    log_format: Optional[str] = None,
) -> RepeatedMessageFilter:
    """Configure root logging handlers once per process."""

    root = logging.getLogger()
    if log_format is None:
        log_format = "[%(asctime)s] [R%(process)d] %(levelname)s %(name)s: %(message)s"
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    # Rank-aware level adjustments
    if rank != 0 and not debug:
        logging.getLogger().setLevel(logging.WARNING)

    # Reduce external noise on non-zero ranks
    if rank != 0:
        for name in ["transformers", "torch", "urllib3"]:
            logging.getLogger(name).setLevel(logging.ERROR)

    # Honor TRANSFORMERS_VERBOSITY env if set
    try:
        from transformers.utils import logging as hf_logging

        level = logging.DEBUG if debug else logging.ERROR if rank != 0 else logging.INFO
        hf_logging.set_verbosity(level)
    except Exception:
        pass

    repeat_filter = RepeatedMessageFilter(limit=1)
    for handler in root.handlers:
        handler.addFilter(repeat_filter)
    return repeat_filter
