#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public configuration API for src_post.

This module currently mirrors the legacy interfaces while the refactor
progressively introduces sectioned schemas, layered loaders, and enhanced
validation.
"""
from __future__ import annotations

from .config import RLRunnerConfig
from .loader import (
    ConfigValidationError,
    load_and_validate_config,
    load_config,
    load_raw_config,
)


__all__ = [
    "ConfigValidationError",
    "RLRunnerConfig",
    "load_config",
    "load_and_validate_config",
    "load_raw_config",
]
