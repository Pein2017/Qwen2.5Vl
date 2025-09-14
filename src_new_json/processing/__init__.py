#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing module for Qwen2.5-VL HuggingFace-first architecture.

This module provides clean processing components for:
- HuggingFace-first conversation processing
- Focused coordinate token conversion
- Centralized prompt constants
"""

from .coordinate_converter import CoordinateTokenConverter
from .templates import CONSTANTS
from .token_processor import TokenProcessor


__all__ = [
    "CoordinateTokenConverter",
    "TokenProcessor",
    "CONSTANTS",
]
