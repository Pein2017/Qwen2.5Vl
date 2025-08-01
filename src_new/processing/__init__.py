#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing module for Qwen2.5-VL new architecture.

This module provides stateless processing components for:
- Chat conversation building and tokenization
- Coordinate token handling and tokenizer extension
- Template management for Chinese prompts
"""

from .chat_processor import ChatProcessor
from .token_processor import TokenProcessor
from .templates import TemplateManager

__all__ = [
    "ChatProcessor",
    "TokenProcessor", 
    "TemplateManager",
]