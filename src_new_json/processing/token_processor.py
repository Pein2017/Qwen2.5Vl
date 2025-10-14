#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token processor (JSON mode).

In pure JSON mode, this component becomes a no-op stub preserved only for
interface compatibility. Vocabulary extension and wrapper/coord token utilities
are disabled.
"""

from dataclasses import dataclass


@dataclass
class TokenProcessorConfig:
    """Configuration for token processing (JSON mode)."""

    # JSON mode: no coordinate-token controls; kept minimal for interface compatibility
    pass


def validate_token_processor_config(cfg: TokenProcessorConfig) -> None:
    """No-op in JSON mode; kept for interface compatibility."""
    return
