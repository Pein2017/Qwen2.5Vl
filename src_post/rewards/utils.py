#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for reward configuration."""
from __future__ import annotations

from typing import Iterable, List


def l1_normalize(weights: Iterable[float]) -> List[float]:
    values = [float(w) for w in weights]
    total = sum(abs(w) for w in values)
    if total <= 0.0:
        return values
    return [float(w / total) for w in values]
