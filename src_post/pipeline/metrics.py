#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sliding-window metrics aggregation helpers."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable


class MetricsAggregator:
    """Maintain sliding-window averages for scalar metrics."""

    def __init__(self, window_size: int = 50, suffix: str = "_win") -> None:
        self.window_size = max(1, int(window_size))
        self.suffix = suffix
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, scalars: Dict[str, float]) -> Dict[str, float]:
        """Update buffers with new scalars and return windowed means."""
        aggregated: Dict[str, float] = {}
        for key, value in scalars.items():
            try:
                if isinstance(value, (int, float)):
                    buf = self._buffers[key]
                    buf.append(float(value))
                    aggregated[f"{key}{self.suffix}"] = float(sum(buf) / len(buf))
            except Exception:
                continue
        return aggregated

    def reset(self) -> None:
        """Clear all buffered metrics."""
        self._buffers.clear()

    def tracked_keys(self) -> Iterable[str]:
        return list(self._buffers.keys())
