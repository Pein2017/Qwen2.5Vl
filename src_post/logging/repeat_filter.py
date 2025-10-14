#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging filter to suppress repeated messages and expose suppression stats."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import DefaultDict, Tuple


class RepeatedMessageFilter(logging.Filter):
    """Filter that only lets the first *limit* occurrences of a message through."""

    def __init__(self, name: str = "", *, limit: int = 1) -> None:
        super().__init__(name)
        self.limit = max(1, int(limit))
        self._seen: DefaultDict[Tuple[int, str], int] = defaultdict(int)
        self._suppressed: DefaultDict[Tuple[int, str], int] = defaultdict(int)

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.levelno, record.getMessage())
        self._seen[key] += 1
        if self._seen[key] <= self.limit:
            return True
        self._suppressed[key] += 1
        return False

    def suppressed_summary(self, reset: bool = False) -> DefaultDict[Tuple[int, str], int]:
        summary = self._suppressed.copy()
        if reset:
            self._suppressed.clear()
        return summary
