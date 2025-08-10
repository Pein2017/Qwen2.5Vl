"""
No-op patches module for src_new.

Provides apply_patches() to satisfy optional test import without affecting runtime.
"""

from typing import Optional


def apply_patches(_: Optional[object] = None) -> None:
    """Apply optional runtime patches. Intentionally does nothing."""
    return
