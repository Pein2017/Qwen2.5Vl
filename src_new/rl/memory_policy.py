"""Memory cleanup policy wrapper (no behavior change by default).

Wraps GPU cache clearing calls behind a tiny policy so we can switch
between always/boundary/never without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src_new.rl.diagnostics.settings import SETTINGS


@dataclass(frozen=True)
class MemoryPolicy:
    mode: str  # "always" | "boundary" | "never"

    def should_clear(self, boundary: bool = False) -> bool:
        if self.mode == "never":
            return False
        if self.mode == "boundary":
            return bool(boundary)
        return True  # default: always


_POLICY = MemoryPolicy(mode=SETTINGS.clear_cache_policy)


def maybe_clear_gpu_cache(boundary: bool = False) -> None:
    if not torch.cuda.is_available():
        return
    if not _POLICY.should_clear(boundary=boundary):
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


__all__ = ["MemoryPolicy", "maybe_clear_gpu_cache"]
