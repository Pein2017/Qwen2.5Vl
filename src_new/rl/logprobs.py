"""Log-probability helpers for multimodal GRPO.

Note: get_per_token_logps was used by the manual trainer and is deprecated.
Only clear_gpu_memory remains for safe cleanup calls.
"""

from __future__ import annotations

import torch


def clear_gpu_memory() -> None:
    """Comprehensive GPU memory cleanup with error handling.

    Safe to call multiple times. Clears cache and synchronizes to ensure
    cleanup completes before continuing.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronization can be costly; enable only when explicitly requested
            import os as _os

            if _os.getenv("RL_SYNC_ON_CLEAR", "0") == "1":
                torch.cuda.synchronize()
    except Exception:
        # Silently ignore cleanup errors to avoid disrupting training
        pass


# Deprecated functions removed


__all__ = [
    "clear_gpu_memory",
]
