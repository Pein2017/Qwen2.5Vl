"""Tensor utilities for RL helpers (behavior-preserving).

Currently provides THW normalization used by buffer/generation paths.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def normalize_thw(thw: Optional[Any]) -> Optional[torch.Tensor]:
    """Normalize THW metadata to shape [N, 3] or return None.

    Accepts shapes: [3], [1,3,1], [N,3], [1,N,3] and general viewable forms.
    Returns None for None or empty inputs. Detaches tensors to avoid autograd.
    """
    if thw is None:
        return None
    if not torch.is_tensor(thw):
        try:
            thw = torch.tensor(thw)
        except Exception:
            return None
    if thw.numel() == 0:
        return None
    thw = thw.detach()
    if thw.dim() == 1 and thw.numel() == 3:
        return thw.view(1, 3)
    if thw.dim() == 3 and thw.size(0) == 1:
        return thw.squeeze(0)
    if thw.dim() == 2 and thw.size(-1) == 3:
        return thw
    return thw.view(-1, 3)


__all__ = ["normalize_thw"]
