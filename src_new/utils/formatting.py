from typing import Any, Dict, Iterable, Optional, Union

import torch


NumberLike = Union[float, int, torch.Tensor]


def _to_float(value: NumberLike) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.item()) if value.numel() == 1 else float(value.mean().item())
    try:
        return float(value)
    except Exception:
        return None


def _trim_trailing_zeros(s: str) -> str:
    if "e" in s or "E" in s:
        return s
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def format_float_compact(
    value: NumberLike,
    sci_threshold: float = 1e-3,
    decimals_std: int = 4,
    decimals_sci: int = 2,
) -> str:
    """
    Format a numeric value with conditional scientific notation:
    - If abs(value) < sci_threshold and value != 0 -> scientific (e.g., 1.23e-04)
    - Else -> standard decimal with fixed precision (trim trailing zeros)
    Always returns a string; if value cannot be parsed, returns str(value).
    """
    f = _to_float(value)
    if f is None:
        return str(value)
    if f == 0.0:
        return f"{0.0:.{decimals_std}f}"
    if abs(f) < sci_threshold:
        return f"{f:.{decimals_sci}e}"
    return _trim_trailing_zeros(f"{f:.{decimals_std}f}")


def format_losses_in_logs(
    logs: Dict[str, Any],
    keys: Iterable[str],
    sci_threshold: float = 1e-3,
    decimals_std: int = 4,
    decimals_sci: int = 2,
) -> Dict[str, Any]:
    """
    Return a new logs dict with selected numeric keys formatted via format_float_compact.
    Non-existent keys are ignored.
    """
    for k in keys:
        if k in logs and logs[k] is not None:
            logs[k] = format_float_compact(
                logs[k],
                sci_threshold=sci_threshold,
                decimals_std=decimals_std,
                decimals_sci=decimals_sci,
            )
    return logs
