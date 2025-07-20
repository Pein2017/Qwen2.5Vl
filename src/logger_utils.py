"""Minimal, project-wide logging helpers with rank-aware filtering.

All modules should obtain a logger via::

    from src.logger_utils import get_logger
    logger = get_logger(__name__)

`configure_global_logging()` *must* be called exactly once from the entry
point (e.g. ``scripts/train.py``, ``src/inference.py``) **before** any heavy
work starts so that every module writes to the same log file.  Subsequent
calls are ignored.

The logger automatically handles distributed training by only showing logs
on rank 0 (main process) for INFO/DEBUG levels, while ERROR/WARNING levels
are shown on all ranks for safety.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable


_CONFIGURED: bool = False
_CURRENT_RANK: int = 0
_IS_MAIN_PROCESS: bool = True


def _detect_current_rank() -> tuple[int, bool]:
    """Detect current process rank in distributed training.

    Returns:
        Tuple of (rank, is_main_process)
    """
    try:
        import torch

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            return rank, rank == 0
    except (ImportError, RuntimeError):
        pass

    # Fallback: check environment variables set by torchrun
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # For single GPU training, both should be 0
    is_main = rank == 0
    return rank, is_main


class RankAwareFilter(logging.Filter):
    """Custom filter that controls log output based on process rank.

    - ERROR/WARNING: Always shown (safety-critical)
    - INFO/DEBUG: Only shown on rank 0 (main process)
    """

    def __init__(self, is_main_process: bool = True):
        super().__init__()
        self.is_main_process = is_main_process

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on rank and level."""
        # Always allow ERROR and WARNING on all ranks for safety
        if record.levelno >= logging.WARNING:
            return True

        # INFO and DEBUG only on main process (rank 0)
        return self.is_main_process


def configure_global_logging(
    *,
    log_dir: str = "logs",
    log_file: str = "run.log",
    log_level: str | int = "INFO",
    verbose: bool = False,
    overwrite: bool = True,
    **_: Any,
) -> None:
    """Configure root logger with rank-aware filtering.

    Parameters
    ----------
    log_dir
        Directory that will contain *log_file*.
    log_file
        File name (inside *log_dir*) that receives all log messages.
    log_level
        Logging level. INFO for production, DEBUG for development.
    verbose
        If *True* timestamps are included in console output.
    overwrite
        If *True* the log file is truncated each run, otherwise we append.

    Notes
    -----
    Rank-aware filtering is automatically applied:
    - ERROR/WARNING: Shown on all ranks (safety-critical)
    - INFO/DEBUG: Only shown on rank 0 (main process)
    """

    global _CONFIGURED, _CURRENT_RANK, _IS_MAIN_PROCESS
    if _CONFIGURED:
        return

    # Detect current rank for distributed training
    _CURRENT_RANK, _IS_MAIN_PROCESS = _detect_current_rank()

    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Convert level strings to ints
    log_level_int = (
        logging.getLevelName(log_level.upper())
        if isinstance(log_level, str)
        else int(log_level)
    )

    # Clear existing handlers
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # Create rank-aware filter
    rank_filter = RankAwareFilter(_IS_MAIN_PROCESS)

    # Formatters
    base_fmt = "%(levelname)s - %(name)s - %(message)s"
    if verbose:
        base_fmt = "%(asctime)s - " + base_fmt
    if not _IS_MAIN_PROCESS:
        # Add rank info for non-main processes (for ERROR/WARNING only)
        base_fmt = f"[Rank {_CURRENT_RANK}] " + base_fmt

    formatter = logging.Formatter(base_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # File handler (always UTF-8) - logs everything for main process only
    file_mode = "w" if overwrite else "a"
    fh = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    fh.setLevel(log_level_int)
    fh.setFormatter(formatter)
    fh.addFilter(rank_filter)

    # Console handler - rank-aware filtering
    ch = logging.StreamHandler()
    ch.setLevel(log_level_int)
    ch.setFormatter(formatter)
    ch.addFilter(rank_filter)

    # Root logger setup
    root.setLevel(log_level_int)
    root.addHandler(fh)
    root.addHandler(ch)

    _CONFIGURED = True


def get_current_rank() -> int:
    """Get the current process rank (0 for single GPU)."""
    return _CURRENT_RANK


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return _IS_MAIN_PROCESS


def rank0_only(func):
    """Decorator to only execute function on rank 0 (main process)."""

    def wrapper(*args, **kwargs):
        if _IS_MAIN_PROCESS:
            return func(*args, **kwargs)
        return None

    return wrapper


# ---------------------------------------------------------------------------
# Convenience helpers & aliases
# ---------------------------------------------------------------------------


def get_logger(name: str | None = None) -> logging.Logger:  # noqa: D401
    """Return a module-specific :class:`logging.Logger`."""

    return logging.getLogger(name or "main")


# Alias type for backward compatibility (was a wrapper class before)
PrefixedLogger = logging.Logger  # type: ignore


# Simple generator for alias functions e.g. get_training_logger()
def _make_getter(alias: str) -> Callable[[], logging.Logger]:
    return lambda: get_logger(alias)


# Common aliases referenced throughout the codebase -------------------------
get_training_logger = _make_getter("training")
get_model_logger = _make_getter("model")
get_data_logger = _make_getter("data")
get_config_logger = _make_getter("config")
get_inference_logger = _make_getter("inference")
get_loss_logger = _make_getter("loss")
get_attention_logger = _make_getter("attention")
get_monitor_logger = _make_getter("monitor")
get_stability_logger = _make_getter("stability")
get_callback_logger = _make_getter("callback")
get_chat_logger = _make_getter("chat")
get_utils_logger = _make_getter("utils")
get_diagnostics_logger = _make_getter("diagnostics")
get_tokens_logger = _make_getter("tokens")
get_patches_logger = _make_getter("patches")

# Legacy names kept for painless migration ----------------------------------
get_raw_data_logger = _make_getter("raw_data")
get_debug_logger = _make_getter("debug")
get_sample_logger = _make_getter("sample")
get_detection_logger = _make_getter("detection")

# Back-compat alias – some scripts import this name directly
configure_logging = configure_global_logging
