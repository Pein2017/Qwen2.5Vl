"""Minimal, project-wide logging helpers.

All modules should obtain a logger via::

    from src.logger_utils import get_logger
    logger = get_logger(__name__)

`configure_global_logging()` *must* be called exactly once from the entry
point (e.g. ``scripts/train.py``, ``src/inference.py``) **before** any heavy
work starts so that every module writes to the same log file.  Subsequent
calls are ignored.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

_CONFIGURED: bool = False


def configure_global_logging(
    *,
    log_dir: str = "logs",
    log_file: str = "run.log",
    log_level: str | int = "INFO",
    console_level: str | int | None = None,
    verbose: bool = False,
    overwrite: bool = True,
    **_: Any,
) -> None:
    """Configure root logger with a single file + console handler.

    Parameters
    ----------
    log_dir
        Directory that will contain *log_file*.
    log_file
        File name (inside *log_dir*) that receives all log messages.
    log_level, console_level
        Levels for file handler and console handler respectively.  If
        *console_level* is *None* the same level as *log_level* is used.
    verbose
        If *True* timestamps are included in console output.
    overwrite
        If *True* the log file is truncated each run, otherwise we append.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Convert level strings to ints
    log_level_int = (
        logging.getLevelName(log_level.upper())
        if isinstance(log_level, str)
        else int(log_level)
    )
    console_level_int = (
        logging.getLevelName(console_level.upper())
        if isinstance(console_level, str)
        else (log_level_int if console_level is None else int(console_level))
    )

    # Clear existing handlers
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # Formatters
    fmt = (
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        if verbose
        else "%(levelname)s - %(name)s - %(message)s"
    )
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # File handler (always UTF-8)
    file_mode = "w" if overwrite else "a"
    fh = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    fh.setLevel(log_level_int)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level_int)
    ch.setFormatter(formatter)

    # Root logger setup
    root.setLevel(log_level_int)
    root.addHandler(fh)
    root.addHandler(ch)

    _CONFIGURED = True


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
