#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank-aware logging utilities for distributed training in Qwen2.5-VL.

This module provides centralized rank-aware logging that eliminates duplicate
logging across multiple ranks while maintaining necessary debugging information.

Key Features:
- Master rank only logging for general training information
- Rank-specific logging for errors and distributed communication issues
- Clean log organization with rank identification
- Decorator-based logging control
- Integration with existing logging infrastructure

Usage:
    from src_new_json.utils.rank_aware_logging import get_rank_aware_logger, rank0_only

    logger = get_rank_aware_logger(__name__)

    # This will only log on rank 0
    logger.info("Training progress: step 100")

    # This will log on all ranks (for errors)
    logger.error("Critical error occurred")

    # Decorator usage
    @rank0_only
    def log_training_metrics():
        logger.info("Logging training metrics...")
"""

import functools
import logging
import os
from typing import Optional, Union


# Global rank detection state
_CURRENT_RANK: Optional[int] = None
_WORLD_SIZE: Optional[int] = None
_IS_MAIN_PROCESS: Optional[bool] = None
_RANK_DETECTION_ATTEMPTED: bool = False

# Global log level management
_GLOBAL_LOG_LEVEL: int = logging.INFO
_GLOBAL_CONFIG_APPLIED: bool = False


def _detect_distributed_info() -> tuple[int, int, bool]:
    """
    Detect current rank, world size, and main process status.

    Returns:
        Tuple of (rank, world_size, is_main_process)
    """
    global _RANK_DETECTION_ATTEMPTED
    _RANK_DETECTION_ATTEMPTED = True

    # Method 1: Try PyTorch distributed
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            return rank, world_size, rank == 0
    except (ImportError, RuntimeError):
        pass

    # Method 2: Check environment variables (set by torchrun/deepspeed)
    rank_env = os.environ["RANK"] if ("RANK" in os.environ) else (os.environ["LOCAL_RANK"] if ("LOCAL_RANK" in os.environ) else "0")
    rank = int(rank_env)
    world_size_env = os.environ["WORLD_SIZE"] if ("WORLD_SIZE" in os.environ) else "1"
    world_size = int(world_size_env)

    # For single GPU training, both should be 0 and 1 respectively
    is_main = rank == 0
    return rank, world_size, is_main


def get_rank_info() -> tuple[int, int, bool]:
    """
    Get current distributed training information.

    Returns:
        Tuple of (rank, world_size, is_main_process)
    """
    global _CURRENT_RANK, _WORLD_SIZE, _IS_MAIN_PROCESS, _RANK_DETECTION_ATTEMPTED

    if not _RANK_DETECTION_ATTEMPTED:
        _CURRENT_RANK, _WORLD_SIZE, _IS_MAIN_PROCESS = _detect_distributed_info()

    return _CURRENT_RANK, _WORLD_SIZE, _IS_MAIN_PROCESS


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    _, _, is_main = get_rank_info()
    return is_main


def get_current_rank() -> int:
    """Get the current process rank."""
    rank, _, _ = get_rank_info()
    return rank


def get_world_size() -> int:
    """Get the total number of processes."""
    _, world_size, _ = get_rank_info()
    return world_size


class RankAwareFilter(logging.Filter):
    """
    Logging filter that implements rank-aware logging rules.

    Rules:
    - ERROR/WARNING: Always logged on all ranks (safety-critical)
    - INFO/DEBUG: Only logged on rank 0 (main process)
    - CRITICAL: Always logged on all ranks
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records based on rank and log level.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False otherwise
        """
        rank, _, is_main = get_rank_info()

        # Always log ERROR, WARNING, and CRITICAL on all ranks
        if record.levelno >= logging.WARNING:
            # Add rank information to critical messages for debugging
            if record.levelno >= logging.ERROR and get_world_size() > 1:
                record.msg = f"[RANK {rank}] {record.msg}"
            return True

        # Only log INFO and DEBUG on main process
        if record.levelno >= logging.INFO:
            return is_main

        # Default: log on main process only
        return is_main


def get_rank_aware_logger(name: str) -> logging.Logger:
    """
    Get a rank-aware logger with automatic filtering.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger with rank-aware filtering
    """
    logger = logging.getLogger(name)

    # Apply global log level if configured
    if _GLOBAL_CONFIG_APPLIED:
        logger.setLevel(_GLOBAL_LOG_LEVEL)

    # Check if we already have a rank-aware filter
    has_rank_filter = any(isinstance(f, RankAwareFilter) for f in logger.filters)

    if not has_rank_filter:
        # Add rank-aware filter
        rank_filter = RankAwareFilter()
        logger.addFilter(rank_filter)

        # If logger has no handlers, add a basic one
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            handler.setLevel(
                _GLOBAL_LOG_LEVEL if _GLOBAL_CONFIG_APPLIED else logging.INFO
            )
            logger.addHandler(handler)
            # Prevent propagation to parent to avoid duplicate messages
            logger.propagate = False

    return logger


def rank0_only(func):
    """
    Decorator to only execute function on rank 0 (main process).

    Args:
        func: Function to decorate

    Returns:
        Decorated function that only runs on rank 0
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return wrapper


def all_ranks(func):
    """
    Decorator to execute function on all ranks (explicit marker).

    This is mainly for documentation purposes to make it clear
    when a function should run on all ranks.

    Args:
        func: Function to decorate

    Returns:
        Original function (no modification)
    """
    return func


def log_distributed_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log distributed training information (rank 0 only).

    Args:
        logger: Logger to use (creates default if None)
    """
    if logger is None:
        logger = get_rank_aware_logger(__name__)

    rank, world_size, is_main = get_rank_info()

    if is_main:
        logger.info("🌐 Distributed Training Information:")
        logger.info(f"   World Size: {world_size}")
        logger.info(f"   Current Rank: {rank}")
        logger.info(f"   Is Main Process: {is_main}")
        logger.info(
            f"   Logging Strategy: INFO/DEBUG on rank 0 only, ERROR/WARNING on all ranks"
        )


def _to_log_level(level: Union[str, int]) -> int:
    """Convert a string or int log level to a valid logging level int."""
    if isinstance(level, str):
        level_upper = level.upper()
        if not hasattr(logging, level_upper):
            raise ValueError(f"Invalid log level string: {level}")
        return getattr(logging, level_upper)
    return int(level)


def configure_rank_aware_logging(
    log_level: Union[str, int] = logging.INFO, format_string: Optional[str] = None
) -> None:
    """
    Configure rank-aware logging for the entire application.

    Args:
        log_level: Logging level to set
        format_string: Custom format string for log messages
    """
    global _GLOBAL_LOG_LEVEL, _GLOBAL_CONFIG_APPLIED

    if format_string is None:
        format_string = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    # Allow environment override if caller passes None explicitly
    if log_level is None:  # type: ignore[truthy-bool]
        env_level = os.getenv("BBU_LOG_LEVEL")
        log_level = env_level if env_level is not None else logging.INFO

    # Convert level to int deterministically
    level_int: int = _to_log_level(log_level)

    # Set global log level state
    _GLOBAL_LOG_LEVEL = level_int
    _GLOBAL_CONFIG_APPLIED = True

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_int)

    # Add rank-aware filter to root logger if not present
    has_rank_filter = any(isinstance(f, RankAwareFilter) for f in root_logger.filters)
    if not has_rank_filter:
        rank_filter = RankAwareFilter()
        root_logger.addFilter(rank_filter)

    # Configure handlers if needed
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        handler.setLevel(level_int)
        root_logger.addHandler(handler)
    else:
        # Ensure all existing root handlers follow the configured level
        for handler in root_logger.handlers:
            handler.setLevel(level_int)

    # Propagate the level change to any existing loggers that already opted-in
    # to rank-aware behavior (i.e., have RankAwareFilter installed)
    for name in logging.Logger.manager.loggerDict:  # type: ignore[attr-defined]
        logger = logging.getLogger(name)
        if hasattr(logger, "filters") and any(
            isinstance(f, RankAwareFilter) for f in logger.filters
        ):
            logger.setLevel(level_int)
            for handler in logger.handlers:
                handler.setLevel(level_int)


def set_global_log_level(log_level: Union[str, int]) -> None:
    """
    Set global log level for all rank-aware loggers.

    Args:
        log_level: Logging level to set (string or int)
    """
    global _GLOBAL_LOG_LEVEL, _GLOBAL_CONFIG_APPLIED

    # Convert level to int deterministically
    level_int: int = _to_log_level(log_level)

    _GLOBAL_LOG_LEVEL = level_int
    _GLOBAL_CONFIG_APPLIED = True

    # Update root logger
    logging.getLogger().setLevel(level_int)

    # Update all existing loggers with rank-aware filters
    for name in logging.Logger.manager.loggerDict:  # type: ignore[attr-defined]
        logger = logging.getLogger(name)
        if hasattr(logger, "filters") and any(
            isinstance(f, RankAwareFilter) for f in logger.filters
        ):
            logger.setLevel(level_int)
            for handler in logger.handlers:
                handler.setLevel(level_int)


# Convenience function to initialize from environment without callers
# needing to manually parse log level.
# Honors BBU_LOG_LEVEL and BBU_LOG_FORMAT if present.


def initialize_logging_from_env() -> None:
    """Initialize rank-aware logging from environment variables.

    Environment variables:
    - BBU_LOG_LEVEL: one of DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive)
    - BBU_LOG_FORMAT: optional logging format string
    """
    level_str = os.getenv("BBU_LOG_LEVEL")
    fmt = os.getenv("BBU_LOG_FORMAT")
    if level_str is None:
        # Fall back to current defaults; still ensure root has our filter/handler
        configure_rank_aware_logging(log_level=_GLOBAL_LOG_LEVEL, format_string=fmt)
    else:
        configure_rank_aware_logging(log_level=level_str, format_string=fmt)


# Convenience function for backward compatibility
def get_logger(name: str) -> logging.Logger:
    """Alias for get_rank_aware_logger for backward compatibility."""
    return get_rank_aware_logger(name)
