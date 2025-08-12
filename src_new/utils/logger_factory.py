#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized logger factory for Qwen2.5-VL project.

This module provides a single, consistent way to create loggers across the entire
codebase, eliminating duplicate logger initialization patterns and ensuring
consistent logging behavior.

Key Features:
- Centralized logger creation with consistent fallback behavior
- Rank-aware logging support for distributed training
- Consistent formatter and handler configuration
- Fail-fast approach with clear error messages
- Integration with existing logging infrastructure

Usage:
    from src_new.utils.logger_factory import get_module_logger

    logger = get_module_logger(__name__)
    logger.info("This will use rank-aware logging if available")
"""

import logging
from typing import Set


# Global state for tracking configured loggers
_CONFIGURED_LOGGERS: Set[str] = set()
_GLOBAL_LOG_LEVEL: int = logging.INFO
_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def get_module_logger(module_name: str, force_debug: bool = False) -> logging.Logger:
    """
    Get a properly configured logger for a module.

    This is the main entry point for all logger creation in the project.
    It provides consistent behavior with automatic fallback handling.

    Args:
        module_name: Name of the module (typically __name__)
        force_debug: If True, force DEBUG level regardless of global settings

    Returns:
        Configured logger instance

    Raises:
        RuntimeError: If logger configuration fails completely
    """
    try:
        # First try: Use rank-aware logging (preferred)
        return _create_rank_aware_logger(module_name, force_debug)
    except ImportError:
        # Second try: Use config system fallback
        try:
            return _create_config_system_logger(module_name, force_debug)
        except ImportError:
            # Final fallback: Standard logging
            try:
                return _create_standard_logger(module_name, force_debug)
            except Exception as e:
                # Complete failure - this should not happen in normal operation
                raise RuntimeError(
                    f"Failed to create logger for module '{module_name}': {e}"
                ) from e
    except Exception as e:
        # Complete failure - this should not happen in normal operation
        raise RuntimeError(
            f"Failed to create logger for module '{module_name}': {e}"
        ) from e


def _create_rank_aware_logger(
    module_name: str, force_debug: bool = False
) -> logging.Logger:
    """Create logger using rank-aware logging system."""
    from .rank_aware_logging import get_rank_aware_logger

    logger = get_rank_aware_logger(module_name)

    if force_debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    return logger


def _create_config_system_logger(
    module_name: str, force_debug: bool = False
) -> logging.Logger:
    """Create logger using centralized config system."""
    from ..config.config import _CONFIGURED_LOGGERS as config_loggers
    from ..config.config import _GLOBAL_LOG_LEVEL as config_level

    logger = logging.getLogger(module_name)

    # Only configure if not already configured
    if module_name not in config_loggers and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set level
        level = logging.DEBUG if force_debug else config_level
        logger.setLevel(level)
        handler.setLevel(level)

        config_loggers.add(module_name)

    return logger


def _create_standard_logger(
    module_name: str, force_debug: bool = False
) -> logging.Logger:
    """Create logger using standard Python logging (final fallback)."""
    logger = logging.getLogger(module_name)

    # Only configure if not already configured
    if module_name not in _CONFIGURED_LOGGERS and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set level - use root logger's level or INFO as default
        if force_debug:
            level = logging.DEBUG
        else:
            root_level = logging.getLogger().getEffectiveLevel()
            level = root_level if root_level != logging.NOTSET else _GLOBAL_LOG_LEVEL

        logger.setLevel(level)
        handler.setLevel(level)

        _CONFIGURED_LOGGERS.add(module_name)

    return logger


def reconfigure_logger(logger: logging.Logger, force_debug: bool = False) -> None:
    """
    Reconfigure an existing logger with current global settings.

    Args:
        logger: Logger instance to reconfigure
        force_debug: If True, force DEBUG level
    """
    if force_debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    logger.info("Logger reconfigured with current global settings")
    logger.debug(f"Logger level: {logger.level}")
    logger.debug(f"Effective level: {logger.getEffectiveLevel()}")


def set_global_log_level(level: int) -> None:
    """
    Set global log level for future logger creation.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    global _GLOBAL_LOG_LEVEL
    _GLOBAL_LOG_LEVEL = level


def get_configured_loggers() -> Set[str]:
    """Get set of module names that have been configured by this factory."""
    return _CONFIGURED_LOGGERS.copy()


def reset_logger_state() -> None:
    """Reset global logger state (primarily for testing)."""
    global _CONFIGURED_LOGGERS, _GLOBAL_LOG_LEVEL
    _CONFIGURED_LOGGERS.clear()
    _GLOBAL_LOG_LEVEL = logging.INFO


# Convenience functions for common use cases
def get_debug_logger(module_name: str) -> logging.Logger:
    """Get a logger with DEBUG level forced on."""
    return get_module_logger(module_name, force_debug=True)


def get_training_logger(module_name: str) -> logging.Logger:
    """Get a logger specifically for training modules."""
    return get_module_logger(f"training.{module_name}")


def get_inference_logger(module_name: str) -> logging.Logger:
    """Get a logger specifically for inference modules."""
    return get_module_logger(f"inference.{module_name}")


def get_processing_logger(module_name: str) -> logging.Logger:
    """Get a logger specifically for processing modules."""
    return get_module_logger(f"processing.{module_name}")


# Export public API
__all__ = [
    "get_module_logger",
    "reconfigure_logger",
    "set_global_log_level",
    "get_configured_loggers",
    "reset_logger_state",
    "get_debug_logger",
    "get_training_logger",
    "get_inference_logger",
    "get_processing_logger",
]
