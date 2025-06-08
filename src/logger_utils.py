"""
Unified logging system for Qwen2.5-VL training.
Provides a single logger instance with module-specific prefixes for clean, consistent logging.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class UnifiedLogger:
    """
    Unified logger that provides a single logger instance with module-specific prefixes.

    Features:
    - Single logger instance for all modules
    - Module-specific prefixes for easy identification
    - All logs go to the same file
    - Consistent formatting across all modules
    - Distributed training support (rank 0 only console logging)
    """

    _instance = None
    _logger = None
    _configured = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True

    def configure(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        verbose: bool = False,
        console_level: Optional[str] = None,
    ):
        """
        Configure the unified logging system.

        Args:
            log_dir: Directory for log files
            log_level: Global log level (DEBUG, INFO, WARNING, ERROR)
            verbose: Enable verbose logging
            console_level: Console log level (defaults to log_level)
        """
        if self._configured:
            return

        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Set console level
        if console_level is None:
            console_level = log_level

        # Create the unified logger
        self._logger = logging.getLogger("qwen_unified")
        self._logger.setLevel(self._get_logging_level(log_level))

        # Clear existing handlers
        self._logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

        # File handler (always enabled)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"qwen_training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(self._get_logging_level(log_level))
        file_handler.setFormatter(detailed_formatter)
        self._logger.addHandler(file_handler)

        # Console handler (rank 0 only in distributed training)
        if self._should_log_to_console():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._get_logging_level(console_level))
            if verbose:
                console_handler.setFormatter(detailed_formatter)
            else:
                console_handler.setFormatter(simple_formatter)
            self._logger.addHandler(console_handler)

        self._configured = True

        # Log configuration
        self._logger.info("🔧 Unified logging system configured:")
        self._logger.info(f"   Log directory: {log_path}")
        self._logger.info(f"   Log file: {log_file}")
        self._logger.info(f"   Log level: {log_level}")
        self._logger.info(f"   Console level: {console_level}")
        self._logger.info(f"   Verbose: {verbose}")
        if self.is_distributed:
            self._logger.info(
                f"   Distributed: rank {self._get_rank()}/{self._get_world_size()}"
            )

    def get_logger(self, prefix: str = "main") -> "PrefixedLogger":
        """
        Get a logger with the specified prefix.

        Args:
            prefix: Module prefix for log messages

        Returns:
            PrefixedLogger instance with the specified prefix
        """
        if not self._configured:
            # Auto-configure with defaults if not configured
            self.configure()

        return PrefixedLogger(self._logger, prefix)

    def _get_rank(self) -> int:
        """Get distributed training rank."""
        rank_vars = ["RANK", "LOCAL_RANK", "SLURM_PROCID"]
        for var in rank_vars:
            if var in os.environ:
                try:
                    return int(os.environ[var])
                except ValueError:
                    continue
        return 0

    def _get_world_size(self) -> int:
        """Get distributed training world size."""
        world_size_vars = ["WORLD_SIZE", "SLURM_NTASKS"]
        for var in world_size_vars:
            if var in os.environ:
                try:
                    return int(os.environ[var])
                except ValueError:
                    continue
        return 1

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self._get_world_size() > 1

    def _should_log_to_console(self) -> bool:
        """Determine if this process should log to console."""
        return self._get_rank() == 0

    def _get_logging_level(self, level_str: str) -> int:
        """Convert log level string to logging constant."""
        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level_upper = level_str.upper()
        if level_upper not in level_mapping:
            raise ValueError(
                f"Invalid log level: {level_str}. Supported: {list(level_mapping.keys())}"
            )

        return level_mapping[level_upper]


class PrefixedLogger:
    """
    Logger wrapper that adds a prefix to all log messages.
    """

    def __init__(self, logger: logging.Logger, prefix: str):
        self._logger = logger
        self._prefix = prefix

    def _format_message(self, message: str) -> str:
        """Format message with prefix."""
        return f"[{self._prefix}] {message}"

    def debug(self, message: str):
        """Log debug message with prefix."""
        self._logger.debug(self._format_message(message))

    def info(self, message: str):
        """Log info message with prefix."""
        self._logger.info(self._format_message(message))

    def warning(self, message: str):
        """Log warning message with prefix."""
        self._logger.warning(self._format_message(message))

    def error(self, message: str):
        """Log error message with prefix."""
        self._logger.error(self._format_message(message))

    def critical(self, message: str):
        """Log critical message with prefix."""
        self._logger.critical(self._format_message(message))

    # Aliases for compatibility
    warn = warning


# Global unified logger instance
_unified_logger = UnifiedLogger()


def configure_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    verbose: bool = False,
    console_level: Optional[str] = None,
    **kwargs,
):
    """
    Configure the unified logging system.

    Args:
        log_dir: Directory for log files
        log_level: Global log level (DEBUG, INFO, WARNING, ERROR)
        verbose: Enable verbose logging
        console_level: Console log level (defaults to log_level)
    """
    _unified_logger.configure(
        log_dir=log_dir,
        log_level=log_level,
        verbose=verbose,
        console_level=console_level,
    )


def get_logger(prefix: str = "main") -> PrefixedLogger:
    """
    Get a logger with the specified prefix.

    Args:
        prefix: Module prefix for log messages (e.g., "training", "data", "model")

    Returns:
        PrefixedLogger instance with the specified prefix
    """
    return _unified_logger.get_logger(prefix)


# Convenience functions for common prefixes
def get_training_logger() -> PrefixedLogger:
    """Get logger for training operations."""
    return get_logger("training")


def get_model_logger() -> PrefixedLogger:
    """Get logger for model operations."""
    return get_logger("model")


def get_data_logger() -> PrefixedLogger:
    """Get logger for data processing."""
    return get_logger("data")


def get_config_logger() -> PrefixedLogger:
    """Get logger for configuration."""
    return get_logger("config")


def get_inference_logger() -> PrefixedLogger:
    """Get logger for inference operations."""
    return get_logger("inference")


def get_loss_logger() -> PrefixedLogger:
    """Get logger for loss computation."""
    return get_logger("loss")


def get_attention_logger() -> PrefixedLogger:
    """Get logger for attention operations."""
    return get_logger("attention")


def get_monitor_logger() -> PrefixedLogger:
    """Get logger for monitoring operations."""
    return get_logger("monitor")


def get_stability_logger() -> PrefixedLogger:
    """Get logger for stability monitoring."""
    return get_logger("stability")


def get_callback_logger() -> PrefixedLogger:
    """Get logger for callback operations."""
    return get_logger("callback")


def get_chat_logger() -> PrefixedLogger:
    """Get logger for chat processing."""
    return get_logger("chat")


def get_utils_logger() -> PrefixedLogger:
    """Get logger for utility functions."""
    return get_logger("utils")


def get_diagnostics_logger() -> PrefixedLogger:
    """Get logger for diagnostics."""
    return get_logger("diagnostics")


def get_tokens_logger() -> PrefixedLogger:
    """Get logger for token operations."""
    return get_logger("tokens")


def get_patches_logger() -> PrefixedLogger:
    """Get logger for model patches."""
    return get_logger("patches")


# Legacy compatibility functions
def get_raw_data_logger() -> PrefixedLogger:
    """Legacy compatibility: Get logger for raw data."""
    return get_logger("raw_data")


def get_debug_logger() -> PrefixedLogger:
    """Legacy compatibility: Get debug logger."""
    return get_logger("debug")


def get_sample_logger() -> PrefixedLogger:
    """Legacy compatibility: Get sample logger."""
    return get_logger("sample")


# Backward compatibility aliases
configure_global_logging = configure_logging
get_raw_data_logger_legacy = get_raw_data_logger
