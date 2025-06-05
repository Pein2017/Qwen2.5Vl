"""
Global logging configuration for Qwen2.5-VL BBU training.
Provides consistent logging across all modules with centralized configuration.
Environment variables are set by the launcher script.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class GlobalLoggerManager:
    """
    Centralized logger manager that ensures consistent logging across all modules.

    Features:
    - Single point of logger configuration
    - Consistent formatting across modules
    - Automatic log directory creation
    - Support for distributed training (rank 0 only console logging)
    - Module-specific loggers with global configuration
    """

    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._loggers = {}
            self._configured = False

    def configure(
        self,
        log_dir: str,
        log_level: str,
        verbose: bool,
        is_training: bool,
        console_level: Optional[str] = None,
    ):
        """
        Configure global logging settings.

        Args:
            log_dir: Directory for log files
            log_level: Global log level (DEBUG, INFO, WARNING, ERROR)
            verbose: Enable verbose logging
            is_training: Whether this is a training session
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

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

        # File handler (always enabled)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        # Console handler (rank 0 only in distributed training)
        if self._should_log_to_console():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, console_level.upper()))
            if verbose:
                console_handler.setFormatter(detailed_formatter)
            else:
                console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)

        self._configured = True

        # Log configuration
        config_logger = self.get_logger("config")
        config_logger.info("🔧 Global logging configured:")
        config_logger.info(f"   Log directory: {log_path}")
        config_logger.info(f"   Log file: {log_file}")
        config_logger.info(f"   Log level: {log_level}")
        config_logger.info(f"   Console level: {console_level}")
        config_logger.info(f"   Verbose: {verbose}")
        config_logger.info(f"   Training mode: {is_training}")
        if self.is_distributed:
            config_logger.info(
                f"   Distributed: rank {self._get_rank()}/{self._get_world_size()}"
            )

    def get_logger(self, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the given name.

        Args:
            name: Logger name
            log_file: Optional specific log file for this logger

        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)

        # If a specific log file is requested, add a file handler
        if log_file and self._configured:
            log_path = Path("logs") / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self._loggers[name] = logger
        return logger

    def get_raw_data_logger(self) -> logging.Logger:
        """Get logger for raw data samples."""
        return self.get_logger("raw_data", "raw_samples.log")

    def get_training_logger(self) -> logging.Logger:
        """Get logger for training events."""
        return self.get_logger("training")

    def get_model_logger(self) -> logging.Logger:
        """Get logger for model operations."""
        return self.get_logger("model")

    def get_data_logger(self) -> logging.Logger:
        """Get logger for data processing."""
        return self.get_logger("data")

    def _get_rank(self) -> int:
        """Get distributed training rank."""
        # Try multiple environment variables for rank detection
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
        # Try multiple environment variables for world size detection
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

    def log_all_ranks(self, logger_name: str, level: str, message: str):
        """Log message from all ranks (for debugging)."""
        logger = self.get_logger(logger_name)
        rank = self._get_rank()
        getattr(logger, level.lower())(f"[Rank {rank}] {message}")

    def force_console_log(self, logger_name: str, level: str, message: str):
        """Force console logging regardless of rank (for critical messages)."""
        logger = self.get_logger(logger_name)
        # Temporarily add console handler if not present
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        getattr(logger, level.lower())(message)
        logger.removeHandler(console_handler)

    def _should_log_to_console(self) -> bool:
        """Determine if this process should log to console."""
        # Only rank 0 logs to console in distributed training
        return self._get_rank() == 0


# Global logger manager instance
_global_logger_manager = GlobalLoggerManager()


def configure_global_logging(
    log_dir: str,
    log_level: str,
    verbose: bool,
    is_training: bool,
    console_level: Optional[str] = None,
):
    """
    Configure global logging settings.

    Args:
        log_dir: Directory for log files
        log_level: Global log level (DEBUG, INFO, WARNING, ERROR)
        verbose: Enable verbose logging
        is_training: Whether this is a training session
        console_level: Console log level (defaults to log_level)
    """
    _global_logger_manager.configure(
        log_dir=log_dir,
        log_level=log_level,
        verbose=verbose,
        is_training=is_training,
        console_level=console_level,
    )


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Get or create a logger with the given name."""
    return _global_logger_manager.get_logger(name, log_file)


def get_raw_data_logger() -> logging.Logger:
    """Get logger for raw data samples."""
    return _global_logger_manager.get_raw_data_logger()


def get_training_logger() -> logging.Logger:
    """Get logger for training events."""
    return _global_logger_manager.get_training_logger()


def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return _global_logger_manager.get_model_logger()


def get_data_logger() -> logging.Logger:
    """Get logger for data processing."""
    return _global_logger_manager.get_data_logger()


def get_raw_data_logger_legacy():
    """Legacy function for compatibility."""
    return get_raw_data_logger()
