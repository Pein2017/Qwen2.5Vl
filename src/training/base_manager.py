"""
Base Manager for Training Components

This module provides a common base class for all training managers,
establishing consistent patterns for initialization, logging, configuration
access, and error handling across the training system.

Key Features:
- Standardized initialization patterns
- Centralized logging configuration
- Common configuration access methods
- Consistent error handling and validation
- Shared utility methods for all managers
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.logger_utils import get_training_logger


class BaseManager(ABC):
    """
    Abstract base class for all training managers.

    Provides common functionality and patterns that all managers
    should follow for consistency and maintainability.
    """

    def __init__(
        self,
        config: Any,
        logger: Optional[Any] = None,
    ):
        """
        Initialize base manager with common components.

        Args:
            config: Configuration object (domain-specific type)
            logger: Optional logger instance
        """
        if config is None:
            raise ValueError("config is required for all managers")

        self.config = config
        self.logger = logger or get_training_logger()

        # Validate configuration during initialization
        self._validate_configuration()

        # Initialize manager-specific state
        self._initialize_manager_state()

        self.logger.debug(f"✅ {self.__class__.__name__} initialized successfully")

    @abstractmethod
    def _validate_configuration(self) -> None:
        """
        Validate manager-specific configuration.

        Subclasses must implement this to validate their specific
        configuration requirements.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def _initialize_manager_state(self) -> None:
        """
        Initialize manager-specific state.

        Subclasses must implement this to set up their internal
        state after configuration validation.
        """
        pass

    def get_manager_info(self) -> Dict[str, Any]:
        """
        Get basic information about this manager.

        Returns:
            Dictionary containing manager information
        """
        return {
            "manager_type": self.__class__.__name__,
            "config_type": type(self.config).__name__,
            "initialized": True,
        }

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message with manager context."""
        manager_name = self.__class__.__name__
        self.logger.debug(f"[{manager_name}] {message}", **kwargs)

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message with manager context."""
        manager_name = self.__class__.__name__
        self.logger.info(f"[{manager_name}] {message}", **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message with manager context."""
        manager_name = self.__class__.__name__
        self.logger.warning(f"[{manager_name}] {message}", **kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        """Log error message with manager context."""
        manager_name = self.__class__.__name__
        self.logger.error(f"[{manager_name}] {message}", **kwargs)

    def validate_required_attribute(
        self, attr_name: str, expected_type: type = None
    ) -> None:
        """
        Validate that a required attribute exists in configuration.

        Args:
            attr_name: Name of the required attribute
            expected_type: Optional expected type for the attribute

        Raises:
            ValueError: If attribute is missing or wrong type
        """
        if not hasattr(self.config, attr_name):
            raise ValueError(
                f"{self.__class__.__name__} requires '{attr_name}' in configuration"
            )

        if expected_type is not None:
            attr_value = getattr(self.config, attr_name)
            if not isinstance(attr_value, expected_type):
                raise ValueError(
                    f"Configuration attribute '{attr_name}' must be {expected_type.__name__}, "
                    f"got {type(attr_value).__name__}"
                )

    def safe_get_config_value(self, attr_name: str, default_value: Any = None) -> Any:
        """
        Safely get a configuration value with fallback.

        Args:
            attr_name: Name of the configuration attribute
            default_value: Default value if attribute doesn't exist

        Returns:
            Configuration value or default
        """
        return getattr(self.config, attr_name, default_value)

    def reset_manager_state(self) -> None:
        """
        Reset manager to initial state.

        Default implementation calls _initialize_manager_state(),
        but subclasses can override for custom reset behavior.
        """
        self.log_debug("Resetting manager state")
        self._initialize_manager_state()
