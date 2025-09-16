#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common imports module for Qwen2.5-VL project.

This module centralizes frequently used imports across the codebase to reduce
redundancy and provide a single place to manage common dependencies.

Key Features:
- Centralized standard library imports
- Common third-party library imports
- Project-specific utility imports
- Conditional imports with availability checks
- Type checking imports

Usage:
    # Import everything commonly needed
    from src_new_json.utils.common_imports import *

    # Or import specific groups
    from src_new_json.utils.common_imports import (
        # Standard library
        logging, Path, Dict, List, Optional,
        # Third-party
        torch, Image,
        # Project utilities
        get_module_logger
    )
"""

# === Standard Library Imports ===
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union


# === Third-Party Library Imports ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Tensor = None

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedTokenizer = None
    PreTrainedModel = None
    TrainingArguments = None
    Trainer = None
    Qwen2_5_VLForConditionalGeneration = None

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

# === Project-Specific Imports ===
from .error_formatting import ErrorMessageBuilder, format_validation_error
from .rank_aware_logging import get_logger as get_module_logger
from .validation import PathValidationError, PathValidator, ValidationError


# === Type Checking Imports ===
if TYPE_CHECKING:
    pass


# === Availability Checks ===
def check_torch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE


def check_pil_available() -> bool:
    """Check if PIL is available."""
    return PIL_AVAILABLE


def check_transformers_available() -> bool:
    """Check if Transformers is available."""
    return TRANSFORMERS_AVAILABLE


def check_numpy_available() -> bool:
    """Check if NumPy is available."""
    return NUMPY_AVAILABLE


def check_yaml_available() -> bool:
    """Check if PyYAML is available."""
    return YAML_AVAILABLE


def require_torch() -> None:
    """Require PyTorch to be available, raise ImportError if not."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required but not available. Install with: pip install torch"
        )


def require_pil() -> None:
    """Require PIL to be available, raise ImportError if not."""
    if not PIL_AVAILABLE:
        raise ImportError(
            "PIL is required but not available. Install with: pip install Pillow"
        )


def require_transformers() -> None:
    """Require Transformers to be available, raise ImportError if not."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Transformers is required but not available. "
            "Install with: pip install transformers"
        )


def require_numpy() -> None:
    """Require NumPy to be available, raise ImportError if not."""
    if not NUMPY_AVAILABLE:
        raise ImportError(
            "NumPy is required but not available. Install with: pip install numpy"
        )


def require_yaml() -> None:
    """Require PyYAML to be available, raise ImportError if not."""
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required but not available. Install with: pip install PyYAML"
        )


# === Common Constants ===
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# Common file extensions
JSONL_EXTENSIONS = [".jsonl", ".json"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
CONFIG_EXTENSIONS = [".yaml", ".yml", ".json"]

# === Common Type Aliases ===
PathLike = Union[str, Path]
TensorLike = (
    Union[torch.Tensor, np.ndarray] if TORCH_AVAILABLE and NUMPY_AVAILABLE else Any
)
ConfigDict = Dict[str, Any]
SampleDict = Dict[str, Any]


# === Utility Functions ===
def ensure_path(path: PathLike) -> Path:
    """Ensure a path-like object is a Path instance."""
    return Path(path) if not isinstance(path, Path) else path


def ensure_list(item: Union[Any, List[Any]]) -> List[Any]:
    """Ensure an item is a list."""
    return item if isinstance(item, list) else [item]


def safe_import(module_name: str, package: Optional[str] = None):
    """Safely import a module, returning None if not available."""
    try:
        if package:
            return __import__(f"{package}.{module_name}", fromlist=[module_name])
        else:
            return __import__(module_name)
    except ImportError:
        return None


def get_available_device() -> str:
    """Get the best available device (cuda, mps, or cpu)."""
    if not TORCH_AVAILABLE:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_logging(level: Union[str, int] = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Setup logging using centralized rank-aware configuration.

    This avoids basicConfig to prevent duplicate handlers across ranks.
    """
    try:
        from .rank_aware_logging import configure_rank_aware_logging, get_rank_aware_logger

        # Normalize level
        if isinstance(level, str):
            level_norm: Union[str, int] = level.upper()
        else:
            level_norm = int(level)

        configure_rank_aware_logging(log_level=level_norm)
        return get_rank_aware_logger(__name__)
    except Exception:
        # Fallback to minimal standard logging without adding multiple handlers
        if isinstance(level, str):
            level_int = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
        else:
            level_int = int(level)
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
            handler.setFormatter(formatter)
            handler.setLevel(level_int)
            logger.addHandler(handler)
            logger.setLevel(level_int)
            logger.propagate = False
        return logger


# === Export Control ===
# Standard library
__all__ = [
    # Standard library
    "json",
    "logging",
    "os",
    "sys",
    "warnings",
    "dataclass",
    "Path",
    "Any",
    "Dict",
    "List",
    "Optional",
    "Tuple",
    "Union",
    "TYPE_CHECKING",
    # Third-party (conditional)
    "torch",
    "nn",
    "F",
    "Tensor",
    "Image",
    "PreTrainedTokenizer",
    "PreTrainedModel",
    "TrainingArguments",
    "Trainer",
    "Qwen2_5_VLForConditionalGeneration",
    "np",
    "yaml",
    # Project utilities
    "get_module_logger",  # alias to rank-aware get_logger
    "PathValidator",
    "ValidationError",
    "PathValidationError",
    "ErrorMessageBuilder",
    "format_validation_error",
    # Availability checks
    "check_torch_available",
    "check_pil_available",
    "check_transformers_available",
    "check_numpy_available",
    "check_yaml_available",
    "require_torch",
    "require_pil",
    "require_transformers",
    "require_numpy",
    "require_yaml",
    # Constants
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FORMAT",
    "JSONL_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "CONFIG_EXTENSIONS",
    # Type aliases
    "PathLike",
    "TensorLike",
    "ConfigDict",
    "SampleDict",
    # Utility functions
    "ensure_path",
    "ensure_list",
    "safe_import",
    "get_available_device",
    "setup_logging",
    # Availability flags
    "TORCH_AVAILABLE",
    "PIL_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
    "NUMPY_AVAILABLE",
    "YAML_AVAILABLE",
]
