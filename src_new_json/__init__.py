"""
Qwen2.5-VL Training Pipeline - New Architecture

Simplified, maintainable architecture for Qwen2.5-VL vision-language
object detection and captioning training.

This package provides:
- Unified configuration management
- Streamlined data processing
- Modular training components
- Enhanced maintainability

Key Features:
- Direct YAML configuration compatibility
- Fail-fast validation and error handling
- Comprehensive type annotations
- Clean separation of concerns
"""

from typing import TYPE_CHECKING


__version__ = "2.0.0"
__author__ = "Qwen2.5-VL Team"

# Initialize rank-aware logging as early as possible so all modules inherit filters/handlers
try:
    from src_new_json.utils.rank_aware_logging import initialize_logging_from_env

    initialize_logging_from_env()
except Exception:
    # Defer to consumers if rank-aware logging utilities are unavailable during early import
    pass

# Core modules
from src_new_json.config import Config, load_config
from src_new_json.data import Dataset, TeacherPoolManager, create_data_collator


if TYPE_CHECKING:
    # Type-only imports
    pass

__all__ = [
    "Config",
    "load_config",
    "Dataset",
    "create_data_collator",
    "TeacherPoolManager",
]
