"""
Data processing components for Qwen2.5-VL training.

This module provides dataset, collator, and teacher management components
for the Qwen2.5-VL training pipeline.
"""

from typing import TYPE_CHECKING

# Re-export core components
from src_new_json.data.collator import create_data_collator
from src_new_json.data.dataset import Dataset
from src_new_json.data.teacher_pool import TeacherPoolManager


if TYPE_CHECKING:
    # Type-only imports
    pass

__all__ = [
    "Dataset",
    "create_data_collator",
    "TeacherPoolManager",
]
