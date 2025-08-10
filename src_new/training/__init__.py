"""
Training module for Qwen2.5-VL detection system.

This module provides HuggingFace Trainer extensions with local loss aggregation
that eliminates NCCL timeout issues in distributed training.

Key Components:
- BBUTrainer: Clean trainer with local loss aggregation (recommended)
- TrainingStateManager: Local loss component aggregation and metrics
- LossTracker: Component-wise loss tracking and averaging
"""

from .bbu_trainer import BBUTrainer
from .callbacks import LossTracker
from .training_state_manager import TrainingStateManager
from .utils import (
    apply_gradient_scaling,
    compute_memory_usage,
    prepare_model_for_training,
)


__all__ = [
    # Recommended (new architecture)
    "BBUTrainer",
    "TrainingStateManager",
    "LossTracker",
    # Utilities
    "prepare_model_for_training",
    "compute_memory_usage",
    "apply_gradient_scaling",
]
