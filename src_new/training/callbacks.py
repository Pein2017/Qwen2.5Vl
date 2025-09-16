"""
Training callbacks for Qwen2.5-VL.

This module provides callbacks for training including loss tracking and
best checkpoint creation with descriptive naming.

Key Components:
- LossTracker: Tracks loss components with moving averages
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


def get_callback_logger() -> logging.Logger:
    """Get rank-aware logger for callbacks."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("callbacks")
    except ImportError:
        return logging.getLogger("callbacks")


def get_rank_info() -> tuple[int, int]:
    """
    Get current rank and world size for distributed training.

    Returns:
        Tuple of (rank, world_size)
    """
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except (ImportError, RuntimeError):
        pass

    # Fallback for non-distributed training
    return 0, 1


@dataclass
class LossTracker:
    """
    Track and compute moving averages for multi-component loss.

    This class maintains rolling averages of different loss components
    to provide smooth metrics for monitoring training progress.
    """

    # Configuration
    window_size: int = 100

    # Loss component history
    loss_history: Optional[List[float]] = None
    llm_loss_history: Optional[List[float]] = None
    coordinate_loss_history: Optional[List[float]] = None
    teacher_loss_history: Optional[List[float]] = None
    student_loss_history: Optional[List[float]] = None

    # Granular teacher-student loss component history
    teacher_llm_loss_history: Optional[List[float]] = None
    teacher_l1_loss_history: Optional[List[float]] = None
    student_llm_loss_history: Optional[List[float]] = None
    student_l1_loss_history: Optional[List[float]] = None

    def __post_init__(self):
        """Initialize loss history lists."""
        self.loss_history = []
        self.llm_loss_history = []
        self.coordinate_loss_history = []
        self.teacher_loss_history = []
        self.student_loss_history = []

        # Initialize granular teacher-student loss histories
        self.teacher_llm_loss_history = []
        self.teacher_l1_loss_history = []
        self.student_llm_loss_history = []
        self.student_l1_loss_history = []

    def update(self, loss_components: Any) -> None:
        """
        Update loss history with new components.

        Args:
            loss_components: Loss components to track
        """
        # Extract loss components
        main_loss = self._extract_loss_value(
            loss_components.loss if hasattr(loss_components, "loss") else None
        )
        llm_loss = self._extract_loss_value(
            loss_components.llm_loss if hasattr(loss_components, "llm_loss") else None
        )
        coordinate_loss = self._extract_loss_value(
            loss_components.coordinate_loss
            if hasattr(loss_components, "coordinate_loss")
            else None
        )
        teacher_loss = self._extract_loss_value(
            loss_components.teacher_loss
            if hasattr(loss_components, "teacher_loss")
            else None
        )
        student_loss = self._extract_loss_value(
            loss_components.student_loss
            if hasattr(loss_components, "student_loss")
            else None
        )

        # Extract granular teacher-student loss components
        teacher_llm_loss = self._extract_loss_value(
            loss_components.teacher_llm_loss
            if hasattr(loss_components, "teacher_llm_loss")
            else None
        )
        teacher_l1_loss = self._extract_loss_value(
            loss_components.teacher_l1_loss
            if hasattr(loss_components, "teacher_l1_loss")
            else None
        )
        student_llm_loss = self._extract_loss_value(
            loss_components.student_llm_loss
            if hasattr(loss_components, "student_llm_loss")
            else None
        )
        student_l1_loss = self._extract_loss_value(
            loss_components.student_l1_loss
            if hasattr(loss_components, "student_l1_loss")
            else None
        )

        # Update histories
        self._update_history(self.loss_history, main_loss)
        self._update_history(self.llm_loss_history, llm_loss)
        self._update_history(self.coordinate_loss_history, coordinate_loss)
        self._update_history(self.teacher_loss_history, teacher_loss)
        self._update_history(self.student_loss_history, student_loss)

        # Update granular teacher-student loss histories
        self._update_history(self.teacher_llm_loss_history, teacher_llm_loss)
        self._update_history(self.teacher_l1_loss_history, teacher_l1_loss)
        self._update_history(self.student_llm_loss_history, student_llm_loss)
        self._update_history(self.student_l1_loss_history, student_l1_loss)

    def _extract_loss_value(self, loss: Any) -> Optional[float]:
        """
        Extract float value from loss tensor or return None.

        Args:
            loss: Loss value (tensor, float, or None)

        Returns:
            Float value or None
        """
        if loss is None:
            return None

        if torch.is_tensor(loss):
            return loss.detach().item()

        if isinstance(loss, (int, float)):
            return float(loss)

        return None

    def _update_history(self, history: List[float], value: Optional[float]) -> None:
        """
        Update history list with new value.

        Args:
            history: History list to update
            value: New value to add
        """
        if value is not None:
            history.append(value)
            # Keep history within window size
            if len(history) > self.window_size:
                history.pop(0)

    def get_averages(self) -> Dict[str, float]:
        """
        Get moving averages for all loss components.

        Returns:
            Dictionary with loss component averages
        """
        return {
            "loss": self._compute_average(self.loss_history),
            "llm_loss": self._compute_average(self.llm_loss_history),
            "coordinate_loss": self._compute_average(self.coordinate_loss_history),
            "teacher_loss": self._compute_average(self.teacher_loss_history),
            "student_loss": self._compute_average(self.student_loss_history),
            # Granular teacher-student loss components
            "teacher_llm_loss": self._compute_average(self.teacher_llm_loss_history),
            "teacher_l1_loss": self._compute_average(self.teacher_l1_loss_history),
            "student_llm_loss": self._compute_average(self.student_llm_loss_history),
            "student_l1_loss": self._compute_average(self.student_l1_loss_history),
        }

    def _compute_average(self, history: List[float]) -> Optional[float]:
        """
        Compute average of history list.

        Args:
            history: List of values

        Returns:
            Average value or None if empty
        """
        if not history:
            return None
        return sum(history) / len(history)

    def reset(self) -> None:
        """Reset all loss histories."""
        self.loss_history = []
        self.llm_loss_history = []
        self.coordinate_loss_history = []
        self.teacher_loss_history = []
        self.student_loss_history = []

        # Reset granular teacher-student loss histories
        self.teacher_llm_loss_history = []
        self.teacher_l1_loss_history = []
        self.student_llm_loss_history = []
        self.student_l1_loss_history = []


class AugmentationScheduleCallback(TrainerCallback):
    """Switch dataset augmentation preset and rebuild dynamic pairing by epoch via Dataset.set_epoch."""

    def on_epoch_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        trainer = (
            kwargs.get("model")._get_training_trainer()
            if hasattr(kwargs.get("model"), "_get_training_trainer")
            else kwargs.get("trainer")
        )
        dataset = (
            getattr(trainer, "train_dataset", None) if trainer is not None else None
        )
        if dataset is None:
            return
        try:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(int(state.epoch or 0))
        except Exception:
            pass
