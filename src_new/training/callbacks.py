"""
Training callbacks for Qwen2.5-VL.

This module provides callbacks for training including loss tracking and
best checkpoint creation with descriptive naming.

Key Components:
- LossTracker: Tracks loss components with moving averages
- BestCheckpointCallback: Creates best-{step}-{eval_loss} checkpoints
"""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
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
        main_loss = self._extract_loss_value(loss_components.loss if hasattr(loss_components, "loss") else None)
        llm_loss = self._extract_loss_value(loss_components.llm_loss if hasattr(loss_components, "llm_loss") else None)
        coordinate_loss = self._extract_loss_value(
            loss_components.coordinate_loss if hasattr(loss_components, "coordinate_loss") else None
        )
        teacher_loss = self._extract_loss_value(
            loss_components.teacher_loss if hasattr(loss_components, "teacher_loss") else None
        )
        student_loss = self._extract_loss_value(
            loss_components.student_loss if hasattr(loss_components, "student_loss") else None
        )

        # Extract granular teacher-student loss components
        teacher_llm_loss = self._extract_loss_value(
            loss_components.teacher_llm_loss if hasattr(loss_components, "teacher_llm_loss") else None
        )
        teacher_l1_loss = self._extract_loss_value(
            loss_components.teacher_l1_loss if hasattr(loss_components, "teacher_l1_loss") else None
        )
        student_llm_loss = self._extract_loss_value(
            loss_components.student_llm_loss if hasattr(loss_components, "student_llm_loss") else None
        )
        student_l1_loss = self._extract_loss_value(
            loss_components.student_l1_loss if hasattr(loss_components, "student_l1_loss") else None
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


class BestCheckpointCallback(TrainerCallback):
    """
    DEPRECATED: This callback has been replaced by unified checkpoint management in BBUTrainer.

    The functionality of this callback is now integrated directly into BBUTrainer's
    _save_checkpoint() method, eliminating redundant I/O operations and ensuring
    consistency between regular and best checkpoints.

    Migration: Remove this callback from your training setup. BBUTrainer now handles
    best checkpoint creation automatically using UnifiedCheckpointManager.
    """

    def __init__(self):
        import warnings

        warnings.warn(
            "BestCheckpointCallback is deprecated and will be removed in a future version. "
            "Best checkpoint functionality is now integrated into BBUTrainer. "
            "Remove this callback from your training setup.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.current_best_dir = None
        self.current_best_metric = None
        self.logger = get_callback_logger()

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after checkpoint save to check if we have a new best model."""
        # Only process on main process to avoid race conditions
        if not args.should_save:
            return

        # Get current eval_loss from the most recent log entry
        current_eval_loss = self._get_latest_eval_loss(state)
        if current_eval_loss is None:
            return

        # Check if this is a new best (lower eval_loss is better)
        if (
            self.current_best_metric is None
            or current_eval_loss < self.current_best_metric
        ):
            # Get current step
            current_step = state.global_step

            # Find the checkpoint directory that was just saved
            if not args.output_dir:
                return  # No output directory specified
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{current_step}")

            if os.path.exists(checkpoint_dir):
                self._create_best_checkpoint_copy(
                    args.output_dir, checkpoint_dir, current_step, current_eval_loss
                )
                self.current_best_metric = current_eval_loss

    def _get_latest_eval_loss(self, state: TrainerState) -> Optional[float]:
        """Extract the most recent eval_loss from trainer state."""
        if not hasattr(state, "log_history") or not state.log_history:
            return None

        # Search backwards through log history for the most recent eval_loss
        for log_entry in reversed(state.log_history):
            if isinstance(log_entry, dict) and "eval_loss" in log_entry:
                return log_entry["eval_loss"]

        return None

    def _create_best_checkpoint_copy(
        self, output_dir: str, source_checkpoint: str, step: int, eval_loss: float
    ):
        """Create or update the copy of the best checkpoint."""
        # Format eval_loss to 4 decimal places for filename
        loss_str = f"{eval_loss:.4f}"
        new_best_name = f"best-{step}-loss{loss_str}"
        new_best_path = os.path.join(output_dir, new_best_name)

        # Skip if this is already the current best directory
        if self.current_best_dir == new_best_path:
            return

        # Remove old best directory if it exists
        if self.current_best_dir and os.path.exists(self.current_best_dir):
            try:
                shutil.rmtree(self.current_best_dir)
                self.logger.info(
                    f"🗑️ Removed old best checkpoint: {os.path.basename(self.current_best_dir)}"
                )
            except OSError as e:
                self.logger.warning(f"Failed to remove old best checkpoint: {e}")

        # Create new best checkpoint copy
        try:
            shutil.copytree(source_checkpoint, new_best_path)
            self.current_best_dir = new_best_path
            self.logger.info(f"📁 Created best checkpoint copy: {new_best_name}")
            self.logger.info(
                f"📊 Best model at step {step} with eval_loss: {eval_loss:.4f}"
            )

        except OSError as e:
            self.logger.error(f"Failed to create best checkpoint copy: {e}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training to report best checkpoint status."""
        if args.should_save and self.current_best_dir:
            self.logger.info(
                f"✅ Training completed. Best checkpoint available at: {os.path.basename(self.current_best_dir)}"
            )


def create_best_checkpoint_callback() -> BestCheckpointCallback:
    """
    DEPRECATED: Factory function for BestCheckpointCallback.

    This function is deprecated because BestCheckpointCallback has been replaced
    by unified checkpoint management in BBUTrainer.

    Raises:
        NotImplementedError: Always raised with migration instructions
    """
    raise NotImplementedError(
        "BestCheckpointCallback has been deprecated and replaced by unified checkpoint management. "
        "Remove this callback from your training setup. BBUTrainer now handles best checkpoint "
        "creation automatically using UnifiedCheckpointManager. "
        "No code changes are required - just remove the callback from your trainer initialization."
    )
