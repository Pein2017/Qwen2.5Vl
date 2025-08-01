"""
Training State Manager for src_new architecture.

This module provides local loss component aggregation and training state management
without distributed operations, following the proven pattern from src/ that eliminates
NCCL timeout issues.

Key Features:
- Local loss component aggregation (no distributed operations)
- Training metrics collection and formatting
- Integration with DetectionModel loss components
- Clean separation from HuggingFace distributed mechanisms
"""

import time
from typing import Any, Dict, Optional

import torch


class TrainingStateManager:
    """
    Simplified training state manager for local loss aggregation.

    This manager provides local loss component tracking and aggregation without
    any distributed operations, following the proven pattern from src/ that
    eliminates NCCL timeout conflicts.
    """

    def __init__(
        self,
        config: Any,
        model: Any,
        logger: Optional[Any] = None,
    ):
        """
        Initialize training state manager.

        Args:
            config: Training configuration object
            model: The detection model
            logger: Optional logger instance
        """
        self.config = config
        self.model = model
        self.logger = logger

        # Loss component tracking
        self._loss_components_accumulator = {}
        self._loss_components_count = 0

        # Training state
        self._step_count = 0
        self._last_log_time = time.time()

    def accumulate_loss_components(self, loss_components):
        """
        Accumulate loss components locally (no distributed operations).

        Args:
            loss_components: LossComponents dataclass or dictionary of loss component tensors
        """
        # Handle LossComponents dataclass
        if hasattr(loss_components, "__dataclass_fields__"):
            # Convert dataclass to dictionary
            loss_dict = {}
            for field_name in loss_components.__dataclass_fields__:
                value = getattr(loss_components, field_name, None)
                if value is not None:
                    loss_dict[field_name] = value
        elif hasattr(loss_components, "items"):
            # Handle dictionary
            loss_dict = loss_components
        else:
            # Handle other types by trying to extract attributes
            loss_dict = {}
            for attr_name in [
                "loss",
                "llm_loss",
                "coordinate_loss",
                "teacher_loss",
                "student_loss",
                "teacher_llm_loss",
                "teacher_l1_loss",
                "student_llm_loss",
                "student_l1_loss",
            ]:
                value = getattr(loss_components, attr_name, None)
                if value is not None:
                    loss_dict[attr_name] = value

        # Accumulate the loss components
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                # Convert to float for accumulation
                value_float = (
                    value.item() if value.numel() == 1 else value.mean().item()
                )
            else:
                value_float = float(value)

            if key not in self._loss_components_accumulator:
                self._loss_components_accumulator[key] = 0.0

            self._loss_components_accumulator[key] += value_float

        self._loss_components_count += 1

    def get_averaged_losses_and_reset(self) -> Dict[str, float]:
        """
        Get averaged loss components and reset accumulators.

        Returns:
            Dictionary of averaged loss components
        """
        if self._loss_components_count == 0:
            return {}

        # Compute local averages
        averaged_losses = {}
        for key, accumulated_value in self._loss_components_accumulator.items():
            averaged_losses[key] = accumulated_value / self._loss_components_count

        # Reset accumulators
        self._loss_components_accumulator.clear()
        self._loss_components_count = 0

        return averaged_losses

    def log_training_metrics(
        self,
        tr_loss: torch.Tensor,
        grad_norm: Optional[torch.Tensor],
        model: Any,
        start_time: float,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Generate training metrics for logging (local only, no distributed operations).

        Args:
            tr_loss: Training loss tensor
            grad_norm: Gradient norm tensor
            model: The model being trained
            start_time: Training start time
            learning_rate: Current learning rate

        Returns:
            Dictionary of metrics for logging
        """
        # Get averaged loss components
        component_logs = self.get_averaged_losses_and_reset()

        # Build comprehensive logs dictionary
        logs = {}

        # Add main loss
        if torch.is_tensor(tr_loss):
            logs["loss"] = (
                tr_loss.item() if tr_loss.numel() == 1 else tr_loss.mean().item()
            )
        else:
            logs["loss"] = float(tr_loss)

        # Add loss components
        logs.update(component_logs)

        # Add gradient norm
        if grad_norm is not None:
            if torch.is_tensor(grad_norm):
                logs["grad_norm"] = (
                    grad_norm.item()
                    if grad_norm.numel() == 1
                    else grad_norm.mean().item()
                )
            else:
                logs["grad_norm"] = float(grad_norm)

        # Add learning rate
        if learning_rate is not None:
            logs["learning_rate"] = float(learning_rate)

        # Add timing information
        current_time = time.time()
        logs["train_runtime"] = current_time - start_time
        logs["train_samples_per_second"] = (
            self._step_count / (current_time - start_time)
            if (current_time - start_time) > 0
            else 0.0
        )

        # Update step count
        self._step_count += 1

        return logs

    def log_metrics_batch(
        self,
        logs: Dict[str, float],
        lr_scheduler: Any = None,
        args: Optional[Any] = None,
        start_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Process metrics batch for logging with learning rate information.

        Args:
            logs: Base metrics dictionary
            lr_scheduler: Learning rate scheduler
            args: Training arguments
            start_time: Optional start time

        Returns:
            Processed metrics dictionary
        """
        # Remove generic learning rate from logs (we'll add specific ones)
        logs.pop("learning_rate", None)

        # Add learning rate information if scheduler is available
        if lr_scheduler is not None:
            try:
                # Get current learning rates from scheduler
                current_lrs = lr_scheduler.get_last_lr()
                if current_lrs:
                    # Add primary learning rate
                    logs["learning_rate"] = current_lrs[0]

                    # Add meaningful group-specific learning rates following src/ pattern
                    if len(current_lrs) > 1:
                        # Define meaningful group names based on common parameter groups
                        group_names = ["vision", "merger", "llm", "adapter"]

                        for i, lr in enumerate(current_lrs):
                            if i < len(group_names):
                                # Use meaningful names like vision_lr, merger_lr, llm_lr
                                logs[f"{group_names[i]}_lr"] = lr
                            else:
                                # Fallback for additional groups
                                logs[f"learning_rate_group_{i}"] = lr
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to get learning rates: {e}")

        # Add timing information if start_time provided
        if start_time is not None:
            current_time = time.time()
            logs["train_runtime"] = current_time - start_time

        return logs

    def reset_metrics_state(self):
        """Reset metrics state after logging."""
        # Clear any remaining accumulators
        self._loss_components_accumulator.clear()
        self._loss_components_count = 0

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            "step_count": self._step_count,
            "coordinate_tokens_enabled": getattr(
                self.config, "coordinate_tokens_enabled", False
            ),
            "accumulated_components": len(self._loss_components_accumulator),
            "components_count": self._loss_components_count,
        }
