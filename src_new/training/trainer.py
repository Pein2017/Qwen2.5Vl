"""
Trainer implementation with minimal HuggingFace Trainer extension.

This module provides a clean extension to HuggingFace Trainer using composition
rather than destructive inheritance. It focuses on:
- Multi-component loss tracking and logging
- Component-wise loss handling (coordinate, teacher, student)
- Memory optimization and gradient scaling
- Clean integration with existing TrainingArguments
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import Trainer as HFTrainer


if TYPE_CHECKING:
    from src_new.models.wrapper import DetectionModel

from .callbacks import LossTracker


@dataclass
class LossComponents:
    """Multi-component loss structure for HuggingFace Trainer compatibility."""

    loss: torch.Tensor  # Main loss for trainer (required field name)
    llm_loss: Optional[torch.Tensor] = None
    coordinate_loss: Optional[torch.Tensor] = None
    teacher_loss: Optional[torch.Tensor] = None
    student_loss: Optional[torch.Tensor] = None


@dataclass
class ModelOutput:
    """Model output with loss components."""

    loss: torch.Tensor
    logits: torch.Tensor
    loss_components: LossComponents
    hidden_states: Optional[torch.Tensor] = None


class DistributedLossTrainer(HFTrainer):
    """
    HuggingFace Trainer with built-in distributed loss component synchronization.

    This trainer overrides the logging methods to use HuggingFace's proven
    distributed synchronization mechanisms for custom loss components,
    eliminating the need for manual NCCL operations in callbacks.
    """

    def __init__(
        self,
        model: "DetectionModel",
        tokenizer: PreTrainedTokenizer,
        training_args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        compute_metrics=None,
        callbacks: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Initialize trainer with distributed loss component tracking.

        Args:
            model: Detection model to train
            tokenizer: Tokenizer for text processing
            training_args: HuggingFace training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            compute_metrics: Metrics computation function
            callbacks: Additional trainer callbacks
            **kwargs: Additional arguments passed to HF Trainer
        """
        # Initialize loss tracking components
        self.loss_tracker = LossTracker()

        # Use provided callbacks (no filtering needed since deprecated callbacks are removed)
        filtered_callbacks = callbacks or []

        # Initialize parent trainer
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=filtered_callbacks,
            **kwargs,
        )

        # Track loss components for distributed synchronization
        self._loss_components_accumulator = {}
        self._loss_components_count = 0
        self._last_log_time = time.time()
        self._log_interval = 5  # seconds - reduced for better debugging (was 30s)

        # Checkpoint saving guard to prevent NCCL operations during checkpoint saving
        self._is_saving_checkpoint = False

        # Store gradient norm for comprehensive metrics
        self._last_grad_norm = None

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss and capture loss components for distributed synchronization.
        """
        # Get model outputs and loss
        if return_outputs:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=num_items_in_batch,
            )
            outputs = None

        # Extract loss components if available
        if hasattr(model, "get_last_loss_components"):
            loss_components = model.get_last_loss_components()
            if loss_components is not None:
                from transformers.utils import logging

                logger = logging.get_logger(__name__)
                logger.info(
                    f"🔍 CAPTURED loss components: {type(loss_components)} with fields: {list(loss_components.__dict__.keys()) if hasattr(loss_components, '__dict__') else 'N/A'}"
                )
            else:
                from transformers.utils import logging

                logger = logging.get_logger(__name__)
                logger.info("🔍 Model returned None for loss components")
            self._accumulate_loss_components(loss_components)
        else:
            from transformers.utils import logging

            logger = logging.get_logger(__name__)
            logger.info(
                f"⚠️ Model {type(model)} does not have get_last_loss_components method"
            )

        return (loss, outputs) if return_outputs else loss

    def _accumulate_loss_components(self, loss_components):
        """Accumulate loss components for later synchronization."""
        if loss_components is None:
            return

        # Convert to dict if it's a dataclass
        if hasattr(loss_components, "__dict__"):
            components_dict = loss_components.__dict__
        else:
            components_dict = loss_components

        # Accumulate each component
        for key, value in components_dict.items():
            if value is not None and torch.is_tensor(value):
                if key not in self._loss_components_accumulator:
                    self._loss_components_accumulator[key] = 0.0
                self._loss_components_accumulator[key] += value.detach().item()

        self._loss_components_count += 1

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        """
        Override HuggingFace's logging method to include distributed loss component synchronization.

        CRITICAL FIX: Separate logging from checkpoint saving to avoid NCCL timeout conflicts.
        The distributed synchronization is now performed only during logging, not during
        checkpoint saving operations which can conflict with DeepSpeed's distributed operations.
        """
        # Store gradient norm for comprehensive metrics
        if grad_norm is not None:
            self._last_grad_norm = (
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            )

        # FIRST: Handle logging separately from checkpoint saving
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            current_time = time.time()

            # Rate limit custom loss logging to avoid excessive NCCL operations
            if current_time - self._last_log_time >= self._log_interval:
                # Perform distributed loss component synchronization ONLY during logging
                self._log_distributed_loss_components()
                self._last_log_time = current_time

        # SECOND: Call parent method for standard loss logging and checkpoint saving
        # This ensures checkpoint saving happens without interference from our custom NCCL operations
        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate,
        )

    def _log_distributed_loss_components(self):
        """
        Log loss components using HuggingFace's built-in distributed synchronization.

        This method uses the same _nested_gather() mechanism that HuggingFace uses
        for the main training loss, ensuring consistent and reliable synchronization.

        CRITICAL: Skip distributed operations during checkpoint saving to prevent NCCL timeouts.
        """
        if self._loss_components_count == 0:
            return

        # CRITICAL FIX: Skip distributed synchronization during checkpoint saving
        # This prevents NCCL timeout conflicts with DeepSpeed's checkpoint operations
        if getattr(self, "_is_saving_checkpoint", False):
            from transformers.utils import logging

            logger = logging.get_logger(__name__)
            logger.info(
                "⏸️ Skipping loss component synchronization during checkpoint saving to prevent NCCL timeout"
            )
            return

        # Additional safety check: Skip if we're in evaluation mode during checkpoint saving
        if (
            hasattr(self, "control")
            and hasattr(self.control, "should_save")
            and self.control.should_save
        ):
            from transformers.utils import logging

            logger = logging.get_logger(__name__)
            logger.debug("⏸️ Skipping loss synchronization during save operation")
            return

        # Get rank info for logging
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
        except (ImportError, RuntimeError):
            rank = 0
            world_size = 1

        # Compute local averages
        local_averages = {}
        for key, accumulated_value in self._loss_components_accumulator.items():
            local_averages[key] = accumulated_value / self._loss_components_count

        # Synchronize loss components using HuggingFace's proven mechanism
        synchronized_averages = {}
        for key, local_avg in local_averages.items():
            try:
                # Convert to tensor for synchronization
                local_tensor = torch.tensor(
                    local_avg, dtype=torch.float32, device=self.args.device
                )

                # Use HuggingFace's _nested_gather() - same as main loss synchronization
                # Add timeout protection for distributed operations
                gathered_tensor = self._nested_gather(local_tensor)

                if gathered_tensor is not None:
                    # Compute global average (same as HuggingFace does for main loss)
                    synchronized_averages[key] = gathered_tensor.mean().item()
                else:
                    # Fallback to local value if gathering fails
                    synchronized_averages[key] = local_avg

            except Exception as e:
                # CRITICAL: Graceful fallback on any distributed operation failure
                # This prevents training crashes due to NCCL timeouts or other distributed issues
                from transformers.utils import logging

                logger = logging.get_logger(__name__)
                logger.warning(
                    f"⚠️ Distributed synchronization failed for {key}: {e}. Using local value."
                )
                synchronized_averages[key] = local_avg

        # Log comprehensive metrics including loss components and training progress
        if synchronized_averages:
            # Build comprehensive metrics dictionary
            comprehensive_metrics = self._build_comprehensive_metrics(
                synchronized_averages
            )

            # Log formatted metrics
            self._log_formatted_metrics(comprehensive_metrics, rank, world_size)

        # Reset accumulators
        self._loss_components_accumulator.clear()
        self._loss_components_count = 0

    def _build_comprehensive_metrics(self, synchronized_averages):
        """
        Build comprehensive metrics dictionary including loss components and training progress.

        Ports key functionality from TrainingStateManager.log_training_metrics()
        to provide complete training visibility.
        """
        metrics = synchronized_averages.copy()

        # Add learning rate metrics
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            try:
                current_lr = self.lr_scheduler.get_last_lr()
                if isinstance(current_lr, list) and len(current_lr) > 0:
                    metrics["learning_rate"] = current_lr[0]
                    # Add differential learning rates if available
                    if len(current_lr) > 1:
                        metrics["vision_lr"] = (
                            current_lr[1] if len(current_lr) > 1 else current_lr[0]
                        )
                        metrics["merger_lr"] = (
                            current_lr[2] if len(current_lr) > 2 else current_lr[0]
                        )
                        metrics["llm_lr"] = (
                            current_lr[3] if len(current_lr) > 3 else current_lr[0]
                        )
            except Exception as e:
                from transformers.utils import logging

                logger = logging.get_logger(__name__)
                logger.debug(f"Could not get learning rate: {e}")

        # Add training progress metrics
        if hasattr(self, "state") and self.state is not None:
            metrics["epoch"] = getattr(self.state, "epoch", 0.0)
            metrics["global_step"] = getattr(self.state, "global_step", 0)

            # Add time estimation
            if hasattr(self.state, "log_history") and len(self.state.log_history) > 0:
                metrics["remaining_hr"] = self._estimate_remaining_time()

        # Add gradient norm if available
        if hasattr(self, "_last_grad_norm") and self._last_grad_norm is not None:
            metrics["grad_norm"] = self._last_grad_norm

        # Add evaluation metrics if available
        eval_metrics = self._get_latest_eval_metrics()
        if eval_metrics:
            metrics.update(eval_metrics)

        return metrics

    def _estimate_remaining_time(self):
        """Estimate remaining training time in hours."""
        try:
            if (
                not hasattr(self.state, "log_history")
                or len(self.state.log_history) < 2
            ):
                return 0.0

            # Get timing from recent log entries
            recent_logs = self.state.log_history[-5:]  # Last 5 entries
            if len(recent_logs) < 2:
                return 0.0

            # Calculate average time per step
            time_diffs = []
            step_diffs = []

            for i in range(1, len(recent_logs)):
                if (
                    "train_runtime" in recent_logs[i]
                    and "train_runtime" in recent_logs[i - 1]
                ):
                    time_diff = (
                        recent_logs[i]["train_runtime"]
                        - recent_logs[i - 1]["train_runtime"]
                    )
                    step_diff = recent_logs[i].get("step", 0) - recent_logs[i - 1].get(
                        "step", 0
                    )
                    if step_diff > 0:
                        time_diffs.append(time_diff)
                        step_diffs.append(step_diff)

            if not time_diffs:
                return 0.0

            avg_time_per_step = sum(time_diffs) / sum(step_diffs)
            remaining_steps = (
                max(0, self.args.max_steps - self.state.global_step)
                if self.args.max_steps > 0
                else 0
            )

            if remaining_steps == 0 and self.args.num_train_epochs > 0:
                # Estimate based on epochs
                steps_per_epoch = len(self.train_dataset) // (
                    self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps
                )
                remaining_steps = max(
                    0, (self.args.num_train_epochs - self.state.epoch) * steps_per_epoch
                )

            remaining_seconds = remaining_steps * avg_time_per_step
            return remaining_seconds / 3600.0  # Convert to hours

        except Exception as e:
            from transformers.utils import logging

            logger = logging.get_logger(__name__)
            logger.debug(f"Could not estimate remaining time: {e}")
            return 0.0

    def _get_latest_eval_metrics(self):
        """Get the latest evaluation metrics from trainer state."""
        try:
            if not hasattr(self.state, "log_history") or not self.state.log_history:
                return {}

            # Find the most recent evaluation metrics
            eval_metrics = {}
            for log_entry in reversed(self.state.log_history):
                for key, value in log_entry.items():
                    if key.startswith("eval_"):
                        eval_metrics[key] = value
                if eval_metrics:  # Found some eval metrics, use the most recent
                    break

            return eval_metrics

        except Exception as e:
            from transformers.utils import logging

            logger = logging.get_logger(__name__)
            logger.debug(f"Could not get evaluation metrics: {e}")
            return {}

    def _log_formatted_metrics(self, metrics, rank, world_size):
        """
        Log comprehensive metrics in a formatted, readable way.

        Provides detailed training progress visibility similar to the old TrainingStateManager.
        """
        from transformers.utils import logging

        logger = logging.get_logger(__name__)

        # Build formatted log components
        log_components = []

        # Core loss components (always show these first)
        core_losses = [
            "loss",
            "llm_loss",
            "coordinate_loss",
            "teacher_loss",
            "student_loss",
        ]
        for key in core_losses:
            if key in metrics:
                value = metrics[key]
                if key == "loss":
                    log_components.append(f"Loss: {value:.4f}")
                elif key == "llm_loss":
                    log_components.append(f"LLM: {value:.4f}")
                elif key == "coordinate_loss":
                    log_components.append(f"Coord: {value:.4f}")
                elif key == "teacher_loss":
                    log_components.append(f"Teacher: {value:.4f}")
                elif key == "student_loss":
                    log_components.append(f"Student: {value:.4f}")

        # Granular teacher-student losses
        granular_losses = [
            "teacher_llm_loss",
            "teacher_l1_loss",
            "student_llm_loss",
            "student_l1_loss",
        ]
        granular_components = []
        for key in granular_losses:
            if key in metrics:
                value = metrics[key]
                if key == "teacher_llm_loss":
                    granular_components.append(f"T_LLM: {value:.4f}")
                elif key == "teacher_l1_loss":
                    granular_components.append(f"T_L1: {value:.4f}")
                elif key == "student_llm_loss":
                    granular_components.append(f"S_LLM: {value:.4f}")
                elif key == "student_l1_loss":
                    granular_components.append(f"S_L1: {value:.4f}")

        if granular_components:
            log_components.extend(granular_components)

        # Training progress metrics
        progress_components = []
        if "learning_rate" in metrics:
            progress_components.append(f"LR: {metrics['learning_rate']:.2e}")
        if "grad_norm" in metrics:
            progress_components.append(f"GradNorm: {metrics['grad_norm']:.3f}")
        if "epoch" in metrics:
            progress_components.append(f"Epoch: {metrics['epoch']:.2f}")
        if "remaining_hr" in metrics and metrics["remaining_hr"] > 0:
            progress_components.append(f"ETA: {metrics['remaining_hr']:.1f}h")

        # Evaluation metrics
        eval_components = []
        for key, value in metrics.items():
            if key.startswith("eval_"):
                eval_name = key.replace("eval_", "").replace("_", " ").title()
                eval_components.append(f"Eval_{eval_name}: {value:.4f}")

        # Log main metrics
        if log_components:
            logger.info(
                f"📊 [Rank {rank}/{world_size}] Loss Components: {' | '.join(log_components)}"
            )

        # Log progress metrics
        if progress_components:
            logger.info(
                f"📈 [Rank {rank}/{world_size}] Training Progress: {' | '.join(progress_components)}"
            )

        # Log evaluation metrics
        if eval_components:
            logger.info(
                f"📋 [Rank {rank}/{world_size}] Evaluation: {' | '.join(eval_components)}"
            )

        # Log differential learning rates if available
        lr_components = []
        for lr_key in ["vision_lr", "merger_lr", "llm_lr"]:
            if lr_key in metrics:
                lr_name = lr_key.replace("_lr", "").title()
                lr_components.append(f"{lr_name}: {metrics[lr_key]:.2e}")

        if lr_components:
            logger.info(
                f"🎯 [Rank {rank}/{world_size}] Learning Rates: {' | '.join(lr_components)}"
            )

    def _save_checkpoint(self, model, trial):
        """
        Override checkpoint saving to prevent NCCL operations during save.

        This method sets a flag to prevent distributed loss synchronization
        during checkpoint saving, which can cause NCCL timeout conflicts
        with DeepSpeed's own distributed operations.
        """
        # Set checkpoint saving guard
        self._is_saving_checkpoint = True

        try:
            # Call parent checkpoint saving method
            result = super()._save_checkpoint(model, trial)
            return result
        finally:
            # Always clear the checkpoint saving guard
            self._is_saving_checkpoint = False

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Override model saving to prevent NCCL operations during save.
        """
        # Set checkpoint saving guard
        self._is_saving_checkpoint = True

        try:
            # Call parent model saving method
            return super().save_model(output_dir, _internal_call)
        finally:
            # Always clear the checkpoint saving guard
            self._is_saving_checkpoint = False
