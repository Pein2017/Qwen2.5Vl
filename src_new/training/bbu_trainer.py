"""
BBU Trainer for src_new architecture.

This module provides a clean replacement for DistributedLossTrainer that eliminates
NCCL timeout issues by using local loss aggregation and completely overriding
HuggingFace's _maybe_log_save_evaluate() method.

Key Features:
- Local loss aggregation via TrainingStateManager (no distributed operations)
- Complete override of _maybe_log_save_evaluate() to avoid NCCL conflicts
- Standard HuggingFace logging via super().log() only
- No custom distributed synchronization
- Compatible with existing DetectionModel and training pipeline
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import Trainer as HFTrainer


if TYPE_CHECKING:
    from src_new.models.wrapper import DetectionModel

from .training_state_manager import TrainingStateManager


class BBUTrainer(HFTrainer):
    """
    BBU Trainer with local loss aggregation and no distributed conflicts.

    This trainer follows the proven pattern from src/ that eliminates NCCL timeout
    issues by using TrainingStateManager for local loss aggregation and completely
    overriding _maybe_log_save_evaluate() to avoid conflicts with HuggingFace's
    distributed operations.
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
        Initialize BBU trainer with local loss aggregation.

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
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )

        # Store references
        self.detection_model = model
        self.tokenizer_ref = tokenizer

        # Initialize training state manager for local loss aggregation
        self.training_state_manager = TrainingStateManager(
            config=getattr(model, "config", None),
            model=model,
            logger=None,  # Will use HF's logging
        )

        # Training state
        self._micro_batch_count = 0
        self._training_start_time = time.time()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss and capture loss components for local aggregation.
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

        # Extract and accumulate loss components locally
        if hasattr(model, "get_last_loss_components"):
            loss_components = model.get_last_loss_components()
            if loss_components:
                # Accumulate loss components locally (no distributed operations)
                self.training_state_manager.accumulate_loss_components(loss_components)

        # Increment micro batch count
        self._micro_batch_count += 1

        if return_outputs:
            return loss, outputs
        else:
            return loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss: Union[torch.Tensor, float],
        grad_norm: Optional[Union[torch.Tensor, float]],
        model: nn.Module,
        trial: Any,
        epoch: Optional[float],
        ignore_keys_for_eval: Optional[List[str]],
        start_time: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Complete override of HuggingFace's logging method to avoid NCCL conflicts.

        This method uses TrainingStateManager for local loss aggregation and
        standard HuggingFace logging only, eliminating all custom distributed
        operations that cause NCCL timeouts.
        """
        # Generate comprehensive metrics using TrainingStateManager
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            # Get comprehensive metrics from training state manager
            logged_metrics = self.training_state_manager.log_training_metrics(
                tr_loss=tr_loss,
                grad_norm=grad_norm,
                model=model,
                start_time=self._training_start_time,
                learning_rate=learning_rate,
            )

            # Process metrics with learning rate information
            final_logs = self.training_state_manager.log_metrics_batch(
                logs=logged_metrics,
                lr_scheduler=self.lr_scheduler,
                args=self.args,
                start_time=self._training_start_time,
            )

            # Format logs for better readability before logging
            formatted_logs = self._format_logs_for_display(final_logs)

            # Use standard HuggingFace logging only (no custom distributed operations)
            super(BBUTrainer, self).log(formatted_logs)

            # Reset training state manager metrics after logging
            self.training_state_manager.reset_metrics_state()

            # Reset micro batch count
            self._micro_batch_count = 0

        # Handle evaluation if needed
        if self.control.should_evaluate:
            self.evaluate(ignore_keys=ignore_keys_for_eval)

        # Handle checkpoint saving if needed
        if self.control.should_save:
            self._save_checkpoint_with_logging(model, trial, logged_metrics)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log metrics using TrainingStateManager for processing.

        This method processes logs through TrainingStateManager and then uses
        standard HuggingFace logging mechanisms only.
        """
        # Process logs through training state manager
        final_logs = self.training_state_manager.log_metrics_batch(
            logs=logs,
            lr_scheduler=self.lr_scheduler,
            args=self.args,
            start_time=start_time,
        )

        # Format logs for better readability before logging
        formatted_logs = self._format_logs_for_display(final_logs)

        # Use standard HuggingFace logging only
        super(BBUTrainer, self).log(formatted_logs, start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation to include individual loss components in metrics.

        This method ensures that eval_loss and eval loss components are properly
        computed and logged, following the proven pattern from src/.
        """
        # Save current training state to avoid interference
        self._save_evaluation_state()

        try:
            # Run base evaluation using HuggingFace's standard evaluation
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            # Add component loss metrics to evaluation results
            self._add_eval_component_metrics(metrics, eval_dataset, metric_key_prefix)

            return metrics

        finally:
            # Always restore training state
            self._restore_evaluation_state()

    def _save_evaluation_state(self):
        """Save current training state before evaluation."""
        # Save training loss accumulators
        self._saved_training_state = {
            "loss_components_accumulator": self.training_state_manager._loss_components_accumulator.copy(),
            "loss_components_count": self.training_state_manager._loss_components_count,
        }

        # Reset accumulators for evaluation
        self.training_state_manager._loss_components_accumulator.clear()
        self.training_state_manager._loss_components_count = 0

    def _restore_evaluation_state(self):
        """Restore training state after evaluation."""
        if hasattr(self, "_saved_training_state"):
            # Restore training accumulators
            self.training_state_manager._loss_components_accumulator = (
                self._saved_training_state["loss_components_accumulator"]
            )
            self.training_state_manager._loss_components_count = (
                self._saved_training_state["loss_components_count"]
            )
            delattr(self, "_saved_training_state")

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Enhanced prediction step that ensures loss components are accumulated during evaluation.

        This method ensures that compute_loss is called during evaluation to accumulate
        loss components, following the proven pattern from src/.
        """
        # Store original training state
        original_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                # Call compute_loss to accumulate loss components during evaluation
                if prediction_loss_only:
                    loss = self.compute_loss(model, inputs)
                    return (loss, None, None)
                else:
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )

                    # Extract logits and labels for evaluation metrics
                    logits = None
                    labels = None

                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    elif isinstance(outputs, dict) and "logits" in outputs:
                        logits = outputs["logits"]

                    if "labels" in inputs:
                        labels = inputs["labels"]

                    return (loss, logits, labels)

        finally:
            # Restore original training state
            model.train(original_training)

    def _format_logs_for_display(self, logs: Dict[str, float]) -> Dict[str, float]:
        """
        Format logs for better readability while preserving calculation precision.

        This method formats loss values to 4 decimal places and learning rates
        to a more readable format for display purposes only.
        """
        formatted_logs = {}

        for key, value in logs.items():
            if not isinstance(value, (int, float)):
                # Keep non-numeric values as-is
                formatted_logs[key] = value
                continue

            if "loss" in key.lower():
                # Format loss values to 4 decimal places
                formatted_logs[key] = round(float(value), 4)
            elif "lr" in key.lower() or "learning_rate" in key.lower():
                # Format learning rates for better readability
                formatted_logs[key] = self._format_learning_rate(value)
            elif key in [
                "grad_norm",
                "train_runtime",
                "eval_runtime",
                "train_samples_per_second",
                "eval_samples_per_second",
                "eval_steps_per_second",
            ]:
                # Format other metrics to reasonable precision
                formatted_logs[key] = round(float(value), 3)
            elif key in ["epoch"]:
                # Format epoch to 3 decimal places
                formatted_logs[key] = round(float(value), 3)
            else:
                # Keep other values as-is
                formatted_logs[key] = value

        return formatted_logs

    def _format_learning_rate(self, lr_value: float) -> str:
        """
        Format learning rate for better readability.

        Args:
            lr_value: Learning rate value

        Returns:
            Formatted learning rate string
        """
        if lr_value == 0:
            return "0.0"
        elif lr_value >= 1e-3:
            # For larger learning rates, use standard decimal notation
            return f"{lr_value:.6f}"
        else:
            # For small learning rates, use scientific notation with 2 decimal places
            return f"{lr_value:.2e}"

    def _save_checkpoint_with_logging(self, model, trial, metrics: Dict[str, float]):
        """
        Save checkpoint with comprehensive logging.

        This method wraps the standard checkpoint saving with detailed logging
        that includes checkpoint location, training progress, and current metrics.
        """
        import time
        from datetime import datetime

        # Get checkpoint directory
        checkpoint_dir = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"

        # Log pre-save information
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n🔄 [CHECKPOINT SAVE] Starting checkpoint save at {current_time}")
        print(f"📁 Checkpoint location: {checkpoint_dir}")
        print(f"📊 Training step: {self.state.global_step}")
        print(f"📈 Epoch: {self.state.epoch:.3f}")

        # Log current metrics with consistent formatting
        if metrics:
            formatted_metrics = self._format_logs_for_display(metrics)
            print("📋 Current training metrics:")
            for key, value in formatted_metrics.items():
                print(f"   {key}: {value}")

        # Record start time for save duration
        save_start_time = time.time()

        try:
            # Perform the actual checkpoint save
            self._save_checkpoint(model, trial)

            # Calculate save duration
            save_duration = time.time() - save_start_time

            # Log successful save
            print(
                f"✅ [CHECKPOINT SAVE] Completed successfully in {save_duration:.2f}s"
            )
            print(f"💾 Checkpoint saved to: {checkpoint_dir}")

            # Log training statistics
            training_stats = self.get_training_stats()
            if training_stats:
                print("📊 Training statistics:")
                for key, value in training_stats.items():
                    print(f"   {key}: {value}")

            print("─" * 60)

        except Exception as e:
            save_duration = time.time() - save_start_time
            print(f"❌ [CHECKPOINT SAVE] Failed after {save_duration:.2f}s: {e}")
            print(f"🚨 Error details: {type(e).__name__}: {str(e)}")
            print("─" * 60)
            raise  # Re-raise the exception to maintain error handling

    def _add_eval_component_metrics(
        self, metrics: Dict[str, Any], eval_dataset, metric_key_prefix: str
    ):
        """Add component loss metrics to evaluation results."""
        # Get averaged evaluation loss components
        eval_components = self.training_state_manager.get_averaged_losses_and_reset()

        # Add component metrics with eval prefix
        for component_name, component_value in eval_components.items():
            if component_name != "loss":  # eval_loss is already added by HuggingFace
                metrics[f"{metric_key_prefix}_{component_name}"] = round(
                    component_value, 4
                )

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            "micro_batch_count": self._micro_batch_count,
            "training_runtime": time.time() - self._training_start_time,
        }

        # Add stats from training state manager
        if hasattr(self.training_state_manager, "get_training_stats"):
            manager_stats = self.training_state_manager.get_training_stats()
            stats.update(manager_stats)

        return stats
