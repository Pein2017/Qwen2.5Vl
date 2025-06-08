import os
import shutil
from typing import Dict, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class BestCheckpointCallback(TrainerCallback):
    """
    Custom callback that saves only the best N checkpoints based on evaluation loss.

    This callback tracks evaluation loss and maintains only the top N performing checkpoints,
    automatically deleting worse-performing ones to save disk space.
    """

    def __init__(
        self,
        save_total_limit: int = 2,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        """
        Initialize the best checkpoint callback.

        Args:
            save_total_limit: Maximum number of best checkpoints to keep
            metric_name: Name of the metric to track (default: "eval_loss")
            greater_is_better: Whether higher metric values are better (default: False for loss)
        """
        self.save_total_limit = save_total_limit
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_checkpoints = []  # List of (metric_value, checkpoint_path) tuples
        self.logger = None
        self.processed_steps = set()  # Track processed steps to avoid duplicates

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when logging occurs - this is where eval metrics are available."""
        if self.logger is None:
            from src.logger_utils import get_callback_logger

            self.logger = get_callback_logger()

        # Get the current logs
        logs = kwargs.get("logs", {})

        # Only process if this is an evaluation log (contains eval_loss) and we haven't processed this step
        if self.metric_name not in logs or state.global_step in self.processed_steps:
            return

        current_metric = logs[self.metric_name]

        # ✅ FIX 1: Handle NaN evaluation loss gracefully
        if current_metric is None or (
            isinstance(current_metric, float)
            and (
                torch.isnan(torch.tensor(current_metric))
                if isinstance(current_metric, (int, float))
                else False
            )
        ):
            self.logger.warning(
                f"⚠️  Skipping checkpoint management due to NaN/None {self.metric_name} at step {state.global_step}"
            )
            return

        # Mark this step as processed to avoid duplicates
        self.processed_steps.add(state.global_step)

        self.logger.info(
            f"📊 Evaluation {self.metric_name}: {current_metric:.6f} at step {state.global_step}"
        )

        # Check if this step will have a checkpoint saved (based on save_strategy and save_steps)
        will_save_checkpoint = False
        if args.save_strategy == "steps" and state.global_step % args.save_steps == 0:
            will_save_checkpoint = True
        elif args.save_strategy == "epoch":
            # For epoch-based saving, we'd need to check if this is the end of an epoch
            # For now, assume it will be saved
            will_save_checkpoint = True

        if will_save_checkpoint:
            current_checkpoint = f"checkpoint-{state.global_step}"
            current_checkpoint_path = os.path.join(args.output_dir, current_checkpoint)

            self.logger.info(f"📁 Checkpoint will be saved at: {current_checkpoint}")

            # Add current checkpoint to the list
            self.best_checkpoints.append((current_metric, current_checkpoint_path))
        else:
            self.logger.info(
                f"⏭️  No checkpoint will be saved at step {state.global_step} (save_steps={args.save_steps})"
            )
            return

        # Sort checkpoints by metric (best first)
        if self.greater_is_better:
            self.best_checkpoints.sort(
                key=lambda x: x[0], reverse=True
            )  # Higher is better
        else:
            self.best_checkpoints.sort(key=lambda x: x[0])  # Lower is better (for loss)

        # Keep only the best N checkpoints
        if len(self.best_checkpoints) > self.save_total_limit:
            # Remove excess checkpoints
            checkpoints_to_remove = self.best_checkpoints[self.save_total_limit :]
            self.best_checkpoints = self.best_checkpoints[: self.save_total_limit]

            # ✅ FIX 2: Improved checkpoint deletion with better error handling
            for metric_value, checkpoint_path in checkpoints_to_remove:
                self._safe_delete_checkpoint(checkpoint_path, metric_value)

        # Log current best checkpoints
        self.logger.info(f"🏆 Best {len(self.best_checkpoints)} checkpoints:")
        for i, (metric_value, checkpoint_path) in enumerate(self.best_checkpoints):
            rank = i + 1
            checkpoint_name = os.path.basename(checkpoint_path)
            self.logger.info(
                f"   #{rank}: {checkpoint_name} ({self.metric_name}={metric_value:.6f})"
            )

        # Update trainer state with best checkpoint info
        if self.best_checkpoints:
            best_metric, best_checkpoint_path = self.best_checkpoints[0]
            state.best_metric = best_metric
            state.best_model_checkpoint = best_checkpoint_path
            self.logger.info(
                f"✨ Current best: {os.path.basename(best_checkpoint_path)} ({self.metric_name}={best_metric:.6f})"
            )

    def _safe_delete_checkpoint(self, checkpoint_path: str, metric_value: float):
        """
        Safely delete a checkpoint directory with improved error handling.

        Args:
            checkpoint_path: Path to checkpoint directory
            metric_value: Metric value for logging
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(
                f"⚠️  Checkpoint {os.path.basename(checkpoint_path)} already deleted or doesn't exist"
            )
            return

        try:
            # ✅ Use ignore_errors=True to handle race conditions
            shutil.rmtree(checkpoint_path, ignore_errors=True)

            # Double-check if deletion was successful
            if os.path.exists(checkpoint_path):
                # If directory still exists, try individual file deletion
                self._force_delete_checkpoint(checkpoint_path)

            self.logger.info(
                f"🗑️  Deleted checkpoint {os.path.basename(checkpoint_path)} ({self.metric_name}={metric_value:.6f})"
            )

        except Exception as e:
            self.logger.warning(
                f"⚠️  Failed to delete checkpoint {checkpoint_path}: {e}"
            )

    def _force_delete_checkpoint(self, checkpoint_path: str):
        """
        Force delete checkpoint by removing individual files first.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        try:
            # Try to remove individual files first
            for root, dirs, files in os.walk(checkpoint_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except (OSError, FileNotFoundError):
                        pass  # Ignore individual file errors

                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                    except (OSError, FileNotFoundError):
                        pass  # Ignore individual directory errors

            # Finally try to remove the main directory
            try:
                os.rmdir(checkpoint_path)
            except (OSError, FileNotFoundError):
                pass  # Ignore if already removed

        except Exception as e:
            self.logger.warning(
                f"⚠️  Force deletion also failed for {checkpoint_path}: {e}"
            )


class DetectionLossLoggingCallback(TrainerCallback):
    """
    Custom callback that integrates detection losses with standard trainer logging.

    This callback ensures that detection loss components are logged alongside
    standard metrics like train_loss, eval_loss, etc., and appear in TensorBoard
    with the same frequency and format as standard trainer logs.
    """

    def __init__(self):
        """Initialize the detection loss logging callback."""
        self.last_logged_step = -1

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Called when the trainer logs metrics.

        This integrates detection losses with the standard logging mechanism.
        """
        if logs is None:
            logs = {}

        # Check if we have detection losses stored in the trainer state
        if hasattr(state, "detection_losses") and state.detection_losses:
            current_step = state.global_step

            # Only add detection losses if we haven't already logged them for this step
            if (
                current_step != self.last_logged_step
                and current_step in state.detection_losses
            ):
                detection_losses = state.detection_losses[current_step]

                # Add detection losses to the standard logs with proper naming
                for loss_name, loss_value in detection_losses.items():
                    if loss_name == "total_loss":
                        # Map total_loss to detection_loss for clarity
                        log_key = "train_detection_loss"
                    else:
                        # Use standard naming convention: train_{component}_loss
                        log_key = f"train_{loss_name}"
                        if not log_key.endswith("_loss"):
                            log_key += "_loss"

                    logs[log_key] = loss_value

                # Update the last logged step to avoid duplicates
                self.last_logged_step = current_step

                # Clean up old detection losses to prevent memory buildup
                # Keep only the last 100 steps
                if len(state.detection_losses) > 100:
                    old_steps = sorted(state.detection_losses.keys())[:-100]
                    for old_step in old_steps:
                        del state.detection_losses[old_step]

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Called during evaluation to potentially add detection-specific eval metrics.
        """
        # This can be extended to add evaluation-specific detection metrics
        # For now, we just ensure the callback is properly integrated
        pass
