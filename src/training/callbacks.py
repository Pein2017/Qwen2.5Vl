import os
import shutil
from typing import List, Optional, Tuple

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class BestCheckpointCallback(TrainerCallback):
    """
    Custom callback that creates and maintains best checkpoint copies.

    This callback tracks evaluation metrics and creates separate "best-{step}-{metric}"
    checkpoint copies that are independent of the trainer's regular checkpoint management.
    """

    def __init__(
        self,
        save_total_limit: int = 2,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        """Initialize best checkpoint callback."""
        from src.logger_utils import get_training_logger

        self.save_total_limit = save_total_limit
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_checkpoints: List[Tuple[float, str]] = []

        # Initialize logger with fallback
        try:
            self.logger = get_training_logger()
        except Exception:
            import logging

            self.logger = logging.getLogger("training")

        self.logger.info(
            f"🔄 BestCheckpointCallback initialized: "
            f"metric={metric_name}, greater_is_better={greater_is_better}, "
            f"save_limit={save_total_limit}"
        )
        self.processed_steps = set()  # Track processed steps to avoid duplicates

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Called after evaluation - this is where we create best checkpoint copies."""
        if self.logger is None:
            from src.logger_utils import get_callback_logger

            self.logger = get_callback_logger()

        # Get the evaluation metrics
        logs = kwargs.get("logs", {})

        # Only process if this is an evaluation step and we haven't processed this step
        if self.metric_name not in logs or state.global_step in self.processed_steps:
            return

        current_metric = logs[self.metric_name]

        # Handle NaN evaluation loss gracefully
        if current_metric is None or (
            isinstance(current_metric, float)
            and torch.isnan(torch.tensor(current_metric))
        ):
            self.logger.warning(
                f"⚠️  Skipping best checkpoint creation due to NaN/None {self.metric_name} at step {state.global_step}"
            )
            return

        # Mark this step as processed to avoid duplicates
        self.processed_steps.add(state.global_step)

        self.logger.info(
            f"📊 Evaluation {self.metric_name}: {current_metric:.6f} at step {state.global_step}"
        )

        # Check if this qualifies as a best checkpoint
        is_best = self._is_best_checkpoint(current_metric)

        if is_best:
            # Create best checkpoint copy
            trainer = kwargs.get("trainer")
            if trainer is not None:
                best_checkpoint_path = self._create_best_checkpoint(
                    trainer, args, state, current_metric
                )

                if best_checkpoint_path:
                    # Add to our tracking list
                    self.best_checkpoints.append((current_metric, best_checkpoint_path))

                    # Sort and maintain limit
                    self._maintain_best_checkpoints()

        # Log current best checkpoints
        self._log_best_checkpoints(state)

    def _is_best_checkpoint(self, current_metric: float) -> bool:
        """Check if current metric qualifies as a best checkpoint."""
        if len(self.best_checkpoints) < self.save_total_limit:
            return True

        # Check if better than worst current best
        worst_metric = (
            max(self.best_checkpoints, key=lambda x: x[0])[0]
            if not self.greater_is_better
            else min(self.best_checkpoints, key=lambda x: x[0])[0]
        )

        if self.greater_is_better:
            return current_metric > worst_metric
        else:
            return current_metric < worst_metric

    def _create_best_checkpoint(
        self, trainer, args: TrainingArguments, state: TrainerState, metric_value: float
    ) -> Optional[str]:
        """Create a best checkpoint copy with proper naming."""
        # Format metric value for filename (avoid dots in filenames)
        metric_str = f"{metric_value:.6f}".replace(".", "_")
        best_checkpoint_name = f"best-{state.global_step}-{metric_str}"
        # Make sure output_dir is a string
        output_dir = str(args.output_dir) if args.output_dir is not None else ""
        best_checkpoint_path = os.path.join(output_dir, best_checkpoint_name)

        try:
            self.logger.info(f"💾 Creating best checkpoint: {best_checkpoint_name}")

            # Use trainer's save_model method to ensure all components are saved
            trainer.save_model(best_checkpoint_path)

            self.logger.info(f"✅ Best checkpoint saved: {best_checkpoint_name}")
            return best_checkpoint_path

        except Exception as e:
            self.logger.error(
                f"❌ Failed to create best checkpoint {best_checkpoint_name}: {e}"
            )
            return None

    def _maintain_best_checkpoints(self):
        """Sort and maintain the best checkpoints limit."""
        # Sort checkpoints by metric (best first)
        if self.greater_is_better:
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        else:
            self.best_checkpoints.sort(key=lambda x: x[0])

        # Remove excess checkpoints
        if len(self.best_checkpoints) > self.save_total_limit:
            checkpoints_to_remove = self.best_checkpoints[self.save_total_limit :]
            self.best_checkpoints = self.best_checkpoints[: self.save_total_limit]

            # Delete excess best checkpoints
            for metric_value, checkpoint_path in checkpoints_to_remove:
                self._safe_delete_checkpoint(checkpoint_path, metric_value)

    def _log_best_checkpoints(self, state: TrainerState):
        """Log current best checkpoints status."""
        if not self.best_checkpoints:
            return

        self.logger.info(f"🏆 Best {len(self.best_checkpoints)} checkpoints:")
        for i, (metric_value, checkpoint_path) in enumerate(self.best_checkpoints):
            rank = i + 1
            checkpoint_name = os.path.basename(checkpoint_path)
            self.logger.info(
                f"   #{rank}: {checkpoint_name} ({self.metric_name}={metric_value:.6f})"
            )

        # Update trainer state with best checkpoint info
        best_metric, best_checkpoint_path = self.best_checkpoints[0]
        state.best_metric = best_metric
        state.best_model_checkpoint = best_checkpoint_path
        self.logger.info(
            f"✨ Current best: {os.path.basename(best_checkpoint_path)} ({self.metric_name}={best_metric:.6f})"
        )

    def _safe_delete_checkpoint(self, checkpoint_path: str, metric_value: float):
        """
        Safely delete a best checkpoint directory.

        Args:
            checkpoint_path: Path to best checkpoint directory
            metric_value: Metric value for logging
        """
        checkpoint_name = os.path.basename(checkpoint_path)

        if not os.path.exists(checkpoint_path):
            self.logger.info(f"📁 Best checkpoint {checkpoint_name} already removed")
            return

        try:
            # Use ignore_errors=True to handle any race conditions
            shutil.rmtree(checkpoint_path, ignore_errors=True)

            # Verify deletion
            if os.path.exists(checkpoint_path):
                self._force_delete_checkpoint(checkpoint_path)

            self.logger.info(
                f"🗑️  Removed old best checkpoint {checkpoint_name} ({self.metric_name}={metric_value:.6f})"
            )

        except Exception as e:
            self.logger.warning(
                f"⚠️  Failed to remove best checkpoint {checkpoint_path}: {e}"
            )

    def _force_delete_checkpoint(self, checkpoint_path: str) -> str:
        """
        Force delete checkpoint by removing individual files first.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Status message
        """
        try:
            # List all files in the directory
            files = os.listdir(checkpoint_path)

            # Delete each file individually
            for file in files:
                # Make sure paths are strings
                checkpoint_path_str = (
                    str(checkpoint_path) if checkpoint_path is not None else ""
                )
                file_str = str(file) if file is not None else ""
                file_path = os.path.join(checkpoint_path_str, file_str)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    self.logger.error(f"Failed to delete {file_path}: {e}")

            # Try to remove the directory again
            os.rmdir(checkpoint_path)
            return f"Force deleted checkpoint {os.path.basename(checkpoint_path)}"
        except Exception as e:
            return f"Failed to force delete checkpoint: {e}"
