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

import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import Trainer as HFTrainer


if TYPE_CHECKING:
    from src_new.models.wrapper import DetectionModel

from ..utils.rank_aware_logging import (
    get_rank_aware_logger,
    log_distributed_info,
    rank0_only,
)
from .training_state_manager import TrainingStateManager
from .unified_checkpoint_manager import UnifiedCheckpointManager


# Import debug logging utilities
try:
    from ..utils.debug_logging import debug_logger
except ImportError:
    # Fallback if debug logging is not available
    debug_logger = None

# Configure rank-aware logger
logger = get_rank_aware_logger(__name__)


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
        processing_class: PreTrainedTokenizer,
        training_args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        compute_metrics=None,
        callbacks: Optional[list] = None,
        # Backward compatibility - deprecated parameter
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ) -> None:
        """
        Initialize BBU trainer with local loss aggregation.

        Args:
            model: Detection model to train
            processing_class: Processing class (tokenizer) for text processing
            training_args: HuggingFace training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            compute_metrics: Metrics computation function
            callbacks: Additional trainer callbacks
            tokenizer: DEPRECATED - use processing_class instead
            **kwargs: Additional arguments passed to HF Trainer
        """
        # Backward-compat: if tokenizer provided, prefer processing_class strictly
        if tokenizer is not None and processing_class is None:
            processing_class = tokenizer

        # Initialize parent trainer
        super().__init__(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )

        # Store references
        self.detection_model = model
        self.tokenizer_ref = processing_class

        # Store processor for saving (will be set during training setup)
        self.processor = None

        # Initialize training state manager for local loss aggregation
        # Use training_config (contains coordinate_tokens_enabled) instead of model.config (HF model config)
        training_config = model.training_config
        self.training_state_manager = TrainingStateManager(
            config=training_config,
            model=model,
            logger=None,  # Will use HF's logging
        )

        # Initialize unified checkpoint manager
        # Extract checkpoint settings from training_config or use defaults
        best_checkpoint_metric = training_config.best_checkpoint_metric
        best_checkpoint_greater_is_better = (
            training_config.best_checkpoint_greater_is_better
        )

        self.checkpoint_manager = UnifiedCheckpointManager(
            metric_name=best_checkpoint_metric,
            greater_is_better=best_checkpoint_greater_is_better,
        )

        # Training state
        self._micro_batch_count = 0
        self._training_start_time = time.time()

        # Flag to prevent duplicate checkpoint saving
        self._checkpoint_in_progress = False

        # Log distributed training information (rank 0 only)
        log_distributed_info(logger)

    def _extract_current_metrics(self) -> Dict[str, float]:
        """
        Extract current evaluation metrics from trainer state.

        This method searches through the trainer's log history to find the most
        recent evaluation metrics, which are used for best checkpoint determination.

        Returns:
            Dictionary of current evaluation metrics, empty dict if no metrics found
        """
        if not hasattr(self.state, "log_history") or not self.state.log_history:
            logger.debug("📊 No log history available for metric extraction")
            return {}

        # Search backwards through log history for the most recent eval metrics
        for log_entry in reversed(self.state.log_history):
            if isinstance(log_entry, dict):
                # Extract all evaluation metrics (keys starting with 'eval_')
                eval_metrics = {
                    k: v for k, v in log_entry.items() if k.startswith("eval_")
                }
                if eval_metrics:
                    logger.debug(f"📊 Extracted current metrics: {eval_metrics}")
                    return eval_metrics

        logger.debug("📊 No evaluation metrics found in log history")
        return {}

    @property
    def tokenizer(self):
        """
        Backward compatibility property for accessing the tokenizer.

        This property provides access to the processing_class (tokenizer) for
        backward compatibility with existing code that expects self.tokenizer.
        """
        import warnings

        warnings.warn(
            "Accessing 'trainer.tokenizer' is deprecated. Use 'trainer.processing_class' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.processing_class

    def set_processor(self, processor):
        """Set the processor for saving during checkpoints and dataset processing.

        Args:
            processor: Qwen2VLProcessor instance that should be saved with checkpoints
        """
        self.processor = processor

        # Set processor on datasets for HuggingFace-first processing
        if hasattr(self.train_dataset, "set_processor"):
            self.train_dataset.set_processor(processor)
            logger.info("✅ Set HuggingFace processor on training dataset")

        if (
            hasattr(self.eval_dataset, "set_processor")
            and self.eval_dataset is not None
        ):
            self.eval_dataset.set_processor(processor)
            logger.info("✅ Set HuggingFace processor on evaluation dataset")

    def get_eval_dataloader(self, eval_dataset=None) -> torch.utils.data.DataLoader:
        """Return evaluation DataLoader with drop_last forced to False.

        This allows training to use drop_last=True (for efficiency) while
        evaluation keeps all remaining samples when the dataset is small.
        """
        original_drop_last = self.args.dataloader_drop_last
        try:
            self.args.dataloader_drop_last = False
            return super().get_eval_dataloader(eval_dataset)
        finally:
            # Restore original setting for other dataloaders (e.g., train)
            self.args.dataloader_drop_last = original_drop_last

    def train(self, *args, **kwargs):
        """
        Override train method to initialize one-time debug logging.
        """
        # Initialize one-time debug logging for the training run
        if debug_logger is not None:
            debug_logger.reconfigure_logger()
            debug_logger.start_training_run()

        # Call parent train method
        return super().train(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss and capture loss components for local aggregation.
        """
        # DEBUG LOGGING: Log training sample on first training step
        if (
            debug_logger is not None
            and model.training
            and debug_logger.should_log_training_sample()
        ):
            self._log_sample_debug_info(inputs, is_training=True)

        # Get model outputs and loss
        if return_outputs:
            # Legacy mode: do not pass unified assistant_spans; rely on teacher/student spans
            if "assistant_spans" in inputs:
                inputs = {k: v for k, v in inputs.items() if k != "assistant_spans"}
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            # Legacy mode: do not pass unified assistant_spans; rely on teacher/student spans
            if "assistant_spans" in inputs:
                inputs = {k: v for k, v in inputs.items() if k != "assistant_spans"}
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

        # Initialize logged_metrics to ensure it's always defined
        logged_metrics = None

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
                trainer_state=self.state,  # Pass trainer state for remaining_hrs calculation
            )

            # Process metrics with learning rate information (using correct LR mapping)
            final_logs = self.training_state_manager.log_metrics_batch(
                logs=logged_metrics,
                lr_scheduler=None,  # Don't let it add LRs automatically
                args=self.args,
                start_time=self._training_start_time,
            )

            # Add correctly labeled learning rates
            lr_dict = self._get_learning_rates_with_correct_labels()
            final_logs.update(lr_dict)

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
            # Generate metrics for checkpoint logging if not already generated
            if logged_metrics is None:
                logged_metrics = self.training_state_manager.log_training_metrics(
                    tr_loss=tr_loss,
                    grad_norm=grad_norm,
                    model=model,
                    start_time=self._training_start_time,
                    learning_rate=learning_rate,
                )

            # Extract current evaluation metrics for unified checkpoint management
            current_metrics = self._extract_current_metrics()

            # Pass metrics to unified checkpoint saving
            self._save_checkpoint(model, trial, current_metrics)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log metrics using TrainingStateManager for processing.

        This method processes logs through TrainingStateManager and then uses
        standard HuggingFace logging mechanisms only.
        """
        # Process logs through training state manager (using correct LR mapping)
        final_logs = self.training_state_manager.log_metrics_batch(
            logs=logs,
            lr_scheduler=None,  # Don't let it add LRs automatically
            args=self.args,
            start_time=start_time,
        )

        # Add correctly labeled learning rates
        lr_dict = self._get_learning_rates_with_correct_labels()
        final_logs.update(lr_dict)

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
        # Debug logging is handled automatically by the one-time flags
        # No need to start evaluation sessions

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
        # DEBUG LOGGING: Log evaluation sample on first evaluation step
        if (
            debug_logger is not None
            and not model.training
            and debug_logger.should_log_evaluation_sample()
        ):
            self._log_sample_debug_info(inputs, is_training=False)
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

    def _save_checkpoint(
        self, model, trial, current_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Unified checkpoint saving with best checkpoint management.

        This method creates both regular and best checkpoints in a single pass,
        eliminating redundant I/O operations and ensuring consistency.

        Key Features:
        - Single-pass checkpoint creation (no copying)
        - Fast inference-ready checkpoints (30s vs 200-400s)
        - SafeTensors format for 4-6x faster loading
        - Extended tokenizer with coordinate tokens
        - Unified best checkpoint management
        - Proper DeepSpeed coordination when needed

        Args:
            model: Model to save
            trial: Training trial (for hyperparameter tuning)
            current_metrics: Current evaluation metrics for best checkpoint determination
        """
        # Prevent duplicate checkpoint saving
        if self._checkpoint_in_progress:
            logger.debug("🔄 Checkpoint already in progress, skipping duplicate call")
            return

        self._checkpoint_in_progress = True

        try:
            import time
            from datetime import datetime

            # Get checkpoint directory
            checkpoint_dir = (
                f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
            )

            # CRITICAL FIX: Only rank 0 should log checkpoint save messages
            should_log = self.args.should_save  # True only on rank 0

            if should_log:
                # Log pre-save information (only on rank 0)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"\n🚀 [FAST CHECKPOINT] Starting inference-ready checkpoint save at {current_time}"
                )
                logger.info(f"📁 Checkpoint location: {checkpoint_dir}")
                logger.info(f"📊 Training step: {self.state.global_step}")
                logger.info(f"📈 Epoch: {self.state.epoch:.3f}")

            # Record start time for save duration
            save_start_time = time.time()
            # Check if we should use fast inference checkpoint mode
            use_fast_checkpoint = getattr(self.args, "fast_checkpoint_mode", True)

            if use_fast_checkpoint:
                # Use optimized inference-ready checkpoint saving
                self._save_inference_checkpoint(model, trial, checkpoint_dir)
            else:
                # Fall back to full HuggingFace checkpoint (for training resumption)
                logger.info("🐌 Using full checkpoint mode (includes optimizer states)")
                super()._save_checkpoint(model, trial)

            # === UNIFIED CHECKPOINT MANAGEMENT ===
            # Check if this should also be a best checkpoint
            if current_metrics is None:
                current_metrics = self._extract_current_metrics()

            # CRITICAL: Only rank 0 handles best checkpoint creation (simple folder copy)
            if (
                should_log  # Only rank 0
                and current_metrics
                and self.checkpoint_manager.is_new_best(current_metrics)
            ):
                logger.info("🏆 Creating best checkpoint by direct folder copy")

                # Generate best checkpoint name
                best_checkpoint_name = (
                    self.checkpoint_manager.create_best_checkpoint_name(
                        current_metrics, self.state.global_step
                    )
                )
                best_checkpoint_path = f"{self.args.output_dir}/{best_checkpoint_name}"

                logger.info(
                    f"📁 Direct copy: {os.path.basename(checkpoint_dir)} → {best_checkpoint_name}"
                )

                import shutil

                copy_start_time = time.time()

                # SIMPLE FOLDER COPY/PASTE - No checkpoint creation, just copy existing folder
                try:
                    # Remove existing best checkpoint if it exists
                    if os.path.exists(best_checkpoint_path):
                        logger.debug(f"🗑️ Removing existing: {best_checkpoint_name}")
                        shutil.rmtree(best_checkpoint_path)

                    # Simple folder copy operation (like copy/paste in file manager)
                    shutil.copytree(checkpoint_dir, best_checkpoint_path)

                except Exception as e:
                    logger.error(f"❌ Failed to copy checkpoint folder: {e}")
                    raise

                copy_duration = time.time() - copy_start_time
                logger.info(f"✅ Best checkpoint folder copied in {copy_duration:.2f}s")

                # Clean up old best checkpoint before updating tracking
                old_best_path = self.checkpoint_manager.current_best_dir
                if (
                    old_best_path
                    and old_best_path != best_checkpoint_path
                    and os.path.exists(old_best_path)
                ):
                    try:
                        shutil.rmtree(old_best_path)
                        logger.info(
                            f"🗑️ Removed old best: {os.path.basename(old_best_path)}"
                        )
                    except (OSError, PermissionError) as e:
                        logger.warning(f"⚠️ Could not remove old best checkpoint: {e}")

                # Update tracking
                self.checkpoint_manager.update_best_checkpoint(
                    current_metrics, best_checkpoint_path
                )

                metric_name = self.checkpoint_manager.metric_name
                metric_value = current_metrics[metric_name]
                logger.info(f"✅ Best checkpoint ready: {best_checkpoint_name}")
                logger.info(f"📊 Best {metric_name}: {metric_value:.4f}")

                # Rotate checkpoints AFTER creating best checkpoint to avoid deleting the source
                self._rotate_inference_checkpoints()

            # Calculate save duration
            save_duration = time.time() - save_start_time

            if should_log:
                # Log successful save (only on rank 0)
                checkpoint_type = (
                    "FAST INFERENCE" if use_fast_checkpoint else "FULL TRAINING"
                )
                logger.info(
                    f"✅ [{checkpoint_type} CHECKPOINT] Completed successfully in {save_duration:.2f}s"
                )
                logger.info(f"💾 Checkpoint saved to: {checkpoint_dir}")

                # Log training statistics
                training_stats = self.get_training_stats()
                if training_stats:
                    logger.info("📊 Training statistics:")
                    for key, value in training_stats.items():
                        logger.info(f"   {key}: {value}")

                logger.info("─" * 60)

        except Exception as e:
            save_duration = time.time() - save_start_time
            if should_log:
                # Log error (only on rank 0)
                logger.error(
                    f"❌ [CHECKPOINT SAVE] Failed after {save_duration:.2f}s: {e}"
                )
                logger.error(f"🚨 Error details: {type(e).__name__}: {str(e)}")
                logger.error("─" * 60)
            raise  # Re-raise the exception to maintain error handling
        finally:
            # Always reset the checkpoint flag
            self._checkpoint_in_progress = False

    def _save_inference_checkpoint(self, model, trial, checkpoint_dir):
        """
        Save optimized inference-ready checkpoint with only essential components.

        This method creates lightweight checkpoints suitable for inference by excluding:
        - Optimizer states (often 2-3x model size)
        - Scheduler states
        - Random number generator states
        - Training-specific arguments

        Includes only:
        - Model weights (SafeTensors format)
        - Extended tokenizer with coordinate tokens
        - Model configuration
        - Image processor configuration
        - Generation configuration

        Args:
            model: Model to save
            trial: Training trial (for hyperparameter tuning)
            checkpoint_dir: Directory to save checkpoint
        """
        import json
        import os
        import time

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Handle DeepSpeed distributed saving coordination
        if self.is_deepspeed_enabled:
            self._save_deepspeed_inference_checkpoint(model, checkpoint_dir)
        elif self.args.should_save:
            logger.info(
                f"🚀 [RANK 0] Creating inference-ready checkpoint at {checkpoint_dir}"
            )

            # 1. Save model weights in SafeTensors format (fastest loading)
            model_start = time.time()
            logger.info(f"💾 [RANK 0] Saving model weights (SafeTensors format)...")

            # Get the unwrapped model for saving
            unwrapped_model = self._get_unwrapped_model(model)

            # Save with SafeTensors format and optimized sharding
            unwrapped_model.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,  # Use SafeTensors format
                max_shard_size="5GB",  # Optimize shard size for faster loading
                push_to_hub=False,  # Don't push to hub
            )

            model_time = time.time() - model_start
            logger.info(f"✅ [RANK 0] Model weights saved in {model_time:.2f}s")

            # 2. Save extended tokenizer with coordinate tokens
            tokenizer_start = time.time()
            if hasattr(self, "processing_class") and self.processing_class is not None:
                logger.info(f"🔤 [RANK 0] Saving extended tokenizer...")
                self.processing_class.save_pretrained(checkpoint_dir)
                tokenizer_time = time.time() - tokenizer_start
                logger.info(
                    f"✅ [RANK 0] Extended tokenizer saved in {tokenizer_time:.2f}s"
                )

            # 3. Save processor configuration (includes both tokenizer and image processor)
            processor_saved = False
            if hasattr(self, "processor") and self.processor is not None:
                try:
                    processor_start = time.time()
                    logger.info(f"💾 [RANK 0] Saving processor configuration...")
                    self.processor.save_pretrained(checkpoint_dir)
                    processor_time = time.time() - processor_start
                    logger.info(
                        f"✅ [RANK 0] Processor configuration saved in {processor_time:.2f}s"
                    )
                    processor_saved = True
                except Exception as e:
                    logger.error(f"❌ Failed to save processor configuration: {e}")
                    logger.info("🔄 [RANK 0] Attempting fallback processor saving...")

            # 3b. Fallback: Create processor configuration manually if main save failed
            if not processor_saved:
                try:
                    processor_start = time.time()
                    logger.info(
                        f"🔧 [RANK 0] Creating processor configuration manually..."
                    )

                    # Save image processor separately
                    image_processor = None
                    if hasattr(self, "data_collator") and hasattr(
                        self.data_collator, "image_processor"
                    ):
                        image_processor = self.data_collator.image_processor
                    elif hasattr(self, "processor") and hasattr(
                        self.processor, "image_processor"
                    ):
                        image_processor = self.processor.image_processor

                    if image_processor is not None:
                        image_processor.save_pretrained(checkpoint_dir)
                        logger.info(f"✅ [RANK 0] Image processor saved separately")

                    # Create preprocessor_config.json manually
                    import json

                    preprocessor_config = {
                        "image_processor_type": "Qwen2VLImageProcessor",
                        "processor_class": "Qwen2VLProcessor",
                    }

                    preprocessor_config_path = os.path.join(
                        checkpoint_dir, "preprocessor_config.json"
                    )
                    with open(preprocessor_config_path, "w") as f:
                        json.dump(preprocessor_config, f, indent=2)

                    processor_time = time.time() - processor_start
                    logger.info(
                        f"✅ [RANK 0] Manual processor config created in {processor_time:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"❌ Fallback processor saving also failed: {e}")
                    logger.warning(
                        "⚠️ [RANK 0] Checkpoint may be missing processor configuration"
                    )

            # 4. Save generation configuration for inference
            gen_config_start = time.time()
            if (
                hasattr(unwrapped_model, "generation_config")
                and unwrapped_model.generation_config is not None
            ):
                logger.info(f"⚙️ [RANK 0] Saving generation configuration...")
                unwrapped_model.generation_config.save_pretrained(checkpoint_dir)
                gen_config_time = time.time() - gen_config_start
                logger.info(
                    f"✅ [RANK 0] Generation config saved in {gen_config_time:.2f}s"
                )
            else:
                logger.debug("⚠️ [RANK 0] No generation config found to save")

            # 5. Save coordinate token configuration for inference compatibility
            coord_config_start = time.time()
            if (
                hasattr(unwrapped_model, "training_config")
                and unwrapped_model.training_config.coordinate_tokens_enabled
            ):
                logger.info(f"🎯 [RANK 0] Saving coordinate token configuration...")
                coord_config = {
                    "coordinate_tokens_enabled": True,
                    "max_coord_value": unwrapped_model.training_config.max_coord_value,
                    "vocab_size_extended": len(self.processing_class.get_vocab())
                    if self.processing_class
                    else None,
                    "coordinate_token_range": (
                        lambda _tok: (
                            [
                                min(
                                    _tok.get_vocab()[t]
                                    for t in _tok.get_vocab()
                                    if t.startswith("<|coord_")
                                ),
                                max(
                                    _tok.get_vocab()[t]
                                    for t in _tok.get_vocab()
                                    if t.startswith("<|coord_")
                                )
                                + 1,
                            ]
                            if _tok is not None
                            and any(
                                t.startswith("<|coord_")
                                for t in _tok.get_vocab().keys()
                            )
                            else [None, None]
                        )
                    )(self.processing_class),
                }

                coord_config_path = os.path.join(
                    checkpoint_dir, "coordinate_config.json"
                )
                with open(coord_config_path, "w") as f:
                    json.dump(coord_config, f, indent=2)

                coord_config_time = time.time() - coord_config_start
                logger.info(
                    f"✅ [RANK 0] Coordinate config saved in {coord_config_time:.2f}s"
                )

            logger.info(f"🚀 [RANK 0] Inference checkpoint optimization complete!")
            logger.info(
                f"📦 [RANK 0] Checkpoint ready for: DetectionModel.from_pretrained('{checkpoint_dir}')"
            )

        # NOTE: Checkpoint rotation is now handled in _save_checkpoint AFTER best-copy
        # to ensure the just-saved checkpoint exists when copying to best.

    def _get_unwrapped_model(self, model):
        """Get the unwrapped model for saving, handling various wrapper types."""
        # Handle DeepSpeed wrapped models
        if hasattr(model, "module"):
            unwrapped = model.module
        else:
            unwrapped = model

        # Handle additional wrappers (DDP, etc.)
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        # CRITICAL FIX: If this is a DetectionModel wrapper, get the base_model
        if hasattr(unwrapped, "base_model"):
            logger.debug("🔧 Unwrapping DetectionModel to get base_model for saving")
            unwrapped = unwrapped.base_model

        return unwrapped

    def _rotate_inference_checkpoints(self):
        """Rotate inference checkpoints based on save_total_limit."""
        if not self.args.should_save or not hasattr(self.args, "save_total_limit"):
            return

        import glob
        import os

        # Get all checkpoint directories
        if not self.args.output_dir:
            return  # No output directory specified
        checkpoint_pattern = os.path.join(self.args.output_dir, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)

        # Sort by step number (extract from checkpoint-{step})
        def get_step_number(checkpoint_path):
            try:
                return int(os.path.basename(checkpoint_path).split("-")[1])
            except (IndexError, ValueError):
                return 0

        checkpoints.sort(key=get_step_number)

        # Remove old checkpoints if we exceed the limit
        if self.args.save_total_limit and len(checkpoints) > self.args.save_total_limit:
            checkpoints_to_remove = checkpoints[: -self.args.save_total_limit]
            for checkpoint_path in checkpoints_to_remove:
                try:
                    import shutil

                    shutil.rmtree(checkpoint_path)
                    logger.info(f"🗑️ [RANK 0] Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ [RANK 0] Failed to remove checkpoint {checkpoint_path}: {e}"
                    )

    def _save_deepspeed_inference_checkpoint(self, model, checkpoint_dir):
        """
        Optimized DeepSpeed checkpoint saving for inference-ready checkpoints.

        This method handles DeepSpeed model saving while creating lightweight
        inference checkpoints by coordinating across ranks efficiently.

        Args:
            model: DeepSpeed wrapped model
            checkpoint_dir: Directory to save checkpoint
        """
        import time

        if self.args.should_save:
            logger.info(f"🚀 Creating inference-ready checkpoint at {checkpoint_dir}")

        # 1. Save INFERENCE-ONLY model weights (SafeTensors format)
        model_start = time.time()

        # CRITICAL FIX: Extract model weights for inference-only saving
        # DeepSpeed's save_checkpoint() saves full training state (49GB)
        # We need to extract just the model weights and save as SafeTensors (~7GB)
        try:
            if self.args.should_save:
                logger.info(
                    "💾 Extracting model weights for inference-only checkpoint..."
                )

            # Get the unwrapped model from DeepSpeed wrapper
            unwrapped_model = self._get_unwrapped_model(model)

            # Save ONLY model weights in SafeTensors format (no optimizer states)
            unwrapped_model.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,  # CRITICAL: Use SafeTensors format
                max_shard_size="5GB",  # Optimize shard size for faster loading
                push_to_hub=False,  # Don't push to hub
            )

            if self.args.should_save:
                model_time = time.time() - model_start
                logger.info(
                    f"✅ Inference-ready model weights saved in {model_time:.2f}s"
                )

        except Exception as e:
            if self.args.should_save:
                logger.error(f"❌ DeepSpeed inference checkpoint save failed: {e}")
            raise

        # 2. Save auxiliary files only on rank 0 (tokenizer, configs, etc.)
        if self.args.should_save:
            aux_start = time.time()

            # Save extended tokenizer with coordinate tokens
            if hasattr(self, "processing_class") and self.processing_class is not None:
                logger.info("🔤 Saving extended tokenizer...")
                self.processing_class.save_pretrained(checkpoint_dir)

            # Save processor configuration (includes both tokenizer and image processor)
            processor_saved = False
            if hasattr(self, "processor") and self.processor is not None:
                try:
                    logger.info("💾 Saving processor configuration...")
                    self.processor.save_pretrained(checkpoint_dir)
                    processor_saved = True
                except Exception as e:
                    logger.error(f"❌ Failed to save processor configuration: {e}")
                    logger.info("🔄 Attempting fallback processor saving...")

            # Fallback: Create processor configuration manually if main save failed
            if not processor_saved:
                try:
                    logger.info("🔧 Creating processor configuration manually...")

                    # Save image processor separately
                    image_processor = None
                    if hasattr(self, "data_collator") and hasattr(
                        self.data_collator, "image_processor"
                    ):
                        image_processor = self.data_collator.image_processor
                    elif hasattr(self, "processor") and hasattr(
                        self.processor, "image_processor"
                    ):
                        image_processor = self.processor.image_processor

                    if image_processor is not None:
                        image_processor.save_pretrained(checkpoint_dir)
                        logger.info("✅ Image processor saved separately")

                    # Create preprocessor_config.json manually
                    import json

                    preprocessor_config = {
                        "image_processor_type": "Qwen2VLImageProcessor",
                        "processor_class": "Qwen2VLProcessor",
                    }

                    preprocessor_config_path = os.path.join(
                        checkpoint_dir, "preprocessor_config.json"
                    )
                    with open(preprocessor_config_path, "w") as f:
                        json.dump(preprocessor_config, f, indent=2)

                    logger.info("✅ Manual processor config created")

                except Exception as e:
                    logger.error(f"❌ Fallback processor saving also failed: {e}")
                    logger.warning(
                        "⚠️ Checkpoint may be missing processor configuration"
                    )

            # Save coordinate token configuration
            unwrapped_model = self._get_unwrapped_model(model)
            if (
                hasattr(unwrapped_model, "training_config")
                and unwrapped_model.training_config.coordinate_tokens_enabled
            ):
                import json

                logger.info("🎯 Saving coordinate token configuration...")
                coord_config = {
                    "coordinate_tokens_enabled": True,
                    "max_coord_value": unwrapped_model.training_config.max_coord_value,
                    "vocab_size_extended": len(self.processing_class.get_vocab())
                    if self.processing_class
                    else None,
                    "coordinate_token_range": (
                        lambda _tok: (
                            [
                                min(
                                    _tok.get_vocab()[t]
                                    for t in _tok.get_vocab()
                                    if t.startswith("<|coord_")
                                ),
                                max(
                                    _tok.get_vocab()[t]
                                    for t in _tok.get_vocab()
                                    if t.startswith("<|coord_")
                                )
                                + 1,
                            ]
                            if _tok is not None
                            and any(
                                t.startswith("<|coord_")
                                for t in _tok.get_vocab().keys()
                            )
                            else [None, None]
                        )
                    )(self.processing_class),
                }

                coord_config_path = os.path.join(
                    checkpoint_dir, "coordinate_config.json"
                )
                with open(coord_config_path, "w") as f:
                    json.dump(coord_config, f, indent=2)

            aux_time = time.time() - aux_start
            logger.info(f"✅ Auxiliary files saved in {aux_time:.2f}s")
            logger.info(
                f"🚀 Inference checkpoint ready for: DetectionModel.from_pretrained('{checkpoint_dir}')"
            )

        # 3. Synchronize all ranks after checkpoint save
        if dist.is_initialized():
            dist.barrier()

    def _save_checkpoint_with_processor(self, model, trial, checkpoint_dir):
        """
        Consolidated checkpoint saving with all optimizations and processor support.

        This method combines all checkpoint saving functionality:
        1. Optimized model saving (SafeTensors format)
        2. Tokenizer saving with coordinate token support
        3. Processor configuration saving
        4. Training arguments saving
        5. Performance optimizations (skips slow files)

        Args:
            model: Model to save
            trial: Training trial (for hyperparameter tuning)
            checkpoint_dir: Directory to save checkpoint
        """
        import os
        import time

        import torch

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_start = time.time()

        if self.args.should_save:
            logger.info(
                f"⚡ [RANK 0] Starting optimized checkpoint save to {checkpoint_dir}"
            )

            # 1. Save model weights (SafeTensors format) - Fast and inference-optimized
            model_start = time.time()
            logger.info(f"💾 [RANK 0] Saving model weights (SafeTensors format)...")

            if hasattr(model, "save_pretrained"):
                model.save_pretrained(
                    checkpoint_dir,
                    safe_serialization=True,
                    max_shard_size="10GB",  # Force single file for models < 10GB
                )
            else:
                # Fallback for wrapped models
                unwrapped_model = self._wrap_model(model, training=False)
                unwrapped_model.save_pretrained(
                    checkpoint_dir,
                    safe_serialization=True,
                    max_shard_size="10GB",
                )

            model_time = time.time() - model_start
            logger.info(f"✅ [RANK 0] Model weights saved in {model_time:.2f}s")

            # 2. Save tokenizer with coordinate token support - Fast
            tokenizer_start = time.time()
            if hasattr(self, "processing_class") and self.processing_class is not None:
                self.processing_class.save_pretrained(checkpoint_dir)
                tokenizer_time = time.time() - tokenizer_start
                logger.info(f"✅ [RANK 0] Tokenizer saved in {tokenizer_time:.2f}s")

            # 3. Save processor configuration if available
            if self.processor is not None:
                try:
                    processor_start = time.time()
                    logger.info(f"💾 [RANK 0] Saving processor configuration...")
                    self.processor.save_pretrained(checkpoint_dir)
                    processor_time = time.time() - processor_start
                    logger.info(
                        f"✅ [RANK 0] Processor configuration saved in {processor_time:.2f}s"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to save processor configuration: {e}")
                    # Don't raise here to avoid breaking the checkpoint save

            # 4. Save training arguments - Fast
            args_start = time.time()
            torch.save(self.args, os.path.join(checkpoint_dir, "training_args.bin"))
            args_time = time.time() - args_start
            logger.info(f"✅ [RANK 0] Training args saved in {args_time:.2f}s")

            # Performance optimization: Skip slow files that cause 200+ second delays
            logger.info(
                "⚡ [RANK 0] SKIPPED slow files: rng_state, scheduler, trainer_state"
            )

            checkpoint_time = time.time() - checkpoint_start
            logger.info(
                f"⚡ [RANK 0] Complete checkpoint save finished in {checkpoint_time:.2f}s"
            )

    def save_final_model(self, output_dir: str = None) -> str:
        """
        Save final model with descriptive naming based on training metrics.

        Args:
            output_dir: Base output directory. If None, uses self.args.output_dir

        Returns:
            str: Path to the saved final model directory
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Get current training step (more accurate for step-based saving)
        current_step = self.state.global_step

        # Try to get the best evaluation loss and corresponding step from trainer state
        eval_loss = None
        best_step = current_step

        if hasattr(self.state, "best_metric") and self.state.best_metric is not None:
            eval_loss = self.state.best_metric
            # Get the step where the best metric was achieved
            if (
                hasattr(self.state, "best_global_step")
                and self.state.best_global_step is not None
            ):
                best_step = self.state.best_global_step
        elif hasattr(self.state, "log_history") and self.state.log_history:
            # Find the most recent eval_loss in log history
            for log_entry in reversed(self.state.log_history):
                if "eval_loss" in log_entry:
                    eval_loss = log_entry["eval_loss"]
                    # Try to get the step from the log entry
                    if "step" in log_entry:
                        best_step = log_entry["step"]
                    break

        # Create descriptive final model directory name using step-based format
        if eval_loss is not None:
            final_model_dir = f"{output_dir}/best-{best_step}-loss{eval_loss:.4f}"
        else:
            final_model_dir = f"{output_dir}/best-{current_step}-final"

        # Save the final model
        self._save(final_model_dir)

        if self.args.should_save:
            logger.info(
                f"💾 Final model saved with descriptive name: {final_model_dir}"
            )

        return final_model_dir

    @rank0_only
    def _log_distributed_info(self):
        """Log distributed training information for debugging (rank 0 only)."""
        world_size = getattr(self.args, "world_size", 1)
        process_index = getattr(self.args, "process_index", 0)
        local_rank = getattr(self.args, "local_rank", 0)

        logger.info("🌐 Distributed Training Info:")
        logger.info(f"   World Size: {world_size}")
        logger.info(f"   Process Index (Global Rank): {process_index}")
        logger.info(f"   Local Rank: {local_rank}")
        logger.info(f"   Should Save (Rank 0 only): {self.args.should_save}")
        logger.info(f"   DeepSpeed Enabled: {self.is_deepspeed_enabled}")

    def _add_eval_component_metrics(
        self, metrics: Dict[str, Any], eval_dataset, metric_key_prefix: str
    ):
        """Add component loss metrics to evaluation results."""
        # Get averaged evaluation loss components
        eval_components = self.training_state_manager.get_averaged_losses_and_reset()

        # Debug: Log what components we got
        if self.args.should_save:  # Only log on rank 0
            logger.info(
                f"🔍 Evaluation components captured: {list(eval_components.keys())}"
            )
            for comp_name, comp_value in eval_components.items():
                logger.info(f"   {comp_name}: {comp_value:.4f}")

        # Add component metrics with eval prefix
        for component_name, component_value in eval_components.items():
            if component_name != "loss":  # eval_loss is already added by HuggingFace
                metrics[f"{metric_key_prefix}_{component_name}"] = round(
                    component_value, 4
                )

        # If no components were captured, log a warning
        if not eval_components and self.args.should_save:
            logger.warning(
                "⚠️ No evaluation loss components captured - check if model.get_last_loss_components() is working during evaluation"
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

    def create_optimizer(self):
        """Create optimizer with differential learning rates for vision, merger, and LLM components."""
        import torch

        if self.optimizer is None:
            # Get training config for learning rates from the wrapped detection model
            config = getattr(self.detection_model, "training_config", None)

            # Create parameter groups with different learning rates
            param_groups = []

            # Vision parameters (lowest LR)
            vision_params = []
            # Merger parameters (highest LR)
            merger_params = []
            # LLM parameters (medium LR)
            llm_params = []
            # Optional groups
            top_layers_params = []
            coord_slice_params = []

            # Resolve optional staged settings
            top_k_layers = (
                int(getattr(config, "prog_unfreeze_top_k_layers", 0))
                if getattr(config, "prog_unfreeze_enabled", False)
                else 0
            )
            num_layers = int(getattr(config, "model_num_layers", 0) or 0)

            def _extract_layer_index(param_name: str):
                marker = "model.layers."
                if marker not in param_name:
                    return None
                try:
                    after = param_name.split(marker, 1)[1]
                    idx_str = after.split(".", 1)[0]
                    return int(idx_str)
                except Exception:
                    return None

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                # Categorize parameters based on module names (following src/ pattern)
                if "visual.merger" in name:
                    # MLP connector/aligner gets merger_lr (highest)
                    merger_params.append(param)
                elif "visual" in name:
                    # Vision encoder gets vision_lr (lowest)
                    vision_params.append(param)
                elif top_k_layers > 0 and _extract_layer_index(name) is not None:
                    idx = _extract_layer_index(name)
                    if (
                        idx is not None
                        and num_layers > 0
                        and idx >= max(0, num_layers - top_k_layers)
                    ):
                        top_layers_params.append(param)
                    else:
                        llm_params.append(param)
                elif any(
                    key in name for key in ("embed_tokens.weight", "lm_head.weight")
                ):
                    coord_slice_params.append(param)
                else:
                    # Default to LLM parameters (includes model.layers, embed_tokens, lm_head, etc.)
                    llm_params.append(param)

            # Create parameter groups with explicit mapping for correct LR logging
            # Store group mapping for correct learning rate labels
            self._param_group_mapping = []

            # Vision group
            if vision_params:
                param_groups.append(
                    {
                        "params": vision_params,
                        "lr": config.vision_lr
                        if hasattr(config, "vision_lr")
                        else self.args.learning_rate,
                        "name": "vision",
                    }
                )
                self._param_group_mapping.append("vision")

            # Merger group (only if parameters exist)
            if merger_params:
                param_groups.append(
                    {
                        "params": merger_params,
                        "lr": config.merger_lr
                        if hasattr(config, "merger_lr")
                        else self.args.learning_rate,
                        "name": "merger",
                    }
                )
                self._param_group_mapping.append("merger")

            # Top layers group (optional)
            if top_layers_params:
                tlr = getattr(config, "lr_top_layers", None)
                param_groups.append(
                    {
                        "params": top_layers_params,
                        "lr": tlr
                        if (tlr is not None)
                        else getattr(config, "llm_lr", self.args.learning_rate),
                        "name": "top_layers",
                    }
                )
                self._param_group_mapping.append("top_layers")

            # Coord slice group (optional; embeddings/head rows masked by callback)
            if coord_slice_params:
                clr = getattr(config, "lr_coord_slice", None)
                param_groups.append(
                    {
                        "params": coord_slice_params,
                        "lr": clr
                        if (clr is not None)
                        else getattr(config, "llm_lr", self.args.learning_rate),
                        "name": "coord_slice",
                    }
                )
                self._param_group_mapping.append("coord_slice")

            # LLM group
            if llm_params:
                param_groups.append(
                    {
                        "params": llm_params,
                        "lr": getattr(config, "lr_full_model", None)
                        if getattr(config, "lr_full_model", None) is not None
                        else config.llm_lr
                        if hasattr(config, "llm_lr")
                        else self.args.learning_rate,
                        "name": "llm",
                    }
                )
                self._param_group_mapping.append("llm")

            # Create optimizer with parameter groups
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
                "weight_decay": self.args.weight_decay,
            }

            self.optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """Create learning rate scheduler that works with parameter groups."""
        if optimizer is None:
            optimizer = self.optimizer

        # Use HuggingFace's scheduler creation but ensure it works with parameter groups
        return super().create_scheduler(num_training_steps, optimizer)

    def _get_learning_rates_with_correct_labels(self):
        """Get learning rates with correct component labels."""
        if not hasattr(self, "lr_scheduler") or self.lr_scheduler is None:
            return {}

        try:
            current_lrs = self.lr_scheduler.get_last_lr()
            if not current_lrs:
                return {}

            lr_dict = {"learning_rate": current_lrs[0]}

            # Use our explicit mapping instead of the fixed order
            if hasattr(self, "_param_group_mapping"):
                for i, group_name in enumerate(self._param_group_mapping):
                    if i < len(current_lrs):
                        lr_dict[f"{group_name}_lr"] = current_lrs[i]

            return lr_dict
        except Exception as e:
            logger.warning(f"Failed to get learning rates: {e}")
            return {}

    def _log_sample_debug_info(self, inputs: Dict[str, Any], is_training: bool) -> None:
        """
        Log comprehensive debug information for a sample during training or evaluation.

        This method extracts the conversation text from the tokenized inputs and logs
        detailed analysis including pre-tokenization text, token-level analysis,
        and loss mask validation using the enhanced debug logger.

        Args:
            inputs: Model inputs containing input_ids and other tensors
            is_training: Whether this is during training or evaluation
        """
        if debug_logger is None:
            return

        # Extract input_ids from the batch
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        if input_ids is None:
            logger.warning("No input_ids found in inputs for debug logging")
            return

        # Take the first sample from the batch for logging
        if input_ids.dim() > 1:
            sample_input_ids = input_ids[0]
            sample_labels = labels[0] if labels is not None else None
        else:
            sample_input_ids = input_ids
            sample_labels = labels

        # Decode the conversation text
        if hasattr(self, "processing_class") and self.processing_class is not None:
            # Decode the full conversation
            chat_text = self.processing_class.decode(
                sample_input_ids, skip_special_tokens=False
            )

            # Generate sample ID from input hash
            import hashlib

            sample_id = hashlib.md5(
                str(sample_input_ids.tolist()).encode()
            ).hexdigest()[:8]

            # Check if this looks like a teacher-student conversation
            has_teachers = chat_text.count("<|im_start|>assistant") > 1

            # Log pre-tokenization text with enhanced analysis
            debug_logger.log_pre_tokenization_text(
                chat_text=chat_text,
                sample_id=f"trainer_{sample_id}",
                is_training=is_training,
                has_teachers=has_teachers,
            )

            # Log comprehensive loss mask analysis if we have labels
            if sample_labels is not None:
                # Identify teacher/student spans with improved logic
                teacher_spans, student_spans = self._identify_teacher_student_spans(
                    sample_input_ids, sample_labels, chat_text
                )

                # Create coordinate mask if available
                coordinate_mask = None
                if hasattr(self.detection_model, "token_processor"):
                    try:
                        coordinate_mask = (
                            self.detection_model.token_processor.create_coordinate_mask(
                                sample_input_ids, self.processing_class
                            )
                        )
                    except Exception as e:
                        logger.debug(f"Could not create coordinate mask: {e}")

                # Log comprehensive conversation analysis with full text and span mapping
                debug_logger.log_comprehensive_conversation_analysis(
                    full_conversation=chat_text,
                    input_ids=sample_input_ids,
                    labels=sample_labels,
                    teacher_spans=teacher_spans,
                    student_spans=student_spans,
                    sample_id=f"trainer_{sample_id}",
                    is_training=is_training,
                    tokenizer=self.processing_class,
                    coordinate_mask=coordinate_mask,
                )

    def _identify_teacher_student_spans(
        self, input_ids: torch.Tensor, labels: torch.Tensor, chat_text: str
    ) -> tuple[List[tuple[int, int]], List[tuple[int, int]]]:
        """
        Identify teacher and student response spans in the tokenized conversation.

        Args:
            input_ids: Token sequence
            labels: Labels with masking
            chat_text: Decoded conversation text

        Returns:
            Tuple of (teacher_spans, student_spans) where each span is (start, end)
        """
        teacher_spans = []
        student_spans = []

        # Find unmasked regions (where labels != -100)
        unmasked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if len(unmasked_positions) == 0:
            return teacher_spans, student_spans

        # Group consecutive unmasked positions into spans
        spans = []
        start = unmasked_positions[0].item()
        end = start

        for pos in unmasked_positions[1:]:
            if pos.item() == end + 1:
                end = pos.item()
            else:
                spans.append((start, end + 1))  # +1 for exclusive end
                start = pos.item()
                end = start
        spans.append((start, end + 1))

        # Determine if this is a teacher-student conversation
        has_teachers = chat_text.count("<|im_start|>assistant") > 1

        if has_teachers:
            # For teacher-student conversations, distinguish spans based on position
            # Teacher examples come first, student response comes last
            total_spans = len(spans)
            if total_spans > 1:
                # All spans except the last are teacher responses
                # Last span is the student response (or generation target)
                teacher_spans = spans[:-1]
                student_spans = spans[-1:]
            elif total_spans == 1:
                # Single span in teacher-student conversation
                # This could be a generation case where only teacher examples are provided
                # and student response is being generated
                teacher_spans = spans
                student_spans = []
            else:
                # No spans found
                teacher_spans = []
                student_spans = []
        else:
            # Simple conversation - all spans are student responses
            teacher_spans = []
            student_spans = spans

        return teacher_spans, student_spans
