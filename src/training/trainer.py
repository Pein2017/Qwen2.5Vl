"""
Unified BBU Trainer that combines orchestration and training functionality.

Key Improvements:
- Enhanced error handling and recovery mechanisms
- Better performance monitoring and optimization
- Cleaner code structure with separation of concerns
- Improved logging and debugging capabilities
- More robust NaN handling and stability monitoring
- Better component management and initialization
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

from src.config_unified import UnifiedConfig
from src.data import BBUDataset
from src.logger_utils import get_model_logger, get_training_logger
from src.losses import ObjectDetectionLoss, ResponseParser
from src.models.attention import replace_qwen2_vl_attention_class
from src.models.wrapper import ModelWrapper
from src.training.callbacks import BestCheckpointCallback
from src.training.stability import StabilityMonitor


class TrainingError(Exception):
    """Custom exception for training errors."""

    pass


class ComponentInitializationError(Exception):
    """Custom exception for component initialization errors."""

    pass


@dataclass
class TrainingMetrics:
    """Container for training metrics and statistics."""

    total_steps: int = 0
    successful_steps: int = 0
    skipped_steps: int = 0
    nan_loss_count: int = 0
    zero_loss_count: int = 0
    consecutive_nan_count: int = 0
    eval_step_count: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0

    @property
    def success_ratio(self) -> float:
        """Calculate success ratio."""
        return self.successful_steps / max(self.total_steps, 1)

    @property
    def skip_ratio(self) -> float:
        """Calculate skip ratio."""
        return self.skipped_steps / max(self.total_steps, 1)

    @property
    def recovery_success_ratio(self) -> float:
        """Calculate recovery success ratio."""
        return self.successful_recoveries / max(self.recovery_attempts, 1)

    def reset_consecutive_counters(self):
        """Reset consecutive failure counters."""
        self.consecutive_nan_count = 0

    def log_summary(self, logger):
        """Log training metrics summary."""
        logger.info("📊 Training Metrics Summary:")
        logger.info(f"   Total steps: {self.total_steps}")
        logger.info(
            f"   Successful steps: {self.successful_steps} ({self.success_ratio:.2%})"
        )
        logger.info(f"   Skipped steps: {self.skipped_steps} ({self.skip_ratio:.2%})")
        logger.info(f"   NaN losses: {self.nan_loss_count}")
        logger.info(f"   Zero losses: {self.zero_loss_count}")
        logger.info(f"   Recovery attempts: {self.recovery_attempts}")
        logger.info(
            f"   Successful recoveries: {self.successful_recoveries} ({self.recovery_success_ratio:.2%})"
        )


class ComponentManager:
    """Manages initialization and configuration of training components."""

    def __init__(self, config: UnifiedConfig, logger):
        self.config = config
        self.logger = logger

    def setup_model_components(self) -> Tuple[nn.Module, Any, Any]:
        """Setup model, tokenizer, and image processor."""
        self.logger.info("🔧 Setting up model components...")

        try:
            model_wrapper = ModelWrapper(self.config, self.logger)
            model, tokenizer, image_processor = model_wrapper.load_all()

            # Validate components
            self._validate_model_components(model, tokenizer, image_processor)

            return model, tokenizer, image_processor

        except Exception as e:
            raise ComponentInitializationError(f"Failed to setup model components: {e}")

    def _validate_model_components(self, model, tokenizer, image_processor):
        """Validate that model components are properly initialized."""
        if model is None:
            raise ComponentInitializationError("Model is None")

        if tokenizer is None:
            raise ComponentInitializationError("Tokenizer is None")

        if image_processor is None:
            raise ComponentInitializationError("Image processor is None")

        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            raise ComponentInitializationError("Model has no parameters")

        # Check tokenizer functionality
        try:
            test_tokens = tokenizer.encode("test", return_tensors="pt")
            if test_tokens.numel() == 0:
                raise ComponentInitializationError("Tokenizer produces empty output")
        except Exception as e:
            raise ComponentInitializationError(f"Tokenizer validation failed: {e}")

    def setup_attention_optimization(self):
        """Setup flash attention optimization."""
        try:
            replace_qwen2_vl_attention_class()
            self.logger.info("✅ Flash attention enabled successfully")
            self.logger.info("   → Works with both padding and flattened data formats")
            self.logger.info("   → Significant performance improvement expected")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not enable flash attention: {e}")
            self.logger.warning(
                "⚠️ Training will continue with standard attention (slower)"
            )
            self.logger.warning(
                "⚠️ Consider installing flash-attn for better performance"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Flash attention setup failed: {e}")
            self.logger.warning("⚠️ Training will continue with standard attention")

    def setup_datasets(
        self, tokenizer, image_processor
    ) -> Tuple[BBUDataset, Optional[BBUDataset]]:
        """Setup train and validation datasets."""
        self.logger.info("🔧 Setting up datasets...")

        try:
            train_dataset = BBUDataset(
                self.config, tokenizer, image_processor, self.config.train_data_path
            )

            # Validate train dataset
            if len(train_dataset) == 0:
                raise ComponentInitializationError("Train dataset is empty")

            eval_dataset = None
            if hasattr(self.config, "val_data_path") and self.config.val_data_path:
                eval_dataset = BBUDataset(
                    self.config, tokenizer, image_processor, self.config.val_data_path
                )

                if len(eval_dataset) == 0:
                    self.logger.warning("⚠️ Validation dataset is empty")
                    eval_dataset = None

            self.logger.info(f"✅ Train dataset: {len(train_dataset)} samples")
            if eval_dataset:
                self.logger.info(f"✅ Eval dataset: {len(eval_dataset)} samples")

            return train_dataset, eval_dataset

        except Exception as e:
            raise ComponentInitializationError(f"Failed to setup datasets: {e}")

    def setup_data_collator(self, tokenizer):
        """Setup FlattenedDataCollator for optimal Qwen2.5-VL training."""
        self.logger.info("🔧 Setting up FlattenedDataCollator...")

        try:
            from ..data import FlattenedDataCollator

            # Calculate max total length for packed sequences
            # Use 2x model_max_length to allow for batch packing
            max_total_length = getattr(self.config, "max_total_length", None)
            if max_total_length is None:
                max_total_length = self.config.model_max_length * 2

            collator = FlattenedDataCollator(
                tokenizer=tokenizer,
                max_total_length=max_total_length,
            )

            self.logger.info("✅ Using FlattenedDataCollator")
            self.logger.info("   → Optimized for Qwen2.5-VL with mRoPE embeddings")
            self.logger.info("   → Proper attention mask format for flash attention")
            self.logger.info("   → Memory efficient padding approach")
            self.logger.info("   → Compatible with multi-round conversations")
            self.logger.info("   → Only trains on final assistant response")

            # Log configuration details
            self.logger.info("   📊 Configuration:")
            self.logger.info(f"      max_total_length: {max_total_length}")
            self.logger.info(f"      model_max_length: {self.config.model_max_length}")

            return collator

        except Exception as e:
            raise ComponentInitializationError(f"Failed to setup data collator: {e}")


class LossManager:
    """Manages loss computation and detection loss integration."""

    def __init__(self, config: UnifiedConfig, tokenizer, logger):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.loss_type = config.loss_type

        # Lazy initialization of loss components
        self._detection_loss_fn = None
        self._response_parser = None

    def get_detection_loss_fn(self) -> ObjectDetectionLoss:
        """Lazy initialization of detection loss function."""
        if self._detection_loss_fn is None:
            try:
                self._detection_loss_fn = ObjectDetectionLoss(
                    lm_weight=getattr(self.config, "lm_weight", 1.0),
                    bbox_weight=getattr(self.config, "bbox_weight", 0.6),
                    giou_weight=getattr(self.config, "giou_weight", 0.4),
                    class_weight=getattr(self.config, "class_weight", 0.3),
                    hungarian_matching=getattr(self.config, "hungarian_matching", True),
                    ignore_index=-100,
                    detection_mode=getattr(self.config, "detection_mode", "inference"),
                    inference_frequency=getattr(self.config, "inference_frequency", 5),
                    max_generation_length=getattr(
                        self.config, "max_generation_length", 512
                    ),
                    use_semantic_similarity=getattr(
                        self.config, "use_semantic_similarity", True
                    ),
                    early_training_mode=getattr(
                        self.config, "early_training_mode", True
                    ),
                )
            except Exception as e:
                raise TrainingError(
                    f"Failed to initialize detection loss function: {e}"
                )
        return self._detection_loss_fn

    def get_response_parser(self) -> ResponseParser:
        """Lazy initialization of response parser."""
        if self._response_parser is None:
            try:
                early_training = getattr(self.config, "early_training_mode", True)
                self._response_parser = ResponseParser(
                    early_training_mode=early_training
                )
            except Exception as e:
                raise TrainingError(f"Failed to initialize response parser: {e}")
        return self._response_parser

    def compute_detection_loss(self, model, outputs, inputs) -> Dict[str, torch.Tensor]:
        """Compute detection loss with error handling."""
        # Check if detection loss is disabled
        disable_detection_loss = os.getenv(
            "DISABLE_DETECTION_LOSS", "false"
        ).lower() == "true" or getattr(self.config, "disable_detection_loss", False)

        if self.loss_type != "object_detection" or disable_detection_loss:
            # Ensure loss is on the correct device
            loss = outputs.loss
            if loss is not None:
                # Get the model's device
                model_device = next(model.parameters()).device
                if loss.device != model_device:
                    loss = loss.to(model_device)
            return {"total_loss": loss}

        try:
            # Extract ground truth objects from labels
            ground_truth_objects = self._extract_ground_truth_from_labels(
                inputs.get("labels"), inputs
            )

            # Compute detection loss
            detection_loss_fn = self.get_detection_loss_fn()
            loss_dict = detection_loss_fn(
                model=model,
                outputs=outputs,
                tokenizer=self.tokenizer,
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                ground_truth_objects=ground_truth_objects,
                labels=inputs.get("labels"),
            )

            return loss_dict

        except Exception as e:
            self.logger.warning(f"⚠️ Detection loss computation failed: {e}")
            # Fallback to standard loss with device consistency
            loss = outputs.loss
            if loss is not None:
                model_device = next(model.parameters()).device
                if loss.device != model_device:
                    loss = loss.to(model_device)
            return {"total_loss": loss}

    def _extract_ground_truth_from_labels(
        self, labels: torch.Tensor, inputs: Dict
    ) -> List[List[Dict]]:
        """Extract ground truth objects from labels."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to decode the labels back to object format
            # For now, return empty list to avoid errors
            batch_size = labels.shape[0] if labels is not None else 1
            return [[] for _ in range(batch_size)]
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract ground truth objects: {e}")
            return [[]]

    def set_training_mode(self, early_training: bool = True):
        """Set training mode for loss components."""
        try:
            if self._detection_loss_fn is not None:
                self._detection_loss_fn.set_training_mode(early_training)

            if self._response_parser is not None:
                self._response_parser.early_training_mode = early_training

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to set training mode: {e}")


class BBUTrainer(Trainer):
    """
    Enhanced BBU Trainer with improved error handling and monitoring.

    Key improvements:
    - Better component management
    - Enhanced error recovery
    - Comprehensive metrics tracking
    - Robust NaN handling
    """

    def __init__(
        self,
        config: UnifiedConfig,
        training_args: Optional[TrainingArguments] = None,
        **kwargs,
    ):
        self.config = config
        self.logger = get_training_logger()
        self.model_logger = get_model_logger()

        # Initialize metrics and managers
        self.metrics = TrainingMetrics()
        self.component_manager = ComponentManager(config, self.logger)

        # Setup components
        self._setup_components()

        # Initialize loss manager
        self.loss_manager = LossManager(config, self.tokenizer, self.logger)

        # Create training arguments if not provided
        if training_args is None:
            training_args = self._create_training_arguments()

        # Setup callbacks
        callbacks = self._setup_callbacks(training_args, kwargs.get("callbacks", []))
        kwargs["callbacks"] = callbacks

        # Initialize stability monitor
        self.stability_monitor = StabilityMonitor(
            config=self.config, logger=self.logger
        )

        # Configure NaN handling
        self._configure_nan_handling()

        # Initialize parent trainer
        super().__init__(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            **kwargs,
        )

        self.logger.info("✅ BBUTrainer initialized successfully")

    def _setup_components(self):
        """Setup all training components."""
        try:
            # Setup attention optimization
            self.component_manager.setup_attention_optimization()

            # Setup model components
            self.model, self.tokenizer, self.image_processor = (
                self.component_manager.setup_model_components()
            )

            # Setup datasets
            self.train_dataset, self.eval_dataset = (
                self.component_manager.setup_datasets(
                    self.tokenizer, self.image_processor
                )
            )

            # Setup data collator
            self.data_collator = self.component_manager.setup_data_collator(
                self.tokenizer
            )

        except Exception as e:
            raise ComponentInitializationError(f"Failed to setup components: {e}")

    def _setup_callbacks(
        self, training_args: TrainingArguments, existing_callbacks: List
    ) -> List:
        """Setup training callbacks."""
        try:
            callbacks = existing_callbacks.copy()

            # Add best checkpoint callback if evaluation is enabled
            if (
                training_args.evaluation_strategy != "no"
                and self.eval_dataset is not None
            ):
                callbacks.append(
                    BestCheckpointCallback(
                        save_total_limit=getattr(self.config, "save_total_limit", 2),
                        metric_name="eval_loss",
                        greater_is_better=False,
                    )
                )

            return callbacks

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup callbacks: {e}")
            return existing_callbacks

    def _configure_nan_handling(self):
        """Configure NaN handling for training."""
        try:
            # Set environment variables for better NaN detection
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

            # Enable anomaly detection in debug mode
            if getattr(self.config, "debug_mode", False):
                torch.autograd.set_detect_anomaly(True)
                self.logger.info("🔍 Anomaly detection enabled for debugging")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to configure NaN handling: {e}")

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from config."""
        try:
            # Extract training arguments from config
            args_dict = {
                "output_dir": getattr(self.config, "output_dir", "output"),
                "num_train_epochs": getattr(self.config, "epochs", 10),
                "per_device_train_batch_size": getattr(self.config, "batch_size", 2),
                "per_device_eval_batch_size": getattr(self.config, "batch_size", 2),
                "learning_rate": getattr(self.config, "learning_rate", 5e-7),
                "warmup_ratio": getattr(self.config, "warmup_ratio", 0.1),
                "lr_scheduler_type": getattr(
                    self.config, "lr_scheduler_type", "cosine"
                ),
                "max_grad_norm": getattr(self.config, "max_grad_norm", 0.5),
                "weight_decay": getattr(self.config, "weight_decay", 0.01),
                "gradient_checkpointing": getattr(
                    self.config, "gradient_checkpointing", True
                ),
                "bf16": getattr(self.config, "bf16", True),
                "fp16": getattr(self.config, "fp16", False),
                "evaluation_strategy": getattr(self.config, "eval_strategy", "steps"),
                "eval_steps": getattr(self.config, "eval_steps", 20),
                "save_strategy": getattr(self.config, "save_strategy", "steps"),
                "save_steps": getattr(self.config, "save_steps", 20),
                "save_total_limit": getattr(self.config, "save_total_limit", 2),
                "logging_steps": getattr(self.config, "logging_steps", 10),
                "log_level": getattr(self.config, "log_level", "INFO"),
                "report_to": getattr(self.config, "report_to", "tensorboard"),
                "dataloader_num_workers": getattr(
                    self.config, "dataloader_num_workers", 8
                ),
                "dataloader_pin_memory": getattr(self.config, "pin_memory", True),
                "dataloader_prefetch_factor": getattr(
                    self.config, "prefetch_factor", 2
                ),
                "remove_unused_columns": getattr(
                    self.config, "remove_unused_columns", False
                ),
                "run_name": getattr(self.config, "run_name", None),
            }

            # Add DeepSpeed config if enabled
            if getattr(self.config, "deepspeed_enabled", True):
                deepspeed_config = getattr(
                    self.config, "deepspeed_config", "scripts/zero2.json"
                )
                if os.path.exists(deepspeed_config):
                    args_dict["deepspeed"] = deepspeed_config
                else:
                    self.logger.warning(
                        f"⚠️ DeepSpeed config not found: {deepspeed_config}"
                    )

            return TrainingArguments(**args_dict)

        except Exception as e:
            raise ComponentInitializationError(
                f"Failed to create training arguments: {e}"
            )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Enhanced training step with better error handling."""
        self.metrics.total_steps += 1

        try:
            # Validate inputs
            self._validate_inputs(inputs)

            # Compute loss
            loss = self.compute_loss(model, inputs, return_outputs=False)

            # Check for NaN or unstable loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.metrics.nan_loss_count += 1
                self.metrics.consecutive_nan_count += 1

                status = self.stability_monitor.check_loss_stability(loss.item())
                return self._handle_unstable_loss(loss, status, model, inputs)

            # Check for zero loss
            if loss.item() == 0.0:
                self.metrics.zero_loss_count += 1
                self.logger.warning(
                    "⚠️ Zero loss detected - this may indicate a problem"
                )

            # Record successful step
            self._record_successful_step(loss)

            return loss

        except Exception as e:
            self.metrics.skipped_steps += 1
            import traceback

            traceback.print_exc()
            return self._handle_training_exception(e, model)

    def _handle_unstable_loss(self, loss, status, model, inputs):
        """Handle unstable loss with recovery attempts."""
        if status == "critical":
            self.logger.error("💥 Critical loss instability detected")
            self._raise_training_failure(["Critical NaN loss"])

        # Attempt recovery
        self.metrics.recovery_attempts += 1
        recovered_loss = self._attempt_loss_recovery(loss, model, inputs)

        if recovered_loss is not None:
            self.metrics.successful_recoveries += 1
            self.logger.info("✅ Loss recovery successful")
            return recovered_loss

        # Skip this step
        self.metrics.skipped_steps += 1
        return self._skip_training_step(loss, [status])

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Enhanced loss computation with detection loss integration."""
        try:
            # Standard forward pass
            outputs = model(**inputs)

            # Compute detection loss if enabled
            loss_dict = self.loss_manager.compute_detection_loss(model, outputs, inputs)

            # Log losses if needed
            if self._should_log_losses():
                self._log_training_losses(loss_dict, outputs)

            total_loss = loss_dict["total_loss"]

            if return_outputs:
                return total_loss, outputs
            return total_loss

        except Exception as e:
            self.logger.error(f"💥 Loss computation failed: {e}")
            import traceback

            traceback.print_exc()
            raise TrainingError(f"Loss computation failed: {e}")

    def _should_log_losses(self) -> bool:
        """Determine if losses should be logged."""
        return (
            hasattr(self, "state")
            and self.state.global_step % self.args.logging_steps == 0
        )

    def _log_training_losses(self, loss_dict: Dict[str, torch.Tensor], outputs):
        """Log training losses with detailed breakdown."""
        try:
            log_data = {}

            for loss_name, loss_value in loss_dict.items():
                if isinstance(loss_value, torch.Tensor):
                    log_data[f"train/{loss_name}"] = loss_value.item()

            # Add standard loss if available
            if hasattr(outputs, "loss") and outputs.loss is not None:
                log_data["train/lm_loss"] = outputs.loss.item()

            # Log to tensorboard/wandb
            if hasattr(self, "log"):
                self.log(log_data)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log training losses: {e}")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Enhanced prediction step with better error handling."""
        self.metrics.eval_step_count += 1

        try:
            # Validate inputs
            self._validate_inputs(inputs)

            # Standard prediction step
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else None

                # Log evaluation losses if needed
                if self._should_log_eval_losses(model, prediction_loss_only):
                    self._log_evaluation_losses(model, inputs)

                return (loss, None, None)

        except Exception as e:
            self.logger.warning(f"⚠️ Prediction step failed: {e}")
            return (None, None, None)

    def _should_log_eval_losses(self, model, prediction_loss_only) -> bool:
        """Determine if evaluation losses should be logged."""
        return (
            not prediction_loss_only
            and hasattr(self, "state")
            and self.state.global_step % (self.args.logging_steps * 2) == 0
        )

    def _log_evaluation_losses(self, model, inputs):
        """Log evaluation losses with detection loss breakdown."""
        try:
            # Compute detection loss for evaluation
            with torch.no_grad():
                outputs = model(**inputs)
                loss_dict = self.loss_manager.compute_detection_loss(
                    model, outputs, inputs
                )

                log_data = {}
                for loss_name, loss_value in loss_dict.items():
                    if isinstance(loss_value, torch.Tensor):
                        log_data[f"eval/{loss_name}"] = loss_value.item()

                # Log to tensorboard/wandb
                if hasattr(self, "log"):
                    self.log(log_data)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log evaluation losses: {e}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with better error handling."""
        try:
            self.logger.info("🔍 Starting evaluation...")

            # Use temporary training mode for evaluation
            with self.temporary_training_mode(early_training=False):
                results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

            self.logger.info(
                f"✅ Evaluation completed: {results.get('eval_loss', 'N/A')}"
            )
            return results

        except Exception as e:
            self.logger.error(f"💥 Evaluation failed: {e}")
            # Return dummy results to prevent training interruption
            return {"eval_loss": float("inf")}

    def _validate_inputs(self, inputs):
        """Validate training inputs."""
        if not isinstance(inputs, dict):
            raise TrainingError("Inputs must be a dictionary")

        required_keys = ["input_ids"]
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise TrainingError(f"Missing required input keys: {missing_keys}")

        # Validate tensor shapes
        input_ids = inputs["input_ids"]
        if input_ids.numel() == 0:
            raise TrainingError("Empty input_ids tensor")

    def _skip_training_step(self, loss, issues):
        """Skip training step due to issues."""
        self.logger.warning(f"⚠️ Skipping training step due to: {', '.join(issues)}")
        # Use the loss device if available, otherwise use model device
        if loss is not None and hasattr(loss, "device"):
            device = loss.device
        else:
            # Fallback to model device
            device = next(iter(self.model.parameters())).device
        return torch.tensor(0.0, requires_grad=True, device=device)

    def _attempt_loss_recovery(self, loss, model, inputs):
        """Attempt to recover from loss issues."""
        try:
            # Clear gradients
            model.zero_grad()

            # Get model device for consistency
            model_device = next(model.parameters()).device

            # Try recomputing with different precision
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(**inputs)
                recovered_loss = outputs.loss

                if recovered_loss is not None:
                    # Ensure recovered loss is on the correct device
                    if recovered_loss.device != model_device:
                        recovered_loss = recovered_loss.to(model_device)

                    if not torch.isnan(recovered_loss) and not torch.isinf(
                        recovered_loss
                    ):
                        return recovered_loss

        except Exception as e:
            self.logger.warning(f"⚠️ Loss recovery attempt failed: {e}")

        return None

    def _record_successful_step(self, loss):
        """Record successful training step."""
        self.metrics.successful_steps += 1
        self.metrics.reset_consecutive_counters()

    def _handle_training_exception(self, exception, model):
        """Handle training exceptions."""
        self.logger.error(f"💥 Training step exception: {exception}")

        # Try to recover
        try:
            model.zero_grad()
            # Create tensor on the same device as the model
            model_device = next(model.parameters()).device
            return torch.tensor(0.0, requires_grad=True, device=model_device)
        except Exception:
            import traceback

            traceback.print_exc()
            raise TrainingError(f"Unrecoverable training error: {exception}")

    def _raise_training_failure(self, issues):
        """Raise training failure with detailed information."""
        error_msg = f"Training failed due to: {', '.join(issues)}"
        self.logger.error(f"💥 {error_msg}")
        self.metrics.log_summary(self.logger)
        raise TrainingError(error_msg)

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        """Enhanced checkpoint saving with error handling."""
        try:
            super()._save_checkpoint(model, trial, *args, **kwargs)
            self.logger.info("💾 Checkpoint saved successfully")
        except Exception as e:
            self.logger.error(f"💥 Failed to save checkpoint: {e}")

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """Enhanced model saving with additional components."""
        try:
            super().save_model(output_dir, _internal_call)

            if output_dir is not None:
                self._save_additional_components(output_dir)

        except Exception as e:
            self.logger.error(f"💥 Failed to save model: {e}")

    def _save_additional_components(self, output_dir: str):
        """Save additional training components."""
        try:
            import json
            from pathlib import Path

            output_path = Path(output_dir)

            # Save training metrics
            metrics_file = output_path / "training_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "total_steps": self.metrics.total_steps,
                        "successful_steps": self.metrics.successful_steps,
                        "skipped_steps": self.metrics.skipped_steps,
                        "nan_loss_count": self.metrics.nan_loss_count,
                        "zero_loss_count": self.metrics.zero_loss_count,
                        "success_ratio": self.metrics.success_ratio,
                        "skip_ratio": self.metrics.skip_ratio,
                    },
                    f,
                    indent=2,
                )

            self.logger.info(f"💾 Additional components saved to {output_dir}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save additional components: {e}")

    @contextmanager
    def temporary_training_mode(self, early_training: bool):
        """Context manager for temporary training mode changes."""
        try:
            # Save current mode
            original_mode = (
                getattr(
                    self.loss_manager._detection_loss_fn, "early_training_mode", True
                )
                if self.loss_manager._detection_loss_fn
                else True
            )

            # Set new mode
            self.loss_manager.set_training_mode(early_training)

            yield

        finally:
            # Restore original mode
            self.loss_manager.set_training_mode(original_mode)

    def set_early_training_mode(self, early_training: bool = True):
        """Set early training mode for loss components."""
        self.loss_manager.set_training_mode(early_training)

    def test_data_loading(self, num_samples: int = 2):
        """Test data loading functionality."""
        try:
            self.logger.info(f"🧪 Testing data loading with {num_samples} samples...")

            for i, batch in enumerate(self.get_train_dataloader()):
                if i >= num_samples:
                    break

                self.logger.info(f"   Batch {i}: {batch['input_ids'].shape}")

            self.logger.info("✅ Data loading test completed successfully")

        except Exception as e:
            self.logger.error(f"💥 Data loading test failed: {e}")

    def test_model_forward(self):
        """Test model forward pass."""
        try:
            self.logger.info("🧪 Testing model forward pass...")

            # Get a sample batch
            dataloader = self.get_train_dataloader()
            batch = next(iter(dataloader))

            # Move to device
            device = next(self.model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Test forward pass
            with torch.no_grad():
                outputs = self.model(**batch)

            self.logger.info(
                f"✅ Forward pass test completed: loss = {outputs.loss.item()}"
            )

        except Exception as e:
            self.logger.error(f"💥 Forward pass test failed: {e}")

    def log_training_statistics(self):
        """Log comprehensive training statistics."""
        self.metrics.log_summary(self.logger)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self.logger.info("🎯 Training completed!")
        self.log_training_statistics()

    def on_epoch_start(self):
        """Called at the start of each epoch."""
        current_epoch = getattr(self.state, "epoch", 0)

        # Switch to strict training mode after early training epochs
        early_training_epochs = getattr(self.config, "early_training_epochs", 3)
        if current_epoch >= early_training_epochs:
            self.set_early_training_mode(early_training=False)
            self.logger.info(
                f"🔄 Switched to strict training mode at epoch {current_epoch}"
            )


def create_trainer(
    config: UnifiedConfig, training_args: Optional[TrainingArguments] = None, **kwargs
) -> BBUTrainer:
    """Factory function to create a BBU trainer."""
    try:
        return BBUTrainer(config, training_args, **kwargs)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise ComponentInitializationError(f"Failed to create trainer: {e}")


class EvalPrediction:
    """
    Enhanced evaluation prediction container.

    Provides better handling of predictions and labels for evaluation.
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, tuple[np.ndarray]],
        label_ids: Union[np.ndarray, tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        self.losses = losses

    def __iter__(self):
        return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        else:
            raise IndexError("EvalPrediction only has predictions and label_ids")
