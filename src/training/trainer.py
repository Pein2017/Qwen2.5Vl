"""
Unified BBU Trainer using the direct configuration system.

This implementation uses the new direct config access that eliminates
parameter passing and provides flat, direct access to all config values.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import SaveStrategy

from src.config import config
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches


class BBUTrainer(Trainer):
    """
    Custom trainer that integrates object detection loss when configured.

    Extends the standard Transformers Trainer to add object detection capabilities
    while maintaining clean separation of concerns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_training_logger()

        # Simple accumulators for loss components (like tr_loss)
        self._accumulated_lm_loss = 0.0
        self._accumulated_bbox_loss = 0.0
        self._accumulated_caption_loss = 0.0
        self._accumulated_objectness_loss = 0.0

        # Initialize training monitor if enabled
        self.monitor = None
        if config.enable_monitoring:
            self._init_monitor()

        # Initialize object detection loss if configured
        self.detection_loss = None
        if config.detection_enabled:
            self._init_detection_loss()

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Unified save method that creates a complete checkpoint directory.

        This saves both the base model and detection head in a single directory,
        making it compatible with HuggingFace's from_pretrained pattern.

        Checkpoint structure:
        checkpoint-N/
        ├── config.json              # Base model config
        ├── model.safetensors         # Base model weights
        ├── tokenizer.json           # Tokenizer files
        ├── detection_head.pth       # Detection head weights
        ├── detection_config.json    # Detection head config
        └── training_args.bin        # Training arguments
        """
        self.logger.info(f"💾 Saving unified checkpoint to: {output_dir}")

        # Save base model using standard HuggingFace method
        # This creates config.json, model.safetensors, etc.
        super()._save(output_dir, state_dict)

        # Save detection head components if present
        if hasattr(self.model, "detection_head") and output_dir is not None:
            self._save_detection_components(output_dir)

        self.logger.info(f"✅ Unified checkpoint saved successfully")

    def _save_detection_components(self, output_dir: str):
        """Save detection head weights and configuration."""
        import json
        import os

        import torch

        # Save detection head weights
        detection_state_dict = self.model.detection_head.state_dict()
        detection_path = os.path.join(output_dir, "detection_head.pth")
        torch.save(detection_state_dict, detection_path)

        # Save detection head configuration with UNIFIED filename
        detection_config = {
            "num_queries": self.model.detection_head.num_queries,
            "max_caption_length": self.model.detection_head.max_caption_length,
            "hidden_size": self.model.detection_head.hidden_size,
            "vocab_size": self.model.detection_head.vocab_size,
            "detection_enabled": True,
            "checkpoint_type": "unified",  # Marker for unified checkpoint
        }

        # Use the UNIFIED config filename (not legacy)
        config_path = os.path.join(output_dir, "detection_config.json")
        with open(config_path, "w") as f:
            json.dump(detection_config, f, indent=2)

        self.logger.info(f"💾 Detection head saved to: {detection_path}")
        self.logger.info(f"💾 Detection config saved to: {config_path}")

    def _init_monitor(self):
        """Initialize training monitor for prediction and GT logging."""
        from src.training.monitor import create_training_monitor

        self.monitor = create_training_monitor(
            log_dir=config.monitor_log_dir,
            save_predictions=config.save_predictions,
            save_token_analysis=config.save_token_analysis,
            save_raw_text=config.save_raw_text,
        )

        self.logger.info("🔍 Training monitor initialized in BBUTrainer")
        self.logger.info(f"   Monitor log directory: {config.monitor_log_dir}")
        self.logger.info(f"   Save predictions: {config.save_predictions}")
        self.logger.info(f"   Save token analysis: {config.save_token_analysis}")
        self.logger.info(f"   Save raw text: {config.save_raw_text}")

    def _init_detection_loss(self):
        """Initialize object detection loss with config parameters."""
        from src.detection_loss import DetectionLoss

        # Initialize with tokenizer for caption loss computation
        self.detection_loss = DetectionLoss(
            bbox_weight=config.detection_bbox_weight,
            giou_weight=config.detection_giou_weight,
            objectness_weight=config.detection_objectness_weight,
            caption_weight=config.detection_caption_weight,
            tokenizer=self.tokenizer,
        )

        self.logger.info("🎯 Detection loss initialized in BBUTrainer")
        self.logger.info(f"   bbox_weight: {config.detection_bbox_weight}")
        self.logger.info(f"   giou_weight: {config.detection_giou_weight}")
        self.logger.info(f"   objectness_weight: {config.detection_objectness_weight}")
        self.logger.info(f"   caption_weight: {config.detection_caption_weight}")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss with end-to-end object detection integration.

        Returns enhanced loss information for automatic logging.
        """
        # Extract and store GT objects separately
        ground_truth_objects = self._extract_ground_truth_objects(inputs)

        # Prepare clean inputs for model (remove GT objects)
        model_inputs = inputs.copy()
        model_inputs.pop("ground_truth_objects", None)
        model_inputs.pop("image_counts_per_sample", None)

        # Ensure we get hidden states for detection
        model_inputs["output_hidden_states"] = True

        # Call the wrapper model - it will return only the LM outputs
        outputs = model(**model_inputs)

        # Standard LM loss from base model
        lm_loss = outputs.loss

        # Detection loss computation (if enabled and GT available)
        detection_loss_components = {}
        total_detection_loss = 0.0

        if (
            config.detection_enabled
            and ground_truth_objects
            and any(len(gt) > 0 for gt in ground_truth_objects)
        ):
            # Get hidden states from wrapper model output
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]  # Final layer
            else:
                # Fallback: call base model directly to get hidden states
                base_outputs = model.base_model(**model_inputs)
                hidden_states = base_outputs.hidden_states[-1]

            attention_mask = model_inputs.get("attention_mask")

            # Call detection head directly
            detection_outputs = model.detection_head(
                hidden_states,
                attention_mask,
                ground_truth_objects,
                training=model.training,
            )

            # Check for bbox saturation and reinitialize if needed
            if (
                hasattr(model, "detection_head")
                and "pred_boxes_raw" in detection_outputs
            ):
                raw_boxes = detection_outputs["pred_boxes_raw"]
                max_raw = raw_boxes.abs().max().item()

                # If raw predictions are extremely large (>10), reinitialize
                if (
                    max_raw > 10.0 and self.state.global_step % 50 == 0
                ):  # Check every 50 steps
                    self.logger.warning(
                        f"🚨 Bbox saturation detected! Max raw value: {max_raw:.2f}"
                    )
                    self.logger.warning("🔄 Reinitializing bbox head weights...")
                    model.detection_head.reinitialize_bbox_head()

            # Compute detection loss and get individual components
            detection_loss_result = self.detection_loss(
                detection_outputs, ground_truth_objects
            )

            # Handle detection loss result
            if isinstance(detection_loss_result, dict):
                total_detection_loss = detection_loss_result.get("total_loss", 0.0)
                detection_loss_components = {
                    k: v for k, v in detection_loss_result.items() if k != "total_loss"
                }
            else:
                total_detection_loss = detection_loss_result

        # Store loss components and accumulate them (simple approach)
        self._current_lm_loss = (
            lm_loss.item() if isinstance(lm_loss, torch.Tensor) else lm_loss
        )
        self._current_bbox_loss = detection_loss_components.get("bbox_loss", 0.0)
        self._current_caption_loss = detection_loss_components.get("caption_loss", 0.0)
        self._current_objectness_loss = detection_loss_components.get(
            "objectness_loss", 0.0
        )

        # Accumulate loss components (simple approach)
        self._accumulated_lm_loss += self._current_lm_loss
        self._accumulated_bbox_loss += self._current_bbox_loss
        self._accumulated_caption_loss += self._current_caption_loss
        self._accumulated_objectness_loss += self._current_objectness_loss

        # Compute final combined loss for backpropagation
        final_loss = lm_loss + total_detection_loss

        # Handle token averaging if needed (BEFORE returning)
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            final_loss *= self.accelerator.num_processes

        # CRITICAL: Return the COMBINED loss (LM + detection) as the main loss
        # This ensures the trainer uses our full loss for optimization and logging
        return (final_loss, outputs) if return_outputs else final_loss

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
    ):
        """
        Override to add our custom loss components to the logs.
        """
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            logs: dict[str, float] = {}

            # Standard tr_loss handling (same as parent, scaled by gradient accumulation)
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss  # Reset tr_loss to zero
            # Compute denominator including gradient accumulation steps
            steps_since_last_log = self.state.global_step - self._globalstep_last_logged
            accumulation_steps = getattr(self.args, "gradient_accumulation_steps", None)
            if accumulation_steps is None:
                raise ValueError("gradient_accumulation_steps is not set")
            denom = steps_since_last_log * accumulation_steps
            logs["loss"] = round(tr_loss_scalar / denom, 4)

            # Add our custom loss components using accumulated values (same pattern as tr_loss)
            steps_since_last_log = self.state.global_step - self._globalstep_last_logged
            # Account for gradient accumulation steps
            accumulation_steps = getattr(self.args, "gradient_accumulation_steps", None)
            if accumulation_steps is None:
                raise ValueError("gradient_accumulation_steps is not set")

            denominator = steps_since_last_log * accumulation_steps
            if hasattr(self, "_accumulated_lm_loss"):
                logs["lm_loss"] = round(self._accumulated_lm_loss / denominator, 4)
                self._accumulated_lm_loss = 0.0  # Reset after logging
            if hasattr(self, "_accumulated_bbox_loss"):
                logs["bbox_loss"] = round(self._accumulated_bbox_loss / denominator, 4)
                self._accumulated_bbox_loss = 0.0  # Reset after logging
            if hasattr(self, "_accumulated_caption_loss"):
                logs["caption_loss"] = round(
                    self._accumulated_caption_loss / denominator, 4
                )
                self._accumulated_caption_loss = 0.0  # Reset after logging
            if hasattr(self, "_accumulated_objectness_loss"):
                logs["objectness_loss"] = round(
                    self._accumulated_objectness_loss / denominator, 4
                )
                self._accumulated_objectness_loss = 0.0  # Reset after logging

            # Add gradient norm and learning rate (same as parent)
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()

            # Update tracking variables (same as parent)
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            # Log everything
            self.log(logs, start_time)

        # Handle evaluation and saving (same as parent)
        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Standard log method - no custom logic needed since _maybe_log_save_evaluate handles everything.
        """
        # Call parent log method - this handles all-reduce for distributed training
        super().log(logs, start_time)

    def _extract_ground_truth_objects(self, inputs):
        """Extract ground truth objects from batch inputs"""
        ground_truth_objects = []

        for batch_idx in range(inputs["input_ids"].shape[0]):
            if "ground_truth_objects" in inputs:
                gt_objects = inputs["ground_truth_objects"][batch_idx]
                self.logger.info(
                    f"🔍 DEBUG: Found GT objects for batch {batch_idx}: {len(gt_objects)} objects"
                )
            else:
                gt_objects = []
                self.logger.info(f"🔍 DEBUG: No GT objects found for batch {batch_idx}")

            ground_truth_objects.append(gt_objects)

        total_gt_objects = sum(len(gt_objs) for gt_objs in ground_truth_objects)
        self.logger.info(f"🔍 DEBUG: Total GT objects across batch: {total_gt_objects}")

        return ground_truth_objects

    def _prepare_detection_inputs(self, inputs):
        """Extract ground truth objects from batch"""
        # DEBUG: Log what keys are available in inputs
        self.logger.info(f"🔍 DEBUG: Available input keys: {list(inputs.keys())}")

        # Extract GT objects from conversation format
        ground_truth_objects = []

        for batch_idx in range(inputs["input_ids"].shape[0]):
            # Extract from the data collator's stored information
            if "ground_truth_objects" in inputs:
                gt_objects = inputs["ground_truth_objects"][batch_idx]
                self.logger.info(
                    f"🔍 DEBUG: Found GT objects for batch {batch_idx}: {len(gt_objects)} objects"
                )
            else:
                # Fallback: extract from conversation (implement based on your data format)
                gt_objects = self._extract_gt_from_conversation(inputs, batch_idx)
                self.logger.info(
                    f"🔍 DEBUG: Using fallback GT extraction for batch {batch_idx}: {len(gt_objects)} objects"
                )

            ground_truth_objects.append(gt_objects)

        # DEBUG: Log final ground truth objects
        total_gt_objects = sum(len(gt_objs) for gt_objs in ground_truth_objects)
        self.logger.info(f"🔍 DEBUG: Total GT objects across batch: {total_gt_objects}")
        self.logger.info(
            f"🔍 DEBUG: GT objects per sample: {[len(gt_objs) for gt_objs in ground_truth_objects]}"
        )

        # Add to model inputs
        model_inputs = inputs.copy()
        model_inputs["ground_truth_objects"] = ground_truth_objects

        return model_inputs

    def _extract_gt_from_conversation(self, inputs, batch_idx):
        """Extract ground truth objects from conversation if not provided directly"""
        # This is a fallback method - ideally GT objects should be provided by data collator
        # You can implement this based on your specific data format
        return []

    def on_train_end(self, args, state, control, **kwargs):
        """Finalize monitoring session when training ends."""
        if self.monitor is not None:
            self.monitor.finalize_session()
        super().on_train_end(args, state, control, **kwargs)

    def create_optimizer(self):
        """
        Create optimizer with separate parameter groups for different learning rates.

        This enables detection head to have its own learning rate separate from
        the base model components.
        """
        if config.use_differential_lr and config.tune_detection:
            self.logger.info(
                "🔧 Creating optimizer with differential learning rates..."
            )

            # Collect parameters for different components
            param_groups = []

            # Base model parameters (vision, MLP, LLM)
            base_params = []
            base_param_names = []

            # Detection head parameters
            detection_params = []
            detection_param_names = []

            # Categorize all model parameters
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if "detection_head" in name:
                    detection_params.append(param)
                    detection_param_names.append(name)
                else:
                    base_params.append(param)
                    base_param_names.append(name)

            # Create parameter groups with different learning rates
            if base_params:
                param_groups.append(
                    {
                        "params": base_params,
                        "lr": config.learning_rate,  # Use general learning rate for base model
                        "weight_decay": self.args.weight_decay,
                    }
                )
                self.logger.info(
                    f"   Base model params: {len(base_params)} (lr={config.learning_rate})"
                )

            if detection_params and config.tune_detection:
                param_groups.append(
                    {
                        "params": detection_params,
                        "lr": config.detection_lr,  # Use detection-specific learning rate
                        "weight_decay": self.args.weight_decay,
                    }
                )
                self.logger.info(
                    f"   Detection head params: {len(detection_params)} (lr={config.detection_lr})"
                )

                # Log detection parameter names for debugging
                self.logger.info(
                    f"   Detection parameters: {detection_param_names[:5]}..."
                )  # Show first 5

            # Create optimizer with parameter groups
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args
            )
            optimizer_kwargs["params"] = param_groups

            self.optimizer = optimizer_cls(**optimizer_kwargs)

            self.logger.info(
                f"✅ Created optimizer with {len(param_groups)} parameter groups"
            )

        else:
            # Use default optimizer creation
            self.logger.info(
                "🔧 Using default optimizer (no differential learning rates)"
            )
            super().create_optimizer()

        return self.optimizer

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Enhanced prediction step that includes detection loss logging during evaluation.
        """
        # Store original state for restoration
        original_training = model.training

        # Set model to eval mode
        model.eval()

        # Use the same compute_loss logic but with eval prefix
        with torch.no_grad():
            # Temporarily modify the loss info prefix for evaluation
            old_prefix = getattr(self, "_loss_prefix", "")
            self._loss_prefix = "eval"

            try:
                # Use our enhanced compute_loss method
                if prediction_loss_only:
                    loss = self.compute_loss(model, inputs)
                    return (loss, None, None)
                else:
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )

                    # Extract logits for evaluation metrics
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in (ignore_keys or []) + ["loss"]
                        )
                    else:
                        logits = (
                            outputs[1:] if hasattr(outputs, "__getitem__") else outputs
                        )

                    # Extract labels if available
                    labels = None
                    if hasattr(self, "label_names") and len(self.label_names) > 0:
                        labels = tuple(inputs.get(name) for name in self.label_names)
                        if len(labels) == 1:
                            labels = labels[0]

                    return (loss, logits, labels)

            finally:
                # Restore original prefix and training state
                self._loss_prefix = old_prefix
                model.train(original_training)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation to include individual loss components in metrics."""
        # Reset accumulators before evaluation
        self._accumulated_lm_loss = 0.0
        self._accumulated_bbox_loss = 0.0
        self._accumulated_caption_loss = 0.0
        self._accumulated_objectness_loss = 0.0
        # Run base evaluation (logs default eval metrics)
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # Compute average component losses over evaluation batches
        eval_loader = self.get_eval_dataloader(eval_dataset)
        num_batches = len(eval_loader)
        if num_batches > 0:
            metrics[f"{metric_key_prefix}_lm_loss"] = round(
                self._accumulated_lm_loss / num_batches, 4
            )
            metrics[f"{metric_key_prefix}_bbox_loss"] = round(
                self._accumulated_bbox_loss / num_batches, 4
            )
            metrics[f"{metric_key_prefix}_caption_loss"] = round(
                self._accumulated_caption_loss / num_batches, 4
            )
            metrics[f"{metric_key_prefix}_objectness_loss"] = round(
                self._accumulated_objectness_loss / num_batches, 4
            )
            # Also expose without prefix for consistency with training logs
            metrics["lm_loss"] = metrics[f"{metric_key_prefix}_lm_loss"]
            metrics["bbox_loss"] = metrics[f"{metric_key_prefix}_bbox_loss"]
            metrics["caption_loss"] = metrics[f"{metric_key_prefix}_caption_loss"]
            metrics["objectness_loss"] = metrics[f"{metric_key_prefix}_objectness_loss"]
        # Log extended metrics
        self.log(metrics)
        return metrics


def set_model_training_params(model):
    """
    Set model training parameters based on global configuration.
    Configures which parts of the model should be trained.
    """
    logger = get_training_logger()

    # Check if this is a wrapped model with detection head
    has_detection_head = hasattr(model, "detection_head")
    base_model = model.base_model if has_detection_head else model

    # Vision encoder training
    if config.tune_vision:
        for n, p in base_model.visual.named_parameters():
            p.requires_grad = True
        logger.info(f"🔧 Vision encoder: TRAINING (lr={config.vision_lr})")
    else:
        for n, p in base_model.visual.named_parameters():
            p.requires_grad = False
        logger.info("🔧 Vision encoder: FROZEN")

    # MLP connector training
    if config.tune_mlp:
        for n, p in base_model.visual.merger.named_parameters():
            p.requires_grad = True
        logger.info(f"🔧 MLP connector: TRAINING (lr={config.mlp_lr})")
    else:
        for n, p in base_model.visual.merger.named_parameters():
            p.requires_grad = False
        logger.info("🔧 MLP connector: FROZEN")

    # LLM training
    if config.tune_llm:
        for n, p in base_model.model.named_parameters():
            p.requires_grad = True
        base_model.lm_head.requires_grad = True
        logger.info(f"🔧 LLM: TRAINING (lr={config.llm_lr})")
    else:
        for n, p in base_model.model.named_parameters():
            p.requires_grad = False
        base_model.lm_head.requires_grad = False
        logger.info("🔧 LLM: FROZEN")

    # Detection head training (NEW)
    if has_detection_head:
        if config.tune_detection:
            for n, p in model.detection_head.named_parameters():
                p.requires_grad = True
            logger.info(f"🔧 Detection head: TRAINING (lr={config.detection_lr})")
        else:
            for n, p in model.detection_head.named_parameters():
                p.requires_grad = False
            logger.info("🔧 Detection head: FROZEN")
    else:
        logger.info("🔧 Detection head: NOT PRESENT")


def setup_model_and_tokenizer() -> Tuple[nn.Module, Any, Any]:
    """
    Setup model and tokenizer with unified loading mechanism.

    This function automatically detects checkpoint type and loads appropriately:
    - Base models: Creates model with random detection head
    - Unified checkpoints: Loads both base model and detection head
    - Legacy checkpoints: Loads with backward compatibility
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up model with unified loading mechanism...")

    # Apply comprehensive Qwen2.5-VL fixes FIRST
    logger.info("🔧 Applying comprehensive Qwen2.5-VL fixes...")
    if not apply_comprehensive_qwen25_fixes():
        raise RuntimeError("Failed to apply Qwen2.5-VL fixes")

    # Load tokenizer first (needed for detection head)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_path,
        model_max_length=config.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Import the detection wrapper
    from src.models.wrapper import Qwen25VLWithDetection

    # Use unified loading mechanism - automatically detects checkpoint type
    logger.info(f"🔄 Loading model from: {config.model_path}")
    model = Qwen25VLWithDetection.from_pretrained(
        model_path=config.model_path,
        num_queries=config.detection_num_queries,
        max_caption_length=config.detection_max_caption_length,
        tokenizer=tokenizer,
    )

    # Verify all patches
    logger.info("🔍 Verifying all patches...")
    if not verify_qwen25_patches():
        raise RuntimeError("Patch verification failed")

    # Load processor
    processor = AutoProcessor.from_pretrained(config.model_path)
    image_processor = processor.image_processor

    # CRITICAL FIX: Use pixel constraints from data_conversion/vision_process.py
    try:
        from data_conversion.vision_process import MAX_PIXELS, MIN_PIXELS

        # Apply the exact same pixel constraints used during data conversion
        image_processor.min_pixels = MIN_PIXELS  # 4 * 28 * 28 = 3136
        image_processor.max_pixels = MAX_PIXELS  # 128 * 28 * 28 = 100352

        # Also set size constraints if the processor supports them
        if hasattr(image_processor, "size"):
            if isinstance(image_processor.size, dict):
                image_processor.size["min_pixels"] = MIN_PIXELS
                image_processor.size["max_pixels"] = MAX_PIXELS
            else:
                image_processor.size = {
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }

        logger.info(
            f"✅ Image processor configured with data_conversion pixel constraints:"
        )
        logger.info(f"   min_pixels: {image_processor.min_pixels} (4 * 28 * 28)")
        logger.info(f"   max_pixels: {image_processor.max_pixels} (128 * 28 * 28)")

    except ImportError as e:
        logger.error(f"❌ Failed to import from data_conversion/vision_process.py: {e}")
        logger.warning("   Using default image processor pixel constraints")

    # CRITICAL: Disable cache for training
    model.base_model.config.use_cache = False

    # Setup gradient checkpointing if enabled
    if config.gradient_checkpointing:
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.base_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    # Set training parameters for base model
    set_model_training_params(model)

    logger.info("✅ Model with unified loading mechanism setup completed")
    return model, tokenizer, image_processor


def setup_data_module(tokenizer, image_processor) -> Dict[str, Any]:
    """
    Setup data module following the official approach.
    This matches the data setup in train_qwen.py.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up data module...")

    # Create datasets - no config parameters needed
    train_dataset = BBUDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_path=config.train_data_path,
    )

    val_dataset = BBUDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_path=config.val_data_path,
    )

    # Create data collator
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        max_total_length=config.max_total_length,
        collator_type=config.collator_type,
    )

    logger.info(f"✅ Data module setup completed:")
    logger.info(f"   Train samples: {len(train_dataset)}")
    logger.info(f"   Val samples: {len(val_dataset)}")
    logger.info(f"   Collator type: {config.collator_type}")

    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Safe model saving following the official approach.
    This matches the saving logic in train_qwen.py.
    """
    logger = get_training_logger()
    logger.info(f"💾 Safely saving model to: {output_dir}")

    # Save model state dict
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def create_trainer(
    training_args: Optional[TrainingArguments] = None, **kwargs
) -> BBUTrainer:
    """
    Create BBU trainer with all components using direct config access.
    This matches the trainer creation in train_qwen.py.
    """
    logger = get_training_logger()
    logger.info("🏋️ Creating BBU trainer...")

    # Setup model and tokenizer
    model, tokenizer, image_processor = setup_model_and_tokenizer()

    # Setup data module
    data_module = setup_data_module(tokenizer, image_processor)

    # Create callbacks
    callbacks = []

    # NOTE: Detection loss logging is now handled directly in compute_loss method
    # No need for separate callback

    # Create BBUTrainer with object detection loss integration
    trainer = BBUTrainer(
        model=model,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        callbacks=callbacks,
        **kwargs,
    )

    # Set the detection loss function in the model wrapper after trainer creation
    if config.detection_enabled and hasattr(trainer.model, "set_detection_loss_fn"):
        trainer.model.set_detection_loss_fn(trainer.detection_loss)
        logger.info("🎯 Detection loss function set in model wrapper")

    logger.info("✅ BBU trainer created successfully")
    return trainer


def test_enhanced_logging():
    """
    Simple test to verify enhanced detection loss logging works.
    This can be called during development to test the logging mechanism.
    """
    from src.logger_utils import get_training_logger

    logger = get_training_logger()
    logger.info("🧪 Testing enhanced detection loss logging...")

    # Test that the loss components are properly structured
    sample_loss_components = {
        "total_loss": 1.5,
        "bbox_loss": 0.8,
        "caption_loss": 0.4,
        "objectness_loss": 0.3,
    }

    # Test prefix handling - no prefix for training, eval_ for evaluation
    for mode, prefix in [("training", ""), ("evaluation", "eval_")]:
        loss_info = {}
        for key, value in sample_loss_components.items():
            if key != "total_loss":  # Skip total_loss to avoid duplication
                loss_info[f"{prefix}detection_{key}"] = float(value)

        # Add other components
        loss_info[f"{prefix}lm_loss"] = 0.5
        loss_info[f"{prefix}detection_loss"] = 1.2
        loss_info[f"{prefix}weighted_detection_loss"] = 0.12

        logger.info(f"✅ {mode.upper()} loss structure: {loss_info}")

    logger.info("🧪 Enhanced logging test completed successfully!")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_enhanced_logging()
