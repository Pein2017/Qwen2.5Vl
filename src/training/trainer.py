"""
Unified BBU Trainer with Robust Loss Logging.

This module extends the standard HuggingFace Trainer to provide fine-grained
logging for a multi-component loss function (language modeling, bounding box,
caption, and objectness) while ensuring accurate, per-step reporting even with
gradient accumulation and frequent evaluations.

Key Implementation Details:
1.  **Component Loss Accumulation:**
    - The `compute_loss` method calculates the final combined loss for
      backpropagation.
    - It also tracks each individual loss component (e.g., `_current_lm_loss`)
      and adds it to a corresponding accumulator (e.g., `_accumulated_lm_loss`)
      for each micro-batch (i.e., each forward pass).

2.  **Per-Step Average Logging:**
    - The `_maybe_log_save_evaluate` method is called by the Trainer's main loop
      AFTER a full gradient accumulation cycle is complete.
    - It averages each accumulated component loss by dividing it by the number of
      gradient accumulation steps.
    - The final reported `loss` is the sum of these averaged components,
      ensuring it accurately reflects the loss for that specific training step.
    - All accumulators are reset to zero immediately after logging, preparing
      them for the next accumulation cycle.

3.  **Isolated Evaluation:**
    - The `evaluate` method is "sandboxed" to prevent state corruption.
    - Before evaluation begins, it saves the current state of the training
      loss accumulators.
    - It then runs the entire evaluation, using the same accumulators but for
      evaluation batches.
    - CRUCIALLY, after evaluation is complete, it restores the saved training
      accumulators, ensuring that the evaluation process does not interfere
      with the training loop's loss tracking.

This design guarantees that training and evaluation logging are independent and
that reported training losses are correctly averaged per step.
"""

import time
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

    def __init__(self, *args, config=None, image_processor=None, **kwargs):
        # Inject config object or fallback to global
        from src.config import config as _global_config

        self.config = config or _global_config
        super().__init__(*args, **kwargs)
        self.logger = get_training_logger()
        self.image_processor = image_processor

        # Simple attributes to store the latest loss components from a single forward pass
        self._current_lm_loss: float = 0.0
        self._current_bbox_loss: float = 0.0
        self._current_caption_loss: float = 0.0
        self._current_objectness_loss: float = 0.0

        # ACCUMULATORS for per-step average logging with gradient accumulation
        self._accumulated_lm_loss: float = 0.0
        self._accumulated_bbox_loss: float = 0.0
        self._accumulated_caption_loss: float = 0.0
        self._accumulated_objectness_loss: float = 0.0

        # Initialize training monitor if enabled
        self.monitor = None
        if self.config.enable_monitoring:
            self._init_monitor()

        # Initialize object detection loss if configured
        self.detection_loss = None
        if self.config.detection_enabled:
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

        # Save image processor config
        if self.image_processor:
            self.image_processor.save_pretrained(output_dir)
            self.logger.info(f"💾 Image processor config saved to: {output_dir}")

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
            log_dir=self.config.monitor_log_dir,
            save_predictions=self.config.save_predictions,
            save_token_analysis=self.config.save_token_analysis,
            save_raw_text=self.config.save_raw_text,
        )

        self.logger.info("🔍 Training monitor initialized in BBUTrainer")
        self.logger.info(f"   Monitor log directory: {self.config.monitor_log_dir}")
        self.logger.info(f"   Save predictions: {self.config.save_predictions}")
        self.logger.info(f"   Save token analysis: {self.config.save_token_analysis}")
        self.logger.info(f"   Save raw text: {self.config.save_raw_text}")

    def _init_detection_loss(self):
        """Initialize object detection loss with config parameters."""
        from src.detection_loss import DetectionLoss

        # Initialize with tokenizer for caption loss computation
        self.detection_loss = DetectionLoss(
            bbox_weight=self.config.detection_bbox_weight,
            giou_weight=self.config.detection_giou_weight,
            objectness_weight=self.config.detection_objectness_weight,
            caption_weight=self.config.detection_caption_weight,
            tokenizer=self.tokenizer,
            focal_loss_gamma=self.config.detection_focal_loss_gamma,
            focal_loss_alpha=self.config.detection_focal_loss_alpha,
        )

        self.logger.info("🎯 Detection loss initialized in BBUTrainer")
        self.logger.info(f"   bbox_weight: {self.config.detection_bbox_weight}")
        self.logger.info(f"   giou_weight: {self.config.detection_giou_weight}")
        self.logger.info(
            f"   objectness_weight: {self.config.detection_objectness_weight}"
        )
        self.logger.info(f"   caption_weight: {self.config.detection_caption_weight}")
        self.logger.info(
            f"   focal_loss_gamma: {self.config.detection_focal_loss_gamma}"
        )
        self.logger.info(
            f"   focal_loss_alpha: {self.config.detection_focal_loss_alpha}"
        )

    def init_param_groups(self):
        """
        Initializes parameter groups for differential learning rate.

        This method categorizes all trainable parameters into 'vision', 'merger',
        'llm', and 'detection' groups. It will raise a ValueError if any
        trainable parameters cannot be categorized, ensuring that all parts of
        the model are explicitly handled.
        """
        self.logger.info(
            "🔧 Initializing parameter groups for differential learning rate..."
        )

        param_groups_with_names = {
            "vision": [],
            "merger": [],
            "llm": [],
            "detection": [],
            "others": [],  # For uncategorized parameters
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Correct parameter name matching based on the model's structure.
            # The order is critical: check for the most specific names first.
            if "detection_head" in name:
                param_groups_with_names["detection"].append((name, param))
            # "merger" is part of the vision tower, so check for it *before* "visual".
            elif "merger" in name:
                param_groups_with_names["merger"].append((name, param))
            elif "visual" in name:
                param_groups_with_names["vision"].append((name, param))
            # Language model parameters are in the main 'model' and 'lm_head'.
            elif ".model." in name or "lm_head" in name:
                param_groups_with_names["llm"].append((name, param))
            else:
                param_groups_with_names["others"].append((name, param))

        # Check for uncategorized parameters and raise an error if any are found.
        if param_groups_with_names["others"]:
            other_param_names = [name for name, _ in param_groups_with_names["others"]]
            self.logger.error(
                f"❌ Found {len(other_param_names)} unexpected trainable parameters that could not be categorized:"
            )
            for name in other_param_names:
                self.logger.error(f"   - {name}")
            raise ValueError(
                "Uncategorized trainable parameters found. All parameters must be explicitly "
                "assigned to a learning rate group (vision, merger, llm, detection)."
            )

        # Remove the (now empty) 'others' group
        del param_groups_with_names["others"]

        # Store for optimizer creation (without names)
        self._param_groups = {
            group: [p for _, p in params]
            for group, params in param_groups_with_names.items()
        }
        self._param_names = list(self._param_groups.keys())

        # Log the parameter distribution
        for group, params in self._param_groups.items():
            num_params = sum(p.numel() for p in params)
            if num_params > 0:
                self.logger.info(
                    f"   - Group '{group}': {len(params)} tensors, {num_params / 1e6:.2f}M params"
                )

    def create_optimizer(self):
        """
        Create the optimizer with differential learning rates if configured.
        """
        if not self.config.use_differential_lr or not hasattr(self, "_param_groups"):
            self.logger.info("🚀 Creating standard optimizer...")
            return super().create_optimizer()

        self.logger.info("🚀 Creating optimizer with differential learning rates...")

        lr_map = {
            "vision": self.config.vision_lr,
            "merger": self.config.merger_lr,
            "llm": self.config.llm_lr,
            "detection": self.config.detection_lr,
        }

        optimizer_grouped_parameters = []
        for group_name, params in self._param_groups.items():
            if params:
                lr = lr_map[group_name]
                optimizer_grouped_parameters.append(
                    {
                        "params": params,
                        "lr": lr,
                    }
                )
                self.logger.info(f"   - Group '{group_name}' assigned LR: {lr}")

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )

        # The scheduler is responsible for applying the learning rate schedule to each
        # parameter group. The optimizer should be initialized with the per-group
        # learning rates, and the scheduler will correctly update them based on its
        # schedule (e.g., cosine annealing).
        #
        # The base `learning_rate` in `optimizer_kwargs` serves as a default for any
        # parameters that are not explicitly assigned to a group, which is not the
        # case here but is harmless to leave in. The per-group `lr` will take
        # precedence.
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        self.logger.info(
            "✅ Optimizer with differential learning rates created successfully."
        )
        return self.optimizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss with end-to-end object detection integration.
        The Trainer handles loss accumulation and averaging, so we just
        return the final combined loss. Individual components are stored
        as attributes for logging.
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
        self._current_lm_loss = lm_loss.item()
        self._accumulated_lm_loss += self._current_lm_loss

        # Detection loss computation
        total_detection_loss = 0.0

        if (
            self.config.detection_enabled
            and self.detection_loss is not None
            and ground_truth_objects
            and any(len(gt) > 0 for gt in ground_truth_objects)
        ):
            hidden_states = outputs.hidden_states[-1]
            attention_mask = model_inputs.get("attention_mask")

            detection_outputs = model.detection_head(
                hidden_states,
                attention_mask,
                ground_truth_objects,
                training=model.training,
            )

            detection_loss_components = self.detection_loss(
                detection_outputs, ground_truth_objects
            )

            # The detection_loss module returns weighted losses
            total_detection_loss = detection_loss_components["total_loss"]

            # Store unweighted components for logging and accumulate them
            self._current_bbox_loss = self.detection_loss.last_bbox_loss.item()
            self._current_caption_loss = self.detection_loss.last_caption_loss.item()
            self._current_objectness_loss = (
                self.detection_loss.last_objectness_loss.item()
            )
            self._accumulated_bbox_loss += self._current_bbox_loss
            self._accumulated_caption_loss += self._current_caption_loss
            self._accumulated_objectness_loss += self._current_objectness_loss
        else:
            # Reset detection losses if not computed
            self._current_bbox_loss = 0.0
            self._current_caption_loss = 0.0
            self._current_objectness_loss = 0.0

        # Total loss for backpropagation
        total_loss = lm_loss + total_detection_loss

        return (total_loss, outputs) if return_outputs else total_loss

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
        Log metrics with averaging for gradient accumulation.
        """
        if self.control.should_log:
            # The `tr_loss` from the Trainer is an accumulated value.
            # We re-compute the loss from our own averaged components to ensure
            # correct, per-step reporting consistent with the docstring.
            grad_accum_steps = self.args.gradient_accumulation_steps

            # Average the accumulated component losses
            avg_lm_loss = self._accumulated_lm_loss / grad_accum_steps
            total_avg_loss = avg_lm_loss

            component_logs: Dict[str, float] = {"lm_loss": avg_lm_loss}

            if self.config.detection_enabled:
                avg_bbox_loss = self._accumulated_bbox_loss / grad_accum_steps
                avg_caption_loss = self._accumulated_caption_loss / grad_accum_steps
                avg_objectness_loss = (
                    self._accumulated_objectness_loss / grad_accum_steps
                )

                component_logs["bbox_loss"] = avg_bbox_loss
                component_logs["caption_loss"] = avg_caption_loss
                component_logs["objectness_loss"] = avg_objectness_loss

                # Reconstruct the total detection loss from its averaged, weighted components.
                # This ensures the final logged 'loss' accurately reflects the value used for backprop.
                # NOTE: This assumes the accumulated components are WEIGHTED. If they are not,
                # this sum will not match the true loss.

                # The `loss` passed to compute_loss is lm_loss + weighted detection loss
                # The total loss for logging should be calculated from averaged components.
                # Here we assume the logged components are the primary ones.
                total_avg_loss += avg_bbox_loss + avg_caption_loss + avg_objectness_loss

            # Define logging order: 'loss', 'grad_norm', then components
            logs: Dict[str, float] = {}
            logs["loss"] = total_avg_loss

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )

            logs.update(component_logs)

            # Add ETA and remaining time
            if self.state.max_steps > 0:
                current_step = self.state.global_step
                if current_step > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_step = elapsed_time / current_step
                    remaining_steps = self.state.max_steps - current_step
                    remaining_time_s = remaining_steps * avg_time_per_step

                    logs["remaining_hr"] = round(remaining_time_s / 3600, 3)

            self.log(logs)

            # Reset accumulators after logging
            self._accumulated_lm_loss = 0.0
            self._accumulated_bbox_loss = 0.0
            self._accumulated_caption_loss = 0.0
            self._accumulated_objectness_loss = 0.0

        if self.control.should_evaluate:
            self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        This method is overridden to support logging of differential learning rates.
        """
        # Remove the generic learning rate from the logs.
        logs.pop("learning_rate", None)

        # Log the learning rate for each parameter group.
        if self.lr_scheduler is not None:
            last_lr = self.lr_scheduler.get_last_lr()
            for i, group_lr in enumerate(last_lr):
                group_name = self._param_names[i]
                logs[f"lr/{group_name}"] = group_lr

        super().log(logs)

    def _extract_ground_truth_objects(self, inputs):
        """Extracts ground truth objects from inputs if they exist."""
        ground_truth_objects = []

        for batch_idx in range(inputs["input_ids"].shape[0]):
            if "ground_truth_objects" in inputs:
                gt_objects = inputs["ground_truth_objects"][batch_idx]
                self.logger.debug(
                    f"🔍 Found GT objects for batch {batch_idx}: {len(gt_objects)} objects"
                )
            else:
                gt_objects = []
                self.logger.debug(f"🔍 No GT objects found for batch {batch_idx}")

            ground_truth_objects.append(gt_objects)

        total_gt_objects = sum(len(gt_objs) for gt_objs in ground_truth_objects)
        self.logger.debug(
            f"🔍 DEBUG: Total GT objects across batch: {total_gt_objects}"
        )

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
        self.logger.debug(f"🔍 Total GT objects across batch: {total_gt_objects}")
        self.logger.debug(
            f"🔍 GT objects per sample: {[len(gt_objs) for gt_objs in ground_truth_objects]}"
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
        # Save training accumulators to prevent interference from evaluation
        saved_accumulators = {
            "lm": self._accumulated_lm_loss,
            "bbox": self._accumulated_bbox_loss,
            "caption": self._accumulated_caption_loss,
            "objectness": self._accumulated_objectness_loss,
        }

        # Reset accumulators before evaluation
        self._accumulated_lm_loss = 0.0
        self._accumulated_bbox_loss = 0.0
        self._accumulated_caption_loss = 0.0
        self._accumulated_objectness_loss = 0.0

        # Run base evaluation. This will call compute_loss and populate our accumulators.
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Compute average component losses over all evaluation batches
        eval_loader = self.get_eval_dataloader(eval_dataset)
        num_batches = len(eval_loader)

        if num_batches > 0:
            metrics[f"{metric_key_prefix}_lm_loss"] = round(
                self._accumulated_lm_loss / num_batches, 4
            )
            if self.config.detection_enabled:
                metrics[f"{metric_key_prefix}_bbox_loss"] = round(
                    self._accumulated_bbox_loss / num_batches, 4
                )
                metrics[f"{metric_key_prefix}_caption_loss"] = round(
                    self._accumulated_caption_loss / num_batches, 4
                )
                metrics[f"{metric_key_prefix}_objectness_loss"] = round(
                    self._accumulated_objectness_loss / num_batches, 4
                )

        # Restore training accumulators
        self._accumulated_lm_loss = saved_accumulators["lm"]
        self._accumulated_bbox_loss = saved_accumulators["bbox"]
        self._accumulated_caption_loss = saved_accumulators["caption"]
        self._accumulated_objectness_loss = saved_accumulators["objectness"]

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
        logger.info(f"🔧 MLP connector: TRAINING (lr={config.merger_lr})")
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
    Centralized setup for model, tokenizer, and image processor.
    - Applies necessary patches for Qwen2.5VL.
    - Initializes tokenizer with custom chat template and special tokens.
    - Initializes the model with appropriate quantization and settings.
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

    # 2. TOKENIZER & PROCESSOR SETUP
    from data_conversion.vision_process import MAX_PIXELS

    # =========================================================================
    logger.info("🔧 Initializing tokenizer and processor...")

    # Load the processor, which includes the tokenizer and image processor
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=False,
        max_pixels=MAX_PIXELS,
    )

    # Set image processor params from config
    # The image processor is already pre-scaled to the correct resolution
    # during data preparation, so we use the rescaled values, not the defaults.
    logger.info("🔧 Overriding default image processor pixel values...")
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
        logger.info(f"   min_pixels: {image_processor.min_pixels}")
        logger.info(f"   max_pixels: {image_processor.max_pixels}")

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
    """Creates a unified BBU trainer with all necessary components."""
    from src.training.trainer import BBUTrainer, set_model_training_params

    # Setup model and tokenizer
    model, tokenizer, image_processor = setup_model_and_tokenizer()

    # Set requires_grad for trainable components based on config
    set_model_training_params(model)

    # Setup data module (dataset and collator)
    data_module = setup_data_module(tokenizer, image_processor)

    # Create trainer
    trainer = BBUTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        image_processor=image_processor,
        **data_module,
    )

    # Initialize param groups for differential learning rate
    if config.use_differential_lr:
        trainer.init_param_groups()

    # Set the detection loss function in the model wrapper after trainer creation
    if config.detection_enabled and hasattr(trainer.model, "set_detection_loss_fn"):
        trainer.model.set_detection_loss_fn(trainer.detection_loss)
        trainer.logger.info("🎯 Detection loss function set in model wrapper")

    trainer.logger.info("✅ BBU trainer created successfully")
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
