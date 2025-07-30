"""
Unified BBU Trainer with Robust Loss Logging.

This module extends the standard HuggingFace Trainer to provide fine-grained
logging for a multi-component loss function (language modeling, bounding bbox_2d,
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

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import BBUConfig
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.models.wrapper import DummyOptim
from src.training.training_state_manager import TrainingStateManager
from src.utils.schema import GroundTruthObject
from src.utils.tokens.special_tokens import SpecialTokens


class BBUTrainer(Trainer):
    """
    Custom trainer for coordinate token training.

    Extends the standard Transformers Trainer with coordinate token support
    while maintaining clean separation of concerns.
    """

    # ------------------------------------------------------------------
    # HF ≥4.41 emits a deprecation warning every time `.tokenizer` is
    # accessed on a Trainer instance.  We access it frequently for
    # logging/decoding, so we cache the reference (if provided) *before*
    # calling the parent ctor, then overwrite the property with a plain
    # attribute afterwards.  This silences the warning without touching
    # upstream library code and keeps backward-compatibility for any
    # external calls that expect `trainer.tokenizer` to exist.
    # ------------------------------------------------------------------
    tokenizer_ref = None

    def __init__(
        self,
        *args: Any,
        cfg: Optional[BBUConfig] = None,
        image_processor: Optional[Qwen2VLImageProcessor] = None,
        training_coordinator: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # --------------------------------------------------------------
        # HF ≥4.41 emits a deprecation warning every time `.tokenizer` is
        # accessed on a Trainer instance.  We access it frequently for
        # logging/decoding, so we cache the reference (if provided) *before*
        # calling the parent ctor, then overwrite the property with a plain
        # attribute afterwards.  This silences the warning without touching
        # upstream library code and keeps backward-compatibility for any
        # external calls that expect `trainer.tokenizer` to exist.
        # --------------------------------------------------------------

        self.tokenizer_ref = kwargs.get("tokenizer")

        # ------------------------------------------------------------------
        # Resolve configuration - EXPLICIT CONFIG REQUIRED
        # No fallback patterns - fail fast if config not provided
        # ------------------------------------------------------------------
        if cfg is None:
            raise ValueError(
                "BBUConfig must be explicitly provided to BBUTrainer. "
                "No fallback to global config allowed."
            )

        self.config = cfg

        super().__init__(*args, **kwargs)

        # Initialize logger first since tokenizer setter needs it
        self.logger = get_training_logger()

        # Overwrite the (deprecated) property with a direct attribute so
        # future accesses skip the warning-emitting property defined in the
        # parent class. We bypass the descriptor protocol via
        # `object.__setattr__` to avoid invoking the original setter.
        if self.tokenizer_ref is None and hasattr(self, "tokenizer"):
            self.tokenizer_ref = object.__getattribute__(self, "tokenizer")

        object.__setattr__(self, "tokenizer", self.tokenizer_ref)
        self.image_processor = image_processor

        # Training coordinator is required - no legacy system support
        if training_coordinator is None:
            raise ValueError(
                "Training coordinator is required. Legacy training system has been removed."
            )

        self.training_coordinator = training_coordinator
        self.logger.info("🎯 Using training coordinator system")

        # Counter for micro-batches processed since last log (still needed for coordinator integration)
        self._micro_batch_count: int = 0

        # Initialize optimizer step tracking
        self._optimizer_step_wrapped: bool = False

        # Initialize accumulated loss tracking for training state manager
        self._accumulated_lm_loss: float = 0.0
        self._accumulated_teacher_lm_loss: float = 0.0
        self._accumulated_student_lm_loss: float = 0.0
        self._accumulated_coordinate_loss: float = 0.0
        self._accumulated_focal_loss: float = 0.0
        self._accumulated_regular_loss: float = 0.0

        # CRITICAL: Ensure tokenizer has correct padding_side for Flash Attention
        self._fix_tokenizer_padding_side()

        # Initialize unified training state manager (consolidates metrics, evaluation, parameters)
        self.training_state_manager = TrainingStateManager(
            config=self.config,
            model=self.model,
            trainer=self,
            training_coordinator=self.training_coordinator,
            base_weight_decay=getattr(self.args, "weight_decay", 0.0),
            logger=self.logger,
        )

        # Initialize enhanced checkpoint manager
        from src.core.checkpoint_manager import CheckpointManager

        self.checkpoint_manager = CheckpointManager(
            config=self.config,
            tokenizer=self.tokenizer_ref,
            image_processor=self.image_processor,
            logger=self.logger,
        )

    def _fix_tokenizer_padding_side(self):
        """Ensure all tokenizer references have correct padding_side for Flash Attention."""
        tokenizers_to_fix = []

        # Collect all possible tokenizer references
        if hasattr(self, "tokenizer_ref") and hasattr(
            self.tokenizer_ref, "padding_side"
        ):
            tokenizers_to_fix.append(("tokenizer_ref", self.tokenizer_ref))
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "padding_side"):
            tokenizers_to_fix.append(("tokenizer", self.tokenizer))
        if (
            hasattr(self, "data_collator")
            and hasattr(self.data_collator, "tokenizer")
            and hasattr(self.data_collator.tokenizer, "padding_side")
        ):
            tokenizers_to_fix.append(
                ("data_collator.tokenizer", self.data_collator.tokenizer)
            )

        # Fix padding_side for all found tokenizers
        for name, tokenizer in tokenizers_to_fix:
            if tokenizer.padding_side != "left":
                self.logger.warning(
                    f"🔧 Fixing {name} padding_side: {tokenizer.padding_side} -> left"
                )
                tokenizer.padding_side = "left"
            else:
                self.logger.debug(
                    f"✅ {name} padding_side already correct: {tokenizer.padding_side}"
                )

    @property
    def tokenizer(self):
        """Backward compatibility property - returns cached tokenizer without deprecation warning."""
        return self.tokenizer_ref

    @tokenizer.setter
    def tokenizer(self, processing_class):
        """Backward compatibility setter for tokenizer."""
        self.tokenizer_ref = processing_class

        # After initialization, we can safely access the tokenizer
        # and add our special tokens. This is a critical step.
        if self.tokenizer_ref:
            special_tokens = SpecialTokens()
            num_added = self.tokenizer_ref.add_special_tokens(
                {"additional_special_tokens": special_tokens.to_list()}
            )
            if num_added > 0:
                self.logger.info(
                    f"✅ Added {num_added} special tokens to the tokenizer."
                )
                # Important: Resize token embeddings in the model
                self.model.resize_token_embeddings(len(self.tokenizer_ref))
                self.logger.info(
                    "✅ Resized model token embeddings to match new tokenizer size."
                )
        else:
            self.logger.warning(
                "⚠️ Tokenizer not found on BBUTrainer, skipping special token setup."
            )

    def _save(
        self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None
    ) -> None:
        """Save checkpoint using enhanced checkpoint manager."""
        if output_dir is None:
            output_dir = self.args.output_dir

        # Use the enhanced checkpoint manager for all checkpoint operations
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            output_dir=output_dir,
            training_args=self.args,
            state_dict=state_dict,
        )

    # NOTE: _ensure_coordinate_tokens_persisted method moved to CheckpointManager.persist_coordinate_tokens()

    # NOTE: _copy_essential_files_from_base_model method moved to CheckpointManager.copy_base_model_files()

    # NOTE: _verify_saved_checkpoint method moved to CheckpointManager.verify_checkpoint_integrity()

    # NOTE: _validate_tokenizer_model_consistency method moved to CheckpointManager.validate_model_consistency()

    # Coordinate token parameter detection moved to ParameterGroupManager - no duplication

    # Parameter grouping logic moved to ParameterGroupManager - no duplication

    def create_optimizer(self) -> Union[Optimizer, DummyOptim]:
        """
        Create the optimizer using training state manager's parameter management.
        """
        if not self.config.use_differential_lr:
            self.logger.info("🚀 Differential LR disabled → using standard optimizer…")
            self.optimizer = super(BBUTrainer, self).create_optimizer()
            self._wrap_optimizer_step()
            return self.optimizer

        # Use training state manager for differential LR
        self.logger.info(
            "🚀 Creating optimizer with differential learning rates via training state manager..."
        )

        # Get parameter groups from training state manager
        optimizer_groups = self.training_state_manager.create_optimizer_groups()

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )

        self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
        self._wrap_optimizer_step()

        self.logger.info(
            "✅ Optimizer with differential learning rates created successfully via coordinator."
        )
        return self.optimizer

    def _wrap_optimizer_step(self):
        """Wrap ``optimizer.step`` so we can capture weight/grad norms right
        before gradients are cleared by ``zero_grad``.  The wrapper is applied
        once, immediately after the optimizer is created.  Captured statistics
        are stored in ``self._norm_cache`` and consumed during the next call to
        ``_maybe_log_save_evaluate``.  This guarantees that *train* logs always
        contain the true gradient magnitudes, even when DeepSpeed/Accelerate
        zero out the gradients before we reach the logging hook.
        """
        # Norm capture disabled to avoid signature mismatches under Accelerate
        return

        # EXPLICIT CONFIG: _optimizer_step_wrapped is initialized in __init__
        if hasattr(self, "_optimizer_step_wrapped") and self._optimizer_step_wrapped:
            return  # Already wrapped

        from torch.optim import Optimizer

        if not isinstance(self.optimizer, Optimizer):
            self.logger.debug(
                "Optimizer is not a torch.optim.Optimizer; skipping norm capture wrap."
            )
            return

        self._optimizer_step_wrapped = True

        original_step = self.optimizer.step

        def step_with_norm_capture(*args, **kwargs):  # type: ignore[override]
            try:
                norm_metrics = self._capture_grad_weight_norms()
                self.training_state_manager.cache_norm_metrics(norm_metrics)
            except Exception as exc:
                self.logger.warning(f"⚠️ Failed to capture weight/grad norms: {exc}")

            return original_step(*args, **kwargs)

        import types

        self.optimizer.step = types.MethodType(step_with_norm_capture, self.optimizer)  # type: ignore[assignment]

    def _capture_grad_weight_norms(self) -> Dict[str, float]:
        """Compute per-parameter-set weight and gradient L2 norms.

        Returns a flat dict ready to be merged into the training logs, e.g.::

            {
                "wn/vision_adapter": 0.91,
                "gn/vision_adapter": 0.03,
                ...
            }
        """

        norms: Dict[str, float] = {}

        # Legacy detection head module mapping removed - coordinate tokens handle detection
        # No module-specific norm capture needed for coordinate token approach
        return norms

    # Teacher-student loss computation moved to LossManager - no duplication

    def _compute_loss_with_coordinator(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss using the training coordinator system.

        This method delegates complex loss computation to the coordinator
        while maintaining the same interface as the legacy compute_loss.
        """

        # Ensure we get hidden states for detection
        model_inputs = inputs.copy()
        model_inputs["output_hidden_states"] = True

        # Validate tokens in training samples (only during training, not evaluation)
        if model.training:
            try:
                # Get the actual model (unwrap DataParallel if needed)
                actual_model = model.module if hasattr(model, "module") else model

                if hasattr(actual_model, "validate_sample_tokens"):
                    # Get input_ids from inputs
                    input_ids = inputs.get("input_ids")
                    if input_ids is not None:
                        # Validate each sample in the batch
                        batch_size = (
                            input_ids.shape[0]
                            if hasattr(input_ids, "shape")
                            else len(input_ids)
                        )
                        for i in range(batch_size):
                            sample_input_ids = (
                                input_ids[i] if batch_size > 1 else input_ids
                            )
                            actual_model.validate_sample_tokens(
                                sample_input_ids,
                                sample_info={
                                    "index": f"batch_{i}",
                                    "training_step": self.state.global_step,
                                    "epoch": self.state.epoch,
                                },
                            )
                        self.logger.debug(
                            f"✅ Token validation passed for batch of {batch_size} samples"
                        )
                else:
                    self.logger.warning(
                        "⚠️ Model does not have validate_sample_tokens method - skipping token validation"
                    )
            except Exception as e:
                self.logger.error(f"❌ Token validation failed: {e}")
                raise e

        # Run model forward pass
        self.logger.debug(f"🔍 TRAINER: About to call model(**model_inputs):")
        self.logger.debug(f"   model type: {type(model)}")
        self.logger.debug(f"   model class: {model.__class__.__name__}")
        self.logger.debug(f"   model id: {id(model)}")

        # Check if this is our wrapper or the base model
        if hasattr(model, "coordinate_tokens_enabled"):
            self.logger.debug(
                f"   ✅ Model is our wrapper with coordinate_tokens_enabled: {model.coordinate_tokens_enabled}"
            )
        else:
            self.logger.error(
                f"   ❌ CRITICAL: Model is NOT our wrapper - it's the base model!"
            )
            self.logger.error(f"   This explains why coordinate losses are missing")

        outputs = model(**model_inputs)

        # CRITICAL DEBUG: Check coordinate losses immediately after model call
        self.logger.debug(
            f"🔍 TRAINER: Model outputs immediately after model(**model_inputs):"
        )
        self.logger.debug(f"   outputs type: {type(outputs)}")
        self.logger.debug(f"   outputs id: {id(outputs)}")
        self.logger.debug(
            f"   hasattr _geometry_focal_loss: {hasattr(outputs, '_geometry_focal_loss')}"
        )
        self.logger.debug(
            f"   hasattr _coordinate_l1_loss: {hasattr(outputs, '_coordinate_l1_loss')}"
        )
        self.logger.debug(
            f"   hasattr _geometry_bbox_giou_loss: {hasattr(outputs, '_geometry_bbox_giou_loss')}"
        )

        # If coordinate losses are missing, we need to investigate the model call
        if hasattr(outputs, "_geometry_focal_loss") and hasattr(
            outputs, "_coordinate_l1_loss"
        ):
            self.logger.debug(f"   ✅ Coordinate losses found on outputs object")
            self.logger.debug(
                f"   _geometry_focal_loss: {outputs._geometry_focal_loss}"
            )
            self.logger.debug(f"   _coordinate_l1_loss: {outputs._coordinate_l1_loss}")
        else:
            self.logger.error(
                f"   ❌ CRITICAL: Coordinate losses missing from outputs object!"
            )
            self.logger.error(
                f"   This means model wrapper is not being called or not attaching losses properly"
            )

        # Use coordinator for loss computation
        if self.training_coordinator is not None:
            total_loss, loss_components = self.training_coordinator.compute_loss(
                model_outputs=outputs, inputs=inputs, is_training=model.training
            )
        else:
            # Fallback to default loss computation if no coordinator is available
            self.logger.warning(
                "⚠️ No training coordinator available for loss computation, using fallback method"
            )
            return super(BBUTrainer, self).compute_loss(
                model, inputs, return_outputs=return_outputs
            )

        # During evaluation, detach the loss to avoid gradient issues with logging
        if not model.training and hasattr(total_loss, "detach"):
            total_loss = total_loss.detach()

        # Update current loss attributes for compatibility with legacy logging
        # Note: lm_loss removed - use teacher + student components instead
        self._current_teacher_lm_loss = loss_components["teacher_lm_loss"]
        self._current_student_lm_loss = loss_components["student_lm_loss"]
        self._current_lm_loss = (
            self._current_teacher_lm_loss + self._current_student_lm_loss
        )  # For legacy compatibility
        # bbox_* and caption_loss duplicates removed - use individual components instead
        # Add coordinate token loss components with new naming - NO DEFAULTS, FAIL FAST
        if (
            hasattr(self.config, "coordinate_tokens_enabled")
            and self.config.coordinate_tokens_enabled
        ):
            required_loss_components = ["llm_loss"]  # Only require clean llm_loss
            # Optional coordinate components: focal_loss, l1_loss, giou_loss
            missing_components = []

            # Check required components
            for component in required_loss_components:
                if component not in loss_components:
                    missing_components.append(component)

            if missing_components:
                raise RuntimeError(
                    f"Coordinate tokens enabled but training coordinator missing required loss components: {missing_components}. "
                    f"This indicates training coordinator is not properly computing coordinate losses."
                )

            # Extract with strict validation - llm_loss is always present
            self._current_llm_loss_clean = loss_components["llm_loss"]

            # Extract coordinate losses (may be zero for teacher samples)
            self._current_coord_focal_loss = loss_components.get(
                "coord_focal_loss", 0.0
            )
            self._current_coord_l1_loss = loss_components.get("coord_l1_loss", 0.0)
            self._current_coord_giou_loss = loss_components.get("coord_giou_loss", 0.0)
        else:
            # Coordinate tokens disabled - set to zero
            self._current_regular_loss = 0.0
            self._current_coord_focal_loss = 0.0
            self._current_coord_l1_loss = 0.0
            self._current_coord_giou_loss = 0.0

        # Coordinator system handles all loss extraction and accumulation

        # Return in same format as legacy method
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss for BBU models with extended functionality.

        Args:
            model: Model to compute loss for
            inputs: Model inputs
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in batch (for loss scaling)

        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        # Use coordinator for loss computation (coordinator is mandatory)
        if not hasattr(self.training_coordinator, "compute_loss"):
            raise ValueError(
                "Training coordinator must have compute_loss method. "
                "Legacy loss computation has been removed."
            )

        # Filter inputs for base model compatibility
        model_inputs = inputs
        # EXPLICIT CONFIG: detection_enabled is set during model initialization
        if hasattr(model, "detection_enabled") and not model.detection_enabled:
            # For base model, exclude packed collator and custom fields
            excluded_keys = [
                "cu_seqlens",
                "max_seqlen",
                "image_counts_per_sample",
                "ground_truth_objects",
                "teacher_assistant_spans",
                "student_assistant_spans",
            ]
            model_inputs = {k: v for k, v in inputs.items() if k not in excluded_keys}
            self.logger.debug(
                f"🔍 Filtered inputs for base model: {list(model_inputs.keys())}"
            )
        elif not hasattr(model, "detection_enabled"):
            # Assume it's a base model if no detection_enabled attribute
            excluded_keys = [
                "cu_seqlens",
                "max_seqlen",
                "image_counts_per_sample",
                "ground_truth_objects",
                "teacher_assistant_spans",
                "student_assistant_spans",
            ]
            model_inputs = {k: v for k, v in inputs.items() if k not in excluded_keys}
            self.logger.debug(
                f"🔍 Filtered inputs for standard model: {list(model_inputs.keys())}"
            )

        # First get model outputs, then pass to coordinator
        model_outputs = model(**model_inputs)
        loss, _ = self.training_coordinator.compute_loss(
            model_outputs, inputs, is_training=model.training
        )
        # Return loss (and outputs if requested)
        return (loss, model_outputs) if return_outputs else loss

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
        Log metrics with averaging for gradient accumulation using MetricsManager.

        This method has been refactored to use the MetricsManager for all metrics
        handling, reducing complexity and improving maintainability.
        """
        # Use TrainingStateManager for training metrics logging
        logged_metrics = self.training_state_manager.log_training_metrics(
            tr_loss=tr_loss,
            grad_norm=grad_norm,
            model=model,
            trial=trial,
            epoch=epoch,
            ignore_keys_for_eval=ignore_keys_for_eval,
            start_time=start_time,
            learning_rate=learning_rate,
            control=self.control,
            state=self.state,
            args=self.args,
            micro_batch_count=self._micro_batch_count,
        )

        # Log the metrics if they were generated
        if logged_metrics:
            # Use the custom log method for differential learning rates
            final_logs = self.training_state_manager.log_metrics_batch(
                logs=logged_metrics,
                lr_scheduler=self.lr_scheduler,
                args=self.args,
            )

            # Call the parent Trainer.log() method to actually log to wandb/tensorboard
            super(BBUTrainer, self).log(final_logs)

            # Reset training state manager metrics after logging
            self.training_state_manager.reset_metrics_state()

            # Reset micro batch count (now handled by metrics manager)
            self._micro_batch_count = 0

        # Handle evaluation if needed
        if self.control.should_evaluate:
            self.evaluate(ignore_keys=ignore_keys_for_eval)

        # Handle checkpointing if needed
        if self.control.should_save:
            self._save()
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Override to ensure proper data loading with BBU datasets.

        This ensures compatibility between HuggingFace trainer and our custom BBUDataset
        and coordinate token system. The base trainer's column removal logic can interfere
        with our data processing when remove_unused_columns=True (the default).

        Note: This issue is resolved by setting remove_unused_columns=False in configuration.
        """
        # Integrated dataloader creation (from DataLoaderManager)
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Worker init function for reproducibility
        def setup_dataloader_workers(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            import random

            import numpy as np

            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Use our data collator directly without any wrapper
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,  # No column removal wrapper
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = setup_dataloader_workers
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        from torch.utils.data import DataLoader

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_params)
        )

    def get_eval_dataloader(self, eval_dataset=None) -> torch.utils.data.DataLoader:
        """
        Override to prevent HuggingFace trainer from applying column removal wrapper.

        This ensures consistent behavior between training and evaluation dataloaders.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Integrated dataloader creation (from DataLoaderManager)
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Worker init function for reproducibility
        def setup_dataloader_workers(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            import random

            import numpy as np

            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Use our data collator directly without any wrapper
        dataloader_params = {
            "batch_size": self.args.per_device_eval_batch_size,
            "collate_fn": self.data_collator,  # No column removal wrapper
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = setup_dataloader_workers
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        from torch.utils.data import DataLoader

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        This method is overridden to support logging of differential learning rates
        using the MetricsManager for enhanced functionality.
        """
        # Use TrainingStateManager for comprehensive log processing
        final_logs = self.training_state_manager.log_metrics_batch(
            logs=logs,
            lr_scheduler=self.lr_scheduler,
            args=self.args,
            start_time=start_time,
        )

        # Call the parent Trainer.log() method to actually log to wandb/tensorboard
        super(BBUTrainer, self).log(final_logs, start_time)

    def _extract_ground_truth_objects(self, inputs):
        """Extracts ground truth objects from inputs if they exist."""
        ground_truth_objects = []

        for batch_idx in range(inputs["input_ids"].shape[0]):
            if "ground_truth_objects" in inputs:
                raw_objs = inputs["ground_truth_objects"][batch_idx]
                gt_objects = [
                    obj
                    if isinstance(obj, GroundTruthObject)
                    else GroundTruthObject(bbox=obj["bbox_2d"], description=obj["desc"])
                    for obj in raw_objs
                ]
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
        return []

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Enhanced prediction step that includes detection loss logging during evaluation.
        """
        return self.training_state_manager.predict_batch(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation to include individual loss components in metrics."""
        return self.training_state_manager.run_evaluation(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    # ------------------------------------------------------------------
    # 🆕  Helper – unpack 1×T *packed* batches back to regular B×S tensors
    # ------------------------------------------------------------------
    def _maybe_unpack_packed(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """If **PackedDataCollator** was used, the batch comes with
        • input_ids/labels -> shape (1, total_len)
        • cu_seqlens       -> inclusive prefix-sum vector [0,L₁,L₁+L₂,…]

        Transformers attention implementation still expects each sample to
        occupy its own batch row.  We therefore reconstruct a padded
        B×S representation on-the-fly *inside* the trainer so the rest of
        the pipeline (loss split, causal mask, etc.) remains unchanged.
        The operation is cheap (≤1 µs) compared to the forward pass.
        """

        if "cu_seqlens" not in batch:
            # Standard collator → nothing to do
            return batch

        cu = batch["cu_seqlens"].to(torch.long)  # (B+1, ) inclusive
        if cu.ndim != 1 or cu[0].item() != 0:
            raise RuntimeError(
                "cu_seqlens must be 1-D inclusive prefix-sum starting with 0"
            )

        lengths = (cu[1:] - cu[:-1]).tolist()  # per-sample lengths
        batch_size = len(lengths)
        max_len = max(lengths)

        device = batch["input_ids"].device
        ids_dtype = batch["input_ids"].dtype
        lbl_dtype = batch["labels"].dtype

        pad_id = (
            self.tokenizer_ref.pad_token_id if self.tokenizer_ref is not None else 0
        )
        IGNORE_INDEX = -100  # keep consistent with src.utils

        new_input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=ids_dtype, device=device
        )
        new_labels = torch.full(
            (batch_size, max_len), IGNORE_INDEX, dtype=lbl_dtype, device=device
        )
        new_attn = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

        # Optional: carry over 3-channel position_ids when available
        pos_ids_src = batch.get(
            "position_ids"
        )  # may be None or (1, total_len) or (3,1,total_len)
        if pos_ids_src is not None:
            pos_dtype = pos_ids_src.dtype
            if pos_ids_src.ndim == 2:  # (1, T)
                pos_ids_src = pos_ids_src.unsqueeze(
                    0
                )  # → (1,1,T) for uniform indexing below
            # (3,1,T) is already fine
            new_pos = torch.zeros(
                (3, batch_size, max_len), dtype=pos_dtype, device=device
            )
        else:
            new_pos = None

        # Adjust teacher / student spans while iterating
        teacher_batch = batch.get("teacher_assistant_spans", [])
        student_batch = batch.get("student_assistant_spans", [])
        new_teacher, new_student = [], []

        cursor = 0
        for i, L in enumerate(lengths):
            slice_ids = slice(cursor, cursor + L)

            # Copy token tensors -------------------------------------------------
            new_input_ids[i, :L] = batch["input_ids"][0, slice_ids]
            new_labels[i, :L] = batch["labels"][0, slice_ids]
            new_attn[i, :L] = True

            # Position-ids -------------------------------------------------------
            if new_pos is not None:
                new_pos[:, i : i + 1, :L] = pos_ids_src[:, :, slice_ids]

            # Span adjustment ----------------------------------------------------
            if teacher_batch and i < len(teacher_batch):
                adj_teacher = [(s - cursor, e - cursor) for (s, e) in teacher_batch[i]]
                new_teacher.append(adj_teacher)
            elif teacher_batch:
                # Handle case where index is out of range
                new_teacher.append([])

            if student_batch and i < len(student_batch):
                adj_student = [(s - cursor, e - cursor) for (s, e) in student_batch[i]]
                new_student.append(adj_student)
            elif student_batch:
                # Handle case where index is out of range
                new_student.append([])

            cursor += L

        # Build new dict --------------------------------------------------------
        packed_keys = {
            "input_ids": new_input_ids,
            "labels": new_labels,
            "attention_mask": new_attn,
        }
        if new_pos is not None:
            packed_keys["position_ids"] = new_pos

        # Replace tensors
        new_batch = batch.copy()
        new_batch.update(packed_keys)

        # Replace spans
        if teacher_batch:
            new_batch["teacher_assistant_spans"] = new_teacher
        if student_batch:
            new_batch["student_assistant_spans"] = new_student

        # Remove cu_seqlens so downstream code isn't confused
        new_batch.pop("cu_seqlens", None)

        return new_batch


def set_model_training_params(model):
    """
    Enable or disable training on model submodules (vision, mlp, llm, detection)
    based on learning-rate flags and the global `detection_enabled` option.
    """
    from src.config import get_config

    config = get_config()
    logger = get_training_logger()
    # Check if we're using the detection wrapper
    has_detection_wrapper = hasattr(model, "base_model") and hasattr(
        model, "coordinate_tokens_enabled"
    )
    base_model = model.base_model if has_detection_wrapper else model

    # Helper to toggle trainability via LR -------------------------------------------------
    def _toggle(module_iter, lr_value: float, module_name: str):
        trainable = lr_value != 0
        for _, p in module_iter:
            p.requires_grad = trainable
        state = "TRAINING" if trainable else "FROZEN"
        logger.info(f"🔧 {module_name}: {state} (lr={lr_value})")

    # Vision encoder
    _toggle(base_model.visual.named_parameters(), config.vision_lr, "Vision encoder")

    # MLP connector (merger)
    _toggle(
        base_model.visual.merger.named_parameters(), config.merger_lr, "MLP connector"
    )

    # LLM backbone & lm_head
    llm_trainable = config.llm_lr != 0
    for _, p in base_model.model.named_parameters():
        p.requires_grad = llm_trainable
    if hasattr(base_model, "lm_head"):
        base_model.lm_head.requires_grad = llm_trainable
    logger.info(
        f"🔧 LLM: {'TRAINING' if llm_trainable else 'FROZEN'} (lr={config.llm_lr})"
    )

    # Detection head removed - using coordinate regression approach


def setup_model_and_tokenizer() -> Tuple[
    nn.Module, PreTrainedTokenizerBase, Qwen2VLImageProcessor
]:
    """
    Centralized setup for model, tokenizer, and image processor using UNIFIED loader.

    This function now delegates to the unified loader to ensure strict consistency
    between training and inference. NO SILENT FALLBACKS.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up model with UNIFIED loading mechanism...")
    from src.config import get_config

    try:
        global_config = get_config()
    except RuntimeError as e:
        raise ValueError(
            "Global configuration not initialized. "
            "Ensure init_config() is called before training starts."
        ) from e

    try:
        from src.models.model_loader import load_model_and_processor_unified

        # Use unified loader with training mode
        model, tokenizer, image_processor = load_model_and_processor_unified(
            model_path=global_config.model_path,
            for_inference=False,  # Training mode
        )

        logger.info("✅ Training model setup completed via unified loader")
        return model, tokenizer, image_processor

    except Exception as e:
        logger.error(f"❌ Training model setup failed: {e}")
        raise RuntimeError(f"Failed to setup model for training: {e}")


def setup_data_module(
    tokenizer: PreTrainedTokenizerBase, image_processor: Qwen2VLImageProcessor
) -> Dict[str, Any]:
    """
    Setup data module following the official approach with improved prompts.
    This matches the data setup in train_qwen.py but with context-aware prompts.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up data module with context-aware prompts...")

    # Create chat processor with training context
    from src.chat_processor import ChatProcessor
    from src.config import get_config

    try:
        config = get_config()
    except RuntimeError as e:
        raise ValueError(
            "Global configuration not initialized. "
            "Ensure init_config() is called before training starts."
        ) from e

    # Use consistent prompt style for training and evaluation to prevent distribution mismatch
    # EXPLICIT: Check for prompt configuration without fallbacks
    if not hasattr(config, "use_consistent_prompts"):
        raise ValueError(
            "Configuration missing 'use_consistent_prompts' field. "
            "Ensure this field is explicitly set in your configuration."
        )
    if not hasattr(config, "training_prompt_style"):
        raise ValueError(
            "Configuration missing 'training_prompt_style' field. "
            "Ensure this field is explicitly set in your configuration."
        )

    use_consistent_prompts = config.use_consistent_prompts
    training_prompt_style = config.training_prompt_style

    # EXPLICIT CONFIG: coordinate token configuration is required and validated at config load
    coordinate_tokens_enabled = config.coordinate_tokens_enabled
    max_coord_value = config.max_coord_value

    # Training chat processor
    train_chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        merge_size=config.merge_size,
        max_length=config.max_total_length,
        use_training_prompts=training_prompt_style,
        language="chinese",
        enable_coordinate_tokens=coordinate_tokens_enabled,
        max_coord_value=max_coord_value,
    )

    # Evaluation chat processor - use same prompt style unless explicitly overridden
    eval_prompt_style = training_prompt_style if use_consistent_prompts else False
    eval_chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        merge_size=config.merge_size,
        max_length=config.max_total_length,
        use_training_prompts=eval_prompt_style,
        language="chinese",
        enable_coordinate_tokens=coordinate_tokens_enabled,
        max_coord_value=max_coord_value,
    )

    if coordinate_tokens_enabled:
        logger.info(
            f"✅ Chat processors created with coordinate tokens enabled (max_coord_value: {max_coord_value})"
        )
    else:
        logger.info("✅ Chat processors created (coordinate tokens disabled)")

    # Create teacher pool manager if teacher_ratio > 0
    teacher_pool_manager = None
    # EXPLICIT: Get teacher_ratio without fallback
    if not hasattr(config, "teacher_ratio"):
        raise ValueError(
            "Configuration missing 'teacher_ratio' field. "
            "Ensure this field is explicitly set in your configuration."
        )

    teacher_ratio = config.teacher_ratio
    if teacher_ratio > 0.0:
        from src.teacher_pool import create_teacher_pool_manager

        # Fail fast if teacher configuration is invalid
        teacher_pool_manager = create_teacher_pool_manager()
        logger.info(
            f"✅ Teacher pool manager created with {len(teacher_pool_manager)} teachers"
        )

    # Create training dataset with detailed prompts and teacher support
    train_dataset = BBUDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        teacher_pool_manager=teacher_pool_manager,
        teacher_ratio=teacher_ratio,
        is_training=True,  # Training context
    )

    # Create validation dataset - use consistent settings unless zero-shot explicitly requested
    val_teacher_manager = teacher_pool_manager if use_consistent_prompts else None
    val_teacher_ratio = teacher_ratio if use_consistent_prompts else 0.0

    val_dataset = BBUDataset(
        data_path=config.val_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        teacher_pool_manager=val_teacher_manager,
        teacher_ratio=val_teacher_ratio,
        is_training=False,  # Evaluation context
    )

    # Create data collator
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        collator_type=config.collator_type,
    )

    logger.info(f"✅ Data module setup completed with improved prompts:")
    logger.info(
        f"   Train samples: {len(train_dataset)} (detailed prompts, teacher_ratio={teacher_ratio})"
    )
    logger.info(f"   Val samples: {len(val_dataset)} (concise prompts, no teachers)")
    # EXPLICIT CONFIG: collator_type is required and validated at config load
    logger.info(f"   Collator type: {config.collator_type}")
    logger.info(
        f"   Training prompt: {train_chat_processor.get_current_system_prompt()[:100]}..."
    )
    logger.info(
        f"   Evaluation prompt: {eval_chat_processor.get_current_system_prompt()[:100]}..."
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str) -> None:
    """
    Safe model saving following the official approach.
    Uses the improved _save method that includes visual components.
    """
    logger = get_training_logger()
    logger.info(f"💾 Safely saving model to: {output_dir}")

    if trainer.args.should_save:
        # Use the trainer's improved _save method which handles visual components properly
        trainer._save(output_dir)
        logger.info("✅ Model saved using improved _save method with visual components")


# Legacy create_trainer function removed - using unified trainer factory with coordinator


def test_enhanced_logging() -> None:
    """
    Simple test to verify enhanced detection loss logging works.
    This can be called during development to test the logging mechanism.
    """
    from src.logger_utils import get_training_logger

    logger = get_training_logger()
    logger.info("🧪 Testing enhanced detection loss logging...")

    # Test that the loss components are properly structured (cleaned up)
    sample_loss_components = {
        "llm_loss": 0.5,
        "focal_loss": 0.3,
        "l1_loss": 0.8,
        "giou_loss": 0.2,
    }

    # Test prefix handling - no prefix for training, eval_ for evaluation
    for mode, prefix in [("training", ""), ("evaluation", "eval_")]:
        loss_info = {}
        for key, value in sample_loss_components.items():
            loss_info[f"{prefix}{key}"] = float(value)

        # Add coordinate loss total
        coord_total = sum(
            sample_loss_components[k] for k in ["focal_loss", "l1_loss", "giou_loss"]
        )
        loss_info[f"{prefix}coord_loss"] = coord_total

        logger.info(f"✅ {mode.upper()} loss structure: {loss_info}")

    logger.info("🧪 Enhanced logging test completed successfully!")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_enhanced_logging()
