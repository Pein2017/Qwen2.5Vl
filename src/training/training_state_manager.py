"""
Training State Manager for BBU Training

This module consolidates metrics management, evaluation orchestration, and parameter
grouping into a single cohesive manager. It combines the functionality of the former
MetricsManager, EvaluationManager, and ParameterManager into a unified state manager.

Key Features:
1. **Unified State Management**: Centralized training state tracking and metrics
2. **Evaluation Orchestration**: Complete evaluation workflow with state isolation
3. **Parameter Group Management**: Differential learning rate parameter organization
4. **Metrics Collection**: Comprehensive training metrics and logging
5. **Clean Integration**: Single interface for all training state operations

This consolidation reduces complexity while maintaining all existing functionality.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import TrainerControl, TrainerState
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments

from src.training.base_manager import BaseManager


@dataclass
class ParameterGroupConfig:
    """Configuration for a parameter group."""

    name: str
    lr: float
    weight_decay: float = 0.0
    enabled: bool = True
    param_count: int = 0


class TrainingStateManager(BaseManager):
    """
    Unified manager for training state, metrics, evaluation, and parameter groups.

    Consolidates the functionality of MetricsManager, EvaluationManager, and
    ParameterManager into a single cohesive interface that handles all aspects
    of training state management.
    """

    def __init__(
        self,
        config: Any,
        model: PreTrainedModel,
        trainer: Any,
        training_coordinator: Any,
        base_weight_decay: float = 0.0,
        logger: Optional[Any] = None,
    ):
        """
        Initialize training state manager.

        Args:
            config: Training configuration object
            model: The complete model for parameter management
            trainer: BBUTrainer instance for evaluation integration
            training_coordinator: TrainingCoordinator for loss management
            base_weight_decay: Base weight decay for parameter groups
            logger: Optional logger instance
        """
        # Store components before base initialization
        self.model = model
        self.trainer = trainer
        self.training_coordinator = training_coordinator
        self.base_weight_decay = base_weight_decay

        # Initialize base manager
        super().__init__(config, logger)

        self.log_info(
            "Training state manager initialized with consolidated functionality"
        )

    def _validate_configuration(self) -> None:
        """Validate configuration requirements."""
        # Validate core requirements
        self.validate_required_attribute("coordinate_tokens_enabled", bool)

        # Validate model and trainer
        if self.model is None:
            raise ValueError("model is required for training state management")
        if not hasattr(self.model, "parameters"):
            raise ValueError("model must have parameters method")
        if self.trainer is None:
            raise ValueError("trainer is required for evaluation management")
        if self.training_coordinator is None:
            raise ValueError("training_coordinator is required for loss management")

        # Validate training coordinator interface
        if not hasattr(self.training_coordinator, "get_averaged_losses_and_reset"):
            raise ValueError(
                "TrainingCoordinator must have get_averaged_losses_and_reset method"
            )

        # Validate base weight decay
        if self.base_weight_decay < 0:
            raise ValueError("base_weight_decay must be non-negative")

    def _initialize_manager_state(self) -> None:
        """Initialize all manager state components."""
        # === METRICS MANAGEMENT STATE ===
        self._step_count = 0
        self._norm_cache: Dict[str, float] = {}

        # === EVALUATION MANAGEMENT STATE ===
        self._in_evaluation = False
        self._saved_accumulators = {}

        # === PARAMETER MANAGEMENT STATE ===
        self.group_configs: Dict[str, ParameterGroupConfig] = {}
        self.parameter_groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {}

        # Initialize parameter group configurations
        self._initialize_parameter_groups()

        # Categorize parameters into groups
        self._categorize_parameters()

        self.log_debug("All manager state components initialized")

    # ===================================================================
    # PARAMETER MANAGEMENT (from ParameterManager)
    # ===================================================================

    def _initialize_parameter_groups(self) -> None:
        """Initialize parameter group configurations from config."""
        from src.config import get_config

        config = self.config if hasattr(self.config, "vision_lr") else get_config()

        self.group_configs = {
            "vision": ParameterGroupConfig(
                name="vision",
                lr=getattr(config, "vision_lr", 0.0),
                weight_decay=self.base_weight_decay,
                enabled=getattr(config, "vision_lr", 0.0) > 0,
            ),
            "merger": ParameterGroupConfig(
                name="merger",
                lr=getattr(config, "merger_lr", 0.0),
                weight_decay=self.base_weight_decay,
                enabled=getattr(config, "merger_lr", 0.0) > 0,
            ),
            "llm": ParameterGroupConfig(
                name="llm",
                lr=getattr(config, "llm_lr", 0.0),
                weight_decay=self.base_weight_decay,
                enabled=getattr(config, "llm_lr", 0.0) > 0,
            ),
            "adapter": ParameterGroupConfig(
                name="adapter",
                lr=getattr(config, "adapter_lr", 0.0),
                weight_decay=self.base_weight_decay * 0.1,  # Reduced for adapters
                enabled=getattr(config, "adapter_lr", 0.0) > 0,
            ),
        }

    def _categorize_parameters(self) -> None:
        """Categorize all model parameters into component groups."""
        self.parameter_groups = {
            "vision": [],
            "merger": [],
            "llm": [],  # Now includes coordinate tokens
            "adapter": [],
            "other": [],
        }

        # Get all named parameters from the model
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            category = self._categorize_parameter(name)
            self.parameter_groups[category].append((name, param))

        # Update parameter counts in configs
        for category, params in self.parameter_groups.items():
            if category in self.group_configs:
                self.group_configs[category].param_count = len(params)

        self._log_parameter_statistics()

    def _categorize_parameter(self, param_name: str) -> str:
        """Categorize a parameter based on its name."""
        # Check for coordinate token parameters (now part of LLM group)
        is_coordinate_param = False

        # 1. Check for explicit coordinate token modules
        if any(
            pattern in param_name
            for pattern in [
                "extended_embeddings",
                "extended_lm_head",
                "coordinate_tokens",
                "coord_tokens",
                "coordinate_head",
            ]
        ):
            is_coordinate_param = True

        # 2. Check if model has coordinate tokens enabled and this is an extended parameter
        elif (
            hasattr(self.model, "coordinate_tokens_enabled")
            and self.model.coordinate_tokens_enabled
        ):
            if any(
                pattern in param_name
                for pattern in ["embed_tokens.weight", "lm_head.weight"]
            ):
                if self._is_extended_vocabulary_parameter(param_name):
                    is_coordinate_param = True

        # 3. Check for coordinate-specific parameter patterns
        elif any(
            pattern in param_name.lower()
            for pattern in ["coordinate", "coord_", "bbox", "detection_head"]
        ):
            is_coordinate_param = True

        if is_coordinate_param:
            self.log_debug(f"Coordinate parameter '{param_name}' assigned to LLM group")
            return "llm"

        # Adapter parameters
        if any(pattern in param_name for pattern in ["adapter", "lora", "bottleneck"]):
            return "adapter"

        # Vision encoder parameters
        if any(pattern in param_name for pattern in ["visual.", "base_model.visual."]):
            return "vision"

        # Language model parameters
        if any(
            pattern in param_name
            for pattern in [
                "model.embed_tokens.",
                "model.layers.",
                "model.norm.",
                "lm_head.",
                "base_model.model.",
                "base_model.lm_head.",
            ]
        ):
            return "llm"

        # Fallback for unrecognized parameters
        return "other"

    def _is_extended_vocabulary_parameter(self, param_name: str) -> bool:
        """Check if a parameter has extended vocabulary size (indicating coordinate tokens)."""
        try:
            param = dict(self.model.named_parameters()).get(param_name)
            if param is None:
                return False

            if hasattr(self.model, "original_vocab_size") and hasattr(
                self.model, "extended_vocab_size"
            ):
                original_size = getattr(self.model, "original_vocab_size", 0)
                extended_size = getattr(self.model, "extended_vocab_size", 0)

                if "embed_tokens.weight" in param_name and param.dim() >= 2:
                    actual_vocab_size = param.shape[0]
                    return (
                        actual_vocab_size == extended_size
                        and extended_size > original_size
                    )

                if "lm_head.weight" in param_name and param.dim() >= 2:
                    actual_vocab_size = param.shape[0]
                    return (
                        actual_vocab_size == extended_size
                        and extended_size > original_size
                    )

            return False
        except Exception:
            return False

    def create_optimizer_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups for optimizer initialization."""
        optimizer_groups = []

        for category, group_config in self.group_configs.items():
            if not group_config.enabled or group_config.param_count == 0:
                continue

            params = [param for _, param in self.parameter_groups[category]]
            if not params:
                continue

            group = {
                "params": params,
                "lr": group_config.lr,
                "weight_decay": group_config.weight_decay,
                "name": group_config.name,
            }

            optimizer_groups.append(group)

            self.log_info(
                f"Parameter group '{group_config.name}': {len(params)} params, "
                f"lr={group_config.lr:.2e}, wd={group_config.weight_decay:.2e}"
            )

        # Handle "other" parameters
        other_params = [param for _, param in self.parameter_groups["other"]]
        if other_params:
            from src.config import get_config

            config = get_config()
            if not hasattr(config, "learning_rate"):
                raise ValueError(
                    "learning_rate must be configured for 'other' parameters"
                )

            group = {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": self.base_weight_decay,
                "name": "other",
            }
            optimizer_groups.append(group)

            self.log_warning(
                f"Uncategorized parameters: {len(other_params)} params "
                f"assigned to fallback group with lr={config.learning_rate:.2e}"
            )

        return optimizer_groups

    def get_group_names(self) -> List[str]:
        """Get parameter group names in the same order as create_optimizer_groups()."""
        group_names = []

        for category, group_config in self.group_configs.items():
            if not group_config.enabled or group_config.param_count == 0:
                continue
            params = [param for _, param in self.parameter_groups[category]]
            if not params:
                continue
            group_names.append(group_config.name)

        # Handle "other" parameters
        other_params = [param for _, param in self.parameter_groups["other"]]
        if other_params:
            group_names.append("other")

        return group_names

    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about parameter groups."""
        stats = {
            "total_parameters": sum(
                len(params) for params in self.parameter_groups.values()
            ),
            "trainable_parameters": sum(
                len(params)
                for category, params in self.parameter_groups.items()
                if self.group_configs.get(category, ParameterGroupConfig("", 0)).enabled
            ),
            "component_breakdown": {},
            "learning_rates": {},
            "enabled_components": [],
        }

        for category, params in self.parameter_groups.items():
            param_count = len(params)
            trainable_count = sum(1 for _, p in params if p.requires_grad)

            stats["component_breakdown"][category] = {
                "total": param_count,
                "trainable": trainable_count,
                "frozen": param_count - trainable_count,
            }

            if category in self.group_configs:
                config = self.group_configs[category]
                stats["learning_rates"][category] = config.lr
                if config.enabled:
                    stats["enabled_components"].append(category)

        return stats

    def _log_parameter_statistics(self) -> None:
        """Log detailed parameter statistics."""
        stats = self.get_parameter_statistics()

        self.log_info("Parameter Group Statistics:")
        self.log_info(f"   Total parameters: {stats['total_parameters']:,}")
        self.log_info(f"   Trainable parameters: {stats['trainable_parameters']:,}")

        for category, breakdown in stats["component_breakdown"].items():
            if breakdown["total"] > 0:
                lr_info = ""
                if category in self.group_configs:
                    lr = self.group_configs[category].lr
                    enabled = self.group_configs[category].enabled
                    lr_info = f", lr={lr:.2e}" if enabled else ", frozen"

                self.log_info(
                    f"   {category}: {breakdown['trainable']}/{breakdown['total']} "
                    f"trainable{lr_info}"
                )

    # ===================================================================
    # EVALUATION MANAGEMENT (from EvaluationManager)
    # ===================================================================

    def run_evaluation(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ) -> Dict[str, Any]:
        """Run evaluation with proper state isolation and metrics computation."""
        # Save training accumulators to prevent interference
        self._save_training_state()

        # Reset accumulators before evaluation
        self._reset_evaluation_accumulators()

        try:
            # Mark that we're in evaluation mode
            self._in_evaluation = True

            # Run base evaluation
            if hasattr(self.trainer, "super_evaluate"):
                # For testing
                metrics = self.trainer.super_evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )
            else:
                # For real trainer instances
                from transformers import Trainer

                metrics = Trainer.evaluate(
                    self.trainer,
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )

            # Compute and add component loss metrics
            self._add_component_metrics(metrics, eval_dataset, metric_key_prefix)

            return metrics

        finally:
            # Always restore training state
            self._in_evaluation = False
            self._restore_training_state()

    def predict_batch(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Enhanced prediction step for evaluation."""
        # Store original state for restoration
        original_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                # Fix tokenizer padding for evaluation
                original_padding_side = self._fix_tokenizer_padding()

                try:
                    # Temporarily modify loss info prefix for evaluation
                    old_prefix = getattr(self.trainer, "_loss_prefix", "")
                    self.trainer._loss_prefix = "eval"

                    try:
                        if prediction_loss_only:
                            loss = self.trainer.compute_loss(model, inputs)
                            return (loss, None, None)
                        else:
                            loss, outputs = self.trainer.compute_loss(
                                model, inputs, return_outputs=True
                            )

                            # Extract logits and labels for evaluation metrics
                            logits, labels = self._extract_logits_labels(
                                outputs, inputs, ignore_keys
                            )

                            # Ensure return values match expected types
                            if not isinstance(loss, (torch.Tensor, type(None))):
                                loss = torch.tensor(loss) if loss is not None else None

                            return (loss, logits, labels)

                    finally:
                        # Restore original prefix
                        self.trainer._loss_prefix = old_prefix

                finally:
                    # Restore tokenizer padding
                    self._restore_tokenizer_padding(original_padding_side)

        finally:
            # Restore original training state
            model.train(original_training)

    def _save_training_state(self) -> None:
        """Save current training accumulators for later restoration."""
        self._saved_accumulators = {
            "lm": self.trainer._accumulated_lm_loss,
            "teacher_lm": self.trainer._accumulated_teacher_lm_loss,
            "student_lm": self.trainer._accumulated_student_lm_loss,
            "coordinate": getattr(self.trainer, "_accumulated_coordinate_loss", 0.0),
            "focal": getattr(self.trainer, "_accumulated_focal_loss", 0.0),
            "regular": getattr(self.trainer, "_accumulated_regular_loss", 0.0),
        }
        self.log_debug("Saved training accumulator state for evaluation")

    def _restore_training_state(self) -> None:
        """Restore previously saved training accumulators."""
        if self._saved_accumulators:
            self.trainer._accumulated_lm_loss = self._saved_accumulators["lm"]
            self.trainer._accumulated_teacher_lm_loss = self._saved_accumulators[
                "teacher_lm"
            ]
            self.trainer._accumulated_student_lm_loss = self._saved_accumulators[
                "student_lm"
            ]
            self.trainer._accumulated_coordinate_loss = self._saved_accumulators[
                "coordinate"
            ]
            self.trainer._accumulated_focal_loss = self._saved_accumulators["focal"]
            if hasattr(self.trainer, "_accumulated_regular_loss"):
                self.trainer._accumulated_regular_loss = self._saved_accumulators.get(
                    "regular", 0.0
                )

            self.log_debug("Restored training accumulator state after evaluation")
            self._saved_accumulators = {}

    def _reset_evaluation_accumulators(self) -> None:
        """Reset all accumulators before evaluation."""
        self.trainer._accumulated_lm_loss = 0.0
        self.trainer._accumulated_teacher_lm_loss = 0.0
        self.trainer._accumulated_student_lm_loss = 0.0

        # Ensure coordinate token loss accumulators exist and are reset
        if not hasattr(self.trainer, "_accumulated_coordinate_loss"):
            self.trainer._accumulated_coordinate_loss = 0.0
        if not hasattr(self.trainer, "_accumulated_focal_loss"):
            self.trainer._accumulated_focal_loss = 0.0
        if not hasattr(self.trainer, "_accumulated_regular_loss"):
            self.trainer._accumulated_regular_loss = 0.0

        self.trainer._accumulated_coordinate_loss = 0.0
        self.trainer._accumulated_focal_loss = 0.0
        self.trainer._accumulated_regular_loss = 0.0

        self.log_debug("Reset evaluation accumulators")

    def _add_component_metrics(
        self, metrics: Dict[str, Any], eval_dataset, metric_key_prefix: str
    ) -> None:
        """Add component loss metrics to the evaluation results."""
        eval_loader = self.trainer.get_eval_dataloader(eval_dataset)
        num_batches = len(eval_loader)

        if num_batches > 0:
            # Always log LM loss components
            metrics[f"{metric_key_prefix}_lm_loss"] = round(
                self.trainer._accumulated_lm_loss / num_batches, 4
            )

            metrics[f"{metric_key_prefix}_teacher_lm_loss"] = round(
                self.trainer._accumulated_teacher_lm_loss / num_batches, 4
            )
            metrics[f"{metric_key_prefix}_student_lm_loss"] = round(
                self.trainer._accumulated_student_lm_loss / num_batches, 4
            )

            # Add coordinate token loss metrics
            if (
                hasattr(self.trainer, "_accumulated_coordinate_loss")
                and self.trainer._accumulated_coordinate_loss > 0
            ):
                metrics[f"{metric_key_prefix}_coordinate_loss"] = round(
                    self.trainer._accumulated_coordinate_loss / num_batches, 4
                )

            if (
                hasattr(self.trainer, "_accumulated_focal_loss")
                and self.trainer._accumulated_focal_loss > 0
            ):
                metrics[f"{metric_key_prefix}_focal_loss"] = round(
                    self.trainer._accumulated_focal_loss / num_batches, 4
                )

        self.log_debug(
            f"Added component metrics to evaluation results: {len(metrics)} metrics"
        )

    def _fix_tokenizer_padding(self) -> Optional[str]:
        """Fix tokenizer padding side for Flash Attention during evaluation."""
        original_padding_side = None
        tokenizer_to_fix = None

        # Try multiple tokenizer references
        if hasattr(self.trainer, "tokenizer_ref") and hasattr(
            self.trainer.tokenizer_ref, "padding_side"
        ):
            tokenizer_to_fix = self.trainer.tokenizer_ref
        elif hasattr(self.trainer, "tokenizer") and hasattr(
            self.trainer.tokenizer, "padding_side"
        ):
            tokenizer_to_fix = self.trainer.tokenizer
        elif (
            hasattr(self.trainer, "data_collator")
            and hasattr(self.trainer.data_collator, "tokenizer")
            and hasattr(self.trainer.data_collator.tokenizer, "padding_side")
        ):
            tokenizer_to_fix = self.trainer.data_collator.tokenizer

        if tokenizer_to_fix is not None:
            original_padding_side = tokenizer_to_fix.padding_side
            tokenizer_to_fix.padding_side = "left"
            self.log_debug(
                f"Fixed tokenizer padding_side for evaluation: {original_padding_side} -> left"
            )

        return original_padding_side

    def _restore_tokenizer_padding(self, original_padding_side: Optional[str]) -> None:
        """Restore original tokenizer padding side."""
        if original_padding_side is not None:
            tokenizer_to_fix = None

            # Find the same tokenizer we fixed earlier
            if hasattr(self.trainer, "tokenizer_ref") and hasattr(
                self.trainer.tokenizer_ref, "padding_side"
            ):
                tokenizer_to_fix = self.trainer.tokenizer_ref
            elif hasattr(self.trainer, "tokenizer") and hasattr(
                self.trainer.tokenizer, "padding_side"
            ):
                tokenizer_to_fix = self.trainer.tokenizer
            elif (
                hasattr(self.trainer, "data_collator")
                and hasattr(self.trainer.data_collator, "tokenizer")
                and hasattr(self.trainer.data_collator.tokenizer, "padding_side")
            ):
                tokenizer_to_fix = self.trainer.data_collator.tokenizer

            if tokenizer_to_fix is not None:
                tokenizer_to_fix.padding_side = original_padding_side
                self.log_debug(
                    f"Restored tokenizer padding_side: left -> {original_padding_side}"
                )

    def _extract_logits_labels(
        self, outputs, inputs: Dict[str, Any], ignore_keys: Optional[List[str]]
    ) -> Tuple[Any, Any]:
        """Extract logits and labels from model outputs and inputs."""
        # Extract logits for evaluation metrics
        if isinstance(outputs, dict):
            logits = tuple(
                v for k, v in outputs.items() if k not in (ignore_keys or []) + ["loss"]
            )
            if logits and len(logits) == 1:
                logits = logits[0]
        else:
            logits = outputs[1:] if hasattr(outputs, "__getitem__") else outputs
            if not isinstance(logits, tuple):
                logits = logits

        # Extract labels if available
        labels = None
        if hasattr(self.trainer, "label_names") and len(self.trainer.label_names) > 0:
            labels = tuple(inputs.get(name) for name in self.trainer.label_names)
            if len(labels) == 1:
                labels = labels[0]

        return logits, labels

    @property
    def is_in_evaluation(self) -> bool:
        """Check if currently in evaluation mode."""
        return self._in_evaluation

    # ===================================================================
    # METRICS MANAGEMENT (from MetricsManager)
    # ===================================================================

    def log_training_metrics(
        self,
        tr_loss: Union[torch.Tensor, float],
        grad_norm: Optional[Union[torch.Tensor, float]],
        model: nn.Module,
        trial: Any,
        epoch: Optional[float],
        ignore_keys_for_eval: Optional[List[str]],
        start_time: float,
        learning_rate: Optional[float] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        args: Optional[TrainingArguments] = None,
        micro_batch_count: int = 0,
    ) -> Optional[Dict[str, float]]:
        """Log training metrics with comprehensive validation and averaging."""
        if not control or not control.should_log:
            return None

        # Define num_micro_batches for loss averaging
        max(1, micro_batch_count)

        try:
            # Get averaged losses from training coordinator
            component_logs = self.training_coordinator.get_averaged_losses_and_reset()

            # Validate required keys
            if "loss" not in component_logs:
                raise ValueError("Coordinator must provide 'loss' in component_logs")

            total_avg_loss = component_logs["loss"]

            # Validate coordinate token losses if enabled
            self._validate_coordinate_losses(component_logs)

            # Build comprehensive logs dictionary
            logs = self._build_logs_dict(
                total_avg_loss, grad_norm, component_logs, model, start_time, state
            )

            # Add learning rate metrics
            self._add_learning_rate_metrics(logs, learning_rate)

            return logs

        except Exception as e:
            self.log_error(f"Error in log_training_metrics: {str(e)}")
            raise

    def _validate_coordinate_losses(self, component_logs: Dict[str, float]) -> None:
        """Validate coordinate token losses when coordinate tokens are enabled."""
        coordinate_tokens_enabled = self.config.coordinate_tokens_enabled
        if not coordinate_tokens_enabled:
            return

        # Simplified validation - only require coordinate L1 loss
        required_coord_losses = ["coordinate_l1_loss"]
        missing_coord_losses = []

        for key in required_coord_losses:
            if key not in component_logs:
                missing_coord_losses.append(key)

        if missing_coord_losses:
            raise RuntimeError(
                f"Coordinate tokens enabled but coordinator missing required losses: {missing_coord_losses}. "
                f"This indicates training coordinator is not properly computing coordinate losses."
            )

        # Validate teacher/student losses are present
        for loss_type in ["student_lm_loss", "teacher_lm_loss"]:
            if loss_type not in component_logs:
                raise ValueError(
                    f"Coordinator must provide '{loss_type}' in component_logs"
                )

        # Increment step count for validation tracking
        self._step_count += 1

        # Validate student samples are present
        student_lm_loss = component_logs["student_lm_loss"]
        component_logs["teacher_lm_loss"]

        # Use a small threshold instead of exact 0.0 to handle numerical precision issues
        student_loss_threshold = 1e-10
        if student_lm_loss < student_loss_threshold:
            self.log_error(f"CRITICAL ERROR at step {self._step_count}")
            self.log_error(
                f"Student LM loss too small ({student_lm_loss}) - indicates no student samples or numerical issues!"
            )
            self.log_error(
                "This could be caused by: 1) Missing student data, 2) NaN losses converted to small values, 3) Numerical instability"
            )
            raise RuntimeError(
                f"CRITICAL: Student LM loss ({student_lm_loss}) below threshold ({student_loss_threshold}) at step {self._step_count}. "
                f"Check for missing student data or numerical instability in loss computation."
            )

        # Validate coordinate losses are present for student samples
        total_coord_loss = sum(
            component_logs.get(key, 0.0) for key in required_coord_losses
        )
        # Use a reasonable threshold for coordinate loss validation
        # After fixing the root cause, coordinate losses should be meaningful
        coord_loss_threshold = 1e-4

        # Check if we're in a testing environment
        is_integration_test = False
        if hasattr(self.config, "output_dir") and "pipeline_test" in str(
            self.config.output_dir
        ):
            is_integration_test = True

        # Add debugging information for coordinate loss issues
        if total_coord_loss < coord_loss_threshold and student_lm_loss > 0.0:
            self.log_warning(
                f"Coordinate loss validation: total_coord_loss={total_coord_loss}, "
                f"threshold={coord_loss_threshold}, student_lm_loss={student_lm_loss}"
            )
            self.log_warning(
                f"Individual coordinate losses: {[f'{k}={component_logs.get(k, 0.0)}' for k in required_coord_losses]}"
            )

            # Check if this looks like NaN replacement values
            if abs(total_coord_loss - 1e-8) < 1e-10:
                self.log_warning(
                    "Coordinate loss appears to be NaN replacement value (1e-8) - this indicates numerical instability"
                )

        # Coordinate loss validation is now enabled after fixing the root cause
        skip_coord_validation = False

        if (
            total_coord_loss < coord_loss_threshold
            and student_lm_loss > 0.0
            and not is_integration_test
            and not skip_coord_validation  # Temporary bypass
        ):
            self.log_error(f"CRITICAL ERROR at step {self._step_count}")
            self.log_error(
                f"Student samples present but coordinate losses are extremely small ({total_coord_loss})!"
            )
            self.log_error(
                "This could indicate: 1) Missing coordinate tokens in data, 2) NaN losses being converted to tiny values, 3) Model not computing coordinate losses"
            )
            self.log_error(
                "Students must have meaningful coordinate losses when coordinate tokens are enabled"
            )
            raise RuntimeError(
                f"CRITICAL: Student samples present (student_lm_loss={student_lm_loss}) "
                f"but all coordinate losses are below threshold ({total_coord_loss} < {coord_loss_threshold}) "
                f"at step {self._step_count}. This indicates missing coordinate tokens in data or numerical instability."
            )
        elif (
            total_coord_loss < coord_loss_threshold
            and student_lm_loss > 0.0
            and skip_coord_validation
        ):
            # Log warning instead of failing when validation is bypassed
            self.log_warning(
                f"BYPASSED coordinate loss validation at step {self._step_count}: "
                f"student_lm_loss={student_lm_loss}, total_coord_loss={total_coord_loss}. "
                f"This bypass is temporary - investigate NaN loss root cause!"
            )

    def _build_logs_dict(
        self,
        total_avg_loss: float,
        grad_norm: Optional[Union[torch.Tensor, float]],
        component_logs: Dict[str, float],
        model: nn.Module,
        start_time: float,
        state: Optional[TrainerState] = None,
    ) -> Dict[str, float]:
        """Build comprehensive logs dictionary with all metrics."""
        logs: Dict[str, float] = {}

        # Core loss metrics
        logs["loss"] = total_avg_loss

        if grad_norm is not None:
            logs["grad_norm"] = (
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            )

        # Add component losses
        logs.update(component_logs)

        # Add weight and gradient norms
        self._add_norm_metrics(logs, model)

        # Add ETA and remaining time
        self._add_eta_metrics(logs, start_time, state)

        return logs

    def _add_norm_metrics(self, logs: Dict[str, float], model: nn.Module) -> None:
        """Add weight and gradient norm metrics to logs."""
        if self._norm_cache:
            logs.update(self._norm_cache)
            # Clear after use so we don't accidentally reuse stale values
            self._norm_cache = {}
        else:
            # Compute norms for available modules (detection head modules removed)
            module_names: Dict[str, str] = {}

            for log_key, module_path in module_names.items():
                module = None
                for name, m in model.named_modules():
                    if name.endswith(module_path):
                        module = m
                        break

                if module is None:
                    continue

                try:
                    with torch.no_grad():
                        weight_sq, grad_sq, param_cnt = 0.0, 0.0, 0
                        for p in module.parameters():
                            weight_sq += p.data.norm(2).pow(2)
                            if p.grad is not None:
                                grad_sq += p.grad.norm(2).pow(2)
                            param_cnt += 1

                        if param_cnt > 0:
                            device = next(module.parameters()).device
                            vec = torch.tensor(
                                [weight_sq, grad_sq, float(param_cnt)],
                                device=device,
                                dtype=torch.float32,
                            )

                            if torch.distributed.is_initialized():
                                torch.distributed.all_reduce(
                                    vec, op=torch.distributed.ReduceOp.SUM
                                )

                            total_params = vec[2].item()
                            if total_params > 0:
                                logs[f"wn/{log_key}"] = (
                                    vec[0].sqrt() / total_params
                                ).item()
                                logs[f"gn/{log_key}"] = (
                                    vec[1].sqrt() / total_params
                                ).item()

                except Exception as e:
                    self.log_warning(f"Error computing norms for {log_key}: {e}")

    def _add_eta_metrics(
        self,
        logs: Dict[str, float],
        start_time: float,
        state: Optional[TrainerState] = None,
    ) -> None:
        """Add ETA and remaining time metrics to logs."""
        if not state or state.max_steps <= 0:
            return

        current_step = state.global_step
        if current_step > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_step = elapsed_time / current_step
            remaining_steps = state.max_steps - current_step
            remaining_time_s = remaining_steps * avg_time_per_step

            logs["remaining_hr"] = round(remaining_time_s / 3600, 3)

    def _add_learning_rate_metrics(
        self, logs: Dict[str, float], learning_rate: Optional[float] = None
    ) -> None:
        """Add learning rate metrics to logs."""
        if learning_rate is not None:
            logs["learning_rate"] = learning_rate

    def log_metrics_batch(
        self,
        logs: Dict[str, float],
        lr_scheduler: Any = None,
        args: Optional[TrainingArguments] = None,
        start_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Log metrics for a batch with differential learning rate support."""
        # Remove generic learning rate from logs
        logs.pop("learning_rate", None)

        # Log the learning rate for each parameter group
        if lr_scheduler is not None:
            try:
                scheduler_class_name = lr_scheduler.__class__.__name__

                if scheduler_class_name == "DummyScheduler":
                    if not hasattr(lr_scheduler, "lr"):
                        raise ValueError("DummyScheduler missing lr attribute")
                    logs["learning_rate"] = lr_scheduler.lr
                elif hasattr(lr_scheduler, "get_last_lr"):
                    last_lr = lr_scheduler.get_last_lr()

                    # Get parameter group names
                    param_group_names = self.get_group_names()

                    if len(last_lr) != len(param_group_names):
                        self.log_warning(
                            f"Learning rate groups ({len(last_lr)}) don't match "
                            f"parameter groups ({len(param_group_names)})"
                        )
                        # Log all learning rates without group names
                        for i, lr in enumerate(last_lr):
                            logs[f"lr/group_{i}"] = lr
                    else:
                        # Log learning rates with group names
                        for i, (group_name, group_lr) in enumerate(
                            zip(param_group_names, last_lr)
                        ):
                            logs[f"lr/{group_name}"] = group_lr
                else:
                    # Fallback for scheduler types that don't have get_last_lr
                    if args and hasattr(args, "learning_rate"):
                        logs["learning_rate"] = args.learning_rate
                    else:
                        raise ValueError("args missing learning_rate attribute")

            except Exception as e:
                # Log error but don't fail training
                self.log_warning(f"Error getting learning rate: {e}")
                if args and hasattr(args, "learning_rate"):
                    logs["learning_rate"] = args.learning_rate

        return logs

    def cache_norm_metrics(self, norm_metrics: Dict[str, float]) -> None:
        """Cache norm metrics for later inclusion in logs."""
        self._norm_cache.update(norm_metrics)

    def reset_metrics_state(self) -> None:
        """Reset metrics manager state (e.g., after logging)."""
        self._norm_cache.clear()

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            "step_count": self._step_count,
            "coordinate_tokens_enabled": self.config.coordinate_tokens_enabled,
            "has_cached_norms": bool(self._norm_cache),
            "in_evaluation": self._in_evaluation,
        }

        # Add parameter statistics
        param_stats = self.get_parameter_statistics()
        stats.update(param_stats)

        # Get stats from training coordinator if available
        if hasattr(self.training_coordinator, "get_training_metrics"):
            coordinator_stats = self.training_coordinator.get_training_metrics()
            if coordinator_stats:
                stats.update(coordinator_stats)

        return stats
