"""
Training Coordinator for BBU Multi-Task Learning

This module orchestrates the complex training loop for the BBU detection system,
coordinating between multiple components:

- Loss computation and tracking
- Parameter group management
- Teacher-student learning coordination
- Detection head training scheduling
- Training state management and recovery

Key Features:
- Clean separation of training orchestration from HuggingFace integration
- Centralized training state management
- Component-wise training control and monitoring
- Robust error handling and recovery
- Integration with domain-specific configurations
"""

from typing import Any, Dict, List, Tuple

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.config import TrainingConfig, CoordinateConfig
from src.logger_utils import get_training_logger
from src.training.loss_manager import LossManager
from src.training.parameter_manager import ParameterGroupManager


class TrainingCoordinator:
    """
    Coordinates multi-task training for BBU detection system.

    Acts as the central orchestrator that manages training state,
    coordinates between different managers, and provides clean
    interfaces for the trainer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_config: TrainingConfig,
        coordinate_config: CoordinateConfig,
    ):
        """
        Initialize training coordinator with typed domain configs.

        Args:
            model: The complete model
            tokenizer: Tokenizer for processing
            training_config: Training-specific configuration
            coordinate_config: Coordinate token configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logger = get_training_logger()

        # Store typed domain configs
        self.training_config = training_config
        self.coordinate_config = coordinate_config
        
        self.logger.debug(f"🔍 COORDINATOR: Using typed domain configs")
        self.logger.debug(
            f"🔍 COORDINATOR: coordinate_tokens_enabled = {coordinate_config.coordinate_tokens_enabled}"
        )

        # Initialize managers
        self.loss_manager = self._create_loss_manager()
        self.parameter_manager = self._create_parameter_manager()

        # Training state - all from typed configs
        self.current_epoch = 0
        self.global_step = 0
        self.detection_training_enabled = self.coordinate_config.coordinate_tokens_enabled
        self._training_metrics = {}

        self.logger.info("✅ Training coordinator initialized with typed configs")
        self.logger.info("   Using domain-specific configuration classes")

    # Domain configs have built-in validation - no additional validation needed

    def _create_loss_manager(self) -> LossManager:
        """Create and configure loss manager."""
        return LossManager(
            tokenizer=self.tokenizer,
            model=self.model,
            teacher_loss_weight=self.training_config.teacher_loss_weight,
            student_loss_weight=self.training_config.student_loss_weight,
            coordinate_tokens_enabled=self.coordinate_config.coordinate_tokens_enabled,
        )

    def _create_parameter_manager(self) -> ParameterGroupManager:
        """Create and configure parameter manager."""
        # Create a basic config object for parameter manager backward compatibility
        # TODO: Refactor ParameterGroupManager to use domain configs
        from types import SimpleNamespace
        legacy_config = SimpleNamespace()
        legacy_config.weight_decay = self.training_config.weight_decay
        legacy_config.vision_lr = self.training_config.vision_lr
        legacy_config.merger_lr = self.training_config.merger_lr
        legacy_config.llm_lr = self.training_config.llm_lr
        legacy_config.coordinate_lr = self.training_config.coordinate_lr
        legacy_config.adapter_lr = self.training_config.adapter_lr
        
        return ParameterGroupManager(
            model=self.model,
            base_weight_decay=self.training_config.weight_decay,
            config=legacy_config,
        )

    def setup_training(self) -> Dict[str, Any]:
        """
        Setup training with parameter groups and initial state.

        Returns:
            Dictionary with optimizer parameter groups and training info
        """
        # Create optimizer parameter groups
        optimizer_groups = self.parameter_manager.create_optimizer_groups()

        # Get parameter statistics
        param_stats = self.parameter_manager.get_parameter_statistics()

        # Log training setup
        self.logger.info("🚀 Training setup completed:")
        self.logger.info(f"   Parameter groups: {len(optimizer_groups)}")
        self.logger.info(
            f"   Trainable parameters: {param_stats['trainable_parameters']:,}"
        )
        self.logger.info(
            f"   Detection training: {'enabled' if self.detection_training_enabled else 'disabled'}"
        )

        return {
            "optimizer_groups": optimizer_groups,
            "parameter_statistics": param_stats,
            "detection_enabled": self.detection_training_enabled,
        }

    def compute_loss(
        self, model_outputs: Any, inputs: Dict[str, Any], is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss using loss manager.

        Args:
            model_outputs: Output from model forward pass
            inputs: Batch inputs
            is_training: Whether in training mode

        Returns:
            Tuple of (total_loss, loss_components)
        """
        total_loss, loss_components = self.loss_manager.compute_total_loss(
            model_outputs=model_outputs,
            inputs=inputs,
            is_training=is_training,
            detection_training_enabled=self.detection_training_enabled,
        )

        # Log performance metrics periodically
        if hasattr(self, "global_step") and self.global_step % 50 == 0:
            self._log_performance_metrics(loss_components)

        return total_loss, loss_components

    def _log_performance_metrics(self, loss_components: Dict[str, float]):
        """Log performance metrics for monitoring."""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.logger.debug(
                f"GPU Memory: {memory_mb:.1f}MB, Step: {self.global_step}"
            )

        # Log key loss components concisely
        key_losses = [
            "llm_loss",
            "coordinate_l1_loss",
            "teacher_lm_loss",
            "student_lm_loss",
        ]
        loss_summary = {
            k: f"{loss_components.get(k, 0.0):.3f}"
            for k in key_losses
            if k in loss_components
        }
        if loss_summary:
            self.logger.debug(f"Losses: {loss_summary}")

    def step_update(self, step: int, epoch: int):
        """
        Update coordinator state after training step.

        Args:
            step: Global training step
            epoch: Current epoch
        """
        self.global_step = step
        self.current_epoch = epoch

        # Update training metrics
        self._update_training_metrics()

    def _update_training_metrics(self):
        """Update training metrics for monitoring."""
        current_losses = self.loss_manager.get_current_losses()

        self._training_metrics.update(
            {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "detection_training_enabled": self.detection_training_enabled,
                **current_losses,
            }
        )

        # Enhanced coordinate loss logging
        if self._validate_coordinate_token_config():
            self._log_coordinate_metrics_summary(current_losses)

    def get_averaged_losses_and_reset(self) -> Dict[str, float]:
        """Get averaged losses with enhanced coordinate token support."""
        averaged_losses = self.loss_manager.get_averaged_losses()

        # Validate and ensure coordinate token losses are properly included
        coordinate_tokens_enabled = self._validate_coordinate_token_config()

        if coordinate_tokens_enabled:
            # Use simplified coordinate loss component
            required_coord_losses = ["coordinate_l1_loss"]
            missing_losses = []

            for key in required_coord_losses:
                if key not in averaged_losses:
                    averaged_losses[key] = 0.0
                    missing_losses.append(key)

            if missing_losses:
                self.logger.debug(
                    f"🔧 Added missing coordinate losses: {missing_losses}"
                )

            # Log coordinate loss summary
            total_coord_loss = sum(
                averaged_losses.get(key, 0.0) for key in required_coord_losses
            )
            self.logger.debug(
                f"📊 Averaged coordinate losses: total={total_coord_loss:.6f}"
            )

        # Add the main 'loss' field that the trainer expects
        llm_loss = averaged_losses.get("llm_loss", 0.0)
        required_coord_losses = ["coordinate_l1_loss"]
        coord_loss = sum(averaged_losses.get(key, 0.0) for key in required_coord_losses)
        averaged_losses["loss"] = llm_loss + coord_loss

        return averaged_losses

    def save_evaluation_state(self) -> Dict[str, Any]:
        """Save training state before evaluation."""
        return {
            "loss_manager_state": self.loss_manager.save_training_state(),
            "coordinator_state": {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "detection_training_enabled": self.detection_training_enabled,
            },
        }

    def restore_evaluation_state(self, state: Dict[str, Any]):
        """Restore training state after evaluation."""
        self.loss_manager.restore_training_state(state["loss_manager_state"])

        coordinator_state = state["coordinator_state"]
        self.global_step = coordinator_state["global_step"]
        self.current_epoch = coordinator_state["current_epoch"]
        self.detection_training_enabled = coordinator_state[
            "detection_training_enabled"
        ]

    def get_component_gradients(self) -> Dict[str, float]:
        """Get gradient norms for each component."""
        return self.parameter_manager.get_component_gradients()

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return self._training_metrics.copy()

    def freeze_components(self, components: List[str]):
        """Freeze specified model components."""
        self.parameter_manager.freeze_components(components)

    def unfreeze_components(self, components: List[str]):
        """Unfreeze specified model components."""
        self.parameter_manager.unfreeze_components(components)

    def update_learning_rates(
        self, optimizer: torch.optim.Optimizer, scale_factor: float
    ):
        """Update learning rates in optimizer."""
        self.parameter_manager.update_learning_rates(optimizer, scale_factor)

    def validate_configuration(self) -> List[str]:
        """Validate training configuration and return warnings."""
        warnings = []

        # Validate parameter configuration
        param_warnings = self.parameter_manager.validate_configuration()
        warnings.extend(param_warnings)

        # Validate coordinate token configuration
        coordinate_enabled = self.coordinate_config.coordinate_tokens_enabled
        if coordinate_enabled and self.training_config.coordinate_lr <= 0:
            warnings.append(
                "Coordinate tokens enabled but coordinate_lr is 0 - coordinate tokens will not be trained"
            )

        # Additional validation could be added for teacher-student here if needed

        return warnings

    def _validate_coordinate_token_config(self) -> bool:
        """Validate coordinate token configuration and return if enabled."""
        try:
            # Use domain configs - already validated
            coordinate_enabled = self.coordinate_config.coordinate_tokens_enabled
            coordinate_lr = self.training_config.coordinate_lr

            self.logger.debug(
                f"🔍 CONFIG_DEBUG: coordinate_tokens_enabled = {coordinate_enabled}"
            )
            self.logger.debug(f"🔍 CONFIG_DEBUG: coordinate_lr = {coordinate_lr}")

            if coordinate_enabled:
                # Additional validation
                if coordinate_lr <= 0:
                    self.logger.warning(
                        "⚠️ Coordinate tokens enabled but coordinate_lr is 0"
                    )
                    return False

                self.logger.debug("✅ Coordinate token configuration validated")
                return True
            else:
                self.logger.debug("ℹ️ Coordinate tokens disabled in configuration")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error validating coordinate token config: {e}")
            return False

    def _log_coordinate_metrics_summary(self, current_losses: Dict[str, float]):
        """Log summary of coordinate token metrics."""
        try:
            # Use simplified coordinate loss component
            loss_components = ["coordinate_l1_loss"]

            llm_loss = current_losses.get("llm_loss", 0.0)

            # Build loss summary from configured components
            coord_losses = {}
            total_coord_loss = llm_loss

            for component in loss_components:
                loss_value = current_losses.get(component, 0.0)
                coord_losses[component] = loss_value
                total_coord_loss += loss_value

            if total_coord_loss > 0:
                loss_summary = ", ".join(
                    [f"{k}={v:.4f}" for k, v in coord_losses.items()]
                )
                self.logger.info(
                    f"📊 Coordinate metrics: llm={llm_loss:.4f}, {loss_summary}, total={total_coord_loss:.4f}"
                )
            else:
                self.logger.debug("📊 All coordinate losses are zero for this step")

        except Exception as e:
            self.logger.error(f"❌ Error logging coordinate metrics: {e}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary for logging."""
        param_stats = self.parameter_manager.get_parameter_statistics()
        current_losses = self.loss_manager.get_current_losses()

        return {
            "training_state": {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "detection_training_enabled": self.detection_training_enabled,
            },
            "parameter_statistics": param_stats,
            "current_losses": current_losses,
            "enabled_components": param_stats["enabled_components"],
            "configuration_warnings": self.validate_configuration(),
        }
