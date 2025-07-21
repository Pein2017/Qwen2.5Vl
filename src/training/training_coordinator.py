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
        config_obj=None,
    ):
        """
        Initialize training coordinator.

        Args:
            model: The complete model
            tokenizer: Tokenizer for processing
            config_obj: Configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logger = get_training_logger()

        # Get configuration
        if config_obj is None:
            from src.config import config

            self.config = config
        else:
            self.config = config_obj

        # Using unified flat configuration
        self.use_domain_config = False

        # Initialize managers
        self.loss_manager = self._create_loss_manager()
        self.parameter_manager = self._create_parameter_manager()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.detection_training_enabled = True
        self._training_metrics = {}

        self.logger.info("✅ Training coordinator initialized")
        self.logger.info("   Using unified flat configuration")

    def _create_loss_manager(self) -> LossManager:
        """Create and configure loss manager."""
        # Legacy detection parameters (ignored - using coordinate tokens instead)
        return LossManager(tokenizer=self.tokenizer)

    def _create_parameter_manager(self) -> ParameterGroupManager:
        """Create and configure parameter manager."""
        weight_decay = getattr(self.config, "weight_decay", 0.01)
        return ParameterGroupManager(model=self.model, base_weight_decay=weight_decay)

    def setup_training(self) -> Dict[str, Any]:
        """
        Setup training with parameter groups and initial state.

        Returns:
            Dictionary with optimizer parameter groups and training info
        """
        # Create optimizer parameter groups
        optimizer_groups = self.parameter_manager.create_optimizer_groups()

        # Setup detection training schedule
        self._setup_detection_schedule()

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

    def _setup_detection_schedule(self):
        """Setup detection training schedule based on configuration."""
        freeze_epochs = getattr(self.config, "detection_freeze_epochs", 0)

        if freeze_epochs > 0:
            self.logger.info(
                f"🔒 Detection head will be frozen for first {freeze_epochs} epochs"
            )
            # Initially disable detection training if freeze epochs specified
            self.detection_training_enabled = False
        else:
            self.detection_training_enabled = True

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
        # Simplified loss computation - validation moved to config level

        total_loss, loss_components = self.loss_manager.compute_total_loss(
            model_outputs=model_outputs,
            inputs=inputs,
            is_training=is_training,
            detection_training_enabled=self.detection_training_enabled,
        )

        # Simplified logging - excessive validation removed

        return total_loss, loss_components

    def step_update(self, step: int, epoch: int):
        """
        Update coordinator state after training step.

        Args:
            step: Global training step
            epoch: Current epoch
        """
        self.global_step = step
        self.current_epoch = epoch

        # Check if detection training should be enabled
        self._update_detection_training_state(epoch)

        # Update training metrics
        self._update_training_metrics()

    def _update_detection_training_state(self, epoch: int):
        """Update detection training state based on epoch and schedule."""
        freeze_epochs = getattr(self.config, "detection_freeze_epochs", 0)

        # Enable detection training after freeze period
        if (
            freeze_epochs > 0
            and epoch >= freeze_epochs
            and not self.detection_training_enabled
        ):
            self.detection_training_enabled = True
            self.logger.info(f"🔓 Detection training enabled at epoch {epoch}")

            # Unfreeze coordinate token parameters
            self.parameter_manager.unfreeze_components(["coordinate"])

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
            # Ensure all coordinate loss components are present
            required_coord_losses = ["focal_loss", "l1_loss", "giou_loss"]
            missing_losses = []

            for key in required_coord_losses:
                if key not in averaged_losses:
                    averaged_losses[key] = 0.0
                    missing_losses.append(key)

            if missing_losses:
                self.logger.debug(f"🔧 Added missing coordinate losses: {missing_losses}")

            # Log coordinate loss summary
            total_coord_loss = sum(averaged_losses.get(key, 0.0) for key in required_coord_losses)
            self.logger.debug(f"📊 Averaged coordinate losses: total={total_coord_loss:.6f}")

        # Add the main 'loss' field that the trainer expects
        llm_loss = averaged_losses.get("llm_loss", 0.0)
        coord_loss = sum(averaged_losses.get(key, 0.0) for key in ["focal_loss", "l1_loss", "giou_loss"])
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
        coordinate_enabled = getattr(
            self.config, "coordinate_tokens_enabled", False
        )
        coordinate_lr = getattr(self.config, "coordinate_lr", 0.0)

        if coordinate_enabled and coordinate_lr <= 0:
            warnings.append(
                "Coordinate tokens enabled but coordinate_lr is 0 - coordinate tokens will not be trained"
            )

        # Validate teacher-student configuration
        teacher_ratio = getattr(self.config, "teacher_ratio", 0.0)
        num_teachers = getattr(self.config, "num_teacher_samples", 0)

        if teacher_ratio > 0 and num_teachers == 0:
            warnings.append(
                "Teacher ratio > 0 but num_teacher_samples is 0 - no teachers will be used"
            )

        return warnings

    def _validate_coordinate_token_config(self) -> bool:
        """Validate coordinate token configuration and return if enabled."""
        try:
            # Check if coordinate tokens are enabled in flat config
            coordinate_enabled = getattr(self.config, "coordinate_tokens_enabled", False)

            if coordinate_enabled:
                # Additional validation
                coordinate_lr = getattr(self.config, "coordinate_lr", 0)
                if coordinate_lr <= 0:
                    self.logger.warning("⚠️ Coordinate tokens enabled but coordinate_lr is 0 or missing")
                    return False

                self.logger.debug("✅ Coordinate token configuration validated")
                return True
            else:
                self.logger.debug("🔧 Coordinate tokens disabled in configuration")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error validating coordinate token config: {e}")
            return False

    def _log_coordinate_metrics_summary(self, current_losses: Dict[str, float]):
        """Log summary of coordinate token metrics."""
        try:
            coord_loss = current_losses.get("coordinate_loss", 0.0)
            focal_loss = current_losses.get("focal_loss", 0.0)
            regular_loss = current_losses.get("regular_loss", 0.0)
            l1_loss = current_losses.get("l1_loss", 0.0)
            giou_loss = current_losses.get("giou_loss", 0.0)

            total_coord_loss = coord_loss + focal_loss + regular_loss + l1_loss + giou_loss

            if total_coord_loss > 0:
                self.logger.info(f"📊 Coordinate metrics: coord={coord_loss:.4f}, focal={focal_loss:.4f}, regular={regular_loss:.4f}, l1={l1_loss:.4f}, giou={giou_loss:.4f}, total={total_coord_loss:.4f}")
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
