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

from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import config, get_config_manager
from src.logger_utils import get_training_logger
from .loss_manager import LossManager
from .parameter_manager import ParameterGroupManager


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
        use_domain_config: bool = False
    ):
        """
        Initialize training coordinator.
        
        Args:
            model: The complete model (base + detection head)
            tokenizer: Tokenizer for processing
            use_domain_config: Whether to use new domain-specific config system
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_domain_config = use_domain_config
        self.logger = get_training_logger()
        
        # Get configuration (either legacy or new system)
        self.config = self._get_configuration()
        
        # Initialize managers
        self.loss_manager = self._create_loss_manager()
        self.parameter_manager = self._create_parameter_manager()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.detection_training_enabled = True
        self._training_metrics = {}
        
        self.logger.info("✅ Training coordinator initialized")
    
    def _get_configuration(self) -> Any:
        """Get configuration using appropriate system."""
        if self.use_domain_config:
            try:
                manager = get_config_manager()
                return manager
            except RuntimeError:
                self.logger.warning(
                    "⚠️  Domain config manager not initialized, falling back to legacy config"
                )
                return config
        else:
            return config
    
    def _create_loss_manager(self) -> LossManager:
        """Create and configure loss manager."""
        if self.use_domain_config and hasattr(self.config, 'detection'):
            # Use domain-specific config
            detection_config = {
                "bbox_weight": self.config.detection.detection_bbox_weight,
                "giou_weight": self.config.detection.detection_giou_weight,
                "objectness_weight": self.config.detection.detection_objectness_weight,
                "caption_weight": self.config.detection.detection_caption_weight,
                "focal_loss_gamma": self.config.detection.detection_focal_loss_gamma,
                "focal_loss_alpha": self.config.detection.detection_focal_loss_alpha,
            }
            detection_enabled = self.config.detection.detection_enabled
        else:
            # Use legacy config
            detection_config = {
                "bbox_weight": self.config.detection_bbox_weight,
                "giou_weight": self.config.detection_giou_weight,
                "objectness_weight": self.config.detection_objectness_weight,
                "caption_weight": self.config.detection_caption_weight,
                "focal_loss_gamma": self.config.detection_focal_loss_gamma,
                "focal_loss_alpha": self.config.detection_focal_loss_alpha,
            }
            detection_enabled = self.config.detection_enabled
        
        return LossManager(
            tokenizer=self.tokenizer,
            detection_enabled=detection_enabled,
            **detection_config
        )
    
    def _create_parameter_manager(self) -> ParameterGroupManager:
        """Create and configure parameter manager."""
        if self.use_domain_config and hasattr(self.config, 'training'):
            weight_decay = self.config.training.weight_decay
        else:
            weight_decay = self.config.weight_decay
        
        return ParameterGroupManager(
            model=self.model,
            base_weight_decay=weight_decay
        )
    
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
        self.logger.info(f"   Trainable parameters: {param_stats['trainable_parameters']:,}")
        self.logger.info(f"   Detection training: {'enabled' if self.detection_training_enabled else 'disabled'}")
        
        return {
            "optimizer_groups": optimizer_groups,
            "parameter_statistics": param_stats,
            "detection_enabled": self.detection_training_enabled
        }
    
    def _setup_detection_schedule(self):
        """Setup detection training schedule based on configuration."""
        if self.use_domain_config and hasattr(self.config, 'detection'):
            freeze_epochs = self.config.detection.detection_freeze_epochs
        else:
            freeze_epochs = getattr(self.config, 'detection_freeze_epochs', 0)
        
        if freeze_epochs > 0:
            self.logger.info(
                f"🔒 Detection head will be frozen for first {freeze_epochs} epochs"
            )
            # Initially disable detection training if freeze epochs specified
            self.detection_training_enabled = False
        else:
            self.detection_training_enabled = True
    
    def compute_loss(
        self,
        model_outputs: Any,
        inputs: Dict[str, Any],
        is_training: bool = True
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
        return self.loss_manager.compute_total_loss(
            model_outputs=model_outputs,
            inputs=inputs,
            is_training=is_training,
            detection_training_enabled=self.detection_training_enabled
        )
    
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
        if self.use_domain_config and hasattr(self.config, 'detection'):
            freeze_epochs = self.config.detection.detection_freeze_epochs
        else:
            freeze_epochs = getattr(self.config, 'detection_freeze_epochs', 0)
        
        # Enable detection training after freeze period
        if freeze_epochs > 0 and epoch >= freeze_epochs and not self.detection_training_enabled:
            self.detection_training_enabled = True
            self.logger.info(f"🔓 Detection training enabled at epoch {epoch}")
            
            # Unfreeze detection parameters
            self.parameter_manager.unfreeze_components(["detection"])
    
    def _update_training_metrics(self):
        """Update training metrics for monitoring."""
        current_losses = self.loss_manager.get_current_losses()
        
        self._training_metrics.update({
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "detection_training_enabled": self.detection_training_enabled,
            **current_losses
        })
    
    def get_averaged_losses_and_reset(self) -> Dict[str, float]:
        """Get averaged losses and reset accumulators."""
        return self.loss_manager.get_averaged_losses()
    
    def save_evaluation_state(self) -> Dict[str, Any]:
        """Save training state before evaluation."""
        return {
            "loss_manager_state": self.loss_manager.save_training_state(),
            "coordinator_state": {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "detection_training_enabled": self.detection_training_enabled,
            }
        }
    
    def restore_evaluation_state(self, state: Dict[str, Any]):
        """Restore training state after evaluation."""
        self.loss_manager.restore_training_state(state["loss_manager_state"])
        
        coordinator_state = state["coordinator_state"]
        self.global_step = coordinator_state["global_step"]
        self.current_epoch = coordinator_state["current_epoch"]
        self.detection_training_enabled = coordinator_state["detection_training_enabled"]
    
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
    
    def update_learning_rates(self, optimizer: torch.optim.Optimizer, scale_factor: float):
        """Update learning rates in optimizer."""
        self.parameter_manager.update_learning_rates(optimizer, scale_factor)
    
    def validate_configuration(self) -> List[str]:
        """Validate training configuration and return warnings."""
        warnings = []
        
        # Validate parameter configuration
        param_warnings = self.parameter_manager.validate_configuration()
        warnings.extend(param_warnings)
        
        # Validate loss configuration
        if self.use_domain_config and hasattr(self.config, 'detection'):
            detection_enabled = self.config.detection.detection_enabled
            detection_lr = self.config.training.detection_lr
        else:
            detection_enabled = self.config.detection_enabled
            detection_lr = self.config.detection_lr
        
        if detection_enabled and detection_lr <= 0:
            warnings.append(
                "Detection enabled but detection_lr is 0 - detection head will not be trained"
            )
        
        # Validate teacher-student configuration
        if self.use_domain_config and hasattr(self.config, 'data'):
            teacher_ratio = self.config.data.teacher_ratio
            num_teachers = self.config.data.num_teacher_samples
        else:
            teacher_ratio = getattr(self.config, 'teacher_ratio', 0.0)
            num_teachers = getattr(self.config, 'num_teacher_samples', 0)
        
        if teacher_ratio > 0 and num_teachers == 0:
            warnings.append(
                "Teacher ratio > 0 but num_teacher_samples is 0 - no teachers will be used"
            )
        
        return warnings
    
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
            "configuration_warnings": self.validate_configuration()
        }