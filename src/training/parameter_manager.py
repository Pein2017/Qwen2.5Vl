"""
Parameter Group Manager for Differential Learning Rates

This module handles the complex logic of organizing model parameters into
groups for differential learning rate training. It supports fine-grained
control over which components of the Qwen2.5-VL model are trained and
at what learning rates.

Key Features:
- Component-wise parameter categorization (vision, merger, LLM, detection, adapter)
- Automatic parameter group creation for optimizers
- Learning rate validation and scaling
- Parameter freezing support
- Comprehensive parameter statistics and debugging
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from src.config import config
from src.logger_utils import get_training_logger


@dataclass
class ParameterGroupConfig:
    """Configuration for a parameter group."""
    name: str
    lr: float
    weight_decay: float = 0.0
    enabled: bool = True
    param_count: int = 0
    

class ParameterGroupManager:
    """
    Manager for organizing model parameters into groups for differential training.
    
    Handles the complex logic of identifying and categorizing parameters from
    the Qwen2.5-VL model and detection head for differential learning rates.
    """
    
    def __init__(self, model: PreTrainedModel, base_weight_decay: float = 0.0):
        """
        Initialize parameter group manager.
        
        Args:
            model: The complete model (base + detection head)
            base_weight_decay: Base weight decay for parameter groups
        """
        self.model = model
        self.base_weight_decay = base_weight_decay
        self.logger = get_training_logger()
        
        # Parameter group configurations
        self.group_configs: Dict[str, ParameterGroupConfig] = {}
        self._initialize_group_configs()
        
        # Categorized parameters
        self.parameter_groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {}
        self._categorize_parameters()
    
    def _initialize_group_configs(self):
        """Initialize parameter group configurations from global config."""
        self.group_configs = {
            "vision": ParameterGroupConfig(
                name="vision",
                lr=config.vision_lr,
                weight_decay=self.base_weight_decay,
                enabled=config.vision_lr > 0
            ),
            "merger": ParameterGroupConfig(
                name="merger", 
                lr=config.merger_lr,
                weight_decay=self.base_weight_decay,
                enabled=config.merger_lr > 0
            ),
            "llm": ParameterGroupConfig(
                name="llm",
                lr=config.llm_lr,
                weight_decay=self.base_weight_decay,
                enabled=config.llm_lr > 0
            ),
            "detection": ParameterGroupConfig(
                name="detection",
                lr=config.detection_lr,
                weight_decay=self.base_weight_decay,
                enabled=config.detection_lr > 0 and config.detection_enabled
            ),
            "adapter": ParameterGroupConfig(
                name="adapter",
                lr=config.adapter_lr,
                weight_decay=self.base_weight_decay * 0.1,  # Reduced weight decay for adapters
                enabled=config.adapter_lr > 0
            )
        }
    
    def _categorize_parameters(self):
        """Categorize all model parameters into component groups."""
        self.parameter_groups = {
            "vision": [],
            "merger": [], 
            "llm": [],
            "detection": [],
            "adapter": [],
            "other": []
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
        """
        Categorize a parameter based on its name.
        
        Args:
            param_name: Full parameter name from model.named_parameters()
            
        Returns:
            Category name ("vision", "merger", "llm", "detection", "adapter", "other")
        """
        # Detection head parameters (highest priority)
        if any(pattern in param_name for pattern in [
            "detection_head", "object_queries", "bbox_head", "objectness_head",
            "caption_head", "caption_decoder", "detection_adapter"
        ]):
            return "detection"
        
        # Adapter parameters
        if any(pattern in param_name for pattern in [
            "adapter", "lora", "bottleneck"
        ]):
            return "adapter"
        
        # Vision encoder parameters
        if any(pattern in param_name for pattern in [
            "visual", "vision", "patch_embed", "pos_embed", "vision_tower",
            "image_processor", "vision_encoder"
        ]):
            return "vision"
        
        # Vision-language merger/connector parameters
        if any(pattern in param_name for pattern in [
            "merger", "connector", "mm_projector", "multi_modal_projector",
            "vision_proj", "visual_proj"
        ]):
            return "merger"
        
        # Language model parameters (catch-all for transformer layers)
        if any(pattern in param_name for pattern in [
            "language_model", "transformer", "layers", "embed_tokens",
            "norm", "lm_head", "output_projection"
        ]):
            return "llm"
        
        # Fallback for unrecognized parameters
        return "other"
    
    def create_optimizer_groups(self) -> List[Dict[str, Any]]:
        """
        Create parameter groups for optimizer initialization.
        
        Returns:
            List of parameter group dictionaries for optimizer
        """
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
                "name": group_config.name  # For debugging/logging
            }
            
            optimizer_groups.append(group)
            
            self.logger.info(
                f"✅ Parameter group '{group_config.name}': {len(params)} params, "
                f"lr={group_config.lr:.2e}, wd={group_config.weight_decay:.2e}"
            )
        
        # Handle any "other" parameters if they exist
        other_params = [param for _, param in self.parameter_groups["other"]]
        if other_params:
            fallback_lr = config.learning_rate if hasattr(config, 'learning_rate') else 1e-5
            group = {
                "params": other_params,
                "lr": fallback_lr,
                "weight_decay": self.base_weight_decay,
                "name": "other"
            }
            optimizer_groups.append(group)
            
            self.logger.warning(
                f"⚠️  Uncategorized parameters found: {len(other_params)} params "
                f"assigned to fallback group with lr={fallback_lr:.2e}"
            )
        
        return optimizer_groups
    
    def freeze_components(self, components: List[str]):
        """
        Freeze specified model components.
        
        Args:
            components: List of component names to freeze ("vision", "merger", etc.)
        """
        for component in components:
            if component not in self.parameter_groups:
                self.logger.warning(f"⚠️  Unknown component '{component}' cannot be frozen")
                continue
            
            for name, param in self.parameter_groups[component]:
                param.requires_grad = False
            
            # Update config to reflect frozen state
            if component in self.group_configs:
                self.group_configs[component].enabled = False
            
            self.logger.info(
                f"🔒 Frozen component '{component}': "
                f"{len(self.parameter_groups[component])} parameters"
            )
    
    def unfreeze_components(self, components: List[str]):
        """
        Unfreeze specified model components.
        
        Args:
            components: List of component names to unfreeze
        """
        for component in components:
            if component not in self.parameter_groups:
                self.logger.warning(f"⚠️  Unknown component '{component}' cannot be unfrozen")
                continue
            
            for name, param in self.parameter_groups[component]:
                param.requires_grad = True
            
            # Update config to reflect unfrozen state
            if component in self.group_configs:
                self.group_configs[component].enabled = True
            
            self.logger.info(
                f"🔓 Unfrozen component '{component}': "
                f"{len(self.parameter_groups[component])} parameters"
            )
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about parameter groups.
        
        Returns:
            Dictionary containing parameter statistics
        """
        stats = {
            "total_parameters": sum(
                len(params) for params in self.parameter_groups.values()
            ),
            "trainable_parameters": sum(
                len(params) for category, params in self.parameter_groups.items()
                if self.group_configs.get(category, ParameterGroupConfig("", 0)).enabled
            ),
            "component_breakdown": {},
            "learning_rates": {},
            "enabled_components": []
        }
        
        for category, params in self.parameter_groups.items():
            param_count = len(params)
            trainable_count = sum(1 for _, p in params if p.requires_grad)
            
            stats["component_breakdown"][category] = {
                "total": param_count,
                "trainable": trainable_count,
                "frozen": param_count - trainable_count
            }
            
            if category in self.group_configs:
                config = self.group_configs[category]
                stats["learning_rates"][category] = config.lr
                if config.enabled:
                    stats["enabled_components"].append(category)
        
        return stats
    
    def validate_configuration(self) -> List[str]:
        """
        Validate parameter group configuration and return any warnings.
        
        Returns:
            List of validation warning messages
        """
        warnings = []
        
        # Check for components with learning rate but no parameters
        for category, group_config in self.group_configs.items():
            if group_config.enabled and group_config.param_count == 0:
                warnings.append(
                    f"Component '{category}' has lr > 0 but no parameters found"
                )
        
        # Check for very high learning rate ratios
        lrs = [grp_cfg.lr for grp_cfg in self.group_configs.values() if grp_cfg.enabled]
        if lrs:
            max_lr = max(lrs)
            min_lr = min(lrs)
            if max_lr / min_lr > 1000:
                warnings.append(
                    f"Large learning rate ratio detected: {max_lr:.2e} / {min_lr:.2e} = "
                    f"{max_lr/min_lr:.1f}x"
                )
        
        # Check for detection components
        if (
            config.detection_enabled 
            and self.group_configs["detection"].enabled
            and self.group_configs["detection"].param_count == 0
        ):
            warnings.append(
                "Detection enabled in config but no detection parameters found in model"
            )
        
        return warnings
    
    def _log_parameter_statistics(self):
        """Log detailed parameter statistics."""
        stats = self.get_parameter_statistics()
        
        self.logger.info("📊 Parameter Group Statistics:")
        self.logger.info(f"   Total parameters: {stats['total_parameters']:,}")
        self.logger.info(f"   Trainable parameters: {stats['trainable_parameters']:,}")
        
        for category, breakdown in stats["component_breakdown"].items():
            if breakdown["total"] > 0:
                lr_info = ""
                if category in self.group_configs:
                    lr = self.group_configs[category].lr
                    enabled = self.group_configs[category].enabled
                    lr_info = f", lr={lr:.2e}" if enabled else ", frozen"
                
                self.logger.info(
                    f"   {category}: {breakdown['trainable']}/{breakdown['total']} "
                    f"trainable{lr_info}"
                )
        
        # Log any validation warnings
        warnings = self.validate_configuration()
        for warning in warnings:
            self.logger.warning(f"⚠️  {warning}")
    
    def update_learning_rates(self, optimizer: torch.optim.Optimizer, scale_factor: float):
        """
        Update learning rates in optimizer parameter groups.
        
        Args:
            optimizer: The optimizer to update
            scale_factor: Factor to scale learning rates by
        """
        for group in optimizer.param_groups:
            group["lr"] *= scale_factor
            
        self.logger.info(f"🔄 Scaled all learning rates by factor {scale_factor:.3f}")
    
    def get_component_gradients(self) -> Dict[str, float]:
        """
        Get gradient norms for each component.
        
        Returns:
            Dictionary mapping component names to gradient norms
        """
        grad_norms = {}
        
        for category, params in self.parameter_groups.items():
            if not params:
                continue
            
            total_norm = 0.0
            param_count = 0
            
            for _, param in params:
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2) ** 2
                    param_count += 1
            
            if param_count > 0:
                grad_norms[category] = (total_norm ** 0.5).item()
            else:
                grad_norms[category] = 0.0
        
        return grad_norms