"""
Models Module for New Qwen2.5-VL Architecture

This module provides the core model components for the new simplified architecture:
- DetectionModel: Composition-based model wrapper
- LossManager: Multi-component loss computation
- CoordinateProcessor: Coordinate token processing utilities
- Data structures: LossComponents, ModelOutput
- Patches: Qwen2.5-VL compatibility fixes
"""

from .loss_manager import LossComponents, LossManager, ModelOutput
from .patches import (
    apply_comprehensive_qwen25_fixes,
    fixed_apply_multimodal_rotary_pos_emb,
    patch_qwen25_attention_implementation,
    patch_qwen25_forward_method,
    patch_qwen25_multimodal_rotary_pos_emb,
    patch_qwen25_prepare_inputs_for_generation,
    register_custom_attention_implementation,
)
from .wrapper import CoordinateProcessor, DetectionModel


__all__ = [
    # Core model classes
    "DetectionModel",
    "LossManager",
    "CoordinateProcessor",
    # Data structures
    "LossComponents",
    "ModelOutput",
    # Patch functions
    "apply_comprehensive_qwen25_fixes",
    "fixed_apply_multimodal_rotary_pos_emb",
    "patch_qwen25_attention_implementation",
    "patch_qwen25_forward_method",
    "patch_qwen25_multimodal_rotary_pos_emb",
    "patch_qwen25_prepare_inputs_for_generation",
    "register_custom_attention_implementation",
]
