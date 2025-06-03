"""
Model components for Qwen2.5-VL training.

Includes model wrapper, patches, and attention optimizations.

CRITICAL: Smart mRoPE Fix
========================
This module includes a critical fix for Qwen2.5-VL models that resolves
dimension mismatch errors between 3B and 7B model sizes.

The fix is automatically applied when importing ModelWrapper, but you can
also test and verify it manually:

    from src.models.patches import test_mrope_configurations, verify_smart_mrope_patch

    # Test the fix works for both model sizes
    test_mrope_configurations()

    # Verify the patch is applied
    verify_smart_mrope_patch()

For full verification, run: python scripts/verify_mrope_fix.py
"""

from .attention import enable_flash_attention, replace_qwen2_vl_attention_class
from .patches import (
    apply_smart_mrope_fix,
    test_mrope_configurations,
    verify_smart_mrope_patch,
)
from .wrapper import ModelWrapper

__all__ = [
    "ModelWrapper",
    "apply_smart_mrope_fix",
    "verify_smart_mrope_patch",
    "test_mrope_configurations",
    "enable_flash_attention",
    "replace_qwen2_vl_attention_class",
]
