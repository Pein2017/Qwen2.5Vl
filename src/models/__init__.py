"""
Model components for Qwen2.5-VL training.

Includes model wrapper, patches, and attention optimizations.

CRITICAL: Comprehensive Qwen2.5-VL Fixes
=========================================
This module includes critical fixes for Qwen2.5-VL models that resolve:
1. mRoPE dimension mismatch errors during generation
2. Visual processing edge cases that cause reshape failures
3. Proper handling of both 3B and 7B model configurations

The fixes are automatically applied when importing ModelWrapper, but you can
also test and verify them manually:

    from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches

    # Apply all fixes
    apply_comprehensive_qwen25_fixes()

    # Verify the patches are working
    verify_qwen25_patches()

For full verification, run: python src/models/test_fixes.py
"""

from .attention import enable_flash_attention, replace_qwen2_vl_attention_class
from .patches import (
    apply_comprehensive_qwen25_fixes,
    official_apply_multimodal_rotary_pos_emb,
    verify_qwen25_patches,
)
from .wrapper import ModelWrapper

__all__ = [
    "ModelWrapper",
    "apply_comprehensive_qwen25_fixes",
    "verify_qwen25_patches",
    "official_apply_multimodal_rotary_pos_emb",
    "enable_flash_attention",
    "replace_qwen2_vl_attention_class",
]
