"""
Smart mRoPE fix for Qwen2.5-VL models.

ISSUE DESCRIPTION:
The transformers library's apply_multimodal_rotary_pos_emb function always doubles
mrope_section values, but different model sizes have different configurations:

- 3B models: Config already has doubled values [16,24,24,16,24,24] (sum=128) ✅
- 7B models: Config has original values [16,24,24] (sum=64) ❌ needs doubling

This causes dimension mismatches and training failures when the doubled values
don't match the expected head dimension (128).

OUR SOLUTION:
Smart patch that only doubles mrope_section when sum(mrope_section) != head_dim.
This automatically handles both model sizes correctly without manual configuration.

VERIFICATION:
Run `python scripts/verify_mrope_fix.py` to verify the fix works correctly.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def smart_apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section, unsqueeze_dim=1
):
    """
    Smart version that only doubles mrope_section when needed.

    Logic: If sum(mrope_section) != head_dim, then double it.
    This handles both 3B (already doubled) and 7B (needs doubling) models.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine values for RoPE
        sin: Sine values for RoPE
        mrope_section: Multi-dimensional RoPE section configuration
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        Tuple of (rotated_q, rotated_k)
    """

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Smart fix: only double if sum doesn't match expected dimension
    expected_dim = q.size(-1)
    if isinstance(mrope_section, (list, tuple)):
        mrope_sum = sum(mrope_section)
    else:
        mrope_sum = mrope_section.sum().item()

    if mrope_sum != expected_dim:
        # Need to double (7B case: [16,24,24] -> [32,48,48])
        if isinstance(mrope_section, (list, tuple)):
            mrope_section = [x * 2 for x in mrope_section]
        else:
            mrope_section = mrope_section * 2
        logger.debug(
            f"🔧 Doubled mrope_section: {mrope_section} (sum={sum(mrope_section) if isinstance(mrope_section, (list, tuple)) else mrope_section.sum().item()})"
        )
    else:
        pass

    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_smart_mrope_fix():
    """Apply the smart mRoPE fix by replacing the problematic function."""
    try:
        from transformers.models.qwen2_5_vl import (
            modeling_qwen2_5_vl as qwen25_modeling,
        )

        # Replace the buggy function
        qwen25_modeling.apply_multimodal_rotary_pos_emb = (
            smart_apply_multimodal_rotary_pos_emb
        )

        logger.info("✅ Smart mRoPE fix applied successfully")
        logger.info("   - Automatically handles both 3B and 7B models")
        logger.info("   - Only doubles mrope_section when needed")
        logger.info("   - Fixes dimension mismatch errors")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to apply smart mRoPE fix: {e}")
        return False


def verify_smart_mrope_patch():
    """Verify that the smart mRoPE patch is working correctly."""
    try:
        from transformers.models.qwen2_5_vl import (
            modeling_qwen2_5_vl as qwen25_modeling,
        )

        func = qwen25_modeling.apply_multimodal_rotary_pos_emb

        if func == smart_apply_multimodal_rotary_pos_emb:
            logger.info("✅ Smart mRoPE patch verification successful")
            return True
        else:
            logger.warning("⚠️ Smart mRoPE patch not applied")
            logger.warning(f"   Current function: {func}")
            logger.warning(
                f"   Expected function: {smart_apply_multimodal_rotary_pos_emb}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Smart mRoPE patch verification failed: {e}")
        return False


def test_mrope_configurations():
    """
    Test the smart mRoPE fix with different model configurations.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    logger.info("🧪 Testing mRoPE configurations...")

    # Test configurations
    test_cases = [
        {
            "name": "3B Model (already doubled)",
            "mrope_section": [16, 24, 24, 16, 24, 24],
            "expected_sum": 128,
            "should_double": False,
        },
        {
            "name": "7B Model (needs doubling)",
            "mrope_section": [16, 24, 24],
            "expected_sum": 128,
            "should_double": True,
        },
    ]

    all_passed = True

    for test_case in test_cases:
        logger.info(f"   Testing: {test_case['name']}")

        # Create dummy tensors
        head_dim = 128
        q = torch.randn(1, 1, 10, head_dim)
        k = torch.randn(1, 1, 10, head_dim)
        cos = torch.randn(10, head_dim)
        sin = torch.randn(10, head_dim)

        try:
            # Test our smart function
            q_out, k_out = smart_apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, test_case["mrope_section"]
            )

            # Verify output shapes
            if q_out.shape == q.shape and k_out.shape == k.shape:
                logger.info(f"   ✅ {test_case['name']}: Shapes correct")
            else:
                logger.error(f"   ❌ {test_case['name']}: Shape mismatch")
                all_passed = False

        except Exception as e:
            logger.error(f"   ❌ {test_case['name']}: Exception - {e}")
            all_passed = False

    if all_passed:
        logger.info("✅ All mRoPE configuration tests passed")
    else:
        logger.error("❌ Some mRoPE configuration tests failed")

    return all_passed


# Auto-apply the fix when this module is imported
if __name__ != "__main__":
    # Only apply automatically when imported, not when run directly
    apply_smart_mrope_fix()
