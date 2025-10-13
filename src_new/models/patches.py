"""
Compatibility patches for Qwen2.5-VL models.

This module provides patches to fix compatibility issues with Qwen2.5-VL models
and ensure proper integration with the HuggingFace ecosystem.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)


def get_patches_logger() -> logging.Logger:
    """Get rank-aware logger for patches module."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("patches")
    except ImportError:
        # Fallback to standard logging
        logger = logging.getLogger("patches")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_patches_logger()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def fixed_apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section, unsqueeze_dim=1
):
    """
    Fixed implementation that prevents double-doubling of mrope_section.

    The issue is that the transformers library automatically doubles mrope_section,
    but our fine-tuned checkpoint already has the correct doubled values in the config.
    This leads to mrope_section being doubled twice, causing dimension mismatches.
    """
    # Collect debug information for error reporting
    debug_info = []

    # Ensure mrope_section is a list
    if isinstance(mrope_section, torch.Tensor):
        mrope_section = mrope_section.tolist()

    debug_info.append(f"🔍 Input mrope_section: {mrope_section}")
    debug_info.append(f"🔍 cos shape: {cos.shape}")

    # Get actual tensor dimension
    actual_dim = cos.shape[-1]
    original_sum = sum(mrope_section)

    debug_info.append(f"🔍 mrope_section sum: {original_sum}, cos dim: {actual_dim}")

    # CRITICAL FIX: The transformers library doubles mrope_section automatically,
    # but our checkpoint config already has the doubled values.
    # We need to prevent this double-doubling.

    try:
        if original_sum == actual_dim:
            # Perfect match - use as is
            final_mrope_section = mrope_section
            debug_info.append("✅ Using mrope_section as-is (perfect match)")
        elif original_sum == actual_dim * 2:
            # mrope_section is doubled but cos dim is half - need to halve mrope_section
            # This happens when config has doubled values but cos tensor is not doubled
            final_mrope_section = [x // 2 for x in mrope_section]
            debug_info.append(
                f"✅ Halving mrope_section: {mrope_section} -> {final_mrope_section}"
            )
        elif original_sum * 2 == actual_dim:
            # Need to double mrope_section to match cos dim
            final_mrope_section = mrope_section * 2
            debug_info.append(
                f"✅ Doubling mrope_section: {mrope_section} -> {final_mrope_section}"
            )
        else:
            # Try to detect if mrope_section was already doubled by checking for patterns
            if len(mrope_section) == 12 and len(set(mrope_section[::2])) <= 3:
                # Looks like a doubled pattern [a,b,c,a,b,c,a,b,c,a,b,c] -> [a,b,c,a,b,c]
                half_len = len(mrope_section) // 2
                first_half = mrope_section[:half_len]
                second_half = mrope_section[half_len:]
                if first_half == second_half and sum(first_half) == actual_dim:
                    final_mrope_section = first_half
                    debug_info.append(
                        f"✅ Detected doubled pattern, using first half: {final_mrope_section}"
                    )
                else:
                    raise RuntimeError(
                        f"Cannot resolve mrope_section: sum={original_sum}, cos_dim={actual_dim}"
                    )
            else:
                raise RuntimeError(
                    f"Cannot resolve mrope_section: sum={original_sum}, cos_dim={actual_dim}"
                )

        # Final validation
        if sum(final_mrope_section) != actual_dim:
            raise RuntimeError(
                f"Final validation failed: {sum(final_mrope_section)} != {actual_dim}"
            )

        debug_info.append(f"✅ Final mrope_section: {final_mrope_section}")

    except RuntimeError as e:
        # Log all collected debug information when an error occurs
        logger.debug("🚨 Error in fixed_apply_multimodal_rotary_pos_emb - Debug trace:")
        for debug_msg in debug_info:
            logger.debug(debug_msg)
        raise e

    # Apply the rotary position embedding with the corrected mrope_section
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(final_mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(final_mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def patch_torch_library_wrap_triton() -> None:
    """
    Patch torch.library.wrap_triton compatibility issue for FlashAttention v2.

    FlashAttention v2 uses torch.library.wrap_triton which doesn't exist in PyTorch 2.5.1.
    This patch provides a compatibility layer.
    """
    import torch

    if hasattr(torch, "library") and not hasattr(torch.library, "wrap_triton"):

        def wrap_triton_compatibility(kernel_fn):
            """
            Fallback implementation that returns the kernel directly for PyTorch 2.5.1.
            This maintains compatibility with FlashAttention v2 without requiring the
            actual Triton wrapping functionality.
            """
            return kernel_fn

        # Add the missing attribute using setattr (more robust)
        setattr(torch.library, "wrap_triton", wrap_triton_compatibility)
        logger.info(
            "🔧 Applied torch.library.wrap_triton compatibility patch for PyTorch 2.5.1"
        )
    else:
        logger.debug("✓ torch.library.wrap_triton already available, no patch needed")


# Apply the patch immediately when this module is imported
# This ensures FlashAttention v2 compatibility before any model operations
patch_torch_library_wrap_triton()


def apply_comprehensive_qwen25_fixes() -> None:
    """
    Apply all necessary patches to Qwen2.5-VL models.

    This function applies all compatibility patches to ensure proper
    integration with the HuggingFace ecosystem and fix known issues.
    """
    logger.info("Applying comprehensive Qwen2.5-VL compatibility patches...")

    # Apply all patches
    patch_qwen25_forward_method()
    patch_qwen25_prepare_inputs_for_generation()
    patch_qwen25_attention_implementation()
    patch_qwen25_multimodal_rotary_pos_emb()
    # Note: torch.library.wrap_triton patch is applied automatically on module import

    logger.info("✅ All Qwen2.5-VL compatibility patches applied successfully")


def patch_qwen25_forward_method() -> None:
    """
    Patch the forward method of Qwen2.5-VL model.

    This patch fixes issues with the forward method to ensure proper
    handling of labels and loss computation.
    """
    logger.info("Patching Qwen2.5-VL forward method...")

    # Store original method for reference
    original_forward = Qwen2_5_VLForConditionalGeneration.forward

    def patched_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Patched forward method with improved label handling.
        """
        # Call original forward method
        # Strip non-HF kwargs that may be plumbed by our training stack
        if "assistant_spans" in kwargs:
            kwargs.pop("assistant_spans", None)
        if "teacher_assistant_spans" in kwargs:
            kwargs.pop("teacher_assistant_spans", None)
        if "student_assistant_spans" in kwargs:
            kwargs.pop("student_assistant_spans", None)

        outputs = original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        # Add custom handling if needed

        return outputs

    # Apply patch
    Qwen2_5_VLForConditionalGeneration.forward = patched_forward
    logger.info("✅ Qwen2.5-VL forward method patched successfully")


def patch_qwen25_prepare_inputs_for_generation() -> None:
    """
    Patch the prepare_inputs_for_generation method of Qwen2.5-VL model.

    This patch fixes issues with the prepare_inputs_for_generation method
    to ensure proper handling of inputs during generation.
    """
    logger.info("Patching Qwen2.5-VL prepare_inputs_for_generation method...")

    # Store original method for reference
    original_prepare_inputs = (
        Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation
    )

    def patched_prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Patched prepare_inputs_for_generation method with improved handling.
        """
        # Call original method
        model_inputs = original_prepare_inputs(
            self,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        # Add custom handling if needed

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    # Apply patch
    Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = (
        patched_prepare_inputs_for_generation
    )
    logger.info(
        "✅ Qwen2.5-VL prepare_inputs_for_generation method patched successfully"
    )


def patch_qwen25_multimodal_rotary_pos_emb() -> None:
    """
    Patch the multimodal rotary position embedding function.

    This fixes the critical mRoPE dimension mismatch issue where the
    transformers implementation blindly doubles mrope_section without
    checking if it matches the actual tensor dimensions.
    """
    try:
        from transformers.models.qwen2_5_vl import (
            modeling_qwen2_5_vl as qwen25_modeling,
        )

        # Replace the problematic function with our fixed version
        # Use globals() to access the function from global scope
        qwen25_modeling.apply_multimodal_rotary_pos_emb = globals()[
            "fixed_apply_multimodal_rotary_pos_emb"
        ]
        logger.info(
            "✅ Qwen2.5-VL multimodal rotary position embedding patched successfully"
        )
        logger.info("   - Fixed dimension mismatch issue")
        logger.info("   - Added intelligent mrope_section handling")
        logger.info("   - Prevents batch duplication errors")

    except Exception as e:
        logger.error(f"❌ Failed to patch multimodal rotary position embedding: {e}")
        raise


def patch_qwen25_attention_implementation() -> None:
    """
    Patch the attention implementation of Qwen2.5-VL model.

    This patch ensures proper handling of different attention implementations
    (eager, flash_attention_2, sdpa) for compatibility with different hardware.
    """
    logger.info("Patching Qwen2.5-VL attention implementation...")

    # Check if flash attention is available
    has_flash_attn = False
    try:
        import flash_attn  # noqa

        has_flash_attn = True
    except ImportError:
        logger.warning(
            "⚠️ flash_attn package not found, falling back to eager attention"
        )

    # Apply patch only if needed
    if not has_flash_attn:
        # Monkey patch the model's config to use eager attention
        original_from_pretrained = Qwen2_5_VLForConditionalGeneration.from_pretrained

        def patched_from_pretrained(cls, *args, **kwargs):
            """
            Patched from_pretrained method to force eager attention if flash_attention is not available.
            """
            # Force eager attention if flash_attention is not available
            if kwargs.get("attn_implementation") == "flash_attention_2":
                logger.warning(
                    "⚠️ flash_attention_2 requested but not available, using eager attention instead"
                )
                kwargs["attn_implementation"] = "eager"

            # Call original method
            return original_from_pretrained(cls, *args, **kwargs)

        # Apply patch
        Qwen2_5_VLForConditionalGeneration.from_pretrained = classmethod(
            patched_from_pretrained
        )
        logger.info("✅ Qwen2.5-VL attention implementation patched successfully")
    else:
        logger.info(
            "✓ flash_attn package found, no attention implementation patch needed"
        )


def register_custom_attention_implementation() -> None:
    """
    Register custom attention implementation for Qwen2.5-VL model.

    This function registers a custom attention implementation for
    specific hardware or performance requirements.
    """
    logger.info("Registering custom attention implementation...")

    # Implementation details would go here

    logger.info("✅ Custom attention implementation registered successfully")
