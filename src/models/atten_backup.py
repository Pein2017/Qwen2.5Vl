"""
Flash Attention utilities for Qwen2.5-VL training.

This module provides a single fail-fast flash attention implementation that:
- Uses HuggingFace's official flash attention implementation
- Fails fast with clear errors instead of silent fallbacks
- Validates inputs and raises descriptive errors for debugging
- Optimizes memory usage and training speed
"""

import logging
from typing import List, Optional, Tuple

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)

logger = logging.getLogger(__name__)


def _is_flash_attn_available() -> bool:
    """Check if Flash Attention is available."""
    try:
        import flash_attn  # noqa: F401
        from flash_attn.bert_padding import pad_input  # noqa: F401
        from flash_attn.flash_attn_interface import (
            flash_attn_func,
            flash_attn_varlen_func,
        )  # noqa: F401

        return True
    except ImportError:
        return False


# Single availability check at module import
FLASH_ATTENTION_AVAILABLE = _is_flash_attn_available()

if FLASH_ATTENTION_AVAILABLE:
    from transformers.modeling_flash_attention_utils import (
        _flash_attention_forward as _hf_flash_forward,
    )

    print("✅ Flash Attention 2 detected and imported successfully")
else:
    print("⚠️ Flash Attention 2 not available. Flash attention will be disabled.")


def _compute_causal(
    is_causal: bool, use_top_left_mask: bool, query_length: int
) -> bool:
    """
    Compute causal flag for flash attention.

    The query_length != 1 check is needed for RoCm Flash Attention compatibility.
    TODO: Remove query_length != 1 check once Flash Attention for RoCm is bumped to 2.1.
    """
    if not use_top_left_mask:
        return is_causal
    return is_causal and query_length != 1


def _optimized_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Fail-fast flash attention implementation.

    This is a thin wrapper around HuggingFace's official _flash_attention_forward
    that validates inputs and fails fast with clear errors.
    """
    if not FLASH_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "Flash Attention 2 is required but not available. "
            "Install with: pip install flash-attn --no-build-isolation"
        )

    batch_size, num_heads, seq_len, head_dim = query_states.shape

    # FAIL-FAST: Validate attention mask compatibility
    if attention_mask is not None and attention_mask.shape[-1] != seq_len:
        raise ValueError(
            f"Attention mask sequence length mismatch: "
            f"mask_len={attention_mask.shape[-1]}, seq_len={seq_len}. "
            f"This indicates a data preprocessing issue that must be fixed."
        )

    logger.debug(
        f"🚀 Flash attention: batch={batch_size}, seq_len={seq_len}, heads={num_heads}"
    )

    # Transform to HuggingFace flash attention format: [batch, seq, heads, head_dim]
    if query_states.shape[1] == num_heads:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    # Compute causal flag
    causal = _compute_causal(is_causal, use_top_left_mask, query_length)

    # Use HuggingFace's official flash attention implementation
    attn_output = _hf_flash_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=dropout,
        softmax_scale=softmax_scale,
        is_causal=causal,
        use_top_left_mask=use_top_left_mask,
        softcap=softcap,
        **kwargs,
    )

    # Transform back to standard format: [batch, heads, seq, head_dim]
    attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
    return attn_output.transpose(1, 2)


def _update_causal_mask(
    self,
    attention_mask: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple] = None,
    output_attentions: bool = False,
) -> Optional[torch.Tensor]:
    """
    Updated causal mask handling for flash attention.

    Only overrides behavior when using flash_attention_2, otherwise uses default.
    """
    if getattr(self.config, "_attn_implementation", None) == "flash_attention_2":
        if attention_mask is not None and past_key_values is not None:
            is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size(
                0
            )
            if is_padding_right:
                logger.warning(
                    "Batched generation with padding_side='right' may cause issues with Flash Attention. "
                    "Consider using padding_side='left' before tokenizing."
                )

        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For non-flash attention, use default behavior
    return attention_mask


def enable_flash_attention() -> bool:
    """Enable flash attention if available."""
    if not FLASH_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "Flash Attention 2 not available. Install with: pip install flash-attn>=2.1.0"
        )
    return True


# Generic helper for trainable parameters printing
def _print_trainable_indices(
    name: str,
    modules: List[torch.nn.Module],
    extra: Optional[torch.nn.Module] = None,
) -> None:
    """Print trainable parameter indices for a list of modules."""
    trainable = [
        i for i, m in enumerate(modules) if any(p.requires_grad for p in m.parameters())
    ]
    non_trainable = [
        i
        for i, m in enumerate(modules)
        if not any(p.requires_grad for p in m.parameters())
    ]

    print(f"{name} Module - Trainable Indices: {trainable or 'None'}")
    print(f"{name} Module - Non-Trainable Indices: {non_trainable or 'None'}")

    if extra is not None:
        is_extra_trainable = any(p.requires_grad for p in extra.parameters())
        extra_name = extra.__class__.__name__
        print(f"{name} Module - {extra_name} Trainable: {is_extra_trainable}")


def print_trainable_parameters_visual(self) -> None:
    """Print trainable parameters for vision components."""
    _print_trainable_indices("Vision", self.blocks, getattr(self, "merger", None))


def print_trainable_parameters(self) -> None:
    """Print trainable parameters for LLM components."""
    _print_trainable_indices("LLM", self.layers, getattr(self, "embed_tokens", None))


def replace_qwen2_vl_attention_class():
    """
    Replace Qwen2.5-VL attention with fail-fast flash attention implementation.

    This replaces the default flash attention with a version that:
    - Fails fast instead of falling back to standard attention
    - Validates inputs and raises clear errors for debugging
    - Uses the official HuggingFace flash attention implementation directly
    """
    import transformers

    # Replace flash attention implementation for both Qwen2 and Qwen2.5
    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _optimized_flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _optimized_flash_attention_forward
    )

    # Replace causal mask update for both versions
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = _update_causal_mask

    logger.info("✅ Fail-fast flash attention enabled for Qwen2.5-VL")
    logger.info("   → Uses official HuggingFace flash attention implementation")
    logger.info("   → Fails fast with clear errors instead of silent fallbacks")
    logger.info("   → Validates attention mask and sequence length compatibility")


# Apply monkey patches for trainable parameter printing
for vision_cls in (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
):
    vision_cls.print_trainable_parameters = print_trainable_parameters_visual

for model_cls in (
    Qwen2VLModel,
    Qwen2_5_VLModel,
):
    model_cls.print_trainable_parameters = print_trainable_parameters
