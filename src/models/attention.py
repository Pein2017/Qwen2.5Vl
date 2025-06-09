"""
Flash Attention utilities for Qwen2.5-VL training.

This module provides flash attention detection and replacement functionality
to optimize memory usage and training speed.
"""

from typing import Optional

import torch
from flash_attn.flash_attn_interface import flash_attn_func
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)

from src.logger_utils import get_attention_logger

logger = get_attention_logger()


# Direct Flash Attention availability check (more reliable than transformers' CUDA-dependent check)
def _is_flash_attn_available():
    """Check if Flash Attention is available, regardless of CUDA availability."""
    try:
        import importlib.util

        # Check if flash_attn is available
        if importlib.util.find_spec("flash_attn") is None:
            return False

        # Try importing the specific modules we need
        import flash_attn  # noqa: F401
        from flash_attn.bert_padding import (  # noqa: F401
            index_first_axis,
            pad_input,
            unpad_input,
        )
        from flash_attn.flash_attn_interface import (  # noqa: F401
            flash_attn_func,
            flash_attn_varlen_func,
        )

        return True
    except ImportError:
        return False


# Conditional imports for Flash Attention
if _is_flash_attn_available():
    from flash_attn.bert_padding import pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _upad_input

    FLASH_ATTENTION_AVAILABLE = True
    print("✅ Flash Attention 2 detected and imported successfully")
else:
    # Fallback imports or dummy functions
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️  Flash Attention 2 not available. Flash attention will be disabled.")


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    This implementation matches the official Qwen2.5-VL flash attention implementation.
    """
    if not FLASH_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "Flash Attention 2 is not available. Please install flash-attn>=2.1.0: "
            "pip install flash-attn --no-build-isolation"
        )

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Flash attention kwargs
    flash_kwargs = {}
    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
            _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        # No padding, use regular flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool = False,
):
    """
    Updated causal mask handling that matches official Qwen2.5-VL implementation.

    For flash attention, we simply return the original attention mask or None,
    letting the official flash attention implementation handle the details.
    """
    # For flash attention, follow the official Qwen2.5-VL approach
    if hasattr(self, "config") and (
        hasattr(self.config, "_attn_implementation")
        and self.config._attn_implementation == "flash_attention_2"
    ):
        if attention_mask is not None and past_key_values is not None:
            is_padding_right = (
                attention_mask[:, -1].sum().item() != input_tensor.size()[0]
            )
            if is_padding_right:
                logger.warning(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. "
                    "Consider using padding_side='left' before tokenizing the input."
                )
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For other attention implementations, preserve the boolean attention mask
    # and let the standard attention implementation handle it properly
    return attention_mask


def enable_flash_attention():
    """Enable flash attention for Qwen2.5VL."""
    if not FLASH_ATTENTION_AVAILABLE:
        print(
            "⚠️  Flash Attention 2 not available. Please install: pip install flash-attn --no-build-isolation"
        )
        return False

    try:
        # Disable custom replacement for compatibility
        print("✅ Flash attention available (using official implementation)")
        print("   → Custom overrides disabled for stability")
        print("   → Using transformers' native flash attention")
        return True
    except Exception as e:
        print(f"⚠️  Flash attention check failed: {e}")
        return False


"""
Custom Flash Attention implementation optimized for StandardDataCollator.

This module provides flash attention that works with boolean attention masks
instead of cumulative sequence lengths, making it compatible with padded batches.
"""


def _optimized_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    **kwargs,
):
    """
    Simplified flash attention that fails fast instead of falling back.

    This implementation assumes flash attention should work and raises clear errors
    if it doesn't, following the fail-fast principle.
    """
    if not _is_flash_attn_available():
        raise RuntimeError(
            "Flash Attention 2 is required but not available. "
            "Install with: pip install flash-attn --no-build-isolation"
        )

    batch_size, num_heads, seq_len, head_dim = query_states.shape

    # FAIL-FAST: Validate inputs instead of falling back
    if attention_mask is not None and attention_mask.shape[-1] != seq_len:
        raise ValueError(
            f"Attention mask sequence length mismatch: "
            f"mask_len={attention_mask.shape[-1]}, seq_len={seq_len}. "
            f"This indicates a data preprocessing issue that must be fixed."
        )

    logger.debug(
        f"🚀 Using flash attention: batch={batch_size}, seq_len={seq_len}, heads={num_heads}"
    )

    # Ensure correct input format for flash attention: [batch, seq, heads, head_dim]
    if query_states.shape[1] == num_heads:
        # Currently [batch, heads, seq, head_dim], need [batch, seq, heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    # Determine causal setting
    causal = is_causal and (not use_top_left_mask or query_length != 1)

    # Use the official flash attention implementation directly
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=dropout,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        is_causal=causal,
        use_top_left_mask=use_top_left_mask,
        softcap=softcap,
        **kwargs,
    )

    # Output should be [batch, seq, heads * head_dim]
    # Reshape to [batch, seq, heads, head_dim] then transpose to [batch, heads, seq, head_dim]
    attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
    return attn_output.transpose(1, 2)


def replace_qwen2_vl_attention_class():
    """
    Replace Qwen2.5-VL attention with fail-fast flash attention implementation.

    This replaces the default flash attention with a version that:
    - Fails fast instead of falling back to standard attention
    - Validates inputs and raises clear errors for debugging
    - Uses the official Transformers flash attention implementation directly
    """
    import transformers
    import transformers.modeling_flash_attention_utils

    # Replace flash attention implementation
    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _optimized_flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _optimized_flash_attention_forward
    )

    # Replace causal mask update
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = _update_causal_mask

    logger.info("✅ Fail-fast flash attention enabled for Qwen2.5-VL")
    logger.info("   → Uses official Transformers flash attention implementation")
    logger.info("   → Fails fast with clear errors instead of silent fallbacks")
    logger.info("   → Validates attention mask and sequence length compatibility")


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components.
    """
    is_embed_trainable = any(
        param.requires_grad for param in self.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)


# Apply monkey patches for trainable parameter printing only
Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters
