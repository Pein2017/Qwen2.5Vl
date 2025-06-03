from typing import Optional

import torch


# Direct Flash Attention availability check (more reliable than transformers' CUDA-dependent check)
def _is_flash_attn_available():
    """Check if Flash Attention is available, regardless of CUDA availability."""
    try:
        import flash_attn
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
        from flash_attn.flash_attn_interface import (
            flash_attn_func,
            flash_attn_varlen_func,
        )

        return True
    except ImportError:
        return False


# Conditional imports for Flash Attention
if _is_flash_attn_available():
    from flash_attn.bert_padding import pad_input
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _upad_input

    FLASH_ATTENTION_AVAILABLE = True
    print("✅ Flash Attention 2 detected and imported successfully")
else:
    # Fallback imports or dummy functions
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️  Flash Attention 2 not available. Flash attention will be disabled.")

from transformers.models.qwen2_vl.modeling_qwen2_vl import Cache


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
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    """Replace the flash attention implementation in transformers with our optimized version."""
    if not FLASH_ATTENTION_AVAILABLE:
        print("⚠️  Flash Attention 2 not available. Skipping attention replacement.")
        return

    import transformers
    import transformers.modeling_flash_attention_utils

    # Replace for both Qwen2VL and Qwen2.5VL
    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = _update_causal_mask


def enable_flash_attention():
    """Enable flash attention for Qwen2.5VL."""
    if not FLASH_ATTENTION_AVAILABLE:
        print(
            "⚠️  Flash Attention 2 not available. Please install: pip install flash-attn --no-build-isolation"
        )
        return False

    try:
        replace_qwen2_vl_attention_class()
        print("✅ Flash attention enabled")
        return True
    except Exception as e:
        print(f"⚠️  Flash attention failed: {e}")
        return False
