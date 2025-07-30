"""
Compatibility shim for utils.py

This file maintains backward compatibility by re-exporting all utility
functions from the new modular structure.
"""

# Import from new modules
from .data_utils import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    format_object_description,
    format_single_round_conversation,
    format_multi_round_conversation,
    format_conversation,
    load_jsonl,
)

from .training_utils import (
    debug_input_shapes,
    prepare_inputs_for_forward,
    prepare_inputs_for_generate,
    validate_attention_mask_consistency,
    fix_attention_mask_mismatch,
    safe_prepare_inputs,
)

from .model_utils import (
    filter_inputs_for_model,
    filter_inputs_for_generation,
    _validate_and_fix_shapes,
)

# Re-export everything for backward compatibility
__all__ = [
    # Constants
    "DEFAULT_IMAGE_TOKEN",
    "IGNORE_INDEX",
    # Data functions
    "format_object_description",
    "format_single_round_conversation",
    "format_multi_round_conversation", 
    "format_conversation",
    "load_jsonl",
    # Training functions
    "debug_input_shapes",
    "prepare_inputs_for_forward",
    "prepare_inputs_for_generate",
    "validate_attention_mask_consistency",
    "fix_attention_mask_mismatch",
    "safe_prepare_inputs",
    # Model functions
    "filter_inputs_for_model",
    "filter_inputs_for_generation",
    "_validate_and_fix_shapes",
]