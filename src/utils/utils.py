"""
Compatibility shim for utils.py

This file maintains backward compatibility by re-exporting all utility
functions from the new modular structure.
"""

# Import from new modules
from .data_utils import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    format_conversation,
    format_multi_round_conversation,
    format_object_description,
    format_single_round_conversation,
    load_jsonl,
)
from .model_utils import (
    _validate_and_fix_shapes,
    filter_inputs_for_generation,
    filter_inputs_for_model,
)
from .training_utils import (
    debug_input_shapes,
    fix_attention_mask_mismatch,
    prepare_inputs_for_forward,
    prepare_inputs_for_generate,
    safe_prepare_inputs,
    validate_attention_mask_consistency,
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
