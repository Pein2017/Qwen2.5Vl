"""
Compatibility shim for prompt.py

This file maintains backward compatibility by re-exporting all prompt-related
functions from the new data_utils module.
"""

# Import everything from the new location
from .data_utils import (
    CHINESE_EVALUATION_PROMPT,
    CHINESE_FEW_SHOT_SECTION,
    CHINESE_TRAINING_PROMPT,
    ENGLISH_FEW_SHOT_SECTION,
    format_few_shot_prompt,
    get_learning_instruction,
    get_optimized_prompt_for_context,
    get_system_prompt,
    get_user_prompt_prefix,
    validate_prompt_language,
)


# Re-export everything for backward compatibility
__all__ = [
    "CHINESE_TRAINING_PROMPT",
    "CHINESE_EVALUATION_PROMPT",
    "CHINESE_FEW_SHOT_SECTION",
    "ENGLISH_FEW_SHOT_SECTION",
    "get_system_prompt",
    "get_user_prompt_prefix",
    "get_learning_instruction",
    "format_few_shot_prompt",
    "validate_prompt_language",
    "get_optimized_prompt_for_context",
]
