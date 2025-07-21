"""
Utilities Package

This package contains utility modules for BBU training:
- utils: General utility functions (JSONL, tensor debugging, etc.)
- prompt: Prompt templates and conversation formatting
- response_parser: Output parsing and validation
- schema: Type definitions and validation schemas
"""

# Core utilities
# Prompt management
from .prompt import (
    CHINESE_EVALUATION_PROMPT,
    CHINESE_TRAINING_PROMPT,
    get_optimized_prompt_for_context,
    get_system_prompt,
)

# Response parsing
from .response_parser import ResponseParser

# Schema and type definitions
from .schema import (
    ChatMessage,
    ChatProcessorOutput,
    CollatedBatch,
    DetectionLossComponents,
    DetectionPredictions,
    GroundTruthObject,
    ImageSample,
    LossDictType,
    MultiChatSample,
    assert_tensor_shape,
)

# Token management
from .tokens import SpecialTokens, TokenFormatter
from .utils import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    debug_input_shapes,
    format_conversation,
    format_object_description,
    load_jsonl,
    prepare_inputs_for_forward,
    prepare_inputs_for_generate,
)


__all__ = [
    # Constants
    "IGNORE_INDEX",
    "DEFAULT_IMAGE_TOKEN",
    "CHINESE_TRAINING_PROMPT",
    "CHINESE_EVALUATION_PROMPT",
    # Functions
    "load_jsonl",
    "debug_input_shapes",
    "prepare_inputs_for_forward",
    "prepare_inputs_for_generate",
    "format_object_description",
    "format_conversation",
    "get_system_prompt",
    "get_optimized_prompt_for_context",
    # Classes
    "ResponseParser",
    "ChatMessage",
    "ImageSample",
    "MultiChatSample",
    "ChatProcessorOutput",
    "CollatedBatch",
    "GroundTruthObject",
    "DetectionPredictions",
    "DetectionLossComponents",
    "SpecialTokens",
    "TokenFormatter",
    # Types and decorators
    "LossDictType",
    "assert_tensor_shape",
]
