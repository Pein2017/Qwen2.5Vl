"""
Utilities Package

This package contains utility modules for BBU training:
- utils: General utility functions (JSONL, tensor debugging, etc.)
- prompt: Prompt templates and conversation formatting
- response_parser: Output parsing and validation
- schema: Type definitions and validation schemas
"""

# Core utilities
from .utils import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN, 
    load_jsonl,
    debug_input_shapes,
    prepare_inputs_for_forward,
    prepare_inputs_for_generate,
    format_object_description,
    format_conversation,
)

# Prompt management
from .prompt import (
    CHINESE_TRAINING_PROMPT,
    CHINESE_EVALUATION_PROMPT,
    get_system_prompt,
    get_optimized_prompt_for_context,
)

# Response parsing
from .response_parser import ResponseParser

# Schema and type definitions
from .schema import (
    ChatMessage,
    ImageSample,
    MultiChatSample,
    ChatProcessorOutput,
    CollatedBatch,
    GroundTruthObject,
    DetectionPredictions,
    DetectionLossComponents,
    LossDictType,
    assert_tensor_shape,
)

# Token management
from .tokens import SpecialTokens, TokenFormatter

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