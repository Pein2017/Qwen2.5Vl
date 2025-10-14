"""
Utilities Package

This package contains reorganized utility modules for BBU training:
- data_utils: Data processing, parsing, schemas, and prompt templates
- training_utils: Training helpers, metrics, callbacks, and training schemas
- model_utils: Model loading, patching helpers, and model schemas
- tokens/: Token management (unchanged)

This reorganization maintains backward compatibility while providing better organization.
"""

# Import all functions and classes from the new modular structure
from .data_utils import (
    CHINESE_EVALUATION_PROMPT,
    # Prompt functions
    CHINESE_TRAINING_PROMPT,
    # Constants
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    # Data schema classes
    ChatMessage,
    GroundTruthObject,
    ImageSample,
    MultiChatSample,
    # Response parser
    ResponseParser,
    format_conversation,
    format_few_shot_prompt,
    format_multi_round_conversation,
    format_object_description,
    format_single_round_conversation,
    get_learning_instruction,
    get_optimized_prompt_for_context,
    get_system_prompt,
    get_user_prompt_prefix,
    # Data functions
    load_jsonl,
    validate_prompt_language,
)
from .model_utils import (
    # Type aliases
    AttentionMaskType,
    CollatedGridThwType,
    CollatedPixelType,
    DetectionHeadOutputs,
    ImageGridThwType,
    InputIdsType,
    LabelsType,
    LLMHiddenStates,
    # Model schema classes
    ModelAssets,
    ModelInputs,
    ModelOutput,
    ObjectFeaturesType,
    PixelValuesType,
    PredBoxesRawType,
    PredBoxesType,
    PredObjectnessType,
    VisionFeatures,
    apply_model_patches,
    assert_detection_head_outputs,
    assert_llm_hidden_states,
    # Model validation functions
    assert_model_inputs,
    assert_model_output,
    assert_vision_features,
    # Vision processing functions
    ensure_batched_vision_feats,
    filter_inputs_for_generation,
    # Model functions
    filter_inputs_for_model,
    get_model_memory_usage,
    load_model_assets,
    log_model_info,
    merge_vision_tokens,
    # Model patching functions
    patch_model_for_coordinate_tokens,
    prepare_model_for_training,
)

# Token management (unchanged)
from .tokens import SpecialTokens, TokenFormatter
from .training_utils import (
    # Training schema classes
    ChatProcessorOutput,
    CollatedBatch,
    # Type aliases
    LossDictType,
    TrainingMetrics,
    assert_chat_processor_output,
    assert_collated_batch,
    # Validation functions
    assert_tensor_shape,
    # Metrics functions
    create_training_metrics,
    # Training helper functions
    debug_input_shapes,
    fix_attention_mask_mismatch,
    # Memory utilities
    get_tensor_memory_usage,
    log_batch_memory_usage,
    log_training_metrics,
    optimize_batch_for_memory,
    prepare_inputs_for_forward,
    prepare_inputs_for_generate,
    safe_prepare_inputs,
    validate_attention_mask_consistency,
)


# ==============================================================================
# Backward Compatibility Aliases
# ==============================================================================

# For backward compatibility, maintain aliases to original locations.
# This ensures existing imports continue to work while encouraging migration
# to the new structure.

# Legacy aliases for prompt functions (originally from prompt.py)
CHINESE_EVALUATION_PROMPT = CHINESE_EVALUATION_PROMPT
CHINESE_TRAINING_PROMPT = CHINESE_TRAINING_PROMPT
get_optimized_prompt_for_context = get_optimized_prompt_for_context

# Legacy aliases for utility functions (originally from utils.py)
format_object_description = format_object_description
format_conversation = format_conversation
load_jsonl = load_jsonl
debug_input_shapes = debug_input_shapes
prepare_inputs_for_forward = prepare_inputs_for_forward
prepare_inputs_for_generate = prepare_inputs_for_generate

# Legacy aliases for schema classes (originally from schema.py)
assert_tensor_shape = assert_tensor_shape

# ==============================================================================
# Complete Export List
# ==============================================================================

__all__ = [
    # Constants
    "IGNORE_INDEX",
    "DEFAULT_IMAGE_TOKEN",
    "CHINESE_TRAINING_PROMPT",
    "CHINESE_EVALUATION_PROMPT",

    # Data processing functions
    "load_jsonl",
    "format_object_description",
    "format_single_round_conversation",
    "format_multi_round_conversation",
    "format_conversation",

    # Prompt functions
    "get_system_prompt",
    "get_user_prompt_prefix",
    "get_learning_instruction",
    "format_few_shot_prompt",
    "validate_prompt_language",
    "get_optimized_prompt_for_context",

    # Training helper functions
    "debug_input_shapes",
    "prepare_inputs_for_forward",
    "prepare_inputs_for_generate",
    "validate_attention_mask_consistency",
    "fix_attention_mask_mismatch",
    "safe_prepare_inputs",

    # Model helper functions
    "filter_inputs_for_model",
    "filter_inputs_for_generation",
    "load_model_assets",
    "prepare_model_for_training",
    "get_model_memory_usage",
    "log_model_info",
    "ensure_batched_vision_feats",
    "merge_vision_tokens",
    "patch_model_for_coordinate_tokens",
    "apply_model_patches",

    # Memory utilities
    "get_tensor_memory_usage",
    "log_batch_memory_usage",
    "optimize_batch_for_memory",

    # Metrics functions
    "create_training_metrics",
    "log_training_metrics",

    # Data schema classes
    "ChatMessage",
    "ImageSample",
    "MultiChatSample",
    "GroundTruthObject",

    # Training schema classes
    "ChatProcessorOutput",
    "CollatedBatch",
    "TrainingMetrics",

    # Model schema classes
    "ModelAssets",
    "ModelInputs",
    "ModelOutput",
    "LLMHiddenStates",
    "DetectionHeadOutputs",
    "VisionFeatures",

    # Response parsing
    "ResponseParser",

    # Token management
    "SpecialTokens",
    "TokenFormatter",

    # Type aliases and decorators
    "LossDictType",
    "assert_tensor_shape",
    "AttentionMaskType",
    "PixelValuesType",
    "ImageGridThwType",
    "InputIdsType",
    "LabelsType",
    "PredBoxesType",
    "PredBoxesRawType",
    "PredObjectnessType",
    "ObjectFeaturesType",
    "CollatedPixelType",
    "CollatedGridThwType",

    # Validation functions
    "assert_chat_processor_output",
    "assert_collated_batch",
    "assert_model_inputs",
    "assert_model_output",
    "assert_detection_head_outputs",
    "assert_llm_hidden_states",
    "assert_vision_features",
]
