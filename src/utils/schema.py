"""
Compatibility shim for schema.py

This file maintains backward compatibility by re-exporting all schema-related
classes and functions from the new modular structure.
"""

# Import from new modules
from .data_utils import (
    ChatMessage,
    GroundTruthObject,
    ImageSample,
    MultiChatSample,
)
from .model_utils import (
    DetectionHeadOutputs,
    LLMHiddenStates,
    ModelAssets,
    ModelInputs,
    ModelOutput,
    VisionFeatures,
    assert_detection_head_outputs,
    assert_llm_hidden_states,
    assert_model_inputs,
    assert_model_output,
    assert_vision_features,
    ensure_batched_vision_feats,
)
from .training_utils import (
    ChatProcessorOutput,
    CollatedBatch,
    LossDictType,
    assert_chat_processor_output,
    assert_collated_batch,
    assert_tensor_shape,
)


# Re-export everything for backward compatibility
__all__ = [
    # Data schemas
    "ChatMessage",
    "ImageSample",
    "MultiChatSample",
    "GroundTruthObject",
    # Training schemas
    "ChatProcessorOutput",
    "CollatedBatch",
    "LossDictType",
    # Model schemas
    "ModelAssets",
    "ModelInputs",
    "ModelOutput",
    "LLMHiddenStates",
    "DetectionHeadOutputs",
    "VisionFeatures",
    # Validation functions
    "assert_tensor_shape",
    "assert_chat_processor_output",
    "assert_collated_batch",
    "assert_model_inputs",
    "assert_model_output",
    "assert_detection_head_outputs",
    "assert_llm_hidden_states",
    "assert_vision_features",
    # Utility functions
    "ensure_batched_vision_feats",
]
