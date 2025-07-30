"""
Compatibility shim for schema.py

This file maintains backward compatibility by re-exporting all schema-related
classes and functions from the new modular structure.
"""

# Import from new modules
from .data_utils import (
    ChatMessage,
    ImageSample,
    MultiChatSample,
    GroundTruthObject,
)

from .training_utils import (
    ChatProcessorOutput,
    CollatedBatch,
    LossDictType,
    assert_tensor_shape,
    assert_chat_processor_output,
    assert_collated_batch,
)

from .model_utils import (
    ModelAssets,
    ModelInputs,
    ModelOutput,
    LLMHiddenStates,
    DetectionHeadOutputs,
    VisionFeatures,
    assert_model_inputs,
    assert_model_output,
    assert_detection_head_outputs,
    assert_llm_hidden_states,
    assert_vision_features,
    ensure_batched_vision_feats,
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