#!/usr/bin/env python3
"""
Utilities for Qwen2.5-VL with Clean Architecture

This module provides utilities that work with clean semantic data:
- Data validation and loading
- Legacy format conversion
- Basic conversation formatting (for reference)

The training pipeline handles all special token processing.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.logger_utils import get_utils_logger

logger = get_utils_logger()

# Default image token for clean semantic data
DEFAULT_IMAGE_TOKEN = "<image>"

# Special tokens for reference (used by training pipeline)
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
OBJECT_REF_START = "<object_ref_start>"
OBJECT_REF_END = "<object_ref_end>"

# Default model paths
DEFAULT_BASE_MODEL_PATH = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_7B_MODEL_PATH = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-7B-Instruct"

# Training constants
IGNORE_INDEX = -100  # Standard ignore index for loss calculation


# ============================================================================
# Basic Conversation Formatting (for reference)
# ============================================================================


def format_object_description(obj: Dict[str, Any]) -> str:
    """Format object description with clean syntax."""
    box = obj.get("box", [])
    desc = obj.get("desc", "")

    # Use clean JSON-like format
    return f'{OBJECT_REF_START}{{"box": {box}, "desc": "{desc}"}}{OBJECT_REF_END}'


def format_single_round_conversation(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format single-round conversation from clean semantic data."""
    # User message with image placeholder
    user_content = f"{DEFAULT_IMAGE_TOKEN}\nPlease describe the objects in this image with their locations."

    # Assistant response with object descriptions
    objects = data.get("objects", [])
    if not objects:
        assistant_content = "I don't see any objects in this image."
    else:
        descriptions = [format_object_description(obj) for obj in objects]
        assistant_content = " ".join(descriptions)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_multi_round_conversation(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format multi-round conversation with examples."""
    conversation = []

    # Add examples first
    examples = data.get("examples", [])
    for example in examples:
        example_turns = format_single_round_conversation(example)
        conversation.extend(example_turns)

    # Add main query
    main_turns = format_single_round_conversation(data)
    conversation.extend(main_turns)

    return conversation


def format_conversation(
    data: Dict[str, Any], multi_round: bool = False
) -> List[Dict[str, str]]:
    """Format conversation based on mode."""
    if multi_round:
        return format_multi_round_conversation(data)
    else:
        return format_single_round_conversation(data)


# ============================================================================
# Utility Functions
# ============================================================================


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def debug_input_shapes(
    inputs: Dict[str, Any], prefix: str = "", log_level: str = "DEBUG"
) -> None:
    """
    Debug function to log all input tensor shapes.

    Args:
        inputs: Dictionary of model inputs
        prefix: Prefix for log messages
        log_level: Logging level to use
    """
    # Use the unified logger with appropriate level
    if log_level.upper() == "DEBUG":
        logger.debug(f"🔍 {prefix} INPUT SHAPES DEBUG:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"   {key}: {value.shape} (dtype: {value.dtype})")
            elif isinstance(value, list):
                logger.debug(f"   {key}: list of length {len(value)}")
            else:
                logger.debug(f"   {key}: {type(value)} - {value}")
    elif log_level.upper() == "INFO":
        logger.info(f"🔍 {prefix} INPUT SHAPES DEBUG:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"   {key}: {value.shape} (dtype: {value.dtype})")
            elif isinstance(value, list):
                logger.info(f"   {key}: list of length {len(value)}")
            else:
                logger.info(f"   {key}: {type(value)} - {value}")


# ============================================================================
# Input Preparation for Forward vs Generate
# ============================================================================


def prepare_inputs_for_forward(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare inputs for model.forward() call during training.

    For forward pass:
    - Keep full sequence (input + target)
    - Keep attention_mask matching full sequence
    - Keep labels for loss computation
    - Apply shape validation and fixes

    Args:
        inputs: Raw inputs from data collator

    Returns:
        Dict: Inputs prepared for forward pass
    """
    logger.debug("🔧 Preparing inputs for forward pass")

    # Validate and fix shape mismatches
    validated_inputs = _validate_and_fix_shapes(inputs)

    # Filter to only include model parameters
    forward_inputs = filter_inputs_for_model(validated_inputs)

    logger.debug(f"✅ Forward inputs prepared: {list(forward_inputs.keys())}")
    return forward_inputs


def prepare_inputs_for_generate(
    inputs: Dict[str, Any], prompt_end_indices: Optional[List[int]] = None
) -> Tuple[Dict[str, Any], List[int]]:
    """
    Prepare inputs for model.generate() call during inference.

    For generation:
    - Extract only input part (before generation starts)
    - Create attention_mask matching input part only
    - Remove labels (not needed for generation)
    - Keep vision inputs for first generation step
    - Preserve image_counts_per_sample for proper visual input extraction

    Args:
        inputs: Raw inputs from data collator or trainer
        prompt_end_indices: Optional list of prompt end positions per sample

    Returns:
        Tuple[Dict, List[int]]: (generation_inputs, actual_prompt_end_indices)
    """
    logger.debug("🔧 Preparing inputs for generation")

    if "input_ids" not in inputs:
        raise ValueError("input_ids required for generation")

    input_ids = inputs["input_ids"]
    labels = inputs.get("labels")
    batch_size = input_ids.size(0)

    # Find prompt end positions if not provided
    if prompt_end_indices is None:
        prompt_end_indices = []
        for i in range(batch_size):
            if labels is not None:
                # Find first non-masked label position
                sample_labels = labels[i]
                real_label_mask = sample_labels != IGNORE_INDEX
                if real_label_mask.any():
                    prompt_end_idx = real_label_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    # No labels, use full sequence
                    prompt_end_idx = input_ids.size(1)
            else:
                # No labels, use full sequence
                prompt_end_idx = input_ids.size(1)

            prompt_end_indices.append(prompt_end_idx)

    # Extract input parts and create proper attention masks
    max_prompt_length = max(prompt_end_indices)

    # Create generation inputs
    generation_inputs = {}

    # Extract input_ids up to prompt end
    generation_inputs["input_ids"] = input_ids[:, :max_prompt_length]

    # Create attention mask for input part only
    generation_inputs["attention_mask"] = torch.ones(
        (batch_size, max_prompt_length), dtype=torch.bool, device=input_ids.device
    )

    # Mask padding positions for each sample
    for i, prompt_end_idx in enumerate(prompt_end_indices):
        if prompt_end_idx < max_prompt_length:
            generation_inputs["attention_mask"][i, prompt_end_idx:] = False

    # Add vision inputs if present
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        generation_inputs["pixel_values"] = inputs["pixel_values"]

    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        generation_inputs["image_grid_thw"] = inputs["image_grid_thw"]

    # CRITICAL: Preserve image_counts_per_sample for proper visual input extraction
    if "image_counts_per_sample" in inputs:
        generation_inputs["image_counts_per_sample"] = inputs["image_counts_per_sample"]

    # Add other generation-compatible parameters
    for key in ["position_ids", "use_cache", "second_per_grid_ts"]:
        if key in inputs and inputs[key] is not None:
            generation_inputs[key] = inputs[key]

    logger.debug(f"✅ Generation inputs prepared:")
    logger.debug(f"   input_ids: {generation_inputs['input_ids'].shape}")
    logger.debug(f"   attention_mask: {generation_inputs['attention_mask'].shape}")
    logger.debug(f"   prompt_end_indices: {prompt_end_indices}")
    if "image_counts_per_sample" in generation_inputs:
        logger.debug(
            f"   image_counts_per_sample: {generation_inputs['image_counts_per_sample']}"
        )

    return generation_inputs, prompt_end_indices


def _validate_and_fix_shapes(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix shape mismatches between input_ids and attention_mask.

    This prevents the IndexError in get_rope_index where attention_mask
    and input_ids have different sequence lengths.

    Args:
        inputs: Dictionary containing input_ids and attention_mask

    Returns:
        Dict: Fixed inputs with consistent shapes
    """
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        logger.debug("⚠️ Missing input_ids or attention_mask for validation")
        return inputs

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    logger.debug(f"🔍 SHAPE VALIDATION:")
    logger.debug(f"   input_ids shape: {input_ids.shape}")
    logger.debug(f"   attention_mask shape: {attention_mask.shape}")

    if input_ids.shape == attention_mask.shape:
        logger.debug("✅ Shapes are consistent")
        return inputs

    logger.warning(f"❌ SHAPE MISMATCH DETECTED!")
    logger.warning(f"   input_ids: {input_ids.shape}")
    logger.warning(f"   attention_mask: {attention_mask.shape}")

    # Create a copy to avoid modifying original
    fixed_inputs = inputs.copy()

    # Strategy 1: Truncate attention_mask to match input_ids
    if attention_mask.shape[-1] > input_ids.shape[-1]:
        logger.info("   Strategy: Truncating attention_mask to match input_ids")
        fixed_inputs["attention_mask"] = attention_mask[:, : input_ids.shape[-1]]

    # Strategy 2: Pad attention_mask to match input_ids
    elif input_ids.shape[-1] > attention_mask.shape[-1]:
        logger.info("   Strategy: Padding attention_mask to match input_ids")
        batch_size, seq_len = input_ids.shape
        _, mask_len = attention_mask.shape
        pad_length = seq_len - mask_len

        # Pad with zeros (masked positions)
        padding = torch.zeros(
            batch_size,
            pad_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        fixed_inputs["attention_mask"] = torch.cat([attention_mask, padding], dim=-1)

    logger.info(f"   ✅ Fixed shapes:")
    logger.info(f"      input_ids: {fixed_inputs['input_ids'].shape}")
    logger.info(f"      attention_mask: {fixed_inputs['attention_mask'].shape}")

    return fixed_inputs


def filter_inputs_for_model(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter inputs to only include parameters accepted by the official Qwen2.5-VL model.

    Removes custom parameters like 'image_counts_per_sample' that are used internally
    but not accepted by the model's forward method.

    Args:
        inputs: Raw inputs dictionary

    Returns:
        Dict: Filtered inputs with only valid model parameters
    """
    # Official Qwen2.5-VL forward method parameters
    valid_model_params = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "use_cache",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "rope_deltas",
        "cache_position",
        "second_per_grid_ts",
    }

    # Filter inputs to only include valid parameters
    filtered_inputs = {
        key: value for key, value in inputs.items() if key in valid_model_params
    }

    # Log filtered parameters for debugging
    removed_params = set(inputs.keys()) - set(filtered_inputs.keys())
    if removed_params:
        logger.debug(f"🔧 Filtered out custom parameters: {removed_params}")

    return filtered_inputs


def filter_inputs_for_generation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter inputs for model.generate() calls.

    Excludes training-specific parameters like 'labels' that are not
    needed for generation.

    Args:
        inputs: Raw inputs dictionary

    Returns:
        Dict: Filtered inputs suitable for generation
    """
    # Valid generation parameters (subset of model parameters)
    valid_generation_params = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "use_cache",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    }

    # Filter inputs to only include valid parameters
    filtered_inputs = {
        key: value for key, value in inputs.items() if key in valid_generation_params
    }

    # Log filtered parameters for debugging
    removed_params = set(inputs.keys()) - set(filtered_inputs.keys())
    if removed_params:
        logger.debug(f"🔧 Filtered out non-generation parameters: {removed_params}")

    return filtered_inputs


# ============================================================================
# Legacy Functions (Deprecated)
# ============================================================================


def validate_attention_mask_consistency(inputs: Dict[str, Any]) -> bool:
    """
    DEPRECATED: Use prepare_inputs_for_forward/generate instead.

    Validate that attention_mask and input_ids have consistent shapes.
    """
    logger.warning(
        "⚠️ validate_attention_mask_consistency is deprecated. Use prepare_inputs_for_forward/generate instead."
    )

    if "input_ids" not in inputs or "attention_mask" not in inputs:
        return True

    return inputs["input_ids"].shape == inputs["attention_mask"].shape


def fix_attention_mask_mismatch(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEPRECATED: Use prepare_inputs_for_forward/generate instead.

    Fix attention mask mismatch by ensuring it matches input_ids shape.
    """
    logger.warning(
        "⚠️ fix_attention_mask_mismatch is deprecated. Use prepare_inputs_for_forward/generate instead."
    )
    return _validate_and_fix_shapes(inputs)


def safe_prepare_inputs(
    inputs: Dict[str, Any], validate_shapes: bool = True
) -> Dict[str, Any]:
    """
    DEPRECATED: Use prepare_inputs_for_forward/generate instead.

    Safely prepare inputs with optional shape validation and fixing.
    """
    logger.warning(
        "⚠️ safe_prepare_inputs is deprecated. Use prepare_inputs_for_forward/generate instead."
    )

    if validate_shapes:
        return _validate_and_fix_shapes(inputs)
    return inputs
