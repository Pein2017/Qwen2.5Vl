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
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

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

