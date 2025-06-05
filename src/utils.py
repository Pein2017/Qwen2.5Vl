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


# ============================================================================
# Data Loading and Validation
# ============================================================================


def load_clean_semantic_data(file_path: str) -> List[Dict[str, Any]]:
    """Load clean semantic data from JSONL file."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def validate_semantic_data(data: Dict[str, Any]) -> bool:
    """Validate clean semantic data format."""
    required_fields = ["images", "objects"]

    # Check required fields
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return False

    # Validate images
    images = data["images"]
    if not isinstance(images, list):
        logger.warning("Images field must be a list")
        return False

    # Validate objects
    objects = data["objects"]
    if not isinstance(objects, list):
        logger.warning("Objects field must be a list")
        return False

    for obj in objects:
        if not isinstance(obj, dict):
            logger.warning("Each object must be a dictionary")
            return False

        if "box" not in obj or "desc" not in obj:
            logger.warning("Each object must have 'box' and 'desc' fields")
            return False

    return True


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
# Legacy Support Functions
# ============================================================================


def convert_legacy_to_clean(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy data format to clean semantic format."""
    # Extract basic info
    image_path = legacy_data.get("image", "")
    raw_objects = legacy_data.get("objects", [])

    # Convert objects to clean format
    clean_objects = []
    for obj in raw_objects:
        # Handle different bbox formats
        if "bbox" in obj:
            box = obj["bbox"]
        elif "bbox_2d" in obj:
            box = obj["bbox_2d"]
        else:
            continue

        # Extract description
        desc = ""
        if "description" in obj:
            desc_data = obj["description"]
            if isinstance(desc_data, dict):
                desc = desc_data.get("desc", "")
            else:
                desc = str(desc_data)
        elif "desc" in obj:
            desc = str(obj["desc"])

        clean_obj = {
            "box": box,
            "desc": desc,
            "type": obj.get("type", ""),
            "property": obj.get("property", ""),
            "extra_info": obj.get("extra_info", ""),
        }
        clean_objects.append(clean_obj)

    return {"images": [image_path] if image_path else [], "objects": clean_objects}


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


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_model_path(model_size: str = "7B") -> str:
    """Get model path based on size."""
    if model_size == "3B":
        return DEFAULT_BASE_MODEL_PATH
    elif model_size == "7B":
        return DEFAULT_7B_MODEL_PATH
    else:
        raise ValueError(f"Unsupported model size: {model_size}")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)
