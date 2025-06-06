#!/usr/bin/env python3
"""
Core Modules for Qwen2.5-VL Data Processing Pipeline

This module provides the core functionality shared across all data conversion scripts:
- Token mapping and field standardization
- Data structure processing and validation
- Response type filtering
- Object sorting and formatting

Standardized field names:
- 'label' -> 'object_type'
- 'question' -> 'property'
- 'extra question' -> 'extra_info'
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Union

# Configure logging
logger = logging.getLogger(__name__)

# Regex to detect Chinese characters
CHINESE_CHAR_REGEX = re.compile(r"[\u4e00-\u9fff]")

# Standardized field mapping - ONLY use new field names
FIELD_MAPPING = {
    "label": "object_type",
    "question": "property",
    "extra question": "extra_info",
    "question_ex": "extra_info",
}

# Default response types - use standardized names only
DEFAULT_RESPONSE_TYPES = {"object_type", "property", "extra_info"}


class TokenMapper:
    """Handles token mapping and field standardization."""

    def __init__(self, token_map_path: Union[str, Path]):
        """Initialize with token map file."""
        self.token_map = self._load_token_map(token_map_path)
        self.missing_tokens: Set[str] = set()

    def _load_token_map(self, map_file_path: Union[str, Path]) -> Dict[str, str]:
        """Load token mapping from JSON file."""
        map_file_path = Path(map_file_path)
        if not map_file_path.is_file():
            logger.error(f"Token map file not found: {map_file_path}")
            raise FileNotFoundError(f"Token map file not found: {map_file_path}")

        with open(map_file_path, "r", encoding="utf-8") as f:
            token_map = json.load(f)

        logger.info(f"Loaded {len(token_map)} token mappings from {map_file_path}")
        return token_map

    def map_token(self, token: Union[str, List[str]]) -> Union[str, List[str]]:
        """Map a single token or list of tokens using the token map."""
        if isinstance(token, list):
            return [self._map_single_token(t) for t in token]
        else:
            return self._map_single_token(token)

    def _map_single_token(self, token: str) -> str:
        """Map a single token."""
        if not isinstance(token, str):
            token = str(token)

        if token == "":  # Allow empty strings
            return token
        elif token in self.token_map:
            return self.token_map[token].lower()
        else:
            self.missing_tokens.add(token)
            return token  # Keep original

    def has_chinese_chars(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        if not isinstance(text, str):
            return False
        return bool(CHINESE_CHAR_REGEX.search(text))

    def get_missing_tokens(self) -> Set[str]:
        """Get all missing tokens encountered during mapping."""
        return self.missing_tokens.copy()


class FieldStandardizer:
    """Handles field name standardization and data structure processing."""

    @staticmethod
    def standardize_field_names(data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize field names according to the new naming convention."""
        standardized = {}

        for key, value in data.items():
            # Map field names to standardized versions
            if key in FIELD_MAPPING:
                standardized_key = FIELD_MAPPING[key]
            else:
                standardized_key = key

            standardized[standardized_key] = value

        return standardized

    @staticmethod
    def extract_content_dict(
        source_dict: Dict[str, Any], token_mapper: TokenMapper
    ) -> Dict[str, Any]:
        """Extract and process content dictionary with token mapping."""
        content_dict = {}

        # Extract and map object_type (from label field)
        object_type = source_dict.get("label", "")
        if object_type:
            content_dict["object_type"] = token_mapper.map_token(object_type)
        else:
            content_dict["object_type"] = ""

        # Extract and map property (from question field)
        property_value = source_dict.get("question", "")
        if property_value:
            content_dict["property"] = token_mapper.map_token(property_value)
        else:
            content_dict["property"] = ""

        # Extract and map extra_info (from question_ex field)
        extra_info = source_dict.get("question_ex", "")
        if extra_info:
            content_dict["extra_info"] = token_mapper.map_token(extra_info)
        else:
            content_dict["extra_info"] = ""

        return content_dict


class ResponseFormatter:
    """Handles response formatting and filtering based on response types."""

    @staticmethod
    def format_to_string(
        content_dict: Dict[str, Any], response_types: Set[str] = None
    ) -> str:
        """Convert content dictionary to string format with flexible response types."""
        if response_types is None:
            response_types = DEFAULT_RESPONSE_TYPES

        parts = []

        # Add object_type if requested
        if "object_type" in response_types:
            object_type = content_dict.get("object_type", "")
            if object_type:
                parts.append(f"object_type:{object_type}")
            else:
                parts.append("object_type:none")

        # Add property if requested
        if "property" in response_types:
            property_value = content_dict.get("property", "")
            if isinstance(property_value, list):
                property_value = ", ".join(property_value) if property_value else ""
            elif not isinstance(property_value, str):
                property_value = str(property_value)

            if property_value:
                parts.append(f"property:{property_value}")
            else:
                parts.append("property:none")

        # Add extra_info if requested
        if "extra_info" in response_types:
            extra_info = content_dict.get("extra_info", "")
            if isinstance(extra_info, list):
                extra_info = ", ".join(extra_info) if extra_info else ""
            elif not isinstance(extra_info, str):
                extra_info = str(extra_info)

            if extra_info:
                parts.append(f"extra_info:{extra_info}")
            else:
                parts.append("extra_info:none")

        return ";".join(parts)

    @staticmethod
    def parse_description_string(description: str) -> Dict[str, str]:
        """Parse description string back into components."""
        components = {"object_type": "", "property": "", "extra_info": ""}

        if not description:
            return components

        parts = description.split(";")
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "object_type":
                    components["object_type"] = value
                elif key == "property":
                    components["property"] = value
                elif key == "extra_info":
                    components["extra_info"] = value

        return components

    @staticmethod
    def filter_description_by_response_types(
        description: str, response_types: Set[str]
    ) -> str:
        """Filter existing description string based on response types."""
        components = ResponseFormatter.parse_description_string(description)
        return ResponseFormatter.format_to_string(components, response_types)


class ObjectProcessor:
    """Handles object processing, sorting, and validation."""

    @staticmethod
    def sort_objects_by_position(
        objects_ref: List[Any], objects_bbox: List[List[float]]
    ) -> tuple:
        """Sort objects by bounding box coordinates (top-left to bottom-right)."""
        if not objects_ref or not objects_bbox or len(objects_ref) != len(objects_bbox):
            return objects_ref, objects_bbox

        # Combine ref and bbox for sorting
        combined_objects = list(zip(objects_ref, objects_bbox))

        # Sort by y1 (top to bottom), then by x1 (left to right)
        combined_objects.sort(key=lambda obj: (obj[1][1], obj[1][0]))

        # Separate back into sorted lists
        if combined_objects:
            objects_ref, objects_bbox = zip(*combined_objects)
            return list(objects_ref), list(objects_bbox)
        else:
            return [], []

    @staticmethod
    def validate_bbox(bbox: List[float]) -> bool:
        """Validate bounding box format [x1, y1, x2, y2]."""
        return (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(coord, (int, float)) for coord in bbox)
        )

    @staticmethod
    def scale_bbox(bbox: List[float], scale_x: float, scale_y: float) -> List[int]:
        """Scale bounding box coordinates."""
        x1, y1, x2, y2 = bbox
        return [
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        ]


class DataValidator:
    """Handles data validation and error checking."""

    @staticmethod
    def validate_sample_structure(sample: Dict[str, Any]) -> bool:
        """Validate basic sample structure."""
        required_fields = ["images", "objects"]

        for field in required_fields:
            if field not in sample:
                logger.error(f"Missing required field: {field}")
                return False

        # Validate images field
        images = sample.get("images")
        if not isinstance(images, list) or len(images) == 0:
            logger.error("Field 'images' must be a non-empty list")
            return False

        # Validate objects field
        objects = sample.get("objects")
        if not isinstance(objects, dict):
            logger.error("Field 'objects' must be a dictionary")
            return False

        # Validate objects structure
        ref_items = objects.get("ref", [])
        bbox_items = objects.get("bbox", [])

        if not isinstance(ref_items, list) or not isinstance(bbox_items, list):
            logger.error("Fields 'objects.ref' and 'objects.bbox' must be lists")
            return False

        if len(ref_items) != len(bbox_items):
            logger.error("Mismatch between number of ref items and bbox items")
            return False

        return True

    @staticmethod
    def validate_conversation_structure(sample: Dict[str, Any]) -> bool:
        """Validate conversation format structure."""
        if "conversations" not in sample:
            logger.error("Missing 'conversations' field")
            return False

        conversations = sample.get("conversations")
        if not isinstance(conversations, list):
            logger.error("Field 'conversations' must be a list")
            return False

        # Check for required roles
        roles = [conv.get("role") for conv in conversations]
        if "system" not in roles or "assistant" not in roles:
            logger.error("Conversations must include 'system' and 'assistant' roles")
            return False

        return True


class CompactResponseFormatter:
    """Handles compact response formatting for cleaner LLM training."""

    @staticmethod
    def format_to_compact_string(
        content_dict: Dict[str, Any], response_types: Set[str] = None
    ) -> str:
        """Convert content dictionary to natural language comma-separated format without schema wrappers."""
        if response_types is None:
            response_types = {"object_type", "property", "extra_info"}

        parts = []

        # Add object_type if requested - no "object_type:" prefix
        if "object_type" in response_types:
            object_type = content_dict.get("object_type", "").strip()
            if object_type and object_type != "none":
                parts.append(object_type)

        # Add property if requested and meaningful - no "property:" prefix
        if "property" in response_types:
            property_value = content_dict.get("property", "").strip()
            if property_value and property_value != "none":
                if isinstance(property_value, list):
                    # Join multiple properties with commas
                    property_parts = [
                        str(p).strip()
                        for p in property_value
                        if str(p).strip() and str(p).strip() != "none"
                    ]
                    if property_parts:
                        parts.extend(property_parts)
                elif property_value:
                    # Split on commas if multiple values are comma-separated
                    if "," in property_value:
                        property_parts = [
                            p.strip()
                            for p in property_value.split(",")
                            if p.strip() and p.strip() != "none"
                        ]
                        parts.extend(property_parts)
                    else:
                        parts.append(property_value)

        # Add extra_info if requested and meaningful - no "extra_info:" prefix
        if "extra_info" in response_types:
            extra_info = content_dict.get("extra_info", "").strip()
            if extra_info and extra_info != "none":
                if isinstance(extra_info, list):
                    # Join multiple extra_info with commas
                    extra_parts = [
                        str(e).strip()
                        for e in extra_info
                        if str(e).strip() and str(e).strip() != "none"
                    ]
                    if extra_parts:
                        parts.extend(extra_parts)
                elif extra_info:
                    # Split on commas if multiple values are comma-separated
                    if "," in extra_info:
                        extra_parts = [
                            e.strip()
                            for e in extra_info.split(",")
                            if e.strip() and e.strip() != "none"
                        ]
                        parts.extend(extra_parts)
                    else:
                        parts.append(extra_info)

        # Join with comma separator for natural reading
        # Filter out empty parts and deduplicate
        clean_parts = []
        seen = set()
        for part in parts:
            part = part.strip()
            if part and part not in seen and part != "none":
                clean_parts.append(part)
                seen.add(part)

        return ", ".join(clean_parts) if clean_parts else "unknown"

    @staticmethod
    def parse_compact_string(description: str) -> Dict[str, str]:
        """Parse compact description back into components (best effort)."""
        # This is for validation/debugging - the compact format prioritizes generation simplicity
        components = {"object_type": "", "property": "", "extra_info": ""}

        if not description or description == "unknown":
            return components

        # For natural language format, we assume first part is object_type
        parts = [p.strip() for p in description.split(",") if p.strip()]
        if parts:
            components["object_type"] = parts[0]
            if len(parts) > 1:
                # Remaining parts are properties/details
                components["property"] = ", ".join(parts[1:])

        return components

    @staticmethod
    def convert_from_verbose_format(
        description: str, response_types: Set[str] = None
    ) -> str:
        """Convert from verbose schema format to simplified comma-separated format."""
        if not description:
            return "unknown"

        # Parse the verbose format: "object_type:X;property:Y;extra_info:Z"
        components = {"object_type": "", "property": "", "extra_info": ""}

        # Split by semicolon and parse each part
        parts = description.split(";")
        for part in parts:
            part = part.strip()
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "object_type":
                    components["object_type"] = value
                elif key == "property":
                    components["property"] = value
                elif key == "extra_info":
                    components["extra_info"] = value

        # Then convert to compact format
        return CompactResponseFormatter.format_to_compact_string(
            components, response_types
        )


# Utility functions for backward compatibility and convenience
def load_token_map(token_map_path: Union[str, Path]) -> Dict[str, str]:
    """Convenience function to load token map."""
    mapper = TokenMapper(token_map_path)
    return mapper.token_map


def convert_to_string_format(
    content_dict: Dict[str, Any], response_types: Set[str] = None
) -> str:
    """Convenience function for string formatting."""
    return ResponseFormatter.format_to_string(content_dict, response_types)


def sort_objects_by_bbox(
    objects_ref: List[Any], objects_bbox: List[List[float]]
) -> tuple:
    """Convenience function for object sorting."""
    return ObjectProcessor.sort_objects_by_position(objects_ref, objects_bbox)
