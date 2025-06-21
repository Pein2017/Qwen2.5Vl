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
    def validate_bbox(
        bbox: List[float], image_width: int = None, image_height: int = None
    ) -> bool:
        """
        Validate a single bounding box with enhanced checks.

        Args:
            bbox: A list of 4 numbers [x_min, y_min, x_max, y_max].
            image_width: Optional width of the image to check bounds.
            image_height: Optional height of the image to check bounds.

        Raises:
            ValueError: If the bounding box is invalid.
        """
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Bbox must be a list of 4 elements, but got: {bbox}")

        if not all(isinstance(coord, (int, float)) for coord in bbox):
            raise ValueError(f"Bbox coordinates must be numbers, but got: {bbox}")

        x_min, y_min, x_max, y_max = bbox

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        bbox = [x_min, y_min, x_max, y_max]

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Invalid bbox, x_min must be less than x_max and y_min must be less than y_max, but got: {bbox}"
            )

        if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
            raise ValueError(f"Bbox coordinates cannot be negative, but got: {bbox}")

        if image_width is not None and image_height is not None:
            if x_max > image_width or y_max > image_height:
                raise ValueError(
                    f"Bbox {bbox} exceeds image dimensions ({image_width}x{image_height})"
                )
        return True

    @staticmethod
    def scale_bbox(
        bbox: List[float],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
    ) -> List[int]:
        """
        Scale a bounding box from the original image dimensions to the resized
        dimensions.

        Fail-fast philosophy:
        1. The original bounding box must be fully inside the original image. If
           not, a ValueError is raised.
        2. The scaled bounding box must be fully inside the resized image. If
           not, a ValueError is raised.
        3. No automatic clamping, expansion, or silent correction is applied.
        """
        # Validate the original bounding box first
        ObjectProcessor.validate_bbox(
            bbox, image_width=original_width, image_height=original_height
        )

        x_scale = new_width / original_width
        y_scale = new_height / original_height

        x_min, y_min, x_max, y_max = bbox

        # Scale and round to integer pixel coordinates
        scaled_x1 = int(round(x_min * x_scale))
        scaled_y1 = int(round(y_min * y_scale))
        scaled_x2 = int(round(x_max * x_scale))
        scaled_y2 = int(round(y_max * y_scale))

        # Ensure correct ordering after scaling
        new_x_min = min(scaled_x1, scaled_x2)
        new_y_min = min(scaled_y1, scaled_y2)
        new_x_max = max(scaled_x1, scaled_x2)
        new_y_max = max(scaled_y1, scaled_y2)

        final_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]

        # Validate the scaled bounding box against the new image dimensions
        ObjectProcessor.validate_bbox(
            final_bbox, image_width=new_width, image_height=new_height
        )

        return final_bbox


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
        """Convert content dictionary to a structured, slash-separated format."""
        if response_types is None:
            response_types = {"object_type", "property", "extra_info"}

        parts = []

        # Add object_type if requested
        if "object_type" in response_types:
            object_type = content_dict.get("object_type", "").strip()
            if object_type and object_type != "none":
                parts.append(object_type)

        # Add property if requested
        if "property" in response_types:
            property_value = content_dict.get("property", "")
            if property_value and property_value != "none":
                if isinstance(property_value, list):
                    parts.extend(
                        [
                            str(p).strip()
                            for p in property_value
                            if str(p).strip() and str(p).strip() != "none"
                        ]
                    )
                else:
                    parts.append(str(property_value).strip())

        # Add extra_info if requested
        if "extra_info" in response_types:
            extra_info = content_dict.get("extra_info", "")
            if extra_info and extra_info != "none":
                if isinstance(extra_info, list):
                    parts.extend(
                        [
                            str(e).strip()
                            for e in extra_info
                            if str(e).strip() and str(e).strip() != "none"
                        ]
                    )
                else:
                    parts.append(str(extra_info).strip())

        # Filter out empty parts and deduplicate
        clean_parts = []
        seen = set()
        for part in parts:
            part = part.strip()
            if part and part not in seen and part != "none":
                clean_parts.append(part)
                seen.add(part)

        # Join with slash separator for a structured format
        return "/".join(clean_parts) if clean_parts else "unknown"

    @staticmethod
    def parse_compact_string(description: str) -> Dict[str, str]:
        """Parse compact description back into components (best effort)."""
        components = {"object_type": "", "property": "", "extra_info": ""}

        if not description or description == "unknown":
            return components

        # For the compact format, we assume the first part is object_type
        parts = [p.strip() for p in description.split("/") if p.strip()]
        if parts:
            components["object_type"] = parts[0]
            if len(parts) > 1:
                # The rest are considered properties/extra_info
                components["property"] = "/".join(parts[1:])
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
