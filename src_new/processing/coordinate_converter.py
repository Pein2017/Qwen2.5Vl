#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused coordinate token converter for Qwen2.5-VL.

This is the ONLY custom logic preserved from the 563-line templates.py.
Everything else is handled by official HuggingFace components.

Key Features:
- ONLY coordinate token conversion (no chat formatting)
- Supports bbox_2d, quad, line geometries
- Fail-fast validation with explicit errors
- Clean 50-line implementation
"""

from typing import Any, Dict, List


class CoordinateTokenConverter:
    """
    Focused component that handles ONLY coordinate token conversion.

    This replaces the _format_objects_for_response method from templates.py
    and is the ONLY custom logic we need to preserve.
    """

    def __init__(self, max_coord_value: int):
        """
        Initialize coordinate token converter.

        Args:
            max_coord_value: Maximum coordinate value for clamping (required)

        Raises:
            ValueError: If max_coord_value is missing or invalid
        """
        if max_coord_value is None:
            raise ValueError(
                "max_coord_value is required and must be provided via configuration (YAML)."
            )
        if not isinstance(max_coord_value, int) or max_coord_value <= 0:
            raise ValueError(
                f"max_coord_value must be a positive integer, got {max_coord_value!r}"
            )

        self.max_coord_value = max_coord_value

        # Geometry type mappings (from current _format_objects_for_response)
        self.geometry_tokens = {
            "bbox_2d": (
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|box_start|>",
                "<|box_end|>",
            ),
            "quad": (
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|quad_start|>",
                "<|quad_end|>",
            ),
            "line": (
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|line_start|>",
                "<|line_end|>",
            ),
        }

    def convert_objects_to_tokens(self, objects: List[Dict[str, Any]]) -> str:
        """
        Convert objects list to coordinate token format.

        This replaces the _format_objects_for_response method from templates.py
        and is the ONLY custom logic we need to preserve.

        Args:
            objects: List of object dictionaries with geometry and description

        Returns:
            Formatted string with coordinate tokens

        Raises:
            ValueError: If objects list is empty or contains invalid objects
        """
        if not objects:
            raise ValueError(
                "Empty objects list encountered. This indicates a data preprocessing failure - "
                "empty object lists should have been filtered out earlier in the pipeline."
            )

        if not isinstance(objects, list):
            raise ValueError(f"objects must be a list, got {type(objects)}")

        token_strings = []
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"Object {i} must be a dict, got {type(obj)}")

            token_string = self._convert_single_object(obj, i)
            if token_string:
                token_strings.append(token_string)

        if not token_strings:
            raise ValueError("No valid objects found after conversion")

        return "\n".join(token_strings)

    def _convert_single_object(self, obj: Dict[str, Any], obj_index: int) -> str:
        """
        Convert a single object to coordinate token format.

        Args:
            obj: Object dictionary with geometry and description
            obj_index: Object index for error reporting

        Returns:
            Formatted coordinate token string

        Raises:
            ValueError: If object has unsupported geometry type or invalid coordinates
        """
        # Determine geometry type and coordinates
        geometry_type = None
        coordinates = None

        if "bbox_2d" in obj:
            geometry_type = "bbox_2d"
            coordinates = obj["bbox_2d"]
        elif "quad" in obj:
            geometry_type = "quad"
            coordinates = obj["quad"]

        elif "line" in obj:
            geometry_type = "line"
            coordinates = obj["line"]
        elif all(key in obj for key in ["x1", "y1", "x2", "y2"]):
            # Legacy format - convert to bbox_2d
            geometry_type = "bbox_2d"
            coordinates = [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]
        else:
            # Fail-fast: Raise error for unsupported geometry types
            available_keys = [k for k in obj.keys() if k not in ["desc", "category"]]
            raise ValueError(
                f"Object {obj_index} contains unsupported geometry type. "
                f"Expected one of: bbox_2d, quad, line. "
                f"Found geometry keys: {available_keys}. "
                f"Full object: {obj}"
            )

        # Validate coordinates
        if not isinstance(coordinates, list):
            raise ValueError(
                f"Object {obj_index} coordinates must be a list, got {type(coordinates)}"
            )

        if not coordinates:
            raise ValueError(f"Object {obj_index} has empty coordinates list")

        # Validate coordinate values are numeric
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, (int, float)):
                raise ValueError(
                    f"Object {obj_index} coordinate {i} must be numeric, got {type(coord)}: {coord}"
                )

        # Get tokens for this geometry type
        ref_start, ref_end, geom_start, geom_end = self.geometry_tokens[geometry_type]

        # Convert coordinates to coordinate tokens
        coord_tokens = []
        for coord in coordinates:
            clamped_coord = max(0, min(int(coord), self.max_coord_value))
            coord_tokens.append(f"<|coord_{clamped_coord}|>")

        coord_string = ", ".join(coord_tokens)
        # Use empty string if description is missing (graceful handling)
        description = obj.get("desc", "")

        # Format complete token string (exact same format as current system)
        return (
            f"{ref_start}{description}{ref_end}{geom_start}[{coord_string}]{geom_end}"
        )
