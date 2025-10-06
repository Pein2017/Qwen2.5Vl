"""Simplified coordinate converter without legacy coordinate tokens."""

from typing import Any, Dict, List

from .special_tokens import GEOMETRY_TOKENS


class CoordinateTokenConverter:
    """Simplified coordinate token converter that only uses special tokens."""

    def __init__(
        self,
        format_mode: str | None = None,
    ):
        """
        Initialize coordinate token converter without legacy coordinate tokens.

        Args:
            format_mode: Only "special_tokens" is supported.
        """
        if format_mode is not None and format_mode != "special_tokens":
            raise ValueError(f"Unsupported format_mode: {format_mode}")

        # Use centralized canonical geometry tokens
        self.geometry_tokens = GEOMETRY_TOKENS
        # Global rendering mode
        self._mode = "special_tokens"

    def convert_objects_list(self, objects: List[Dict[str, Any]]) -> str:
        """Convert a list of objects to coordinate tokens format."""
        if not objects:
            return ""
        
        out_lines: List[str] = []
        for i, obj in enumerate(objects):
            # Extract geometry and description
            geometry_type, coordinates = self._extract_geometry(obj, i)
            ref_start, ref_end = self._get_ref_tokens()
            geom_start, geom_end = self._get_geom_tokens(geometry_type)
            coord_texts: List[str] = []
            for j, coord in enumerate(coordinates):
                int_coord = int(coord)
                # Basic coordinate validation (no upper limit)
                if int_coord < 0:
                    raise ValueError(f"Object {i} coordinate {j} value {int_coord} must be non-negative")
                coord_texts.append(str(int_coord))
            coord_string = ", ".join(coord_texts)
            out_lines.append(f"{geom_start}[{coord_string}]{geom_end}")
        return "\n".join(out_lines)
    
    def convert_objects_to_tokens(self, objects: List[Dict[str, Any]]) -> str:
        """Convert objects to coordinate tokens format."""
        return self.convert_objects_list(objects)
    
    def convert_objects_to_desc_only(self, objects: List[Dict[str, Any]]) -> str:
        """Convert objects to description-only format."""
        if not objects:
            return ""
        
        ref_start, ref_end = self._get_ref_tokens()
        out_lines = []
        for obj in objects:
            description = obj.get("desc", "")
            out_lines.append(f"{ref_start}{description}{ref_end}")
        return "\n".join(out_lines)
    
    def convert_objects_to_geometry_only(self, objects: List[Dict[str, Any]]) -> str:
        """Convert objects to geometry-only format."""
        if not objects:
            return ""
        
        out_lines = []
        for i, obj in enumerate(objects):
            geometry_type, coordinates = self._extract_geometry(obj, i)
            geom_start, geom_end = self._get_geom_tokens(geometry_type)
            coord_texts = []
            for j, coord in enumerate(coordinates):
                int_coord = int(coord)
                if int_coord < 0:
                    raise ValueError(f"Object {i} coordinate {j} value {int_coord} must be non-negative")
                coord_texts.append(str(int_coord))
            coord_string = ", ".join(coord_texts)
            out_lines.append(f"{geom_start}[{coord_string}]{geom_end}")
        return "\n".join(out_lines)

    # ---------- Public API for user rendering ----------
    def format_geometry_for_user(self, obj: Dict[str, object]) -> str:
        """Format a single object's geometry for inclusion in a user message (wrapper tokens)."""
        if "line" in obj:
            coords = ", ".join(str(int(v)) for v in obj["line"])  
            return f"<|line_start|>[{coords}]<|line_end|>"
        if "bbox_2d" in obj:
            coords = ", ".join(str(int(v)) for v in obj["bbox_2d"])  
            return f"<|box_start|>[{coords}]<|box_end|>"
        if "quad" in obj:
            coords = ", ".join(str(int(v)) for v in obj["quad"])  
            return f"<|quad_start|>[{coords}]<|quad_end|>"
        return ""

    def format_object_ref_for_user(self, desc: str) -> str:
        """Format a description-only object for inclusion in a user message (wrapper tokens)."""
        safe_desc = desc if isinstance(desc, str) else str(desc)
        return f"<|object_ref_start|>{safe_desc}<|object_ref_end|>"

    # ---------- Internals ----------
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
        geometry_type, coordinates = self._extract_geometry(obj, obj_index)

        # Get tokens for this geometry type
        ref_start, ref_end, geom_start, geom_end = self.geometry_tokens[geometry_type]

        # Convert coordinates to raw integers
        coord_texts: List[str] = []
        for i, coord in enumerate(coordinates):
            int_coord = int(coord)
            if int_coord < 0:
                raise ValueError(f"Object {obj_index} coordinate {i} value {int_coord} must be non-negative")
            coord_texts.append(str(int_coord))

        coord_string = ", ".join(coord_texts)
        # Use empty string if description is missing (graceful handling)
        description = obj.get("desc", "")

        # Format complete token/numeric string (same structure)
        return (
            f"{ref_start}{description}{ref_end}{geom_start}[{coord_string}]{geom_end}"
        )

    def _extract_geometry(self, obj: Dict[str, Any], obj_index: int):
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
            geometry_type = "bbox_2d"
            coordinates = [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]
        else:
            available_keys = [k for k in obj.keys() if k not in ["desc", "category"]]
            raise ValueError(
                f"Object {obj_index} contains unsupported geometry type. "
                f"Expected one of: {list(self.geometry_tokens.keys())}. "
                f"Found geometry keys: {available_keys}. "
                f"Full object: {obj}"
            )
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError(f"Object {obj_index} has invalid coordinates list")
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, (int, float)):
                raise ValueError(
                    f"Object {obj_index} coordinate {i} must be numeric, got {type(coord)}: {coord}"
                )
        return geometry_type, coordinates

    def _get_ref_tokens(self) -> Any:
        # Any geometry type shares the same ref tokens; use bbox_2d as representative
        ref_start, ref_end, _, _ = self.geometry_tokens["bbox_2d"]
        return ref_start, ref_end

    def _get_geom_tokens(self, geometry_type: str) -> Any:
        _, _, geom_start, geom_end = self.geometry_tokens[geometry_type]
        return geom_start, geom_end
