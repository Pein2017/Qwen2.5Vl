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

from typing import Any, Dict, List, Optional, Tuple

from src_new.processing.special_tokens import GEOMETRY_TOKENS


class CoordinateTokenConverter:
    """
    Focused component that handles ONLY coordinate token conversion.

    This replaces the _format_objects_for_response method from templates.py
    and is the ONLY custom logic we need to preserve.
    """

    def __init__(
        self,
        max_coord_value: int,
        coordinate_tokens_enabled: bool,
        plain_text_mode_enabled: bool = False,
        format_mode: str | None = None,
    ):
        """
        Initialize coordinate token converter.

        Args:
            max_coord_value: Maximum coordinate value for clamping (required)
            coordinate_tokens_enabled: If False, emit raw numeric coordinates instead of <|coord_*|> tokens
            plain_text_mode_enabled: If True, emit JSON Lines without any wrapper tokens
            format_mode: One of {"special_tokens", "plain", "coord_tokens"}. If provided, overrides the two flags.

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
        if not isinstance(coordinate_tokens_enabled, bool):
            raise ValueError(
                f"coordinate_tokens_enabled must be a bool, got {type(coordinate_tokens_enabled)}: {coordinate_tokens_enabled!r}"
            )
        if not isinstance(plain_text_mode_enabled, bool):
            raise ValueError(
                f"plain_text_mode_enabled must be a bool, got {type(plain_text_mode_enabled)}: {plain_text_mode_enabled!r}"
            )
        if format_mode is not None and format_mode not in {"special_tokens", "plain", "coord_tokens"}:
            raise ValueError(f"Unsupported format_mode: {format_mode}")

        self.max_coord_value = max_coord_value
        self.coordinate_tokens_enabled = coordinate_tokens_enabled
        # Use centralized canonical geometry tokens
        self.geometry_tokens = GEOMETRY_TOKENS
        # Global rendering mode
        if format_mode is not None:
            self._mode = format_mode
        else:
            self._mode = "plain" if plain_text_mode_enabled else ("coord_tokens" if coordinate_tokens_enabled else "special_tokens")
        self.plain_text_mode_enabled = (self._mode == "plain")

    def _use_coord_tokens(self) -> bool:
        """Return True iff coordinates should be rendered as <|coord_*|> tokens.

        Coord tokens are only used in wrapper mode when mode == 'coord_tokens'.
        """
        return self._mode == "coord_tokens" and not self.plain_text_mode_enabled

    # ---------- Public API for assistant rendering ----------
    def convert_objects_to_tokens(self, objects: List[Dict[str, Any]]):
        """
        Convert objects list to assistant text.

        Wrapper mode: returns a string with wrappers.
        Plain-text mode: returns a dict with keys:
            {"text": str, "group_char_spans": {"caption": [(s,e),...], "grounding": [(s,e),...]}}
        where spans are relative to the assistant content (start=0) and end-exclusive.

        Raises on empty/invalid inputs.
        """
        if not objects:
            raise ValueError(
                "Empty objects list encountered. This indicates a data preprocessing failure - "
                "empty object lists should have been filtered out earlier in the pipeline."
            )
        if not isinstance(objects, list):
            raise ValueError(f"objects must be a list, got {type(objects)}")

        if not self.plain_text_mode_enabled:
            token_strings: List[str] = []
            for i, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    raise ValueError(f"Object {i} must be a dict, got {type(obj)}")
                token_string = self._convert_single_object(obj, i)
                if token_string:
                    token_strings.append(token_string)
            if not token_strings:
                raise ValueError("No valid objects found after conversion")
            return "\n".join(token_strings)

        # Plain text JSON mode
        out_lines: List[str] = []
        caption_spans: List[Tuple[int, int]] = []
        grounding_spans: List[Tuple[int, int]] = []
        cursor = 0  # relative to assistant content
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"Object {i} must be a dict, got {type(obj)}")
            geom_type, coords = self._extract_geometry(obj, i)
            desc_val = obj.get("desc", "")
            line_text, cap_sp, grd_sp = self._build_json_line_with_spans(
                geom_type, coords, desc_val
            )
            out_lines.append(line_text)
            # Offset spans by current cursor
            for s, e in cap_sp:
                caption_spans.append((cursor + s, cursor + e))
            for s, e in grd_sp:
                grounding_spans.append((cursor + s, cursor + e))
            cursor += len(line_text) + 1  # +1 for the newline that will join lines
        assistant_text = "\n".join(out_lines)
        return {"text": assistant_text, "group_char_spans": {"caption": caption_spans, "grounding": grounding_spans}}

    def convert_objects_to_desc_only(self, objects: List[Dict[str, Any]]):
        """Assistant text: description-only.

        Wrapper mode: returns lines like <|object_ref_start|>desc<|object_ref_end|>.
        Plain-text mode: returns dict with JSON lines {"desc":"..."} and caption spans only.
        """
        if not objects:
            raise ValueError("Empty objects list encountered in convert_objects_to_desc_only")
        if not self.plain_text_mode_enabled:
            out_lines: List[str] = []
            ref_start, ref_end = self._get_ref_tokens()
            for i, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    raise ValueError(f"Object {i} must be a dict, got {type(obj)}")
                description = obj.get("desc", "")
                out_lines.append(f"{ref_start}{description}{ref_end}")
            return "\n".join(out_lines)

        # Plain text: {"desc":"..."}
        from json import dumps as _jd

        plain_lines: List[str] = []
        caption_spans: List[Tuple[int, int]] = []
        cursor = 0
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"Object {i} must be a dict, got {type(obj)}")
            desc_val = obj.get("desc", "")
            desc_json = _jd(str(desc_val), ensure_ascii=False)
            line = '{"desc":' + desc_json + '}'
            # desc_json includes quotes; capture inside-quotes span only
            start = line.index('"desc":') + len('"desc":') + 1
            end = start + len(desc_json) - 2
            plain_lines.append(line)
            caption_spans.append((cursor + start, cursor + end))
            cursor += len(line) + 1
        assistant_text = "\n".join(plain_lines)
        return {"text": assistant_text, "group_char_spans": {"caption": caption_spans, "grounding": []}}

    def convert_objects_to_geometry_only(self, objects: List[Dict[str, Any]]):
        """Assistant text: geometry-only.

        Wrapper mode: returns <|geom_start|>[coords]<|geom_end|> lines.
        Plain-text mode: returns dict with JSON lines {"<geom>":[...]} and grounding spans only.
        """
        if not objects:
            raise ValueError("Empty objects list encountered in convert_objects_to_geometry_only")
        if not self.plain_text_mode_enabled:
            out_lines: List[str] = []
            for i, obj in enumerate(objects):
                geometry_type, coordinates = self._extract_geometry(obj, i)
                geom_start, geom_end = self._get_geom_tokens(geometry_type)
                coord_texts: List[str] = []
                for j, coord in enumerate(coordinates):
                    int_coord = int(coord)
                    if int_coord < 0 or int_coord > self.max_coord_value:
                        raise ValueError(
                            f"Object {i} coordinate {j} value {int_coord} out of valid range [0, {self.max_coord_value}]"
                        )
                    if self._use_coord_tokens():
                        coord_texts.append(f"<|coord_{int_coord}|>")
                    else:
                        coord_texts.append(str(int_coord))
                coord_string = ", ".join(coord_texts)
                out_lines.append(f"{geom_start}[{coord_string}]{geom_end}")
            return "\n".join(out_lines)

        # Plain text JSON: only geometry
        plain_lines: List[str] = []
        grounding_spans: List[Tuple[int, int]] = []
        cursor = 0
        for i, obj in enumerate(objects):
            geometry_type, coordinates = self._extract_geometry(obj, i)
            line, _cap, grd = self._build_json_line_with_spans(geometry_type, coordinates, None, include_desc=False)
            plain_lines.append(line)
            for s, e in grd:
                grounding_spans.append((cursor + s, cursor + e))
            cursor += len(line) + 1
        assistant_text = "\n".join(plain_lines)
        return {"text": assistant_text, "group_char_spans": {"caption": [], "grounding": grounding_spans}}

    # ---------- Public API for user rendering ----------
    def format_geometry_for_user(self, obj: Dict[str, object]) -> str:
        """Format a single object's geometry for inclusion in a user message."""
        if self.plain_text_mode_enabled:
            geom_type, coords = self._extract_geometry(obj, 0)
            values = ",".join(str(int(v)) for v in coords)
            return '{"' + geom_type + '":[' + values + ']}'
        # Wrapper mode falls back to legacy helpers
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
        """Format a description-only object for inclusion in a user message."""
        if self.plain_text_mode_enabled:
            from json import dumps as _jd

            return '{"desc":' + _jd(str(desc), ensure_ascii=False) + '}'
        safe_desc = desc if isinstance(desc, str) else str(desc)
        return f"<|object_ref_start|>{safe_desc}<|object_ref_end|>"

    # ---------- Internals ----------
    def _build_json_line_with_spans(
        self, geom_type: str, coordinates: List[Any], desc_val: Optional[str], include_desc: bool = True
    ) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Build a single JSON line (no spaces) and compute relative char spans.

        Returns: (line_text, caption_spans, grounding_spans)
        Spans are relative to the start of this line (not including trailing newline).
        - caption spans cover the inside of desc value quotes (without quotes) when include_desc=True
        - grounding spans cover the geometry key string and each coordinate value (exclude [ ], ,)
        """
        from json import dumps as _jd

        # geometry key
        geom_key = geom_type
        # coordinate values
        val_segments: List[str] = []
        for coord in coordinates:
            int_coord = int(coord)
            if int_coord < 0 or int_coord > self.max_coord_value:
                raise ValueError(
                    f"Coordinate value {int_coord} out of valid range [0, {self.max_coord_value}]"
                )
            # Plain JSON mode never uses coord tokens; always numeric
            val_segments.append(str(int_coord))
        values_str = ",".join(val_segments)
        # Assemble progressively to obtain exact offsets
        prefix = '{"' + geom_key + '":['
        suffix = ']'
        if include_desc:
            desc_json = _jd(str(desc_val if desc_val is not None else ""), ensure_ascii=False)
            middle = '],"desc":' + desc_json + '}'
        else:
            middle = ']}'
        line = prefix + values_str + middle

        # Grounding spans: geometry key string and each value segment
        grounding_spans: List[Tuple[int, int]] = []
        key_start = 1  # after leading '{'
        key_token = '"' + geom_key + '"'
        grounding_spans.append((key_start, key_start + len(key_token)))
        # coordinate token spans
        base = len(prefix)
        pos = base
        for seg in val_segments:
            grounding_spans.append((pos, pos + len(seg)))
            pos += len(seg) + 1  # +1 for comma except last; harmless to overshoot, we'll drop last -1 below
        # Fix the last added +1: if any segments exist, reduce last span end increment compensation
        if val_segments:
            # remove the final comma accounting
            pass  # spans already recorded precisely; commas are excluded by construction

        # Caption spans: inside desc quotes
        caption_spans: List[Tuple[int, int]] = []
        if include_desc:
            # desc_json includes surrounding quotes
            desc_key_pos = line.index('"desc":')
            val_start = desc_key_pos + len('"desc":')
            # value starts with a quote
            if val_start >= len(line) or line[val_start] != '"':
                # Defensive guard; fail fast to avoid silent misalignment
                raise ValueError("Plain-text JSON builder: desc value must be a quoted JSON string")
            cap_start = val_start + 1
            # find closing quote: use length of desc_json
            desc_json = _jd(str(desc_val if desc_val is not None else ""), ensure_ascii=False)
            cap_end = val_start + len(desc_json) - 1
            if cap_end < cap_start:
                cap_end = cap_start
            caption_spans.append((cap_start, cap_end))
        return line, caption_spans, grounding_spans

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

        # Convert coordinates either to tokens or to raw integers
        coord_texts: List[str] = []
        for i, coord in enumerate(coordinates):
            int_coord = int(coord)
            if int_coord < 0 or int_coord > self.max_coord_value:
                raise ValueError(
                    f"Object {obj_index} coordinate {i} value {int_coord} out of valid range [0, {self.max_coord_value}]"
                )
            if self._use_coord_tokens():
                coord_texts.append(f"<|coord_{int_coord}|>")
            else:
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
