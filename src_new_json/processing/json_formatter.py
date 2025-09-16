#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from src_new_json.types.json_schema import JsonSchema


class JsonGeometryFormatter:
    def __init__(self, schema: JsonSchema | None = None) -> None:
        self._schema = schema or JsonSchema()

    # Public API used by variants
    def build_dense_caption(self, objects: List[Dict[str, Any]]) -> str:
        payload = [self._object_to_item(o) for o in objects]
        return self._pretty_dump(payload)

    def build_coords_to_desc_user(self, objects: List[Dict[str, Any]]) -> str:
        # Geometry only; omit label
        payload = [self._object_to_item(o, include_label=False) for o in objects]
        return self._pretty_dump(payload)

    def build_desc_to_coords_user(self, objects: List[Dict[str, Any]]) -> str:
        # Label only to drive model to return geometry
        items: List[Dict[str, Any]] = []
        for o in objects:
            if (not isinstance(o, dict)) or ("desc" not in o):
                raise ValueError("Object missing required 'desc' field for desc_to_coords variant")
            label = str(o["desc"]) if (o["desc"] is not None) else ""
            items.append({self._schema.label_key: label})
        return self._pretty_dump(items)

    def build_desc_only(self, objects: List[Dict[str, Any]]) -> str:
        # Assistant output: desc only (label field) with no geometry
        items: List[Dict[str, Any]] = []
        for o in objects:
            if (not isinstance(o, dict)) or ("desc" not in o):
                raise ValueError("Object missing required 'desc' field for desc_only variant")
            items.append({self._schema.label_key: str(o["desc"])})
        return self._pretty_dump(items)

    def build_coords_only(self, objects: List[Dict[str, Any]]) -> str:
        # Assistant output: geometry only with no label
        payload = [self._object_to_item(o, include_label=False) for o in objects]
        return self._pretty_dump(payload)

    # Backward-compatible alias used by conversation builders for assistant text
    def convert_objects_to_tokens(self, objects: List[Dict[str, Any]]) -> str:
        return self.build_dense_caption(objects)

    # Internals
    def _object_to_item(self, o: Dict[str, Any], include_label: bool = True) -> Dict[str, Any]:
        if self._schema.is_compact():
            ty, coords = self._extract_geometry(o)
            type_coordinate = [ty] + [[int(x), int(y)] for (x, y) in coords]
            item: Dict[str, Any] = {self._schema.type_coordinate_key: type_coordinate}
            if include_label:
                if (not isinstance(o, dict)) or ("desc" not in o):
                    raise ValueError("Object missing required 'desc' field when include_label=True")
                item[self._schema.label_key] = str(o["desc"]) if (o["desc"] is not None) else ""
            return item

        # Expanded (default): explicit *_points keys
        item: Dict[str, Any] = {}
        ty, coords = self._extract_geometry(o)
        key = (
            self._schema.box_points_key
            if ty == "box"
            else self._schema.quadrilateral_points_key
            if ty == "quadrilateral"
            else self._schema.line_points_key
        )
        item[key] = [[int(x), int(y)] for (x, y) in coords]
        if include_label:
            if (not isinstance(o, dict)) or ("desc" not in o):
                raise ValueError("Object missing required 'desc' field when include_label=True")
            item[self._schema.label_key] = str(o["desc"]) if (o["desc"] is not None) else ""
        return item

    def _extract_geometry(self, o: Dict[str, Any]) -> Tuple[str, List[Tuple[int, int]]]:
        # Input schema: bbox_2d | quad | line
        if "bbox_2d" in o:
            bb = o["bbox_2d"]
            if not (isinstance(bb, list) and len(bb) == 4):
                raise ValueError("bbox_2d must be a list of 4 integers [x1,y1,x2,y2]")
            x1, y1, x2, y2 = [int(v) for v in bb]
            if not (x1 < x2 and y1 < y2):
                raise ValueError(f"Invalid bbox_2d: {bb}")
            return "box", [(x1, y1), (x2, y2)]
        if "quad" in o:
            q = o["quad"]
            if not (isinstance(q, list) and len(q) == 8):
                raise ValueError("quad must be a list of 8 integers [x1,y1,...,x4,y4]")
            ints = [int(v) for v in q]
            coords = [(ints[0], ints[1]), (ints[2], ints[3]), (ints[4], ints[5]), (ints[6], ints[7])]
            return "quadrilateral", coords
        if "line" in o:
            l = o["line"]
            if not (isinstance(l, list) and len(l) >= 4 and len(l) % 2 == 0):
                raise ValueError("line must be a list of even-length integers with at least 2 points")
            ints = [int(v) for v in l]
            coords = [(ints[i], ints[i + 1]) for i in range(0, len(ints), 2)]
            return "line", coords
        raise ValueError("Object missing geometry: expected one of bbox_2d, quad, line")

    def _pretty_dump(self, payload: Any) -> str:
        # Compact JSON to avoid newlines for coordinate arrays
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = ["JsonGeometryFormatter"]
