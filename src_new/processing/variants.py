#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant handler registry for conversation building.

This registry provides pluggable handlers for the three core variants:
- dense_caption
- coords_to_desc
- desc_to_coords

Handlers return two callables:
- build_user_text(objects) -> Optional[str]
- build_assistant_text(objects) -> str
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from .geometry_text import format_geometry_for_user, format_object_ref
from src_new.types import ConversationVariant
from .coordinate_converter import CoordinateTokenConverter
from .templates import CONSTANTS


class VariantHandler(Protocol):
    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[Union[str, Dict[str, Any]]]:
        ...

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        ...


class DenseCaptionHandler:
    def __init__(self, converter: CoordinateTokenConverter) -> None:
        self.converter = converter

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        return None  # image-only user

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        return self.converter.convert_objects_to_tokens(objects)


class CoordToDescHandler:
    def __init__(self, converter: CoordinateTokenConverter) -> None:
        self.converter = converter
        self._header = CONSTANTS.get("COORD_TO_DESC_USER_PROMPT", "请描述以下坐标中的物体")

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        # Use converter to format geometry for user when in plain mode
        try:
            fmt = []
            for o in objects:
                if hasattr(self.converter, "format_geometry_for_user"):
                    fmt.append(self.converter.format_geometry_for_user(o))
                else:
                    fmt.append(format_geometry_for_user(o))
            body = "\n".join(fmt)
            return f"{self._header}\n{body}" if body else self._header
        except Exception:
            lines = [format_geometry_for_user(o) for o in objects]
            body = "\n".join(lines)
            return f"{self._header}\n{body}" if body else self._header

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        res = self.converter.convert_objects_to_desc_only(objects)
        return res["text"] if isinstance(res, dict) else res


class DescToCoordHandler:
    def __init__(self, converter: CoordinateTokenConverter) -> None:
        self.converter = converter
        self._header = CONSTANTS.get("DESC_TO_COORD_USER_PROMPT", "请返回以下描述的物体的坐标")

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        try:
            fmt = []
            for o in objects:
                desc = o.get("desc", "")
                if hasattr(self.converter, "format_object_ref_for_user"):
                    fmt.append(self.converter.format_object_ref_for_user(desc))
                else:
                    fmt.append(format_object_ref(desc))
            body = "\n".join(fmt)
            return f"{self._header}\n{body}" if body else self._header
        except Exception:
            lines = [format_object_ref(o.get("desc", "")) for o in objects]
            body = "\n".join(lines)
            return f"{self._header}\n{body}" if body else self._header

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        res = self.converter.convert_objects_to_geometry_only(objects)
        return res["text"] if isinstance(res, dict) else res


class WrapperReconstructionHandler:
    def __init__(self, converter: CoordinateTokenConverter) -> None:
        self.converter = converter
        self._header = CONSTANTS.get(
            "WRAPPER_RECON_USER_PROMPT",
            "请忽略图像，仅根据文本重新输出标准包裹格式",
        )
        self._prefix = CONSTANTS.get("WRAPPER_RECON_OBJECT_PREFIX", "对象")

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not objects:
            raise ValueError("Wrapper reconstruction variant requires non-empty objects list")
        lines: List[str] = []
        for idx, obj in enumerate(objects, start=1):
            geometry_type, coordinates = self._extract_geometry(obj, idx - 1)
            desc = obj.get("desc", "")
            coord_text = ", ".join(str(int(v)) for v in coordinates)
            lines.append(
                f"{self._prefix}{idx}: 描述: {desc}\n   几何: {geometry_type} [{coord_text}]"
            )
        body = "\n".join(lines)
        text = f"{self._header}\n\n{body}" if body else self._header
        return {"text": text, "include_image": False}

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        return self.converter.convert_objects_to_tokens(objects)

    @staticmethod
    def _extract_geometry(obj: Dict[str, Any], obj_index: int) -> Tuple[str, List[int]]:
        if "bbox_2d" in obj:
            coords = obj["bbox_2d"]
            gtype = "bbox_2d"
        elif "quad" in obj:
            coords = obj["quad"]
            gtype = "quad"
        elif "line" in obj:
            coords = obj["line"]
            gtype = "line"
        else:
            raise ValueError(
                f"Object {obj_index} missing supported geometry keys ('bbox_2d', 'quad', 'line'): {obj}"
            )
        if not isinstance(coords, list) or not coords:
            raise ValueError(f"Object {obj_index} has invalid coordinates list: {coords}")
        return gtype, [int(v) for v in coords]


class VariantRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[str, VariantHandler] = {}

    def register(self, key: str, handler: VariantHandler) -> None:
        k = str(getattr(key, "value", key)).strip().lower()
        if not k:
            raise ValueError("Variant key cannot be empty")
        self._handlers[k] = handler

    def get(self, key: str) -> VariantHandler:
        k = str(getattr(key, "value", key)).strip().lower()
        if k not in self._handlers:
            raise ValueError(f"Unsupported variant: {key}")
        return self._handlers[k]



def create_default_variant_registry(converter: CoordinateTokenConverter) -> VariantRegistry:
    reg = VariantRegistry()
    reg.register(ConversationVariant.DENSE_CAPTION, DenseCaptionHandler(converter))
    reg.register(ConversationVariant.COORDS_TO_DESC, CoordToDescHandler(converter))
    reg.register(ConversationVariant.DESC_TO_COORDS, DescToCoordHandler(converter))
    reg.register(ConversationVariant.WRAPPER_RECONSTRUCTION, WrapperReconstructionHandler(converter))
    # Note: 'summary' variant is handled directly by ConversationProcessor using precomputed sample['summary']
    return reg
