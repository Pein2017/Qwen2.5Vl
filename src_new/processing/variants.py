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
import json

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


class TextOnlyHandler:
    def __init__(self, converter: CoordinateTokenConverter) -> None:
        self.converter = converter
        self._header = CONSTANTS.get(
            "TEXT_ONLY_USER_PROMPT",
            "Rewrite the JSON objects using dense wrapper format",
        )

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not objects:
            raise ValueError("Wrapper reconstruction variant requires non-empty objects list")
        lines: List[str] = []
        for idx, obj in enumerate(objects, start=1):
            # The model receives raw object JSON to rebuild wrappers exactly as dense captioning.
            raw_json = json.dumps(obj, ensure_ascii=False, separators=(",", ": "))
            lines.append(f"Object {idx}: {raw_json}")
        body = "\n".join(lines)
        text = f"{self._header}\n\n{body}" if body else self._header
        return {"text": text, "include_image": True}

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        return self.converter.convert_objects_to_tokens(objects)

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
    reg.register(ConversationVariant.TEXT_ONLY, TextOnlyHandler(converter))
    # Note: 'summary' variant is handled directly by ConversationProcessor using precomputed sample['summary']
    return reg
