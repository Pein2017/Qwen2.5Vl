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

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .geometry_text import format_geometry_for_user, format_object_ref
from .coordinate_converter import CoordinateTokenConverter
from .templates import CONSTANTS


class VariantHandler(Protocol):
    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
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
        lines = [format_geometry_for_user(o) for o in objects]
        body = "\n".join(lines)
        return f"{self._header}\n{body}" if body else self._header

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        return self.converter.convert_objects_to_desc_only(objects)


class DescToCoordHandler:
    def __init__(self, converter: CoordinateTokenConverter) -> None:
        self.converter = converter
        self._header = CONSTANTS.get("DESC_TO_COORD_USER_PROMPT", "请返回以下描述的物体的坐标")

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        lines = [format_object_ref(o.get("desc", "")) for o in objects]
        body = "\n".join(lines)
        return f"{self._header}\n{body}" if body else self._header

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        return self.converter.convert_objects_to_geometry_only(objects)


class VariantRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[str, VariantHandler] = {}

    def register(self, key: str, handler: VariantHandler) -> None:
        k = key.strip().lower()
        if not k:
            raise ValueError("Variant key cannot be empty")
        self._handlers[k] = handler

    def get(self, key: str) -> VariantHandler:
        k = key.strip().lower()
        if k not in self._handlers:
            raise ValueError(f"Unsupported variant: {key}")
        return self._handlers[k]


def create_default_variant_registry(converter: CoordinateTokenConverter) -> VariantRegistry:
    reg = VariantRegistry()
    reg.register("dense_caption", DenseCaptionHandler(converter))
    reg.register("coords_to_desc", CoordToDescHandler(converter))
    reg.register("desc_to_coords", DescToCoordHandler(converter))
    return reg
