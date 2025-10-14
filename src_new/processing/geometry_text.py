#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared geometry and user-text formatting helpers.

These helpers centralize how geometry and object references are rendered in
user prompts to ensure consistent formatting across dataset conversion and
conversation builders.
"""
from typing import Dict


def format_geometry_for_user(obj: Dict[str, object]) -> str:
    """Format a single object's geometry for inclusion in a user message.

    Supported keys (first match wins): "line", "bbox_2d", "quad".
    Values are expected to be sequences of numeric values.
    """
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


def format_object_ref(desc: str) -> str:
    """Wrap a natural-language description in object-ref tokens."""
    safe_desc = desc if isinstance(desc, str) else str(desc)
    return f"<|object_ref_start|>{safe_desc}<|object_ref_end|>"
