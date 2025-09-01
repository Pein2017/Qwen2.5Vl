#!/usr/bin/env python3
"""
Centralized constants for the data conversion pipeline.

Contains the canonical set of object types and the default label hierarchy
fallback used by the unified processor in Chinese-only mode.
"""

from typing import Dict, List, Set

# Canonical set of supported object types
OBJECT_TYPES: Set[str] = {
    "bbu",
    "bbu_shield",
    "label",
    "fiber",
    "wire",
    "connect_point",
}

# Default hierarchy matching the v2 data structure (Chinese-only mode)
DEFAULT_LABEL_HIERARCHY: Dict[str, List[str]] = {
    "螺丝、光纤插头": ["BBU安装螺丝", "BBU端光纤插头"],
    "标签": [],
    "BBU设备": ["华为"],
    "光纤": [],
    "电线": [],
    "挡风板": ["华为"],
}
