from __future__ import annotations

from typing import Optional


# Map of Chinese label prefixes to canonical types
_PREFIX_TO_TYPE = {
    "BBU设备": "bbu",
    "挡风板": "bbu_shield",
    "螺丝、光纤插头": "connect_point",
    "标签": "label",
    "光纤": "fiber",
    "电线": "wire",
}


def resolve_object_type(desc: Optional[str]) -> Optional[str]:
    """Resolve object type from the Chinese prefix of desc.

    Returns a canonical type string or None if unknown.
    """
    if not isinstance(desc, str) or not desc:
        return None
    for prefix, typ in _PREFIX_TO_TYPE.items():
        if desc.startswith(prefix):
            return typ
    return None
