#!/usr/bin/env python3
"""
Sanitizers for text/description cleanup in the data conversion pipeline.
"""

from typing import Optional


def strip_occlusion_tokens(desc: Optional[str]) -> Optional[str]:
    """
    Remove tokens containing '遮挡' per level/token while preserving separators.

    - Split by '/' into levels, by ',' within levels
    - Drop any token containing '遮挡'
    - Rejoin; drop empty levels
    """
    if not desc or not isinstance(desc, str):
        return desc
    levels = [lvl.strip() for lvl in desc.split("/")]
    kept_levels = []
    for lvl in levels:
        if not lvl:
            continue
        tokens = [t.strip() for t in lvl.split(",")]
        kept_tokens = [t for t in tokens if t and ("遮挡" not in t)]
        if kept_tokens:
            kept_levels.append(",".join(kept_tokens))
    return "/".join(kept_levels)
