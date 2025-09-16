#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


JsonFormat = Literal["compact", "expanded"]


@dataclass(frozen=True)
class JsonSchema:
    """Schema configuration for JSON geometry rendering and parsing.

    - format: 'compact' uses a unified key 'type_coordinate' and 'label'.
      - type_coordinate = ["box"|"quadrilateral"|"line", [x,y], ...]
    - format: 'expanded' (default) uses separate explicit keys:
      - box_points_key, quadrilateral_points_key, line_points_key, and 'label'.
    """

    # Selection between unified and legacy shapes
    format: JsonFormat = "expanded"

    # Unified (compact) keys
    label_key: str = "label"
    type_coordinate_key: str = "type_coordinate"

    # Expanded (default) keys
    box_points_key: str = "box_points"
    quadrilateral_points_key: str = "quadrilateral_points"
    line_points_key: str = "line_points"

    def is_compact(self) -> bool:
        return self.format == "compact"

    def is_expanded(self) -> bool:
        return self.format == "expanded"


__all__ = ["JsonSchema", "JsonFormat"]
