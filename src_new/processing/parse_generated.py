"""Utilities for parsing generated dense-caption outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src_new.processing.special_tokens import GEOMETRY_TOKENS
from src_new.utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger("processing.parse_generated")


def _sanitize_description(desc: object) -> str:
    """Normalize description strings emitted around geometry wrappers."""

    if not isinstance(desc, str):
        return ""

    cleaned = desc.strip()

    # If upstream output duplicated object_ref_start, keep suffix after last occurrence.
    for token in ("<|object_ref_start|>", "<|obj_ref_start|>"):
        if token in cleaned:
            cleaned = cleaned.split(token)[-1]

    # Truncate at first closing wrapper to avoid bleed from subsequent blocks.
    for token in ("<|object_ref_end|>", "<|obj_ref_end|>"):
        if token in cleaned:
            cleaned = cleaned.split(token)[0]

    return cleaned.strip()


def _extract_raw_numbers(coords_section: str) -> List[int]:
    number_pattern = r"-?\d+"
    matches = re.findall(number_pattern, coords_section.strip())
    if not matches:
        raise ValueError("No numbers found in coordinates section")
    try:
        return [int(m) for m in matches]
    except ValueError as exc:
        raise ValueError(f"Failed to convert numbers to integers: {exc}") from exc


def _extract_coordinate_tokens(coords_section: str) -> List[int]:
    pattern = r"<\|coord_(\d+)\|>"
    matches = re.findall(pattern, coords_section)
    if not matches:
        raise ValueError("No coordinate tokens found in coordinates section")
    return [int(m) for m in matches]


def deduplicate_objects(objects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for obj in objects:
        key: Optional[Tuple[Any, ...]] = None
        if "bbox_2d" in obj:
            key = ("bbox_2d", tuple(obj["bbox_2d"]), obj.get("desc", ""))
        elif "quad" in obj:
            key = ("quad", tuple(obj["quad"]), obj.get("desc", ""))
        elif "line" in obj:
            key = ("line", tuple(obj["line"]), obj.get("desc", ""))
        if key is None or key in seen:
            continue
        seen.add(key)
        unique.append(obj)
        if len(unique) >= 50:
            break
    return unique


def parse_coordinate_response(response: str) -> List[Dict[str, Any]]:
    """Parse coordinate-token responses (strict training format)."""

    geometry_patterns = {
        "bbox": re.compile(
            r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|box_start\|>\[(.*?)\]<\|box_end\|>"
        ),
        "quad": re.compile(
            r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|quad_start\|>\[(.*?)\]<\|quad_end\|>"
        ),
        "line": re.compile(
            r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|line_start\|>\[(.*?)\]<\|line_end\|>"
        ),
    }

    objects: List[Dict[str, Any]] = []
    consumed_spans: List[Tuple[int, int]] = []
    found_any = False

    for geom_type, pattern in geometry_patterns.items():
        for match in pattern.finditer(response):
            span = match.span()
            if any(not (span[1] <= s or span[0] >= e) for s, e in consumed_spans):
                raise ValueError("Overlapping geometry spans detected in response")

            desc, coords_section = match.group(1), match.group(2)
            try:
                coords = _extract_coordinate_tokens(coords_section)
            except ValueError as exc:
                logger.warning("Failed to extract coordinate tokens: %s", exc)
                continue

            clean_desc = _sanitize_description(desc)
            if not clean_desc:
                logger.warning("Discarding geometry block with empty description after sanitization")
                continue

            if geom_type == "bbox":
                if len(coords) != 4:
                    logger.warning("Invalid bbox length: expected 4, got %d", len(coords))
                    continue
                objects.append({"bbox_2d": coords, "desc": clean_desc})
            elif geom_type == "quad":
                if len(coords) != 8:
                    logger.warning("Invalid quad length: expected 8, got %d", len(coords))
                    continue
                objects.append({"quad": coords, "desc": clean_desc})
            else:  # line
                if len(coords) < 4 or len(coords) % 2 != 0:
                    logger.warning("Invalid line length: got %d", len(coords))
                    continue
                objects.append({"line": coords, "desc": clean_desc})

            consumed_spans.append(span)
            found_any = True

    if not found_any or not objects:
        return []
    return deduplicate_objects(objects)


def parse_geometry_response(response: str, *, tolerant: bool = False) -> List[Dict[str, Any]]:
    """Parse geometry blocks with raw integer coordinates."""

    if tolerant:
        patterns = {
            "bbox": re.compile(
                r"(?:<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|>)?\s*<\|box_start\|>\s*\[(.*?)\s*(?:\]|\))?\s*<\|box_end\|>",
                re.DOTALL,
            ),
            "quad": re.compile(
                r"(?:<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|>)?\s*<\|quad_start\|>\s*\[(.*?)\s*(?:\]|\))?\s*<\|quad_end\|>",
                re.DOTALL,
            ),
            "line": re.compile(
                r"(?:<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|>)?\s*<\|line_start\|>\s*\[(.*?)\s*(?:\]|\))?\s*<\|line_end\|>",
                re.DOTALL,
            ),
        }
    else:
        patterns = {
            "bbox": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|box_start\|>\[(.*?)\]<\|box_end\|>",
                re.DOTALL,
            ),
            "quad": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|quad_start\|>\[(.*?)\]<\|quad_end\|>",
                re.DOTALL,
            ),
            "line": re.compile(
                r"<\|(?:object_ref_start|obj_ref_start)\|>(.*?)<\|(?:object_ref_end|obj_ref_end)\|><\|line_start\|>\[(.*?)\]<\|line_end\|>",
                re.DOTALL,
            ),
        }

    objects: List[Dict[str, Any]] = []

    for geom_type, pattern in patterns.items():
        for match in pattern.finditer(response):
            desc = match.group(1) if match.lastindex and match.lastindex >= 1 else None
            coords_section = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            try:
                coords = _extract_raw_numbers(coords_section)
            except Exception:
                continue

            clean_desc = _sanitize_description(desc)

            if geom_type == "bbox":
                if len(coords) < 4:
                    continue
                objects.append({"bbox_2d": coords[:4], "desc": clean_desc})
            elif geom_type == "quad":
                if len(coords) < 8:
                    continue
                objects.append({"quad": coords[:8], "desc": clean_desc})
            else:
                if len(coords) < 4:
                    continue
                if len(coords) % 2 == 1:
                    coords = coords[:-1]
                objects.append({"line": coords, "desc": clean_desc})

    return deduplicate_objects(objects)


def convert_objects_to_vis(objects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert parsed objects to visualization schema (label + geometry)."""

    vis_objects: List[Dict[str, Any]] = []
    for obj in objects:
        label = obj.get("desc") or obj.get("label") or ""
        vis_obj: Dict[str, Any] = {"label": label}
        for key in GEOMETRY_TOKENS:
            if key in obj and isinstance(obj[key], list):
                vis_obj[key] = [int(v) for v in obj[key]]
                break
        if any(g in vis_obj for g in GEOMETRY_TOKENS):
            vis_objects.append(vis_obj)
    return vis_objects


def convert_response_to_vis_json(
    response: str,
    *,
    coordinate_mode: bool,
    tolerant_geometry: bool = False,
) -> str:
    """Parse response and dump a visualization-friendly JSON list."""

    if coordinate_mode:
        objects = parse_coordinate_response(response)
    else:
        objects = parse_geometry_response(response, tolerant=tolerant_geometry)
    vis_objects = convert_objects_to_vis(objects)
    try:
        return json.dumps(vis_objects, ensure_ascii=False)
    except Exception:
        return response


__all__ = [
    "convert_objects_to_vis",
    "convert_response_to_vis_json",
    "deduplicate_objects",
    "parse_coordinate_response",
    "parse_geometry_response",
]
