from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from src_new_json.augmentation.object_registry import resolve_object_type
from src_new_json.augmentation.standardize import validate_sample_after_transform
from src_new_json.config.augmentation_config import CriteriaConfig


try:
    from src_new_json.utils.rank_aware_logging import get_rank_aware_logger as _get_logger

    _logger = _get_logger(__name__)
except Exception:
    _logger = logging.getLogger(__name__)


def apply_occlusion_criterion(
    sample: Dict[str, Any], criteria: CriteriaConfig
) -> Dict[str, Any]:
    occ = criteria.occlusion
    if occ is None or not occ.enabled:
        return sample
    width = int(sample["width"])  # fail-fast
    height = int(sample["height"])  # fail-fast

    # Dynamic knobs derived from image size with safe lower-bounds
    short_edge = max(1, min(width, height))
    dyn_line_w = max(2, int(round(short_edge * 0.002)))
    line_w = max(occ.line_width_px, dyn_line_w)
    dyn_down = max(1, short_edge // 512)  # ~512 on short edge
    down = max(1, dyn_down)

    w_low = max(1, width // down)
    h_low = max(1, height // down)

    # Build per-object low-res masks and metadata
    # Also decide occluder participation based on type
    masks: List[
        Tuple[str, Image.Image, str, bool]
    ] = []  # (geom_type, mask, type_name, is_occluder)

    # Which types contribute as occluders
    OCCLUDER_TYPES = {"bbu_shield", "fiber", "wire", "connect_point"}

    for o in sample["objects"]:
        typ = resolve_object_type(o["desc"]) if (isinstance(o, dict) and ("desc" in o)) else "unknown"
        geom = "other"
        m = Image.new("L", (w_low, h_low), 0)
        if "quad" in o:
            q = o["quad"]
            pts = [(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])]
            hi = Image.new("L", (width, height), 0)
            ImageDraw.Draw(hi).polygon(pts, outline=255, fill=255)
            m = hi.resize((w_low, h_low), Image.NEAREST)
            geom = "area"
        elif "bbox_2d" in o:
            b = o["bbox_2d"]
            hi = Image.new("L", (width, height), 0)
            ImageDraw.Draw(hi).rectangle(
                [b[0], b[1], b[2], b[3]], outline=255, fill=255
            )
            m = hi.resize((w_low, h_low), Image.NEAREST)
            geom = "area"
        elif "line" in o:
            l = o["line"]
            pts = [(l[i], l[i + 1]) for i in range(0, len(l), 2)]
            hi = Image.new("L", (width, height), 0)
            ImageDraw.Draw(hi).line(pts, fill=255, width=line_w)
            m = hi.resize((w_low, h_low), Image.NEAREST)
            geom = "line"
        masks.append((geom, m, typ, typ in OCCLUDER_TYPES))

    # Compute occluder union (exclude non-occluders like label)
    union_base = Image.new("L", (w_low, h_low), 0)
    for _geom, m, typ, is_occ in masks:
        if is_occ:
            union_base = ImageChops.lighter(union_base, m)
    # Small dilation to be robust to thin gaps
    union = union_base.filter(ImageFilter.MaxFilter(size=3))

    # Tiny object ignore threshold (in low-res pixels)
    MIN_OBJ_AREA_LOWRES = 8

    objs = sample["objects"]
    updated: List[Dict[str, Any]] = []
    num_flagged = 0

    def _threshold_for_target(tname: str, geom: str) -> float:
        # Start from config defaults
        base = (
            occ.min_overlap_fraction_line
            if geom == "line"
            else occ.min_overlap_fraction_bbox
        )
        # Type-specific nudges
        if tname == "bbu_shield":
            return max(0.10, min(0.15, base))
        if tname == "connect_point":
            return max(0.15, min(0.20, base))
        return base

    for i, o in enumerate(objs):
        typ_i = resolve_object_type(o["desc"]) if (isinstance(o, dict) and ("desc" in o)) else "unknown"
        geom_i, mask_i, _typ_meta, _is_occ = masks[i]
        if geom_i not in ("area", "line"):
            updated.append(o)
            continue

        # Ignore tiny objects
        area_sum = sum(mask_i.getdata())
        if area_sum < MIN_OBJ_AREA_LOWRES:
            updated.append(o)
            continue

        inter = ImageChops.multiply(mask_i, union)
        inter_sum = sum(inter.getdata())
        frac = (inter_sum / area_sum) if area_sum > 0 else 0.0
        thresh = _threshold_for_target(typ_i, geom_i)

        if frac >= thresh:
            # monitoring only: do not mutate desc since occlusion tokens are stripped upstream
            updated.append(o)
            num_flagged += 1
        else:
            updated.append(o)

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            "occlusion_criterion: width=%d height=%d down=%d line_w=%d flagged=%d/%d",
            width,
            height,
            down,
            line_w,
            num_flagged,
            len(objs),
        )

    out = dict(sample)
    out["objects"] = updated
    return out


def validate_geometry_bounds(sample: Dict[str, Any]) -> None:
    validate_sample_after_transform(sample, int(sample["width"]), int(sample["height"]))
