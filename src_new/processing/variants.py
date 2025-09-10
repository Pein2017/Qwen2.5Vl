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
from src_new.types import ConversationVariant
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


class SummaryHandler:
    def __init__(self) -> None:
        self._header = CONSTANTS.get("SUMMARY_USER_PROMPT", "请只输出一行摘要：")

    def _extract_summary(self, objects: List[Dict[str, Any]]) -> str:
        bad_screw: List[str] = []
        bad_fiber = False
        bad_wire = False
        shield_need_bad = False
        shield_dir_bad = False
        bbu_partial = False
        label_clear: Optional[bool] = None
        remarks_pool: List[str] = []

        def split_commas(seg: str) -> List[str]:
            seg = seg.strip()
            if not seg:
                return []
            # support both Chinese '，' and ASCII ','
            parts = [p.strip() for p in seg.replace("，", ",").split(",")]
            return [p for p in parts if p]

        for o in objects:
            desc = str(o.get("desc", "")).strip()
            if not desc:
                continue
            parts = [p.strip() for p in desc.split("/")]
            if not parts:
                continue
            kind = parts[0]

            # BBU设备: [kind, lvl1(brand,visibility,windshield_requirement), [windshield_conformity], [remarks]]
            if kind == "BBU设备" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if "只显示部分" in lvl1:
                    bbu_partial = True
                need_wshield = any(x.endswith("需要安装") for x in lvl1)
                # Determine conformity segment index if required
                base_count = 2 + (1 if need_wshield else 0)
                if need_wshield:
                    # if conformity present and indicates missing/nonconformant, mark violation
                    if len(parts) >= 3:
                        conformity = parts[2].strip()
                        if conformity and ("未按要求配备" in conformity):
                            shield_need_bad = True
                    else:
                        # required but no conformity segment present → treat as missing
                        shield_need_bad = True
                # Remarks if extra segment exists
                if len(parts) > base_count:
                    candidate = parts[-1].strip()
                    if candidate:
                        remarks_pool.append(candidate)

            # 挡风板: [kind, lvl1(brand,visibility,obstruction,install_direction), [remarks]]
            elif kind == "挡风板" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if any(x.endswith("安装方向错误") or x == "安装方向错误" for x in lvl1):
                    shield_dir_bad = True
                if len(parts) > 2:
                    candidate = parts[-1].strip()
                    if candidate:
                        remarks_pool.append(candidate)

            # 螺丝、光纤插头: [kind, lvl1(type,visibility,compliance), [specific_issues], [remarks]]
            elif kind == "螺丝、光纤插头" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                noncompliant = any(x == "不符合要求" for x in lvl1)
                base_count = 2 + (1 if noncompliant else 0)
                if noncompliant and len(parts) >= 3:
                    issues_seg = parts[2].strip() if len(parts) >= 3 else ""
                    issues = split_commas(issues_seg)
                    for it in issues:
                        if it in ("未拧紧", "露铜", "复接", "生锈"):
                            bad_screw.append(it)
                # remarks
                if len(parts) > base_count:
                    candidate = parts[-1].strip()
                    if candidate:
                        remarks_pool.append(candidate)

            # 光纤: [kind, lvl1(obstruction,protection,bend_radius), [protection_details], [remarks]]
            elif kind == "光纤" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if any("弯曲半径不合理" in x for x in lvl1):
                    bad_fiber = True
                protected = any(x == "有保护措施" or x.startswith("有保护措施") for x in lvl1)
                base_count = 2 + (1 if protected else 0)
                if len(parts) > base_count:
                    candidate = parts[-1].strip()
                    if candidate:
                        remarks_pool.append(candidate)

            # 电线: [kind, lvl1(obstruction,organization), [remarks]]
            elif kind == "电线" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if any(x == "分布散乱" for x in lvl1):
                    bad_wire = True
                if len(parts) > 2:
                    candidate = parts[-1].strip()
                    if candidate:
                        remarks_pool.append(candidate)

            # 标签: [kind, [text]]
            elif kind == "标签":
                text = parts[1].strip() if len(parts) >= 2 else ""
                label_clear = bool(text)

        parts_out: List[str] = []
        if bad_screw:
            parts_out.append("、".join(sorted(set(bad_screw))))
        if bad_fiber:
            parts_out.append("光纤弯曲半径不合理")
        if bad_wire:
            parts_out.append("电线分布散乱")
        if shield_need_bad:
            parts_out.append("需安装挡风板未按要求配备")
        if shield_dir_bad:
            parts_out.append("挡风板安装方向错误")
        if bbu_partial:
            parts_out.append("BBU只显示部分")
        if label_clear is not None:
            parts_out.append("标签清晰" if label_clear else "标签不清晰")

        if not parts_out:
            parts_out = ["关键项正常", "光纤弯曲合理", "电线捆扎整齐", "标签清晰"]

        if remarks_pool:
            # Prefer the shortest remark to preserve brevity
            parts_out.append(min(remarks_pool, key=len))

        summary = "，".join(parts_out)
        summary = summary.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
        return summary[:40]

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        return self._header

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        return self._extract_summary(objects)


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
    # New summary variant: image -> one-line summary (assistant-only; user carries fixed header)
    reg.register(ConversationVariant.SUMMARY, SummaryHandler())
    return reg
