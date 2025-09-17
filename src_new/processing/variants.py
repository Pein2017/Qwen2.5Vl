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
        # Allowed canonical tokens (from hierarchical_attribute_mapping.json)
        BBU_REQ_NEED = "机柜空间充足需要安装"
        BBU_REQ_NONEED = "无需安装"
        BBU_CONF_OK = "这个BBU设备按要求配备了挡风板"
        BBU_CONF_BAD = "这个BBU设备未按要求配备挡风板"

        SHIELD_DIR_OK = "安装方向正确"
        SHIELD_DIR_BAD = "安装方向错误"

        CP_COMPLY_OK = "符合要求"
        CP_COMPLY_BAD = "不符合要求"
        CP_ISSUES = {"未拧紧", "露铜", "复接", "生锈"}

        FIB_PROTECT_NONE = "无保护措施"
        FIB_PROTECT_HAVE = "有保护措施"
        FIB_PROTECT_DETAILS = {"蛇形管", "铠装", "同时有蛇形管和铠装"}
        FIB_BEND_OK = "弯曲半径合理"
        FIB_BEND_BAD = "弯曲半径不合理（弯曲半径<4cm或者成环）"

        WIRE_NEAT = "捆扎整齐"
        WIRE_MESS = "分布散乱"

        # Aggregated outputs (deduplicated, order by importance)
        out_tokens: List[str] = []
        seen: set[str] = set()

        # Local collectors
        cp_issues: List[str] = []
        cp_noncompliant = False
        fib_protection: Optional[str] = None
        fib_details: Optional[str] = None
        fib_bend: Optional[str] = None
        wire_org: Optional[str] = None
        bbu_req: Optional[str] = None
        bbu_conf: Optional[str] = None
        shield_dir: Optional[str] = None
        label_clear: Optional[bool] = None

        def split_commas(seg: str) -> List[str]:
            seg = seg.strip()
            if not seg:
                return []
            return [p.strip() for p in seg.replace("，", ",").split(",") if p.strip()]

        for o in objects:
            desc = str(o.get("desc", "")).strip()
            if not desc:
                continue
            parts = [p.strip() for p in desc.split("/")]
            if not parts:
                continue
            kind = parts[0]

            # BBU设备: [kind, lvl1(brand,completeness,windshield_requirement), [windshield_conformity], [special_text]]
            if kind == "BBU设备" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                # Extract requirement (ignore brand/completeness)
                if BBU_REQ_NONEED in lvl1:
                    bbu_req = BBU_REQ_NONEED
                    # '无需安装' → ignore any downstream conformity and remarks
                    bbu_conf = None
                elif BBU_REQ_NEED in lvl1:
                    bbu_req = BBU_REQ_NEED
                    # Try conformity if present
                    if len(parts) >= 3:
                        conf = parts[2].strip()
                        if conf == BBU_CONF_OK:
                            bbu_conf = BBU_CONF_OK
                        elif conf == BBU_CONF_BAD:
                            bbu_conf = BBU_CONF_BAD
                        else:
                            # Missing or unknown → treat as non‑conformant conservatively
                            bbu_conf = BBU_CONF_BAD

            # 挡风板: [kind, lvl1(brand,completeness,obstruction,install_direction), [special_text]]
            elif kind == "挡风板" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if SHIELD_DIR_BAD in lvl1:
                    shield_dir = SHIELD_DIR_BAD
                elif SHIELD_DIR_OK in lvl1:
                    shield_dir = SHIELD_DIR_OK if shield_dir is None else shield_dir

            # 螺丝、光纤插头: [kind, lvl1(type,completeness,compliance), [specific_issues], [special_text]]
            elif kind == "螺丝、光纤插头" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if CP_COMPLY_BAD in lvl1:
                    cp_noncompliant = True
                    if len(parts) >= 3:
                        issues = split_commas(parts[2])
                        for it in issues:
                            if it in CP_ISSUES:
                                cp_issues.append(it)

            # 光纤: [kind, lvl1(obstruction,protection,bend_radius), [protection_details], [special_text]]
            elif kind == "光纤" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if FIB_PROTECT_NONE in lvl1:
                    fib_protection = FIB_PROTECT_NONE
                    fib_details = None
                elif FIB_PROTECT_HAVE in lvl1:
                    fib_protection = FIB_PROTECT_HAVE
                    if len(parts) >= 3:
                        det = parts[2].strip()
                        if det in FIB_PROTECT_DETAILS:
                            fib_details = det
                if (FIB_BEND_BAD in lvl1) or ("弯曲半径不合理(弯曲半径<4cm或者成环)" in lvl1):
                    fib_bend = FIB_BEND_BAD
                elif FIB_BEND_OK in lvl1 and fib_bend is None:
                    fib_bend = FIB_BEND_OK

            # 电线: [kind, lvl1(obstruction,organization), [special_text]]
            elif kind == "电线" and len(parts) >= 2:
                lvl1 = split_commas(parts[1])
                if WIRE_MESS in lvl1:
                    wire_org = WIRE_MESS
                elif WIRE_NEAT in lvl1 and wire_org is None:
                    wire_org = WIRE_NEAT

            # 标签: [kind, [text_content]] → clarity only
            elif kind == "标签":
                text = parts[1].strip() if len(parts) >= 2 else ""
                label_clear = bool(text)

        # Compose output by priority
        def add(tok: Optional[str]) -> None:
            if tok and tok not in seen:
                out_tokens.append(tok)
                seen.add(tok)

        # 1) BBU 挡风板需求/符合性（核心决策链）
        if bbu_req == BBU_REQ_NONEED:
            add(BBU_REQ_NONEED)
        elif bbu_req == BBU_REQ_NEED:
            add(BBU_REQ_NEED)
            add(bbu_conf or BBU_CONF_BAD)

        # 2) 挡风板安装方向（任务相关）
        add(shield_dir)

        # 3) 连接点合规与细项
        if cp_noncompliant:
            add(CP_COMPLY_BAD)
            if cp_issues:
                for it in sorted(set(cp_issues)):
                    add(it)
        # 4) 光纤保护/弯曲半径（与任务强相关）
        add(fib_protection)
        if fib_protection == FIB_PROTECT_HAVE:
            add(fib_details)
        add(fib_bend)

        # 5) 电线整齐度
        add(wire_org)

                    # 6) 标签（输出可以识别/无法识别）
        if label_clear is False:
            add("标签/无法识别")
        elif label_clear is True:
            add("标签/可以识别")

        # Fallback minimal positive phrasing when nothing extracted
        if not out_tokens:
            out_tokens = [WIRE_NEAT, FIB_BEND_OK, "标签/可以识别"]

        summary = "，".join(out_tokens)
        summary = summary.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
        # Keep a reasonable cap to ensure one-line brevity
        return summary

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
