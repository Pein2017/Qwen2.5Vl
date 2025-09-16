#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant handler registry for conversation building (JSON-first, four variants):
- dense_caption
- coords_to_desc
- desc_to_coords
- summary

Handlers return two callables:
- build_user_text(objects) -> Optional[str]
- build_assistant_text(objects) -> str

The JSON output schema is governed by JsonGeometryFormatter (compact by default).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from src_new_json.types import ConversationVariant
from .templates import CONSTANTS
from .json_formatter import JsonGeometryFormatter
import re


class VariantHandler(Protocol):
    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        ...

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        ...


class DenseCaptionHandler:
    def __init__(self, formatter: JsonGeometryFormatter) -> None:
        self.formatter = formatter

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        return None  # image-only user

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        # Dense caption is the only variant that returns geometry + desc together
        return self.formatter.build_dense_caption(objects)


class CoordToDescHandler:
    def __init__(self, formatter: JsonGeometryFormatter) -> None:
        self.formatter = formatter
        self._header = CONSTANTS["COORD_TO_DESC_USER_PROMPT"] if "COORD_TO_DESC_USER_PROMPT" in CONSTANTS else "请描述以下坐标中的物体"

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        # User: geometry only
        body = self.formatter.build_coords_to_desc_user(objects)
        return f"{self._header}\n{body}" if body else self._header

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        # Assistant: desc only (matching src_new logic)
        return self.formatter.build_desc_only(objects)


class DescToCoordHandler:
    def __init__(self, formatter: JsonGeometryFormatter) -> None:
        self.formatter = formatter
        self._header = CONSTANTS["DESC_TO_COORD_USER_PROMPT"] if "DESC_TO_COORD_USER_PROMPT" in CONSTANTS else "请返回以下描述的物体的坐标"

    def build_user_text(self, objects: List[Dict[str, Any]]) -> Optional[str]:
        # User: desc only
        body = self.formatter.build_desc_to_coords_user(objects)
        return f"{self._header}\n{body}" if body else self._header

    def build_assistant_text(self, objects: List[Dict[str, Any]]) -> str:
        # Assistant: coords only (matching src_new logic)
        return self.formatter.build_coords_only(objects)


class SummaryHandler:
    def __init__(self) -> None:
        self._header = CONSTANTS["SUMMARY_USER_PROMPT"] if "SUMMARY_USER_PROMPT" in CONSTANTS else "请只输出一行摘要："

    @staticmethod
    def _split_commas(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        return [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]

    @staticmethod
    def _collect_remarks(objects: List[Dict[str, Any]]) -> List[str]:
        remarks: List[str] = []
        seen: set[str] = set()
        pattern = re.compile(r"备注[:：]\s*(.+)$")
        for obj in objects:
            desc = str(obj["desc"]).strip() if (isinstance(obj, dict) and "desc" in obj) else ""
            if not desc:
                continue
            for seg in desc.split("/"):
                m = pattern.search(seg)
                if not m:
                    continue
                content = m.group(1).strip()
                if not content:
                    continue
                # normalize ending punctuation
                content = content.strip("；，。;,")
                if content and content not in seen:
                    seen.add(content)
                    remarks.append(content)
        return remarks

    def _extract_summary(self, objects: List[Dict[str, Any]]) -> str:
        # Canonical tokens
        BBU = "BBU设备"
        SHIELD = "挡风板"
        CP = "螺丝、光纤插头"
        FIB = "光纤"
        WIRE = "电线"
        LABEL = "标签"

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
        FIB_BEND_BAD = "弯曲半径不合理(弯曲半径<4cm或者成环)"

        WIRE_NEAT = "捆扎整齐"
        WIRE_MESS = "分布散乱"

        # Aggregators (group key -> count)
        counts: Dict[Tuple[str, ...], int] = {}
        def inc(key: Tuple[str, ...]) -> None:
            counts[key] = (counts[key] + 1) if key in counts else 1

        # Presence flags (for summarization emphasis)
        bbu_present = 0
        shield_need = 0
        shield_ok = 0
        shield_bad = 0

        # Parse
        for o in objects:
            desc = str(o["desc"]).strip() if (isinstance(o, dict) and "desc" in o) else ""
            if not desc:
                continue
            parts = [p.strip() for p in desc.split("/")]
            if not parts:
                continue
            kind = parts[0]

            if kind == BBU and len(parts) >= 2:
                bbu_present += 1
                lvl1 = self._split_commas(parts[1])
                if BBU_REQ_NEED in lvl1:
                    shield_need += 1
                    if len(parts) >= 3:
                        conf = parts[2].strip()
                        if conf == BBU_CONF_OK:
                            inc((SHIELD, "按要求配备"))
                            shield_ok += 1
                        elif conf == BBU_CONF_BAD:
                            inc((SHIELD, "未按要求配备"))
                            shield_bad += 1
                        else:
                            # 未明确给出配备性，保守计入未按要求
                            inc((SHIELD, "未按要求配备"))
                            shield_bad += 1
                elif BBU_REQ_NONEED in lvl1:
                    inc((BBU, "无需挡风板"))

            elif kind == SHIELD and len(parts) >= 2:
                lvl1 = self._split_commas(parts[1])
                if SHIELD_DIR_BAD in lvl1:
                    inc((SHIELD, SHIELD_DIR_BAD))
                elif SHIELD_DIR_OK in lvl1:
                    inc((SHIELD, SHIELD_DIR_OK))

            elif kind == CP and len(parts) >= 2:
                lvl1 = self._split_commas(parts[1])
                if CP_COMPLY_BAD in lvl1:
                    issues: List[str] = []
                    if len(parts) >= 3:
                        issues = [it for it in self._split_commas(parts[2]) if it in CP_ISSUES]
                    if issues:
                        # 细项分组计数
                        for it in sorted(set(issues)):
                            inc((CP, CP_COMPLY_BAD, it))
                    else:
                        inc((CP, CP_COMPLY_BAD))
                elif CP_COMPLY_OK in lvl1:
                    inc((CP, CP_COMPLY_OK))

            elif kind == FIB and len(parts) >= 2:
                lvl1 = self._split_commas(parts[1])
                # 保护
                if FIB_PROTECT_NONE in lvl1:
                    inc((FIB, FIB_PROTECT_NONE))
                elif FIB_PROTECT_HAVE in lvl1:
                    # 细分保护类型
                    det = parts[2].strip() if len(parts) >= 3 else ""
                    if det in FIB_PROTECT_DETAILS:
                        inc((FIB, FIB_PROTECT_HAVE, det))
                    else:
                        inc((FIB, FIB_PROTECT_HAVE))
                # 弯曲半径
                if FIB_BEND_BAD in lvl1:
                    inc((FIB, FIB_BEND_BAD))
                elif FIB_BEND_OK in lvl1:
                    inc((FIB, FIB_BEND_OK))

            elif kind == WIRE and len(parts) >= 2:
                lvl1 = self._split_commas(parts[1])
                if WIRE_MESS in lvl1:
                    inc((WIRE, WIRE_MESS))
                elif WIRE_NEAT in lvl1:
                    inc((WIRE, WIRE_NEAT))

            elif kind == LABEL:
                text = parts[1].strip() if len(parts) >= 2 else ""
                if not text:
                    inc((LABEL, "无法识别"))
                else:
                    inc((LABEL, "清晰"))

        # Compose one-line summary with grouping and ×N
        segments: List[str] = []

        # 先给出 BBU 存在与挡风板链路的关键提示
        if bbu_present > 0:
            segments.append(f"BBU×{bbu_present}")
        if shield_need > 0 and shield_bad > 0:
            segments.append(f"挡风板未按要求配备×{shield_bad}")
        if shield_need > 0 and shield_ok > 0:
            segments.append(f"挡风板按要求配备×{shield_ok}")

        # 连接点（不合规优先，其次合规）
        # 排序保证稳定输出
        for key in sorted(counts.keys()):
            cnt = counts[key]
            if cnt <= 0:
                continue
            # 优先输出负向项
            if key[:2] == (CP, CP_COMPLY_BAD):
                if len(key) == 3:  # 带细项
                    segments.append(f"连接点不合规-{key[2]}×{cnt}")
                else:
                    segments.append(f"连接点不合规×{cnt}")
        # 合规数放后
        if (CP, CP_COMPLY_OK) in counts:
            segments.append(f"连接点合规×{counts[(CP, CP_COMPLY_OK)]}")

        # 光纤保护/弯曲（负向优先）
        if (FIB, FIB_PROTECT_NONE) in counts:
            segments.append(f"光纤无保护×{counts[(FIB, FIB_PROTECT_NONE)]}")
        for det in sorted(FIB_PROTECT_DETAILS):
            k = (FIB, FIB_PROTECT_HAVE, det)
            if k in counts:
                segments.append(f"光纤{det}×{counts[k]}")
        if (FIB, FIB_PROTECT_HAVE) in counts and all((FIB, FIB_PROTECT_HAVE, d) not in counts for d in FIB_PROTECT_DETAILS):
            segments.append(f"光纤有保护×{counts[(FIB, FIB_PROTECT_HAVE)]}")
        if (FIB, FIB_BEND_BAD) in counts:
            segments.append(f"光纤弯曲不合理×{counts[(FIB, FIB_BEND_BAD)]}")
        if (FIB, FIB_BEND_OK) in counts:
            segments.append(f"光纤弯曲合理×{counts[(FIB, FIB_BEND_OK)]}")

        # 电线
        if (WIRE, WIRE_MESS) in counts:
            segments.append(f"电线分布散乱×{counts[(WIRE, WIRE_MESS)]}")
        if (WIRE, WIRE_NEAT) in counts:
            segments.append(f"电线捆扎整齐×{counts[(WIRE, WIRE_NEAT)]}")

        # 标签
        if (LABEL, "无法识别") in counts:
            segments.append(f"标签无法识别×{counts[(LABEL, '无法识别')]}")
        if (LABEL, "清晰") in counts:
            segments.append(f"标签清晰×{counts[(LABEL, '清晰')]}")

        # 挡风板安装方向
        if (SHIELD, SHIELD_DIR_BAD) in counts:
            segments.append(f"挡风板安装方向错误×{counts[(SHIELD, SHIELD_DIR_BAD)]}")
        if (SHIELD, SHIELD_DIR_OK) in counts:
            segments.append(f"挡风板安装方向正确×{counts[(SHIELD, SHIELD_DIR_OK)]}")

        # 若仍为空，构造最小肯定表达
        if not segments:
            segments = ["电线捆扎整齐", "光纤弯曲合理", "标签清晰"]

        summary = "，".join(segments)

        # 合并备注（末尾一次性追加）
        remarks = self._collect_remarks(objects)
        if remarks:
            remark_str = "；".join(remarks)
            summary = f"{summary}，备注: {remark_str}" if summary else f"备注: {remark_str}"

        # 清理潜在特殊字符
        summary = summary.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
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

    def __contains__(self, key: object) -> bool:
        k = str(getattr(key, "value", key)).strip().lower()
        return k in self._handlers

    def __getitem__(self, key: str) -> VariantHandler:
        return self.get(key)


def create_default_variant_registry(formatter: Optional[JsonGeometryFormatter] = None) -> VariantRegistry:
    """Create registry with JSON-first handlers (default schema via formatter).

    If formatter is None, a default JsonGeometryFormatter() is created.
    """
    fmt = formatter or JsonGeometryFormatter()
    reg = VariantRegistry()
    reg.register(ConversationVariant.DENSE_CAPTION, DenseCaptionHandler(fmt))
    reg.register(ConversationVariant.COORDS_TO_DESC, CoordToDescHandler(fmt))
    reg.register(ConversationVariant.DESC_TO_COORDS, DescToCoordHandler(fmt))
    reg.register(ConversationVariant.SUMMARY, SummaryHandler())
    return reg
