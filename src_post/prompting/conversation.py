#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import Qwen2VLProcessor

from src_new_json.processing.templates import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT


# import json  # removed: no longer reading table.json at runtime
# import os    # removed: no longer resolving external table path


STAGE_A_SYSTEM_PROMPT: str = SUMMARY_SYSTEM_PROMPT


def _normalize_mission(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(str(name).strip().lower().split())


DISPLAY_VARIANTS = {"显示完整", "只显示部分", "遮挡"}


def _normalize_desc_item(text: str) -> str:
    parts = [seg.strip() for seg in str(text).split("/") if str(seg).strip()]
    filtered = [seg for seg in parts if seg not in DISPLAY_VARIANTS]
    return "/".join(filtered)


def _load_annotation_table_json() -> Optional[Dict[str, Dict[str, List[str]]]]:
    """Load mission→{pass_items, fail_items} from group_annotation/table.json.

    Returns None when file not found or parsing fails.
    """
    try:
        root = Path(__file__).resolve().parent.parent  # src_post/
        table_path = root.parent / "group_annotation" / "table.json"
        if not table_path.exists():
            return None
        data = json.loads(table_path.read_text(encoding="utf-8"))
        mapping: Dict[str, Dict[str, List[str]]] = {}
        for row in data:
            mission = _normalize_mission(row.get("mission"))
            gp = str(row.get("global_pass", "")).strip()
            items_raw = [
                str(x).strip()
                for x in (row.get("desc_summary") or [])
                if str(x).strip()
            ]
            items = [
                _normalize_desc_item(x) for x in items_raw if _normalize_desc_item(x)
            ]
            if mission not in mapping:
                mapping[mission] = {"pass_items": [], "fail_items": []}
            if gp == "通过":
                mapping[mission]["pass_items"].extend(items)
            elif gp == "不通过":
                mapping[mission]["fail_items"].extend(items)
        for m in list(mapping.keys()):
            mapping[m]["pass_items"] = sorted(
                list({x for x in mapping[m]["pass_items"]})
            )
            mapping[m]["fail_items"] = sorted(
                list({x for x in mapping[m]["fail_items"]})
            )
        return mapping
    except Exception:
        return None


def _find_best_mission_key(
    mapping: Dict[str, Any], mission: Optional[str]
) -> Optional[str]:
    norm = _normalize_mission(mission)
    if not norm:
        return None
    if norm in mapping:
        return norm
    for k in mapping.keys():
        if k.startswith(norm) or norm.startswith(k):
            return k
    return None


def _build_stage_b_system_prompt(
    mission: Optional[str], pass_items: List[str], fail_items: List[str]
) -> str:
    lines: List[str] = [
        "你是通信机房质检助手。目标：基于多张图片的单行摘要，做一个简明的工单级判断说明。",
        "输出格式严格为两行：",
        "  第一行：总评: 通过 或 总评: 不通过",
        "  第二行：原因: <用自然语言简述关键依据，可合并表达；可包含‘备注: …’>",
        "仅做格式约束，不强制引用固定词表；不要输出坐标/特殊标记。",
    ]
    if mission:
        lines.append(f"当前任务：{mission}")
    # 关键检查项（只列出本 mission 负项要点，过滤可见性描述）
    focus: List[str] = []
    for raw in (fail_items or [])[:]:  # 防止过长
        if not raw:
            continue
        item = _normalize_desc_item(str(raw))
        if not item:
            continue
        focus.append(item)
    focus = sorted(list({x for x in focus}))[:8]
    if focus:
        lines.append(
            "关键检查项（与本任务密切相关；可见性类‘遮挡/只显示部分/显示完整’不作为直接依据）："
        )
        for it in focus:
            lines.append(f"  - {it}")
        lines.append(
            "若关键项不可确认或命中负项，请输出‘不通过’，并简要说明原因；非本任务范围的异常可忽略。"
        )
    return "\n".join(lines)


def _inline_annotation_items() -> Dict[str, Dict[str, List[str]]]:
    """Inline mission→{pass_items, fail_items} table built from group_annotation/table.json.

    Items are normalized by removing display variants like ‘显示完整’/‘只显示部分’.
    """
    rows: List[Dict[str, Any]] = [
        {
            "mission": "BBU安装方式检查（正装）",
            "global_pass": "通过",
            "desc_summary": [
                "螺丝、光纤插头/BBU安装螺丝/显示完整/符合要求",
                "螺丝、光纤插头/BBU安装螺丝/只显示部分/符合要求",
            ],
        },
        {
            "mission": "BBU安装方式检查（正装）",
            "global_pass": "不通过",
            "desc_summary": [
                "螺丝、光纤插头/BBU安装螺丝/显示完整/不符合要求",
                "螺丝、光纤插头/BBU安装螺丝/只显示部分/不符合要求",
            ],
        },
        {
            "mission": "BBU接地线检查",
            "global_pass": "通过",
            "desc_summary": [
                "螺丝、光纤插头/机柜处接地螺丝/显示完整/符合要求",
                "螺丝、光纤插头/机柜处接地螺丝/只显示部分/符合要求",
                "螺丝、光纤插头/地排处接地螺丝/显示完整/符合要求",
                "螺丝、光纤插头/地排处接地螺丝/只显示部分/符合要求",
                "电线/捆扎整齐",
            ],
        },
        {
            "mission": "BBU接地线检查",
            "global_pass": "不通过",
            "desc_summary": [
                "螺丝、光纤插头/机柜处接地螺丝/显示完整/不符合要求",
                "螺丝、光纤插头/机柜处接地螺丝/只显示部分/不符合要求",
                "螺丝、光纤插头/地排处接地螺丝/显示完整/不符合要求",
                "螺丝、光纤插头/地排处接地螺丝/只显示部分/不符合要求",
                "电线/分布散乱",
            ],
        },
        {
            "mission": "BBU线缆布放要求",
            "global_pass": "通过",
            "desc_summary": [
                "螺丝、光纤插头/BBU端光纤插头/显示完整/符合要求",
                "螺丝、光纤插头/BBU端光纤插头/只显示部分/符合要求",
                "螺丝、光纤插头/ODF端光纤插头/显示完整/符合要求",
                "螺丝、光纤插头/ODF端光纤插头/只显示部分/符合要求",
                "光纤/有保护措施/蛇形管/弯曲半径合理",
                "光纤/有保护措施/铠装/弯曲半径合理",
                "光纤/有保护措施/同时有蛇形管和铠装/弯曲半径合理",
            ],
        },
        {
            "mission": "BBU线缆布放要求",
            "global_pass": "不通过",
            "desc_summary": [
                "螺丝、光纤插头/BBU端光纤插头/显示完整/不符合要求",
                "螺丝、光纤插头/BBU端光纤插头/只显示部分/不符合要求",
                "螺丝、光纤插头/ODF端光纤插头/显示完整/不符合要求",
                "螺丝、光纤插头/ODF端光纤插头/只显示部分/不符合要求",
                "光纤/无保护措施/弯曲半径合理",
                "光纤/无保护措施/弯曲半径不合理（弯曲半径<4cm或者成环）",
                "光纤/有保护措施/蛇形管/弯曲半径不合理（弯曲半径<4cm或者成环）",
                "光纤/有保护措施/铠装/弯曲半径不合理（弯曲半径<4cm或者成环）",
                "光纤/有保护措施/同时有蛇形管和铠装/弯曲半径不合理（弯曲半径<4cm或者成环）",
            ],
        },
        {
            "mission": "挡风板安装检查",
            "global_pass": "通过",
            "desc_summary": [
                "BBU设备/华为/显示完整/这个BBU设备按要求配备了挡风板",
                "BBU设备/华为/只显示部分/这个BBU设备按要求配备了挡风板",
                "BBU设备/中兴/显示完整/这个BBU设备按要求配备了挡风板",
                "BBU设备/中兴/只显示部分/这个BBU设备按要求配备了挡风板",
                "BBU设备/华为/显示完整/无需安装",
                "BBU设备/华为/只显示部分/无需安装",
                "BBU设备/中兴/显示完整/无需安装",
                "BBU设备/中兴/只显示部分/无需安装",
                "BBU设备/爱立信/显示完整/无需安装",
                "BBU设备/爱立信/只显示部分/无需安装",
                "挡风板/显示完整/安装方向正确",
                "挡风板/只显示部分/安装方向正确",
            ],
        },
        {
            "mission": "挡风板安装检查",
            "global_pass": "不通过",
            "desc_summary": [
                "BBU设备/华为/显示完整/这个BBU设备未按要求配备挡风板",
                "BBU设备/华为/只显示部分/这个BBU设备未按要求配备挡风板",
                "BBU设备/中兴/显示完整/这个BBU设备未按要求配备挡风板",
                "BBU设备/中兴/只显示部分/这个BBU设备未按要求配备挡风板",
                "挡风板/显示完整/安装方向错误",
                "挡风板/只显示部分/安装方向错误",
            ],
        },
    ]
    mapping: Dict[str, Dict[str, List[str]]] = {}
    for row in rows:
        mission = _normalize_mission(row.get("mission"))
        gp = str(row.get("global_pass", "")).strip()
        items_raw = [
            str(x).strip() for x in (row.get("desc_summary") or []) if str(x).strip()
        ]
        items = [_normalize_desc_item(x) for x in items_raw if _normalize_desc_item(x)]
        if mission not in mapping:
            mapping[mission] = {"pass_items": [], "fail_items": []}
        if gp == "通过":
            mapping[mission]["pass_items"].extend(items)
        elif gp == "不通过":
            mapping[mission]["fail_items"].extend(items)
    for m in list(mapping.keys()):
        mapping[m]["pass_items"] = sorted(list({x for x in mapping[m]["pass_items"]}))
        mapping[m]["fail_items"] = sorted(list({x for x in mapping[m]["fail_items"]}))
    return mapping


class GroupQCConversationBuilder:
    """Builds Stage-A and Stage-B conversations for Group QC RL.

    This builder only constructs messages and raw text via `apply_chat_template`.
    Image tensorization should be handled by `Qwen2VLProcessor` in the runner.
    """

    def __init__(
        self, processor: Qwen2VLProcessor, annotation_path: Optional[str] = None
    ) -> None:
        self.processor = processor
        # Inline annotation table (do not read external files)
        self._annotation_path = "<inline>"
        self._mission_items = _inline_annotation_items()
        self._last_mission_key: Optional[str] = None
        self._last_mission_name: Optional[str] = None

    def rules_for_mission(self, mission: Optional[str]) -> List[str]:
        """Return table-driven items for the mission (pass+fail) as generic hints."""
        key = _find_best_mission_key(self._mission_items, mission)
        self._last_mission_key = key
        self._last_mission_name = mission
        if key is None:
            return []
        items = self._mission_items[key]
        return list(items.get("pass_items", [])) + list(items.get("fail_items", []))

    def build_stage_a_messages(
        self, mission: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build Stage-A messages for single-image summary generation.

        Returns a messages list with one `<image>` placeholder in user content.
        The caller should pass a list with exactly one image when encoding.
        """
        system_text = STAGE_A_SYSTEM_PROMPT
        user_text = SUMMARY_USER_PROMPT
        key = _find_best_mission_key(self._mission_items, mission)
        if key is not None:
            # Retain mission context line only; no checklist expansion
            lines: List[str] = [
                f"任务概览：{mission}。请仅陈述可见事实，优先描述关键信息；如不确定，句末用‘，备注: …’精简说明。"
            ]
            user_text = user_text + "\n" + "\n".join(lines)
        generic_lines = [
            "只输出一行中文，语句简洁自然，避免堆砌或重复。",
            "不要写坐标、几何或特殊标记；与任务无关的现象忽略。",
            # 业务说明：‘遮挡/显示部分/显示完整’仅作为可见性描述，不能作为决定性依据，不要据此直接下最终结论
            "若存在无法确认/信息缺失/标签不可识别/安装方向不明等情况，请在末尾补充‘，备注: …’，并明确指出不确定性。",
        ]
        user_text = user_text + "\n" + "\n".join(generic_lines)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": user_text}],
            },
        ]
        GroupQCConversationBuilder.validate_typed_image_count(messages, 1)
        return messages

    def build_stage_b_messages(
        self, summary_lines: List[str], checklist_lines: List[str]
    ) -> List[Dict[str, str]]:
        """Build Stage-B messages for text aggregation and final decision (table-only)."""
        summaries_text = "\n".join(summary_lines)
        user_text = (
            "请基于下方摘要做两行输出（第一行‘总评: 通过/不通过’，第二行‘原因: …’），"
            "不要输出坐标或特殊标记；原因用自然语言简述关键依据，可合并表达；如不确定，可在末尾补充‘备注: …’。\n"
            f"摘要列表：\n{summaries_text}"
        )
        if self._last_mission_key and self._last_mission_key in self._mission_items:
            items = self._mission_items[self._last_mission_key]
            pass_items = items.get("pass_items", [])
            fail_items = items.get("fail_items", [])
            system_prompt = _build_stage_b_system_prompt(
                self._last_mission_name, pass_items, fail_items
            )
        else:
            system_prompt = _build_stage_b_system_prompt(None, [], [])
        # 增加多样化与不确定性提示
        diversity_hint = "请避免重复用语/模板化句式，允许多种合理表达；"
        uncertainty_hint = "若关键项不可确认（如标签缺失/安装方向不明），请输出‘不通过’，并简要说明原因。"
        system_full = system_prompt + "\n" + diversity_hint + "\n" + uncertainty_hint
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_full},
            {"role": "user", "content": user_text},
        ]
        return messages

    def build_stage_b_messages_minimal(
        self, summary_lines: List[str]
    ) -> List[Dict[str, str]]:
        """Build Stage-B messages without checklist (table-only)."""
        summaries_text = "\n".join(summary_lines)
        user_text = (
            "请基于下方摘要做两行输出（第一行‘总评: 通过/不通过’，第二行‘原因: …’），"
            "不要输出坐标或特殊标记；原因用自然语言简述关键依据，可合并表达；如不确定，可在末尾补充‘备注: …’。\n"
            f"摘要列表：\n{summaries_text}"
        )
        if self._last_mission_key and self._last_mission_key in self._mission_items:
            items = self._mission_items[self._last_mission_key]
            pass_items = items.get("pass_items", [])
            fail_items = items.get("fail_items", [])
            system_prompt = _build_stage_b_system_prompt(
                self._last_mission_name, pass_items, fail_items
            )
        else:
            system_prompt = _build_stage_b_system_prompt(None, [], [])
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return messages

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        text: str = self.processor.apply_chat_template(
            messages=messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text

    @staticmethod
    def validate_image_placeholder_count(
        rendered_text: str, expected_num_images: int
    ) -> None:
        count = rendered_text.count("<image>")
        if count != expected_num_images:
            raise ValueError(
                f"Image placeholder count mismatch: found {count}, expected {expected_num_images}."
            )

    @staticmethod
    def validate_typed_image_count(
        messages: List[Dict[str, Any]], expected_num_images: int
    ) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Messages must be a non-empty list for typed image validation"
            )
        typed_count = 0
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
            ):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        typed_count += 1
        if int(typed_count) != int(expected_num_images):
            raise ValueError(
                f"Typed image count mismatch: found {typed_count}, expected {expected_num_images}."
            )
