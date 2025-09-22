#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import Qwen2VLProcessor
from src_new_json.processing.templates import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT

import json
import os


STAGE_A_SYSTEM_PROMPT: str = SUMMARY_SYSTEM_PROMPT


def _normalize_mission(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(str(name).strip().lower().split())


DISPLAY_VARIANTS = {"显示完整", "只显示部分"}


def _normalize_desc_item(text: str) -> str:
    parts = [seg.strip() for seg in str(text).split("/") if str(seg).strip()]
    filtered = [seg for seg in parts if seg not in DISPLAY_VARIANTS]
    return "/".join(filtered)


def _load_annotation_table(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation table not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    mapping: Dict[str, Dict[str, List[str]]] = {}
    for row in rows:
        mission = _normalize_mission(row.get("mission"))
        gp = str(row.get("global_pass", "")).strip()
        items_raw = [str(x).strip() for x in (row.get("desc_summary") or []) if str(x).strip()]
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


def _find_best_mission_key(mapping: Dict[str, Any], mission: Optional[str]) -> Optional[str]:
    norm = _normalize_mission(mission)
    if not norm:
        return None
    if norm in mapping:
        return norm
    for k in mapping.keys():
        if k.startswith(norm) or norm.startswith(k):
            return k
    return None


def _build_stage_b_system_prompt(mission: Optional[str], pass_items: List[str], fail_items: List[str]) -> str:
    lines: List[str] = [
        "你是通信机房质检助手（仅依赖标注表进行判断）。",
        "判定规则（只允许使用下方表内词项进行匹配，不得引入表外词项）：",
        "  1) 若摘要出现任意‘不通过类’词项 → 总评: 不通过",
        "  2) 否则，若‘通过类’词项全部出现（每项至少出现一次） → 总评: 通过",
        "  3) 否则（缺少任意通过类词项） → 总评: 不通过",
        "输出格式严格为两行：",
        "  第一行：总评: 通过 或 总评: 不通过",
        "  第二行：原因: <自然语言简述（一到两句，允许合并描述，不需列表/编号；若因未覆盖导致不通过，请说明缺失的通过类词项）>（不得引入表外词项）",
    ]
    if mission:
        lines.append(f"当前任务：{mission}")
    lines.append("表内词项（仅供匹配）：")
    if pass_items:
        lines.append("  通过类（全部需覆盖）：")
        lines.extend([f"    - {x}" for x in pass_items])
    else:
        lines.append("  通过类：<无>")
    if fail_items:
        lines.append("  不通过类（出现任意一项即判不通过）：")
        lines.extend([f"    - {x}" for x in fail_items])
    else:
        lines.append("  不通过类：<无>")
    return "\n".join(lines)


class GroupQCConversationBuilder:
    """Builds Stage-A and Stage-B conversations for Group QC RL.

    This builder only constructs messages and raw text via `apply_chat_template`.
    Image tensorization should be handled by `Qwen2VLProcessor` in the runner.
    """

    def __init__(self, processor: Qwen2VLProcessor, annotation_path: Optional[str] = None) -> None:
        self.processor = processor
        # Resolve annotation table path
        root_guess = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidate_paths = [
            (annotation_path or os.path.join(root_guess, "group_annotation", "table.json")),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "group_annotation", "table.json"),
            os.path.join(os.getcwd(), "group_annotation", "table.json"),
        ]
        resolved = None
        for p in candidate_paths:
            if os.path.exists(p):
                resolved = p
                break
        if not resolved:
            raise FileNotFoundError("group_annotation/table.json not found in expected locations")
        self._annotation_path = resolved
        self._mission_items = _load_annotation_table(self._annotation_path)
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

    def build_stage_a_messages(self, mission: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build Stage-A messages for single-image summary generation.

        Returns a messages list with one `<image>` placeholder in user content.
        The caller should pass a list with exactly one image when encoding.
        """
        system_text = STAGE_A_SYSTEM_PROMPT
        user_text = SUMMARY_USER_PROMPT
        key = _find_best_mission_key(self._mission_items, mission)
        if key is not None:
            pass_items = self._mission_items[key].get("pass_items", [])
            fail_items = self._mission_items[key].get("fail_items", [])
            lines: List[str] = [
                f"任务概览：{mission}。请逐张图核验清单里的词项，不要引入表外描述。",
                "先把每个通过类词项确认一遍：看到就直接照原词写‘×1’；如果没看到或角度不清，请写‘未见<词项>×1’或‘<词项>/不符合要求×1’，方便最终判定。",
                "一旦发现不通过类词项，也要照原词写出并补充位置。",
                "清单如下（已去掉“显示完整”“只显示部分”等辅助描述）：",
                "- 通过类（全部需要覆盖）：",
            ] + [f"  - {x}" for x in pass_items]
            lines += [
                "- 不通过类（出现任意一项即判不通过）：",
            ] + [f"  - {x}" for x in fail_items]
            user_text = user_text + "\n" + "\n".join(lines)
        generic_lines = [
            "只输出一行中文，语句精简自然，避免堆砌或重复。",
            "看不清或被遮挡，请直接说明真实情况，不要猜测。",
            "不要写坐标、几何数据或特殊标记；与当前任务无关的现象直接忽略。",
            "仅引用或同义转述表内词项；无需重复‘显示完整/只显示部分’等修饰语。",
            "发现不通过类词项时，请在描述中注明位置或对应的图像编号。",
            "每个要点末尾都要写‘×数量’，数量为1也要写×1，禁止单独写‘×’或留空。",
            "要点之间使用全角逗号，例如：机柜处接地螺丝/符合要求×1，电线/捆扎整齐×1。",
        ]
        user_text = user_text + "\n" + "\n".join(generic_lines)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        ]
        GroupQCConversationBuilder.validate_typed_image_count(messages, 1)
        return messages

    def build_stage_b_messages(
        self, summary_lines: List[str], checklist_lines: List[str]
    ) -> List[Dict[str, str]]:
        """Build Stage-B messages for text aggregation and final decision (table-only)."""
        summaries_text = "\n".join(summary_lines)
        user_text = (
            "请仅依据下方‘表内词项’进行匹配与判断，输出严格两行（总评/原因），不得引入表外词项。第二行用自然语言简述原因。\n"
            f"摘要列表：\n{summaries_text}"
        )
        if self._last_mission_key and self._last_mission_key in self._mission_items:
            items = self._mission_items[self._last_mission_key]
            pass_items = items.get("pass_items", [])
            fail_items = items.get("fail_items", [])
            system_prompt = _build_stage_b_system_prompt(self._last_mission_name, pass_items, fail_items)
        else:
            system_prompt = _build_stage_b_system_prompt(None, [], [])
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return messages

    def build_stage_b_messages_minimal(self, summary_lines: List[str]) -> List[Dict[str, str]]:
        """Build Stage-B messages without checklist (table-only)."""
        summaries_text = "\n".join(summary_lines)
        user_text = (
            "请仅依据下方‘表内词项’进行匹配与判断，输出严格两行（总评/原因），不得引入表外词项。第二行用自然语言简述原因。\n"
            f"摘要列表：\n{summaries_text}"
        )
        if self._last_mission_key and self._last_mission_key in self._mission_items:
            items = self._mission_items[self._last_mission_key]
            pass_items = items.get("pass_items", [])
            fail_items = items.get("fail_items", [])
            system_prompt = _build_stage_b_system_prompt(self._last_mission_name, pass_items, fail_items)
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
    def validate_image_placeholder_count(rendered_text: str, expected_num_images: int) -> None:
        count = rendered_text.count("<image>")
        if count != expected_num_images:
            raise ValueError(
                f"Image placeholder count mismatch: found {count}, expected {expected_num_images}."
            )

    @staticmethod
    def validate_typed_image_count(messages: List[Dict[str, Any]], expected_num_images: int) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Messages must be a non-empty list for typed image validation")
        typed_count = 0
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        typed_count += 1
        if int(typed_count) != int(expected_num_images):
            raise ValueError(
                f"Typed image count mismatch: found {typed_count}, expected {expected_num_images}."
            )
