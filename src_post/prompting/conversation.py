#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import Qwen2VLProcessor
from src_new_json.processing.templates import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT

STAGE_A_SYSTEM_PROMPT: str = SUMMARY_SYSTEM_PROMPT

STAGE_B_SYSTEM_PROMPT: str = (
    "你是通信机房质检助手（BBU场景专用）。下面给出若干图像的摘要列表（每行对应一张图像），以及该 mission 的提示要点（MISSION_HINTS，作为参考）。\n"
    "判定准则（默认通过，但需覆盖要点）：若摘要未出现与本任务直接相关的明确负面项，且不缺少 mission 要求的必要识别项，则判定为通过。\n"
    "两类可判为不通过的情况（任一满足即可）：\n"
    "  A) 摘要中出现明确负面项（标准词项）：未拧紧/露铜/复接/生锈/弯曲半径不合理(弯曲半径<4cm或者成环)/分布散乱/安装方向错误/无保护措施。\n"
    "  B) 对照 MISSION_HINTS 的必要识别项（例如：BBU设备/挡风板/光纤/电线/接地螺丝等），摘要中未体现（缺项）或明显缺失。\n"
    "“标签/无法识别”不单独作为不通过依据；仅当满足 A 或 B 时，才可判定为不通过。\n"
    "输出格式严格为两行：\n"
    "  第一行：总评: 通过 或 总评: 不通过\n"
    "  第二行：原因: <1–2条来自摘要或缺项比对的关键依据>（必须简洁、具体，且与本任务直接相关；不得凭空引入新的负面项；缺项时写‘缺少: <项名>’）\n"
    "禁止使用非规范决策词（例如：需要进一步确认/可以/可通过/合格/不合格/否）。\n"
    "禁止输出‘无需额外信息/无需添加/无’等无信息短语。"
)


# Mission-specific rules (reference only)
MISSION_RULES: Dict[str, List[str]] = {
    # bbu安装方式检查
    "bbu安装方式检查": [
        "BBU正面及4颗螺丝齐全且连接符合要求",
        "标签内容清晰可见",
        "电线是否横平竖直",
        "光纤弯曲半径是否合理",
    ],
    # bbu接地线检查
    "bbu接地线检查": [
        "设备柜接地端子正反面检查，不漏铜",
        "接地相关标签文字清晰可见",
    ],
    # bbu电线布放要求
    "bbu电线布放要求": [
        "BBU与DCDU接地线及等电位连接规范，地排处连接规范，相关标签清晰可见",
        "BBU尾纤是否有保护（传输尾纤套蛇形管/铠装不用套管）、绑扎是否清晰、弯曲半径是否合理",
        "BBU尾纤与ODF连接点、CPRI与BBU连接点的标签是否清晰",
    ],
    # 挡风板安装检查
    "挡风板安装检查": [
        "BBU上下1U是否空置",
        "相邻BBU间是否都安装挡风板",
        "挡风板进/出风道及开孔是否无遮挡",
        "若空间允许是否已加装挡风板；满配情况是否有整体照备注",
        "爱立信设备是否按需间隔1U（有空间）",
    ],
}

# Mission-specific Stage-A/B hint bullets (actual prompt text)
MISSION_HINTS: Dict[str, List[str]] = {
    "bbu安装方式检查": [
        "至少识别到：BBU设备",
        "螺丝合规性：符合要求 / 不符合要求（未拧紧/露铜/复接/生锈）",
        "电线整齐度：捆扎整齐 / 分布散乱",
        "光纤弯曲半径：弯曲半径合理 / 弯曲半径不合理(弯曲半径<4cm或者成环)",
    ],
    "bbu接地线检查": [
        "连接点（螺丝、光纤插头）合规性：符合要求 / 不符合要求（未拧紧/露铜/复接/生锈）",
        "标签可读性：清晰（可读）/ 无法识别（不清晰时应在摘要中体现）",
    ],
    "bbu电线布放要求": [
        "标签可读性：清晰（可读）/ 无法识别（不清晰时应在摘要中体现）",
        "光纤保护措施：无保护措施 / 有保护措施（蛇形管/铠装/同时有蛇形管和铠装）",
        "光纤弯曲半径：弯曲半径合理 / 弯曲半径不合理(弯曲半径<4cm或者成环)",
    ],
    "挡风板安装检查": [
        "挡风板需求（结合机柜空间）：机柜空间充足需要安装 / 无需安装",
        "挡风板安装方向：安装方向正确 / 安装方向错误",
    ],
}

# Coverage mapping retained for reference only (unchanged)
# status: "covered" | "partial" | "not_covered"
MISSION_CHECKS_COVERAGE: Dict[str, List[Dict[str, str]]] = {
    "bbu安装方式检查": [
        {"check": "挡风板需求（机柜空间充足需要安装/无需安装）", "status": "covered", "by": "BBU设备.windshield_requirement (机柜空间充足需要安装 | 无需安装)"},
        {"check": "挡风板符合性（如需安装：按要求配备/未按要求配备）", "status": "covered", "by": "BBU设备.windshield_conformity (这个BBU设备按要求配备了挡风板 | 这个BBU设备未按要求配备挡风板)"},
        {"check": "连接点合规与细项（未拧紧/露铜/复接/生锈）", "status": "covered", "by": "connect_point.compliance + specific_issues (不符合要求 时的 未拧紧/露铜/复接/生锈)"},
        {"check": "电线整齐度（捆扎整齐/分布散乱）", "status": "covered", "by": "wire.organization"},
        {"check": "光纤弯曲半径（合理/不合理）", "status": "covered", "by": "fiber.bend_radius"},
        {"check": "标签可读性（无法识别时输出）", "status": "covered", "by": "label.text_content 为空 → 摘要中输出 标签/无法识别；有文字时不输出标签项"},
    ],
    "bbu接地线检查": [
        {"check": "接地连接是否规范（未拧紧/露铜/复接/生锈）", "status": "covered", "by": "connect_point.compliance + specific_issues (不符合要求 时的 细项)"},
        {"check": "标签可读性（无法识别时输出）", "status": "partial", "by": "label.text_content 为空 → 标签/无法识别；否则不输出"},
    ],
    "bbu电线布放要求": [
        {"check": "接地连接是否规范（机柜/地排处；未拧紧/复接/生锈）", "status": "covered", "by": "connect_point.type∈{机柜处接地螺丝, 地排处接地螺丝} + compliance + specific_issues"},
        {"check": "光纤保护措施（无保护措施/有保护措施）", "status": "covered", "by": "fiber.protection"},
        {"check": "光纤保护细节（蛇形管/铠装/同时）", "status": "covered", "by": "fiber.protection_details (当 protection=有保护措施 时)"},
        {"check": "光纤弯曲半径（合理/不合理）", "status": "covered", "by": "fiber.bend_radius"},
        {"check": "电线整齐度（捆扎整齐/分布散乱）", "status": "covered", "by": "wire.organization"},
        {"check": "关键连接点标签可读性（无法识别时输出）", "status": "partial", "by": "label.text_content 为空 → 标签/无法识别；否则不输出"},
    ],
    "挡风板安装检查": [
        {"check": "挡风板需求（机柜空间充足需要安装）", "status": "covered", "by": "BBU设备.windshield_requirement"},
        {"check": "挡风板符合性（按要求配备/未按要求配备）", "status": "covered", "by": "BBU设备.windshield_conformity"},
        {"check": "挡风板安装方向（安装方向正确/安装方向错误）", "status": "covered", "by": "bbu_shield.install_direction"},
    ],
}


class GroupQCConversationBuilder:
    """Builds Stage-A and Stage-B conversations for Group QC RL.

    This builder only constructs messages and raw text via `apply_chat_template`.
    Image tensorization should be handled by `Qwen2VLProcessor` in the runner.
    """

    def __init__(self, processor: Qwen2VLProcessor) -> None:
        self.processor = processor
        # Use mission hints (reference-only rules removed)
        self.mission_hints = MISSION_HINTS

    def rules_for_mission(self, mission: Optional[str]) -> List[str]:
        # Return mission hints for prompts and diagnostics; rules kept as reference only
        if mission and mission in self.mission_hints:
            return list(self.mission_hints[mission])
        return []

    def build_stage_a_messages(self, mission: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build Stage-A messages for single-image summary generation.

        Returns a messages list with one `<image>` placeholder in user content.
        The caller should pass a list with exactly one image when encoding.
        """
        # Base system prompt from SFT summary variant
        system_text = STAGE_A_SYSTEM_PROMPT
        # Build user text: mission-focused guidance lives in the user turn
        user_text = SUMMARY_USER_PROMPT
        if mission and mission in self.mission_hints:
            hints = self.mission_hints.get(mission, [])
            if hints:
                lines = [
                    f"本次任务：{mission}",
                    "仅关注与该任务相关的要点；与本任务无关的问题请忽略。",
                    "不要在摘要中描述与任务无关的问题（例如在“bbu安装方式检查”中，不要写‘挡风板相关的内容’）。",
                    "任务提示（供摘要侧重参考）：",
                ] + [f"- {h}" for h in hints]
                user_text = user_text + "\n" + "\n".join(lines)
        # Generic cautious guidelines (always appended)
        generic_lines = [
            "仅输出一行中文，避免重复与口水话；不要使用引号或方括号。",
            "如无法确认，请如实表述（例如：无法确认/角度不佳/遮挡），不要臆测或编造。",
            "不要输出坐标/几何/特殊标记（<|...|>）。",
            "与当前 mission 无关的现象请忽略。",
            "不要输出未定义占位词（unknown/未知/未标明），无法确认请用自然语言说明。",
        ]
        user_text = user_text + "\n" + "\n".join(generic_lines)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        ]
        # Enforce exactly one typed image for Stage-A
        GroupQCConversationBuilder.validate_typed_image_count(messages, 1)
        return messages

    def build_stage_b_messages(
        self, summary_lines: List[str], checklist_lines: List[str]
    ) -> List[Dict[str, str]]:
        """Build Stage-B messages for text aggregation and final decision.

        summary_lines: list of single-line strings, one per image
        checklist_lines: list of checklist items
        """
        # Present mission hints rather than rules
        summaries_text = "\n".join(summary_lines)
        checklist_text = "\n".join(checklist_lines)
        user_text = (
            "请根据以下摘要列表与提示要点（MISSION_HINTS）输出结论（严格按照规范，仅限 BBU 场景）：\n"
            f"摘要列表：\n{summaries_text}\n"
            f"提示要点：\n{checklist_text}\n"
            "注意：若摘要未出现与当前 mission 直接相关的明确负面项，则判定为通过；不得因为与本任务无关的信息而判为不通过。\n"
            "始终输出两行：第一行‘总评: 通过/不通过’，第二行‘原因: <1–2条来自摘要的关键依据>’；不得引入新的负面项；禁止‘无需额外信息/无需添加/无’等无信息短语。"
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": STAGE_B_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        return messages

    def build_stage_b_messages_minimal(self, summary_lines: List[str]) -> List[Dict[str, str]]:
        """Build Stage-B messages without checklist (minimal prompt)."""
        summaries_text = "\n".join(summary_lines)
        system_prompt = (
            "你是通信机房质检助手（BBU场景专用）。下面给出若干图像的摘要列表（每行对应一张图像）。\n"
            "请仅依据这些摘要判断是否通过；偏好一致、简洁、具体的证据；若未出现明确负面项则默认通过。\n"
            "输出格式严格为两行：第一行‘总评: 通过/不通过’，第二行‘原因: <1–2条来自摘要的关键依据>’；理由仅能引用或同义转述摘要中的负面/关键信息；禁止‘无需额外信息/无需添加/无’等无信息短语。"
        )
        user_text = (
            "请根据以下摘要列表输出结论（严格按照规范，仅限 BBU 场景）：\n"
            f"摘要列表：\n{summaries_text}"
        )
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
        """Render messages to raw text using the processor's chat template.

        Returns a single conversation string. Image placeholders remain as `<image>`.
        """
        text: str = self.processor.apply_chat_template(
            messages=messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text

    @staticmethod
    def validate_image_placeholder_count(rendered_text: str, expected_num_images: int) -> None:
        """Ensure the number of `<image>` placeholders equals the number of images.

        Raises ValueError if mismatch.
        """
        count = rendered_text.count("<image>")
        if count != expected_num_images:
            raise ValueError(
                f"Image placeholder count mismatch: found {count}, expected {expected_num_images}."
            )

    @staticmethod
    def validate_typed_image_count(messages: List[Dict[str, Any]], expected_num_images: int) -> None:
        """Ensure typed user content includes exactly expected_num_images 'image' entries.

        This is used for typed HF messages where '<image>' may not render in text.
        """
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
