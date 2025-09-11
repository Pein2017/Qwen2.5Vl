#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import Qwen2VLProcessor

STAGE_A_SYSTEM_PROMPT: str = (
    "你是一个视觉质检助手。请根据图片仅用中文输出简洁的一行摘要，"
    "只输出一行，聚焦于 BBU 场景；允许的对象仅限：BBU设备、挡风板、螺丝/光纤插头、标签、光纤、电线。"
    "禁止出现与任务无关的设备或术语（如 PPDU、DCDU、ODF、CPRI 等）。"
    "不要输出任何坐标或几何标记；不要出现任意 <|...|> 特殊标记、方括号 [ ] 或坐标数字。如，<|obj_ref_start|>和<|obj_ref_end|>或<|box_start|>和<|box_end|>，<|quad_start|>和<|quad_end|>，<|line_start|>和<|line_end|>等几何标记。"
    "不要使用引号，不要逐字重复词语，不要罗列'螺丝、螺丝、…'等重复清单；"
    "严格遵循：仅围绕下方给定的检查项（与当前 mission 对应）输出相关要点；"
    "忽略未在检查项中的属性（例如：遮挡、品牌、OCR具体文字等）。涉及标签时，只判断'清晰/不清晰'，不要抄写具体文字。"
    "如有影响判定的特殊情况（extra_info），可在摘要末尾简要说明（例如：螺丝不全、无法安装、无法判断、已整改等）。"
    "在检查项范围内，若存在明确不符合项（如螺丝不全/露铜/方向错误/弯曲半径违规/分布散乱），请优先点出。"
    "仅描述当前这张图像的现象，不要输出任何“总评: 通过/不通过”或组级结论，不要跨图片推断。"
)

STAGE_B_SYSTEM_PROMPT: str = (
    "你是通信机房质检助手（BBU场景专用）。下面给出若干图像的摘要列表（每行对应一张图像），以及该 mission 的检查项。\n"
    "请仅基于摘要与检查项判断是否通过：\n"
    "- 若根据摘要与其中的合理说明可判定通过，输出：总评: 通过\n"
    "- 否则输出两行：\n"
    "  第一行：总评: 不通过\n"
    "  第二行：原因: <简短列出你的判断依据（列出未满足的检查项或摘要中的问题；若摘要出现影响判定的特殊情况，也请纳入理由）>\n"
    "严格限制：只依据给定摘要与本 mission 的检查项（MISSION_RULES）；若摘要包含与本 mission 无关的问题，请忽略；"
    "标签仅判断'清晰/不清晰'；不要臆造信息；不要引入与 BBU 场景无关的设备或术语（例如 PPDU、DCDU、ODF、CPRI 等）。"
)

# Derived default QC checklist (from SFT attribute taxonomy/mapping)
DEFAULT_QC_RULES: List[str] = [
    "检测到至少一个BBU设备",
    "如机柜空间充足需要安装，挡风板存在且安装方向正确",
    "螺丝/插头连接符合要求（无未拧紧/露铜/复接/生锈）",
    "光纤有保护措施且弯曲半径合理（无<4cm或成环）",
    "电线捆扎整齐",
    "必要标签可读（如关键信息标签）",
]

# Mission-specific rules (Chinese mission names accepted)
MISSION_RULES: Dict[str, List[str]] = {
    # bbu安装方式检查
    "bbu安装方式检查": [
        "BBU正面及4颗螺丝齐全且连接符合要求",
        "标签内容清晰可见，能看到站点名称",
        "线缆是否横平竖直",
        "光纤弯曲半径是否合理",
    ],
    # bbu接地线检查
    "bbu接地线检查": [
        "设备柜接地端子正反面检查，不漏铜",
        "接地相关标签文字清晰可见",
    ],
    # bbu线缆布放要求
    "bbu线缆布放要求": [
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

# Mission-specific Stage-A hint bullets to steer summaries
MISSION_HINTS: Dict[str, List[str]] = {
    "bbu安装方式检查": [
        "BBU显示是否完整",
        "螺丝/插头连接是否符合要求（未拧紧/露铜/复接/生锈/螺丝不全→不通过）",
        "标签是否清晰（仅判断清晰/不清晰）",
        "电线捆扎是否整齐（分布散乱→不通过）",
        "光纤弯曲半径是否合理（<4cm或成环→不通过）",
        "如有影响判定的特殊情况（extra_info），简要说明（如未拍摄完整/角度问题/无法判断/已整改等）",
    ],
    "bbu接地线检查": [
        "接地端子是否漏铜（露铜→不通过）",
        "接地相关连接是否规范（未拧紧/复接/生锈→不通过）",
        "相关标签是否清晰（仅判断清晰/不清晰）",
        "如有影响判定的特殊情况（extra_info），简要说明（如无法判断/已整改等）",
    ],
    "bbu线缆布放要求": [
        "机柜/地排处接地连接是否规范（未拧紧/复接/生锈→不通过）",
        "光纤是否有保护（蛇形管/铠装/同时），弯曲半径是否合理（<4cm或成环→不通过）",
        "电线捆扎是否整齐（分布散乱→不通过）",
        "关键连接点标签是否清晰（仅判断清晰/不清晰）",
        "如有影响判定的特殊情况（extra_info），简要说明（如保护不完整/部分未套蛇形管/无法判断/已整改等）",
    ],
    "挡风板安装检查": [
        "如需要安装，挡风板是否存在且安装方向正确（方向错误/未按要求配备→不通过）",
        "如有影响判定的特殊情况（extra_info），简要说明（如固定螺丝缺少/无法判断/已整改等）",
    ],
}

# Coverage mapping (SFT data coverage by mission checks)
# status: "covered" | "partial" | "not_covered"
MISSION_CHECKS_COVERAGE: Dict[str, List[Dict[str, str]]] = {
    "bbu安装方式检查": [
        {"check": "BBU显示是否完整", "status": "covered", "by": "bbu.visibility (显示完整/只显示部分)"},
        {"check": "螺丝/插头连接是否符合要求（未拧紧/露铜/复接/生锈/螺丝不全→不通过）", "status": "covered", "by": "connect_point.compliance + specific_issues (未拧紧/露铜/复接/生锈) + extra_info(螺丝不全)"},
        {"check": "标签是否清晰（仅判断清晰/不清晰）", "status": "partial", "by": "label.text_content 自由文本；无显式清晰度布尔"},
        {"check": "电线捆扎是否整齐（分布散乱→不通过）", "status": "covered", "by": "wire.organization (捆扎整齐/分布散乱)"},
        {"check": "光纤弯曲半径是否合理（<4cm或成环→不通过）", "status": "covered", "by": "fiber.bend_radius (合理/不合理)"},
        {"check": "特殊情况（extra_info）", "status": "partial", "by": "special_circumstances/special_situation 自由文本；摘要可提及，训练以RL引导输出"},
    ],
    "bbu接地线检查": [
        {"check": "接地端子是否漏铜（露铜→不通过）", "status": "covered", "by": "connect_point.specific_issues 中的 露铜 (当 compliance=不符合要求 时)"},
        {"check": "接地相关连接是否规范（未拧紧/复接/生锈→不通过）", "status": "covered", "by": "connect_point.compliance + specific_issues (未拧紧/复接/生锈)"},
        {"check": "相关标签是否清晰（仅判断清晰/不清晰）", "status": "partial", "by": "label.text_content 自由文本；无显式清晰度布尔"},
        {"check": "特殊情况（extra_info）", "status": "partial", "by": "special_circumstances/special_situation 自由文本；摘要可提及，训练以RL引导输出"},
    ],
    "bbu线缆布放要求": [
        {"check": "机柜/地排处接地连接是否规范（未拧紧/复接/生锈→不通过）", "status": "covered", "by": "connect_point.type ∈ {机柜处接地螺丝, 地排处接地螺丝} 的 compliance + specific_issues"},
        {"check": "光纤是否有保护（蛇形管/铠装/同时），弯曲半径是否合理（<4cm或成环→不通过）", "status": "covered", "by": "fiber.protection (+ protection_details) 与 fiber.bend_radius"},
        {"check": "电线捆扎是否整齐（分布散乱→不通过）", "status": "covered", "by": "wire.organization"},
        {"check": "关键连接点标签是否清晰（仅判断清晰/不清晰）", "status": "partial", "by": "label.text_content 自由文本；无显式清晰度布尔"},
        {"check": "特殊情况（extra_info）", "status": "partial", "by": "special_circumstances/special_situation 自由文本；摘要可提及，训练以RL引导输出"},
    ],
    "挡风板安装检查": [
        {"check": "如需要安装，挡风板是否存在且安装方向正确（方向错误/未按要求配备→不通过）", "status": "covered", "by": "bbu.windshield_requirement + bbu_shield + bbu_shield.install_direction"},
        # 提示：遮挡属性在推理阶段被忽略，不纳入该 mission 的检查项
        {"check": "特殊情况（extra_info）", "status": "partial", "by": "special_circumstances/special_situation 自由文本；摘要可提及，训练以RL引导输出"},
        # 下列历史检查点未纳入当前提示，且不被数据显式覆盖：
        # - BBU上下1U是否空置 (not_covered)
        # - 相邻BBU间是否都安装挡风板 (需跨对象关系/邻接，not_covered)
        # - 爱立信设备是否按需间隔1U（有空间）(not_covered)
    ],
}


class GroupQCConversationBuilder:
    """Builds Stage-A and Stage-B conversations for Group QC RL.

    This builder only constructs messages and raw text via `apply_chat_template`.
    Image tensorization should be handled by `Qwen2VLProcessor` in the runner.
    """

    def __init__(self, processor: Qwen2VLProcessor) -> None:
        self.processor = processor
        # Default rules and hints
        self.default_rules = DEFAULT_QC_RULES
        self.mission_rules = MISSION_RULES
        self.mission_hints = MISSION_HINTS

    def default_qc_rules(self) -> List[str]:
        # Use universal checks for Stage-A
        return list(DEFAULT_QC_RULES)

    def rules_for_mission(self, mission: Optional[str]) -> List[str]:
        # Stage-B should rely on mission-specific rules only
        if mission and mission in self.mission_rules:
            return list(self.mission_rules[mission])
        return []

    def build_stage_a_messages(self, mission: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build Stage-A messages for single-image summary generation.

        Returns a messages list with one `<image>` placeholder in user content.
        The caller should pass a list with exactly one image when encoding.
        """
        hints = self.default_qc_rules()
        sys = STAGE_A_SYSTEM_PROMPT
        if hints:
            sys = sys + "\n" + "\n".join(hints)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys},
            {"role": "user", "content": "现在给你一张图像，请只输出一行摘要（仅中文自然语言短语，不要出现 < 或 > 或任意 <|...|> 标记）：<image>"},
        ]
        return messages

    def build_stage_b_messages(
        self, summary_lines: List[str], checklist_lines: List[str]
    ) -> List[Dict[str, str]]:
        """Build Stage-B messages for text aggregation and final decision.

        summary_lines: list of single-line strings, one per image
        checklist_lines: list of checklist items
        """
        summaries_text = "\n".join(summary_lines)
        checklist_text = "\n".join(checklist_lines)
        user_text = (
            "请根据以下摘要列表与提示检查项输出结论（严格按照规范，仅限 BBU 场景）：\n"
            f"摘要列表：\n{summaries_text}\n"
            f"提示检查项（本 mission 规则）：\n{checklist_text}"
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": STAGE_B_SYSTEM_PROMPT},
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
