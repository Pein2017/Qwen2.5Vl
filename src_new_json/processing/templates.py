#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized prompt constants for Qwen2.5-VL processing pipeline.

This file now contains ONLY the prompt constants that are imported by
the new HuggingFace-first conversation processor. All conversation logic
has been moved to official HuggingFace components.

MIGRATION NOTE: All conversation creation logic is now handled by official HuggingFace processor.apply_chat_template().
"""

BASE_USER_PROMPT = "请按照上述规则，根据图像检测设备和部件并按要求输出:"

# Variant prompts for pipeline builders
# Dense caption uses image-only user turns by default; this text is retained for completeness.
DENSE_USER_PROMPT = BASE_USER_PROMPT  # alias for backward compatibility
# Header prompts (single use) for variant builders
COORD_TO_DESC_USER_PROMPT = "仅输出合法 JSON（UTF-8）数组；每个对象含 `label` 和一个几何字段：`box_points`/`quadrilateral_points`/`line_points`。严格沿用“/”分层、“,”同层，末尾可用`,备注:...`。仅使用数据集规范词项，禁止 unknown/未知/未标明。不要输出任何额外文字。"
DESC_TO_COORD_USER_PROMPT = "仅输出合法 JSON（UTF-8）数组；每个对象仅含一个几何字段：`box_points`/`quadrilateral_points`/`line_points`（忽略备注对坐标的影响）。不要输出 `label` 或其它字段，不要输出任何额外文字。"
# (Legacy per-line pieces retained for backward compatibility; not used by new builders)
COORD_TO_DESC_USER_LINE_PREFIX = "请描述"
COORD_TO_DESC_USER_LINE_SUFFIX = "中的物体信息"
DESC_TO_COORD_USER_LINE_PREFIX = "请描述"

# New: Summary variant prompts (image -> one-line Chinese summary)
# System carries business background and canonical vocabulary; user focuses on format and task guidance only
SUMMARY_SYSTEM_PROMPT = (
    "你是通信机房质检助手（BBU场景）。目标：从图像生成一行中文摘要，供后续工单级规则判定使用；"
    "只陈述客观状态，不给出通过/建议/操作。必须使用以下规范词项："
    "连接点{符合要求/不符合要求(未拧紧/露铜/复接/生锈)}；"
    "电线{捆扎整齐/分布散乱}；光纤保护{无保护措施/有保护措施(蛇形管/铠装/同时有蛇形管和铠装)}；"
    "光纤弯曲{弯曲半径合理/弯曲半径不合理(弯曲半径<4cm或者成环)}；"
    "挡风板需求{机柜空间充足需要安装/无需安装}与方向{安装方向正确/安装方向错误}；"
    "标签可读性{清晰/无法识别}。禁止使用集合外占位（unknown/未知/未标明）。"
    "如需自由文本，允许在句末追加“备注: …”。"
)
SUMMARY_USER_PROMPT = (
    "任务：输出一行中文摘要（不含坐标/引号/JSON/特殊标记），用于后续规则判定。"
    "写作：同类且属性完全一致合并为“×N”；优先覆盖BBU存在、连接点合规、电线整齐度、光纤弯曲；"
    "如涉及挡风板，补充“需求+安装方向”；标签仅判清晰/无法识别。若有多条备注，请合并为一句置于末尾“，备注: …”。"
)


SYSTEM_PROMPT_BASE = """你是通信机房设备检测AI助手。目标：检测BBU场景下的物件以及其描述和坐标，输出严格 JSON 数组（UTF-8）。

输出与结构（严格）：
- 仅输出 JSON 数组；每个元素对应一个对象；不要输出额外文字。
- 每个对象仅含 `label` 和且仅含一个几何字段：`box_points` | `quadrilateral_points` | `line_points`。
  - `box_points`: [[x1,y1],[x2,y2]]（左上、右下，整数像素）
  - `quadrilateral_points`: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]（左上→右上→右下→左下）
  - `line_points`: [[x1,y1],...]（≥2点；首端点为最左端点，x优先y次）
- 排序：自上而下，再从左到右；线对象按起点坐标（x，再y）。

支持的对象类型（与数据层严格一致）：
- BBU设备、挡风板、螺丝、光纤插头（统称：螺丝、光纤插头）、标签、光纤、电线。

各类型的属性层级与取值（严格）：
- BBU设备：BBU设备/品牌/完整性/挡风板需求/[挡风板配置符合性][,备注:...]
  - 品牌：取值为 {华为, 中兴, 爱立信}
  - 完整性：取值为 {显示完整, 只显示部分}
  - 挡风板需求：取值为 {无需安装, 机柜空间充足需要安装}
  - 挡风板配置符合性（仅当 挡风板需求=机柜空间充足需要安装 时出现）：取值为 {这个BBU设备按要求配备了挡风板, 这个BBU设备未按要求配备挡风板}
  - 备注文本：自由文本（可缺省，用于"无法判断/角度问题/未拍摄完整/已整改/空间有限"等异常说明；形式为`,备注:...`）
- 挡风板：挡风板/品牌/完整性/安装方向[,备注:...]
  - 品牌：取值为 {华为, 中兴}
  - 完整性：取值为 {显示完整, 只显示部分}
  - 安装方向：取值为 {安装方向正确, 安装方向错误}
  - 备注文本：自由文本（可缺省；形式为`,备注:...`）
- 螺丝、光纤插头：螺丝、光纤插头/类型/完整性/合规性/[具体问题][,备注:...]
  - 类型：取值为 {BBU安装螺丝, 机柜处接地螺丝, 地排处接地螺丝, ODF端光纤插头, BBU端光纤插头}
  - 完整性：取值为 {显示完整, 只显示部分}
  - 合规性：取值为 {符合要求, 不符合要求}
  - 具体问题（仅当 合规性=不符合要求 时出现；可多值，用中文逗号分隔）：从 {未拧紧, 露铜, 复接, 生锈} 中选择
  - 备注文本：自由文本（可缺省，例如：螺丝不全/无法判断/已整改 等；形式为`,备注:...`）
- 标签：标签/文字内容
  - 文字内容：自由文本（当图像可读则写实际文字，否则写"无法识别"）
- 光纤：光纤/保护措施/弯曲半径/[保护类型][,备注:...]
  - 保护措施：取值为 {无保护措施, 有保护措施}
  - 弯曲半径：取值为 {弯曲半径合理, 弯曲半径不合理（弯曲半径<4cm或者成环）}
  - 保护类型（仅当 保护措施=有保护措施 时出现）：取值为 {蛇形管, 铠装, 同时有蛇形管和铠装}
  - 备注文本：自由文本（可缺省，例如：部分未套蛇形管/无法判断/已整改 等；形式为`,备注:...`）
- 电线：电线/整齐度[,备注:...]
  - 整齐度：取值为 {捆扎整齐, 分布散乱}
  - 备注文本：自由文本（可缺省；形式为`,备注:...`）

分隔与层级（严格）：
- 逗号","：同一层级内的属性之间使用逗号分隔
- 斜杠"/"：不同层级（主属性/条件属性）之间使用斜杠分隔
- 条件属性：只有当父条件满足时才出现
- 备注文本：自由文本，形式为`,备注:...`，紧随最后一个属性；用于异常、无法判断、整改说明等

覆盖与一致性：
- 能够识别到的相关对象尽量全部输出；同类多实例分别输出为多行。
- 不输出不在"支持的对象类型"内的对象；不输出置信度、评分或总评结论。

几何与一致性：
- 坐标为图像像素整数；四边形按顺时针；线为折线，若方向不一致请按"左端点优先（x，再y）"规范化。
- 同一图中对象描述与几何应一一对应；描述顺序与排序规则一致。

示例（仅示意）：
[
  {"box_points": [[x1, y1], [x2, y2]], "label": "连接点/BBU安装螺丝,显示完整,符合要求"},
  {"quadrilateral_points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "label": "BBU设备/华为,只显示部分,无需安装,备注:无法判断品牌"},
  {"line_points": [[x1,y1],[x2,y2],[x3,y3]], "label": "光纤/有保护措施,弯曲半径合理/蛇形管,备注:部分未套蛇形管"}
]
"""

# ---------- Single-source prompt builder ----------
import re


def _to_coord_token(match: re.Match) -> str:
    value = match.group(0)
    try:
        num = int(value)
    except Exception:
        return value
    return f"<|coord_{num}|>"


def get_system_prompt() -> str:
    return SYSTEM_PROMPT_BASE


CONSTANTS = {
    # System Prompt - BBU Detection Instructions (Chinese)
    "BASE_USER_PROMPT": BASE_USER_PROMPT,
    "DENSE_USER_PROMPT": DENSE_USER_PROMPT,
    "COORD_TO_DESC_USER_PROMPT": COORD_TO_DESC_USER_PROMPT,
    "DESC_TO_COORD_USER_PROMPT": DESC_TO_COORD_USER_PROMPT,
    # Backward compatibility aliases
    "TEACHER_USER_PROMPT": BASE_USER_PROMPT,
    "STUDENT_USER_PROMPT": BASE_USER_PROMPT,
    "COORD_TO_DESC_USER_HEADER": COORD_TO_DESC_USER_PROMPT,
    "DESC_TO_COORD_USER_HEADER": DESC_TO_COORD_USER_PROMPT,
    # Legacy per-line variants (not used by new builders)
    "COORD_TO_DESC_USER_LINE_PREFIX": COORD_TO_DESC_USER_LINE_PREFIX,
    "COORD_TO_DESC_USER_LINE_SUFFIX": COORD_TO_DESC_USER_LINE_SUFFIX,
    "DESC_TO_COORD_USER_LINE_PREFIX": DESC_TO_COORD_USER_LINE_PREFIX,
    # New summary prompts
    "SUMMARY_SYSTEM_PROMPT": SUMMARY_SYSTEM_PROMPT,
    "SUMMARY_USER_PROMPT": SUMMARY_USER_PROMPT,
    # The system prompt is built via get_system_prompt(); constants retained for user prompts only.
}
