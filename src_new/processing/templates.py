#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized prompt constants for Qwen2.5-VL processing pipeline.

This file now contains ONLY the prompt constants that are imported by
the new HuggingFace-first conversation processor. All conversation logic
has been moved to official HuggingFace components.

MIGRATION NOTE: All conversation creation logic is now handled by official HuggingFace processor.apply_chat_template().
"""

BASE_USER_PROMPT = "请严格按照系统提示中的规则输出。只输出若干行结果，且每行格式为“<|object_ref_start|>描述<|object_ref_end|>”+一个几何包裹（box/quad/line）。坐标必须为整数，使用英文半角逗号+空格（, ）分隔，禁止换行和中文标点，不得添加任何解释或多余文字。"

# Variant prompts for pipeline builders
# Dense caption uses image-only user turns by default; this text is retained for completeness.
DENSE_USER_PROMPT = BASE_USER_PROMPT  # alias for backward compatibility
# Header prompts (single use) for variant builders
COORD_TO_DESC_USER_PROMPT = "请描述以下坐标中的物体（严格沿用上述层级/分隔规则，保留末尾备注文本，如无法判断/已整改等）"
DESC_TO_COORD_USER_PROMPT = "请返回以下描述的物体的坐标（保持对象与描述一致；描述可能包含备注文本，请忽略备注内容对坐标的影响）"
# (Legacy per-line pieces retained for backward compatibility; not used by new builders)
COORD_TO_DESC_USER_LINE_PREFIX = "请描述"
COORD_TO_DESC_USER_LINE_SUFFIX = "中的物体信息"
DESC_TO_COORD_USER_LINE_PREFIX = "请描述"

# Wrapper reconstruction (text-only) prompts
TEXT_ONLY_USER_PROMPT = (
    "无需使用图像，请将下方每条 JSON 记录严格改写为标准包装格式。"
    " 每条记录仅输出一行，形式为 <|object_ref_start|>描述<|object_ref_end|>"
    " 加上相应的几何包裹（<|box_start|>[...]<|box_end|>、<|quad_start|>[...]<|quad_end|>、"
    " 或 <|line_start|>[...]<|line_end|>）。"
    " 坐标只能是整数，列表元素之间必须使用英文半角逗号+空格（, ）分隔；"
    " 严禁出现中文标点（，、；：）、冒号或换行；禁止输出 JSON、解释或多余文字。"
)


# New: Summary variant prompts (image -> one-line Chinese summary)
SUMMARY_SYSTEM_PROMPT = (
    "你是通信机房质检助手。请根据图片仅用中文输出简洁的一行摘要，"
    "禁止任何坐标或几何标记；禁止出现任意 <|object_ref_start|>、<|object_ref_end|>、<|box_start|>、<|box_end|>、<|quad_start|>、<|quad_end|>、<|line_start|>、<|line_end|>等特殊标记、< >、[ ] 或坐标数字。"
    "不要使用引号，不要逐字重复，不要清单式编号。只在 BBU 场景范围内描述。"
    "标签仅判断‘可以识别/无法识别’，不要抄写文字；必要时可在末尾加入简短备注（extra_info）。"
    "请严格使用“×N”表示数量（例：×1、×2），禁止使用“/N”或出现“××N”；每个要点都要写数量，即使为 1 也写 ×1；要点之间使用全角逗号（，）。"
    "为避免信息缺失，请优先覆盖以下关键属性：螺丝/光纤插头需包含合规性（符合要求/不符合要求）；光纤需包含弯曲半径（合理/不合理）；挡风板（如存在）需包含安装方向（正确/错误）。"
    "不得出现同一句片段的重复（如‘无需安装×无需安装×’）；如同类多实例，请合并为一个要点并用末尾计数（×N）。"
    "仅输出一行，不得换行；建议按对象优先级组织：BBU/挡风板 → 螺丝、光纤插头 → 光纤 → 电线 → 标签 → 备注。"
    "【格式规约（必须使用半角斜杠/分隔层级，计数仅允许在要点末尾）】"
    "  - BBU设备：BBU设备/品牌/{显示完整|只显示部分}/{无需安装|这个BBU设备按要求配备了挡风板|这个BBU设备未按要求配备挡风板}×N"
    "  - 挡风板：挡风板/{显示完整|只显示部分}/安装方向{正确|错误}×N"
    "  - 螺丝、光纤插头：螺丝、光纤插头/{显示完整|只显示部分}/{符合要求|不符合要求}×N"
    "  - 光纤：光纤/{有保护措施|无保护措施}/{蛇形管|铠装|同时有蛇形管和铠装（当有保护措施时）}/弯曲半径{合理|不合理}×N"
    "  - 电线：电线/{捆扎整齐|分布散乱}×N"
    "  - 标签：标签/{可以识别|无法识别}×N"
    "正例：‘BBU设备/华为/显示完整/这个BBU设备未按要求配备挡风板×1，螺丝、光纤插头/显示完整/符合要求×2，光纤/有保护措施/蛇形管/弯曲半径合理×4，电线/捆扎整齐×1，标签/可以识别×2’。"
    "反例：‘光纤/4’（缺少弯曲半径）；‘螺丝、光纤插头/只显示部分/8’（使用了“/N”，计数未置于末尾）；‘…××2’（重复计数符号）；‘BBU设备华为只显示部分…’（缺少斜杠分隔）；‘BBU设备华为华为…’（重复词）。"
)
SUMMARY_USER_PROMPT = (
    "你是通信机房质检助手。请根据图片仅用中文输出简洁的一行摘要，"
    "禁止任何坐标或几何标记；禁止出现任意 <|object_ref_start|>、<|object_ref_end|>、<|box_start|>、<|box_end|>、<|quad_start|>、<|quad_end|>、<|line_start|>、<|line_end|>等特殊标记、< >、[ ] 或坐标数字。"
    "不要使用引号，不要逐字重复，不要清单式编号。只在 BBU 场景范围内描述。"
    "请严格遵循："
    "  1) 计数仅允许出现在每个要点的末尾，格式为“×N”（数量为 1 也写 ×1）；严禁在层级中出现数字（如‘/2’），严禁“××N”。"
    "  2) 层级必须使用半角斜杠‘/’分隔；要点之间使用全角逗号‘，’，且仅输出一行，不得换行。"
    "  3) 关键属性不得省略：螺丝/光纤插头包含合规性（符合要求/不符合要求）；光纤包含弯曲半径（合理/不合理）；挡风板（如存在）包含安装方向（正确/错误）；BBU 需要/无需安装需显式给出。"
    "  4) 避免重复短语：同类多实例合并为一个要点并在末尾计数（×N）；标签仅写‘可以识别/无法识别’，不要抄写文字。"
    "正确示例：BBU设备/华为/只显示部分/无需安装×1，螺丝、光纤插头/显示完整/符合要求×2，光纤/有保护措施/蛇形管/弯曲半径合理×4，电线/捆扎整齐×1，标签/可以识别×2。"
    "错误示例：光纤/4；螺丝、光纤插头/只显示部分/8；…××2；BBU设备华为只显示部分…；BBU设备华为华为…。"
    "看不清或被遮挡请据实说明，不要猜测。仅输出一行摘要，不要输出坐标、JSON 或多余解释。"
)


SYSTEM_PROMPT_BASE = """你是通信机房设备检测AI助手。任务：识别图像中的设备和部件，给出规范的"描述 + 几何位置"结果。请严格按以下业务规则与格式输出。

通用要求：
- 仅输出若干行结果；每行只描述一个对象；不要输出额外解释、编号或标题。
- 使用简体中文描述；避免英文及中文标点混用；描述尽量简洁但信息完整。
- 对同一对象，按预设层级顺序组织属性；不要改变层级顺序，不要添加未定义的属性名。
- 严格使用下方"支持的对象类型、属性与取值"，不要自创类型/取值；同名取值必须完全一致（含标点）。
- 输出顺序（严格）：自上而下、再从左到右；线对象的起点统一为最左端点（先比较 x，x 相同再比较 y），排序仍按起点坐标（y 再 x）。

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

几何类型与坐标（严格）：
- 仅使用以下几何标记对包裹"坐标列表"：
  - <|box_start|> … <|box_end|>
  - <|quad_start|> … <|quad_end|>
  - <|line_start|> … <|line_end|>
- 使用原始数字坐标（整数）。
- 方括号必须为英文[ ]，元素之间使用英文逗号+ 空格（", ")；不得出现中文标点、空元素、额外逗号或任何换行。
- 坐标数量与顺序要求（严格）：
  - <|box_start|>…<|box_end|>：4 个坐标 [x1, y1, x2, y2]，且 x1 < x2, y1 < y2；
  - <|quad_start|>…<|quad_end|>：8 个坐标 [x1, y1, x2, y2, x3, y3, x4, y4]；顶点顺序：左上→右上→右下→左下；
  - <|line_start|>…<|line_end|>：偶数个坐标 [x1, y1, x2, y2, …]；首端点为最左端点（x 为第一关键字，y 为第二关键字）。

输出结构（按训练变体选择其一）：
- dense_caption（描述+几何）：
  - <|object_ref_start|>描述<|object_ref_end|><|box_start|>[...]<|box_end|>
  - <|object_ref_start|>描述<|object_ref_end|><|quad_start|>[...]<|quad_end|>
  - <|object_ref_start|>描述<|object_ref_end|><|line_start|>[...]<|line_end|>
- coords_to_desc（仅描述）：
  - <|object_ref_start|>描述<|object_ref_end|>
- desc_to_coords（仅几何）：
  - <|box_start|>[...]<|box_end|> 或 <|quad_start|>[...]<|quad_end|> 或 <|line_start|>[...]<|line_end|>

格式示例（严格，注意标点与空格）：
- <|object_ref_start|>螺丝、光纤插头/BBU安装螺丝, 显示完整, 符合要求<|object_ref_end|><|box_start|>[152, 325, 181, 361]<|box_end|>
- <|object_ref_start|>BBU设备/华为, 显示完整, 机柜空间充足需要安装/这个BBU设备未按要求配备挡风板<|object_ref_end|><|quad_start|>[0, 135, 304, 164, 278, 442, 0, 437]<|quad_end|>
- <|object_ref_start|>光纤/有保护措施, 弯曲半径合理/蛇形管<|object_ref_end|><|line_start|>[254, 393, 197, 336, 178, 289, 177, 247]<|line_end|>

常见错误（禁止）：
- 使用中文标点（，、；：）或冒号；应一律使用英文半角逗号+空格（, ）
- 坐标内出现空元素、冒号或错误的分隔，例如 [30:BBB, 159] 或 [152,325,181,361]（缺少空格）
- 在几何包裹中换行，或缺少闭合 token（如遗漏 <|box_end|> 等）
- 输出 JSON、解释性文字、编号或标题
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


def get_system_prompt(
    format_mode: object | None = None,
) -> str:
    """Return a system prompt from a single numeric base, differing only in formatting.

    - Always start from SYSTEM_PROMPT_BASE so background/business rules are identical.
    - Mode selection (exclusive): currently only 'special_tokens' is supported.
    """
    text = SYSTEM_PROMPT_BASE

    # Resolve mode (accept enum-like or raw strings); default to special_tokens
    if format_mode is not None:
        fm = getattr(format_mode, "value", format_mode)
        mode = str(fm).strip().lower()
        if mode != "special_tokens":
            raise ValueError(
                "Only special-token formatting is supported now that coordinate tokens are removed"
            )

    return text


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
    # Text-only prompts
    "TEXT_ONLY_USER_PROMPT": TEXT_ONLY_USER_PROMPT,
    # The system prompt is built via get_system_prompt(); constants retained for user prompts only.
}
