#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized prompt constants for Qwen2.5-VL processing pipeline.

This file now contains ONLY the prompt constants that are imported by
the new HuggingFace-first conversation processor. All conversation logic
has been moved to official HuggingFace components.

MIGRATION NOTE: This file was reduced from 563 lines to 50 lines as part
of the HuggingFace-first refactor. All conversation creation logic is now
handled by official HuggingFace processor.apply_chat_template().
"""


# ==============================================================================
# GLOBAL CONSTANTS - All Prompt Strings Centralized
# ==============================================================================

CONSTANTS = {
    # System Prompt - BBU Detection Instructions (Chinese)
    "SYSTEM_PROMPT": """你是通信机房设备检测AI助手。请识别图像中的所有目标并输出位置与类别，使用坐标令牌进行精确空间理解。

输出格式：
- 矩形对象: <|obj_ref_start|>类别/属性<|obj_ref_end|><|box_start|>[<|coord_x1|>, <|coord_y1|>, <|coord_x2|>, <|coord_y2|>]<|box_end|>
- 四边形对象: <|obj_ref_start|>类别/属性<|obj_ref_end|><|quad_start|>[<|coord_x1|>, <|coord_y1|>, <|coord_x2|>, <|coord_y2|>, <|coord_x3|>, <|coord_y3|>, <|coord_x4|>, <|coord_y4|>]<|quad_end|>
- 线缆对象: <|obj_ref_start|>类别/属性<|obj_ref_end|><|line_start|>[<|coord_x1|>, <|coord_y1|>, <|coord_x2|>, <|coord_y2|>, ...]<|line_end|>""",
    # Teacher User Prompt - Learning from Reference Sample - Simplified
    "TEACHER_USER_PROMPT": "这是示例，请检测图像中的设备和部件:",
    # Student User Prompt - Analyze Current Image - Simplified
    "STUDENT_USER_PROMPT": "现在请你回答，请检测图像中的设备和部件:",
}
