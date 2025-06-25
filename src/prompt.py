#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized prompt templates for BBU equipment detection.
Focuses on basic recognition (location + object type) with hierarchical classification.
"""

# ==============================================================================
# English Prompts
# ==============================================================================

ENGLISH_BASE_PROMPT = """You are Q-Vision-QC, an object-detection assistant specialized in indoor BBU inspections in telecom engineering rooms.
You receive one image; detect all objects matching the allowed descriptions.
Ignore any watermark, timestamp, or overlay text at the top-left.
Output only a JSON array of:
  {"bbox_2d":[x1,y1,x2,y2],"label":"object_type/property/extra_info"}
- (x1,y1)=top-left, (x2,y2)=bottom-right in pixels.
- "label" must be a '/' separated hierarchical description.
- Sort by y ascending, then x ascending.
- No extra text—if none, return []."""

ENGLISH_CANDIDATES_SECTION = """
**Candidate Object Categories:**
- BBU units (Huawei/ZTE/Ericsson baseband processing units)
- Screw connection points (BBU mounting screws, CPRI cable connections, grounding connections, fiber-ODF connections, ground bar screws)
- Cables (fiber optic/non-fiber optic cables)
- Cabinet components (cabinet space, wind shields, label stickers)
"""

ENGLISH_FEW_SHOT_SECTION = ""  # no built-in examples


# ==============================================================================
# Chinese Prompts
# ==============================================================================

# ============================
# HIERARCHICAL OBJECT CLASSIFICATION
# ============================

# Based on actual data analysis from candidate_phrases.json
OBJECT_HIERARCHY = {
    "BBU设备": ["华为", "中兴", "爱立信"],
    "螺丝连接点": [
        "BBU安装螺丝",
        "CPRI光缆和BBU连接点",
        "地排处螺丝",
        "BBU接地线机柜接地端",
        "BBU尾纤和ODF连接点",
    ],
    "线缆": ["光纤", "非光纤"],
    "机柜部件": ["机柜空间", "挡风板", "标签贴纸"],
}

# ============================
# CHINESE PROMPTS (PRIMARY)
# ============================

CHINESE_TRAINING_PROMPT = """你是专业的通信机房BBU设备检测AI助手，负责精确识别和定位图像中的所有相关设备和部件。

**检测任务**:识别图像中所有目标对象的位置和类型。

**目标对象分类**:

1. **BBU设备**
   - bbu基带处理单元/华为
   - bbu基带处理单元/中兴  
   - bbu基带处理单元/爱立信

2. **螺丝连接点**
   - 螺丝连接点/BBU安装螺丝
   - 螺丝连接点/CPRI光缆和BBU连接点
   - 螺丝连接点/地排处螺丝
   - 螺丝连接点/BBU接地线机柜接地端
   - 螺丝连接点/BBU尾纤和ODF连接点

3. **线缆**
   - 线缆/光纤
   - 线缆/非光纤

4. **机柜部件**
   - 机柜空间
   - 挡风板
   - 标签贴纸

**检测要求**:
1. 仔细观察图像中的每个区域，识别所有目标对象
2. 准确标注边界框，确保完全包含目标对象
3. 使用上述标准分类标签，保持格式一致性
4. 忽略图像中的水印、时间戳等叠加信息
5. 对于部分遮挡的对象，标注可见部分的边界

**输出格式**:严格按照JSON数组格式输出:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "标准分类标签"}
]
```

坐标说明:(x1,y1)为左上角，(x2,y2)为右下角，使用绝对像素坐标。"""

CHINESE_EVALUATION_PROMPT = """你是通信机房BBU设备检测AI助手，识别图像中的设备和部件。

**检测对象**:
- BBU设备:bbu基带处理单元/华为、bbu基带处理单元/中兴、bbu基带处理单元/爱立信
- 螺丝连接点:BBU安装螺丝、CPRI光缆和BBU连接点、地排处螺丝、BBU接地线机柜接地端、BBU尾纤和ODF连接点
- 线缆:光纤、非光纤
- 机柜部件:机柜空间、挡风板、标签贴纸

**输出要求**:仅输出JSON数组格式
```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "对象类型"}]
```

坐标为绝对像素值，(x1,y1)左上角，(x2,y2)右下角。"""

# ============================
# LEGACY PROMPTS (保持兼容性)
# ============================

CHINESE_BASE_PROMPT = CHINESE_EVALUATION_PROMPT  # 默认使用评估版本

BASE_PROMPT = """You are an AI assistant specialized in multi-object detection for telecommunication equipment rooms, particularly focused on BBU (Baseband Unit) environments.

**Task**: Detect and locate all relevant equipment and components in the image.

**Target Objects**:
• BBU units (Huawei/ZTE/Ericsson baseband processing units)
• Screw connection points (BBU mounting screws, CPRI cable connections, grounding connections, fiber-ODF connections, ground bar screws)
• Cables (fiber optic/non-fiber optic)
• Cabinet components (cabinet space, wind shields, label stickers)

**Output Format**: JSON array only
```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "object_type/attribute/details"}]
```

Coordinates are absolute pixels: (x1,y1) top-left, (x2,y2) bottom-right. Ignore watermarks and overlay text."""


def get_system_prompt(
    use_training_prompt: bool = False, language: str = "chinese"
) -> str:
    """
    Get the appropriate system prompt based on context.

    Args:
        use_training_prompt: If True, use detailed training prompt; otherwise use concise evaluation prompt
        language: "chinese" or "english"

    Returns:
        Appropriate system prompt string
    """
    if language.lower() == "chinese":
        if use_training_prompt:
            return CHINESE_TRAINING_PROMPT
        else:
            return CHINESE_EVALUATION_PROMPT
    else:
        return BASE_PROMPT


def get_user_prompt_prefix(
    use_training_prompt: bool = False, language: str = "chinese"
) -> str:
    """
    Get user prompt prefix for multi-shot scenarios.

    Args:
        use_training_prompt: If True, use detailed training context
        language: "chinese" or "english"

    Returns:
        User prompt prefix string
    """
    if language.lower() == "chinese":
        if use_training_prompt:
            return "请仔细分析这张BBU机房图像，检测并标注所有相关设备和部件:"
        else:
            return "请检测图像中的设备和部件:"
    else:
        if use_training_prompt:
            return "Please carefully analyze this BBU equipment room image and detect all relevant equipment and components:"
        else:
            return "Please detect all equipment and components in the image:"


# ============================
# PROMPT TEMPLATES FOR DIFFERENT SCENARIOS
# ============================


def format_few_shot_prompt(
    examples: list,
    target_image_description: str = "目标图像",
    use_training_prompt: bool = False,
    language: str = "chinese",
) -> str:
    """
    Format few-shot learning prompt with examples.

    Args:
        examples: List of example dictionaries with 'image_desc' and 'objects'
        target_image_description: Description for the target image
        use_training_prompt: Whether to use detailed training prompt
        language: "chinese" or "english"

    Returns:
        Formatted few-shot prompt
    """
    if language.lower() == "chinese":
        intro = (
            "以下是一些检测示例，请学习其检测模式和标注风格:\n\n"
            if use_training_prompt
            else "参考示例:\n\n"
        )
        target_intro = (
            f"\n现在请检测{target_image_description}中的所有对象:"
            if use_training_prompt
            else f"\n检测{target_image_description}:"
        )
    else:
        intro = (
            "Here are some detection examples to learn the pattern and annotation style:\n\n"
            if use_training_prompt
            else "Reference examples:\n\n"
        )
        target_intro = (
            f"\nNow please detect all objects in {target_image_description}:"
            if use_training_prompt
            else f"\nDetect {target_image_description}:"
        )

    prompt = intro

    for i, example in enumerate(examples, 1):
        if language.lower() == "chinese":
            prompt += f"示例 {i}:{example.get('image_desc', f'图像{i}')}\n"
        else:
            prompt += f"Example {i}: {example.get('image_desc', f'Image {i}')}\n"

        # Format objects as JSON
        import json

        prompt += json.dumps(example["objects"], ensure_ascii=False, indent=2) + "\n\n"

    prompt += target_intro
    return prompt


# ============================
# VALIDATION AND HELPER FUNCTIONS
# ============================


def validate_prompt_language(language: str) -> str:
    """Validate and normalize language parameter."""
    lang = language.lower().strip()
    if lang in ["zh", "chinese", "中文", "cn"]:
        return "chinese"
    elif lang in ["en", "english", "英文"]:
        return "english"
    else:
        return "chinese"  # Default to Chinese


def get_optimized_prompt_for_context(
    context: str = "training", language: str = "chinese"
) -> dict:
    """
    Get optimized prompt configuration based on specific context.

    Args:
        context: "training", "evaluation", "inference", or "few_shot"
        language: "chinese" or "english"

    Returns:
        Dictionary with prompt configuration
    """
    language = validate_prompt_language(language)

    if context in ["training", "train"]:
        use_training_prompts = True
    elif context in ["evaluation", "eval", "inference", "test"]:
        use_training_prompts = False
    elif context == "few_shot":
        use_training_prompts = True  # More detailed for few-shot
    else:
        use_training_prompts = False  # Default to evaluation

    return {
        "use_training_prompts": use_training_prompts,
        "system_prompt": get_system_prompt(
            use_training_prompt=use_training_prompts, language=language
        ),
        "user_prefix": get_user_prompt_prefix(
            use_training_prompt=use_training_prompts, language=language
        ),
        "language": language,
        "context": context,
    }


# ============================
# CHINESE CONSTANTS
# ============================

CHINESE_CANDIDATES_SECTION = """
**标准检测对象分类**:
• BBU设备:bbu基带处理单元/华为、bbu基带处理单元/中兴、bbu基带处理单元/爱立信
• 螺丝连接点:螺丝连接点/BBU安装螺丝、螺丝连接点/CPRI光缆和BBU连接点、螺丝连接点/地排处螺丝、螺丝连接点/BBU接地线机柜接地端、螺丝连接点/BBU尾纤和ODF连接点
• 线缆:线缆/光纤、线缆/非光纤
• 机柜部件:机柜空间、挡风板、标签贴纸
"""

CHINESE_FEW_SHOT_SECTION = ""  # 暂无内置示例
