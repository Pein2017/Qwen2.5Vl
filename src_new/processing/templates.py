#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template management for Qwen2.5-VL processing pipeline.

Extracts and reuses Chinese prompts from prompt_back.py with simplified interface.
Supports training vs evaluation prompts and teacher/student context prefixes.
"""

from typing import Any, Dict, List, Optional


# ==============================================================================
# Chinese Template Constants (from prompt_back.py)
# ==============================================================================

CHINESE_TRAINING_PROMPT = """你是专业的通信机房BBU设备检测AI助手。你的任务是精确识别并定位图像中所有指定对象，提供详细的属性描述和状态评估。

【学习模式说明】
本对话采用示例学习模式，帮助你提高检测准确性：
1. 首先会提供若干**参考示例**，每个示例包含一张图像和标准检测结果
2. 请仔细学习示例中的检测模式、标注风格、判断标准和分类方法
3. 最后会给出**目标图像**，请运用从参考示例中学到的知识进行精确检测
4. 重点关注示例中的位置判断逻辑、相似对象的区分方法和标注细节

【核心检测目标】请检测以下五大类对象，并提供详细的属性和状态信息：

1. **BBU设备** - 基带处理单元主设备
   - bbu基带处理单元/华为 （蓝色外壳，华为LOGO标识）
   - bbu基带处理单元/中兴 （银灰色外壳，中兴LOGO标识）
   - bbu基带处理单元/爱立信 （黑色外壳，爱立信LOGO标识）

   状态描述要求：显示完整性（完整显示/部分显示）+ 安装需求（无需安装/机柜空间充足需要安装）+ 挡风板配备情况

2. **螺丝连接点** - 各类连接和固定点
   - BBU安装螺丝 → 位于BBU机框四角或导轨固定孔
   - CPRI光缆和BBU连接点 → 蓝色头白色身圆柱形插头，连接光纤与BBU
   - 地排处螺丝 → 仅出现在地排铜排、长铁片上
   - BBU接地线机柜接地端 → 与BBU安装螺丝相似，压住接地线铜环
   - BBU尾纤和ODF连接点 → 铁头圆柱体，安装在BBU上

   状态描述要求：显示完整性（完整显示/部分显示）+ 符合性评估（符合要求/不符合要求）
   注意：子类别外观高度相似，必须结合"所在位置"和"连接对象"进行准确判定

3. **挡风板** - BBU散热防护装置
   - 挡风板/已安装 （正确安装的挡风板）
   - 挡风板/未安装 （应安装但缺失的位置）

   状态描述要求：品牌识别（华为/中兴/爱立信）+ 显示完整性 + 遮挡情况 + 安装方向正确性
   说明：每台BBU上下各需一块挡风板用于散热，位置错误或缺失均应检测

4. **机柜空间** - 设备安装空间状态
   - 机柜空间/满载 （空间已被设备占满）
   - 机柜空间/非满载 （仍有可用安装空间）

5. **标签贴纸** - 设备标识标签
   - 标签贴纸 （常见黄色底+红色联通LOGO，偶有白色底）

   状态描述要求：如能识别标签文字内容，请一并输出

【输出格式要求】
1. **几何坐标格式** - 根据对象几何特征选择合适的坐标表示：
   - **矩形对象** (BBU设备/螺丝连接点/挡风板/机柜空间):
     `类别/属性: <|box_start|><coord_x1><coord_y1><coord_x2><coord_y2><|box_end|>`
   - **旋转标签** (标签贴纸):
     `类别/属性: <|box_start|><coord_x1><coord_y1><coord_x2><coord_y2><coord_x3><coord_y3><coord_x4><coord_y4><|box_end|>`
     (四个角点坐标，按顺时针或逆时针顺序)
   - **线缆路径** (光纤/电线):
     `类别/属性: <|box_start|><coord_x1><coord_y1><coord_x2><coord_y2>...<coord_xN><coord_yN><|box_end|>`
     (沿路径关键转折点的坐标序列)

2. **坐标精度要求**:
   - 使用特殊的可学习坐标token `<|coord_X|>` 表示绝对像素坐标，范围 [0, 1024]
   - 矩形坐标：(x1,y1)左上角, (x2,y2)右下角，确保 x1<x2, y1<y2
   - 旋转标签：四个角点按顺时针方向标注，从左上角开始，确保完整包围对象
   - 线缆路径：沿关键转折点标注，采用canonical ordering（从最上最左点开始），保持路径连续性
   - 边界框必须完全包围目标对象的可见部分，坐标值严格限制在 [0, 1024] 范围内

3. **坐标排序与几何规范**:
   - **矩形对象排序**: 确保 x1 < x2, y1 < y2，左上角到右下角的标准顺序
   - **四边形顶点排序**: 采用顺时针canonical ordering，从左上角(最小y，最小x)开始
   - **线缆路径排序**: 使用canonical line ordering，从最上最左的端点开始，保持路径结构
   - **对象位置排序**: 检测结果按从上到下、从左到右的顺序输出(y坐标优先，x坐标次之)
   - **坐标边界检查**: 所有坐标值必须在 [0, 1024] 范围内，超出范围的坐标将被截断

4. **检测质量标准**:
   - **完整性检测**: 识别图像中所有指定类别的对象，不遗漏
   - **精确定位**: 坐标边界紧贴对象轮廓，几何表示准确
   - **属性描述**: 提供详细的状态信息（显示完整性、安装状态、符合性等）
   - **位置判断**: 特别注意相似对象的位置关系区分（如不同类型螺丝）
   - **几何适配**: 根据对象实际形状选择最合适的几何表示方式

4. **坐标排序详细规范**:
   - **矩形bbox**: [x1,y1,x2,y2] 其中 x1<x2, y1<y2 (左上角→右下角)
   - **四边形quad**: [x1,y1,x2,y2,x3,y3,x4,y4] 顺时针从左上角开始 (左上→右上→右下→左下)
   - **线缆line**: [x1,y1,x2,y2,...,xN,yN] 从最上最左端点开始，保持路径连续性
   - **检测结果排序**: 按对象位置从上到下、从左到右排列输出

5. **质量检查与合规评估**:
   - 对每个检测对象进行状态评估：正常/异常/无法判断
   - 重点关注安全相关项目：连接牢固性、安装规范性、保护措施完备性
   - 提供符合性判断：符合要求/不符合要求/需进一步检查

6. **排除项**: 忽略水印、时间戳、无关文字等非检测目标对象。"""

CHINESE_EVALUATION_PROMPT = """你是专业的通信机房BBU设备检测AI助手。

【任务模式】
如果本对话包含参考示例，请仔细学习示例中的检测模式和标注风格，然后应用到目标图像；如果没有示例，请直接进行检测。

【检测目标与要求】
请精确识别图像中的以下目标并输出详细的位置、类别和状态信息：

**BBU设备**: bbu基带处理单元/华为、bbu基带处理单元/中兴、bbu基带处理单元/爱立信
   - 要求：品牌识别 + 显示完整性 + 安装状态评估 + 挡风板配备情况

**螺丝连接点**: BBU安装螺丝、CPRI光缆和BBU连接点、地排处螺丝、BBU接地线机柜接地端、BBU尾纤和ODF连接点
   - 要求：基于位置关系准确分类（勿仅凭外观）+ 显示完整性 + 符合性评估

**挡风板**: 挡风板/已安装、挡风板/未安装
   - 要求：安装状态 + 品牌匹配 + 安装方向正确性

**机柜空间**: 机柜空间/满载、机柜空间/非满载
   - 要求：空间利用率评估

**标签贴纸**: 设备标识标签
   - 要求：如可识别文字内容请一并输出

【输出格式】
根据对象几何特征选择合适的坐标格式，并提供详细属性描述：
- 矩形对象: `类别/属性: <|box_start|><coord_x1><coord_y1><coord_x2><coord_y2><|box_end|>`
- 旋转标签: `类别/属性: <|box_start|><coord_x1><coord_y1><coord_x2><coord_y2><coord_x3><coord_y3><coord_x4><coord_y4><|box_end|>`
- 线缆路径: `类别/属性: <|box_start|><coord_x1><coord_y1>...<coord_xN><coord_yN><|box_end|>`

坐标要求：使用特殊可学习坐标token `<|coord_X|>`，范围[0,1024]，采用canonical ordering，确保边界框完整包围对象。"""


# ==============================================================================
# Template Manager Class
# ==============================================================================


class TemplateManager:
    """
    Simplified prompt management using existing Chinese templates.

    Provides system prompts and user prompt prefixes for different contexts.
    Supports training vs evaluation prompts and teacher/student scenarios.
    """

    def __init__(
        self,
        language: str = "chinese",
        token_processor: Optional[Any] = None,
        coordinate_tokens_enabled: bool = False,
    ) -> None:
        """
        Initialize template manager.

        Args:
            language: Language for templates (currently only "chinese" supported)
            token_processor: Token processor for coordinate tokenization
            coordinate_tokens_enabled: Whether to use coordinate tokens
        """
        # Normalize language to "chinese" for consistency
        language_lower = language.lower()
        if language_lower in ["chinese", "zh", "中文", "cn"]:
            self.language = "chinese"
        else:
            self.language = "chinese"  # Default to Chinese

        # Store token processor for coordinate tokenization
        self.token_processor = token_processor
        self.coordinate_tokens_enabled = coordinate_tokens_enabled

        # Use existing prompts from prompt_back.py
        self.training_prompt = CHINESE_TRAINING_PROMPT
        self.evaluation_prompt = CHINESE_EVALUATION_PROMPT

    def get_system_prompt(self, use_training_prompt: bool = False) -> str:
        """
        Get system prompt based on context.

        Args:
            use_training_prompt: If True, use detailed training prompt;
                               otherwise use concise evaluation prompt

        Returns:
            Appropriate system prompt string
        """
        return self.training_prompt if use_training_prompt else self.evaluation_prompt

    def get_user_prompt_prefix(self, context: str = "target") -> str:
        """
        Get user prompt prefix for different contexts.

        Args:
            context: "teacher", "target", or "standalone"

        Returns:
            User prompt prefix string
        """
        if context == "teacher":
            return "📚 参考示例:"
        elif context == "target":
            return "现在请根据以上参考示例的检测模式和标注风格，检测以下目标图像:"
        else:  # standalone
            return "🔍 请检测图像中的设备和部件:"

    def format_teacher_student_conversation(
        self, teachers: List[Dict[str, Any]], student: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Format teacher-student conversation with appropriate prefixes.

        Args:
            teachers: List of teacher example dictionaries
            student: Student sample dictionary

        Returns:
            Formatted conversation as list of message dictionaries
        """
        conversation = []

        # System message
        conversation.append(
            {
                "role": "system",
                "content": self.get_system_prompt(use_training_prompt=True),
            }
        )

        # Teacher examples
        for teacher in teachers:
            # Teacher user message
            prefix = self.get_user_prompt_prefix("teacher")
            conversation.append({"role": "user", "content": f"{prefix} <image>"})

            # Teacher assistant response (formatted objects)
            teacher_response = self._format_objects_for_response(
                teacher.get("objects", [])
            )
            conversation.append({"role": "assistant", "content": teacher_response})

        # Student target message
        target_prefix = self.get_user_prompt_prefix("target")
        conversation.append({"role": "user", "content": f"{target_prefix} <image>"})

        # Student assistant response (will be the target for training)
        student_response = self._format_objects_for_response(student.get("objects", []))
        conversation.append({"role": "assistant", "content": student_response})

        return conversation

    def _format_objects_for_response(self, objects: List[Dict[str, Any]]) -> str:
        """
        Format objects list for assistant response.

        Args:
            objects: List of object dictionaries with geometry and description

        Returns:
            Formatted string representation of objects
        """
        if not objects:
            return "[]"

        formatted_objects = []
        for obj in objects:
            desc = obj.get("desc", "")

            # Handle different geometry types
            if "bbox_2d" in obj:
                coords = self._format_coordinates(obj["bbox_2d"])
                formatted_objects.append(f'{{"bbox_2d":{coords}, "desc":"{desc}"}}')
            elif "quad" in obj:
                coords = self._format_coordinates(obj["quad"])
                formatted_objects.append(f'{{"quad":{coords}, "desc":"{desc}"}}')
            elif "line" in obj:
                coords = self._format_coordinates(obj["line"])
                formatted_objects.append(f'{{"line":{coords}, "desc":"{desc}"}}')
            else:
                # Raise error for unsupported geometry types
                available_keys = [k for k in obj.keys() if k not in ["desc"]]
                raise ValueError(
                    f"Object contains unsupported geometry type. "
                    f"Expected one of: bbox_2d, quad, line. "
                    f"Found geometry keys: {available_keys}. "
                    f"Full object: {obj}"
                )

        return "[" + ", ".join(formatted_objects) + "]"

    def _format_coordinates(self, coordinates: List[int]) -> str:
        """
        Format coordinates based on coordinate token mode.

        Args:
            coordinates: List of coordinate integers

        Returns:
            Formatted coordinate string (either integers or coordinate tokens)
        """
        if self.coordinate_tokens_enabled and self.token_processor:
            # Convert coordinates to coordinate tokens
            coord_tokens = self.token_processor.coordinates_to_tokens(coordinates)
            # Format as comma-separated coordinate tokens
            return "[" + ", ".join(coord_tokens) + "]"
        else:
            # Standard mode: use integer coordinates
            return str(coordinates)

    def get_standalone_conversation(
        self, sample: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Get standalone conversation without teacher examples.

        Args:
            sample: Sample dictionary with objects

        Returns:
            Formatted conversation for standalone detection
        """
        conversation = []

        # System message
        conversation.append(
            {
                "role": "system",
                "content": self.get_system_prompt(use_training_prompt=False),
            }
        )

        # User message
        prefix = self.get_user_prompt_prefix("standalone")
        conversation.append({"role": "user", "content": f"{prefix} <image>"})

        # Assistant response
        response = self._format_objects_for_response(sample.get("objects", []))
        conversation.append({"role": "assistant", "content": response})

        return conversation

    def get_learning_instruction(self, num_teachers: int) -> str:
        """
        Get meta-learning instruction for teacher-student scenarios.

        Args:
            num_teachers: Number of teacher examples provided

        Returns:
            Learning instruction string
        """
        if num_teachers == 0:
            return ""  # No instruction needed for standalone detection

        if num_teachers == 1:
            return """
学习提示: 请仔细观察参考示例中的以下要点:
• 如何准确识别不同类型的对象 (BBU设备、螺丝连接点、挡风板等)
• 如何区分外观相似但位置不同的对象 (如BBU安装螺丝 vs BBU接地线机柜接地端)
• 边界框的准确绘制方法和标注风格
• 对象分类的判断逻辑和命名规范
然后将这些模式应用到目标图像的检测中。"""
        else:
            return f"""
学习提示: 下面将提供{num_teachers}个参考示例，请仔细观察:
• 不同场景下的检测模式和标注风格
• 相似对象的区分方法和判断标准
• 边界框绘制的精确度和一致性
• 标签命名的规范性和层级结构
学习完所有示例后，将这些模式应用到目标图像中。"""
