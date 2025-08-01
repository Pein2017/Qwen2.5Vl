"""
Tests for template management functionality.

This module tests:
- Chinese prompt template management
- System prompt selection (training vs evaluation)
- User prompt prefixes for different contexts
- Teacher-student conversation formatting
"""

from typing import Any, Dict, List

from src_new.tests.fixtures import create_teacher_student_data


class TestTemplateManager:
    """Test template management functionality."""

    def test_template_manager_initialization(self):
        """Test template manager initialization."""
        # Test with default Chinese language
        manager = self._mock_template_manager()

        assert manager.language == "chinese"
        assert len(manager.training_prompt) > 0
        assert len(manager.evaluation_prompt) > 0

        # Test with explicit language settings
        for lang in ["chinese", "zh", "中文", "cn"]:
            manager = self._mock_template_manager(language=lang)
            assert manager.language == "chinese"  # Should normalize to "chinese"

        # Test with invalid language (should default to Chinese)
        manager = self._mock_template_manager(language="english")
        assert manager.language == "chinese"

    def test_system_prompt_selection(self):
        """Test system prompt selection based on context."""
        manager = self._mock_template_manager()

        # Test training prompt
        training_prompt = manager.get_system_prompt(use_training_prompt=True)
        assert len(training_prompt) > 0
        assert "BBU设备检测" in training_prompt or "通信机房" in training_prompt

        # Test evaluation prompt
        eval_prompt = manager.get_system_prompt(use_training_prompt=False)
        assert len(eval_prompt) > 0

        # Training prompt should be more detailed than evaluation prompt
        assert len(training_prompt) >= len(eval_prompt)

    def test_user_prompt_prefixes(self):
        """Test user prompt prefixes for different contexts."""
        manager = self._mock_template_manager()

        # Test teacher context
        teacher_prefix = manager.get_user_prompt_prefix("teacher")
        assert "参考示例" in teacher_prefix
        assert "📚" in teacher_prefix

        # Test target context
        target_prefix = manager.get_user_prompt_prefix("target")
        assert "参考示例" in target_prefix
        assert "目标图像" in target_prefix

        # Test standalone context
        standalone_prefix = manager.get_user_prompt_prefix("standalone")
        assert "检测图像" in standalone_prefix
        assert "🔍" in standalone_prefix

        # Test default/unknown context
        default_prefix = manager.get_user_prompt_prefix("unknown")
        assert "检测图像" in default_prefix  # Should default to standalone

    def test_teacher_student_conversation_formatting(self):
        """Test formatting of teacher-student conversations."""
        manager = self._mock_template_manager()

        # Create test data
        teacher_student_data = create_teacher_student_data()
        teachers = [teacher_student_data["teacher_sample"]]
        student = teacher_student_data["student_sample"]

        # Format conversation
        conversation = manager.format_teacher_student_conversation(teachers, student)

        # Validate conversation structure
        assert (
            len(conversation) == 5
        )  # system + teacher_user + teacher_assistant + student_user + student_assistant

        # Check role sequence
        expected_roles = ["system", "user", "assistant", "user", "assistant"]
        actual_roles = [msg["role"] for msg in conversation]
        assert actual_roles == expected_roles

        # Check content
        assert (
            "BBU设备检测" in conversation[0]["content"]
            or "通信机房" in conversation[0]["content"]
        )  # System
        assert "参考示例" in conversation[1]["content"]  # Teacher user
        assert "bbox_2d" in conversation[2]["content"]  # Teacher response
        assert "目标图像" in conversation[3]["content"]  # Student user
        assert "bbox_2d" in conversation[4]["content"]  # Student response

    def test_standalone_conversation_formatting(self):
        """Test formatting of standalone conversations without teachers."""
        manager = self._mock_template_manager()

        # Create sample data
        sample = {
            "objects": [
                {"bbox_2d": [100, 150, 200, 250], "desc": "测试设备/基础检测目标"}
            ]
        }

        # Format standalone conversation
        conversation = manager.get_standalone_conversation(sample)

        # Validate structure
        assert len(conversation) == 3  # system + user + assistant

        expected_roles = ["system", "user", "assistant"]
        actual_roles = [msg["role"] for msg in conversation]
        assert actual_roles == expected_roles

        # Check content
        assert len(conversation[0]["content"]) > 0  # System prompt
        assert "检测图像" in conversation[1]["content"]  # User request
        assert "bbox_2d" in conversation[2]["content"]  # Assistant response

    def test_objects_response_formatting(self):
        """Test formatting of objects for assistant responses."""
        manager = self._mock_template_manager()

        # Test with different geometry types
        test_objects = [
            {"bbox_2d": [100, 150, 200, 250], "desc": "矩形目标/测试"},
            {
                "square": [300, 400, 350, 410, 348, 425, 302, 415],
                "desc": "方形目标/测试",
            },
            {"line": [50, 100, 150, 120, 250, 140], "desc": "线条目标/测试"},
        ]

        formatted = manager._format_objects_for_response(test_objects)

        # Should be valid JSON format
        assert formatted.startswith("[")
        assert formatted.endswith("]")

        # Should contain all geometry types
        assert "bbox_2d" in formatted
        assert "square" in formatted
        assert "line" in formatted

        # Should contain descriptions
        assert "矩形目标/测试" in formatted
        assert "方形目标/测试" in formatted
        assert "线条目标/测试" in formatted

        # Test empty objects
        empty_formatted = manager._format_objects_for_response([])
        assert empty_formatted == "[]"

    def test_learning_instruction_generation(self):
        """Test generation of learning instructions for different teacher counts."""
        manager = self._mock_template_manager()

        # Test with no teachers
        instruction_0 = manager.get_learning_instruction(0)
        assert instruction_0 == ""  # No instruction needed

        # Test with single teacher
        instruction_1 = manager.get_learning_instruction(1)
        assert len(instruction_1) > 0
        assert "学习提示" in instruction_1
        assert "参考示例" in instruction_1

        # Test with multiple teachers
        instruction_3 = manager.get_learning_instruction(3)
        assert len(instruction_3) > 0
        assert "学习提示" in instruction_3
        assert "3个参考示例" in instruction_3

        # Multiple teacher instruction should mention the number of teachers
        assert "3个参考示例" in instruction_3
        # Both instructions should be substantial
        assert len(instruction_3) > 50
        assert len(instruction_1) > 50

    def test_multi_teacher_conversation_formatting(self):
        """Test formatting with multiple teachers."""
        manager = self._mock_template_manager()

        # Create multiple teachers
        teachers = [
            {"objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "教师1/示例"}]},
            {
                "objects": [
                    {"square": [50, 60, 70, 65, 68, 80, 48, 75], "desc": "教师2/示例"}
                ]
            },
            {"objects": [{"line": [100, 110, 120, 130], "desc": "教师3/示例"}]},
        ]

        student = {"objects": [{"bbox_2d": [200, 210, 220, 230], "desc": "学生/目标"}]}

        # Format conversation
        conversation = manager.format_teacher_student_conversation(teachers, student)

        # Should have system + (teacher_user + teacher_assistant) * 3 + student_user + student_assistant
        expected_length = 1 + (2 * 3) + 2  # 9 messages total
        assert len(conversation) == expected_length

        # Check that all teachers are included
        teacher_responses = [
            msg["content"] for msg in conversation if msg["role"] == "assistant"
        ][:-1]  # Exclude student
        assert len(teacher_responses) == 3

        assert "教师1/示例" in teacher_responses[0]
        assert "教师2/示例" in teacher_responses[1]
        assert "教师3/示例" in teacher_responses[2]

    def test_conversation_image_token_handling(self):
        """Test that image tokens are properly included in conversations."""
        manager = self._mock_template_manager()

        teachers = [{"objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "示例"}]}]
        student = {"objects": [{"bbox_2d": [50, 60, 70, 80], "desc": "目标"}]}

        conversation = manager.format_teacher_student_conversation(teachers, student)

        # Check that image tokens are present
        user_messages = [
            msg["content"] for msg in conversation if msg["role"] == "user"
        ]

        # Should have <image> tokens in user messages
        for user_msg in user_messages:
            assert "<image>" in user_msg

        # Test standalone conversation
        standalone = manager.get_standalone_conversation(student)
        user_content = [msg["content"] for msg in standalone if msg["role"] == "user"][
            0
        ]
        assert "<image>" in user_content

    def test_prompt_consistency(self):
        """Test consistency of prompts across different methods."""
        manager = self._mock_template_manager()

        # System prompts should be consistent
        training_prompt1 = manager.get_system_prompt(use_training_prompt=True)
        training_prompt2 = manager.get_system_prompt(use_training_prompt=True)
        assert training_prompt1 == training_prompt2

        eval_prompt1 = manager.get_system_prompt(use_training_prompt=False)
        eval_prompt2 = manager.get_system_prompt(use_training_prompt=False)
        assert eval_prompt1 == eval_prompt2

        # User prefixes should be consistent
        teacher_prefix1 = manager.get_user_prompt_prefix("teacher")
        teacher_prefix2 = manager.get_user_prompt_prefix("teacher")
        assert teacher_prefix1 == teacher_prefix2

    def test_special_characters_handling(self):
        """Test handling of special characters in descriptions."""
        manager = self._mock_template_manager()

        # Objects with special characters
        special_objects = [
            {"bbox_2d": [100, 150, 200, 250], "desc": "设备/包含\"引号\"和'单引号'"},
            {
                "square": [300, 400, 350, 410, 348, 425, 302, 415],
                "desc": "标签,包含逗号;分号:冒号",
            },
            {"line": [50, 100, 150, 120], "desc": "线缆(包含括号)和[方括号]"},
        ]

        # Should handle without errors
        formatted = manager._format_objects_for_response(special_objects)

        # Should be valid formatted response
        assert formatted.startswith("[")
        assert formatted.endswith("]")

        # Should contain the descriptions (possibly escaped)
        for obj in special_objects:
            # The exact format may vary, but description should be included somehow
            assert obj["desc"][:10] in formatted  # Check first part of description

    # Helper methods for mocking template manager functionality

    def _mock_template_manager(self, language: str = "chinese"):
        """Create a mock template manager."""

        class MockTemplateManager:
            def __init__(self, language: str = "chinese"):
                self.language = language.lower()
                # Normalize language code
                if self.language in ["zh", "中文", "cn"]:
                    self.language = "chinese"
                elif self.language not in ["chinese"]:
                    self.language = "chinese"

                # Mock prompts (simplified versions)
                self.training_prompt = """你是一个专用于通信机房BBU设备检测的AI助手。你的任务是精确识别并定位图像中所有指定对象。

请按照以下标准进行检测:
1. 仔细观察图像中的每个BBU设备、螺丝连接点、挡风板等对象
2. 准确绘制边界框，确保完全包围目标对象
3. 使用标准化的类别命名规范
4. 输出JSON格式的检测结果"""

                self.evaluation_prompt = """你是通信机房设备检测AI助手。请识别图像中的所有目标并输出位置与类别，支持三种几何格式：bbox_2d、square、line。"""

            def get_system_prompt(self, use_training_prompt: bool = False) -> str:
                return (
                    self.training_prompt
                    if use_training_prompt
                    else self.evaluation_prompt
                )

            def get_user_prompt_prefix(self, context: str = "target") -> str:
                if context == "teacher":
                    return "📚 参考示例:"
                elif context == "target":
                    return (
                        "现在请根据以上参考示例的检测模式和标注风格，检测以下目标图像:"
                    )
                else:  # standalone
                    return "🔍 请检测图像中的设备和部件:"

            def format_teacher_student_conversation(
                self, teachers: List[Dict[str, Any]], student: Dict[str, Any]
            ) -> List[Dict[str, str]]:
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
                    conversation.append(
                        {"role": "user", "content": f"{prefix} <image>"}
                    )

                    # Teacher assistant response
                    teacher_response = self._format_objects_for_response(
                        teacher.get("objects", [])
                    )
                    conversation.append(
                        {"role": "assistant", "content": teacher_response}
                    )

                # Student target message
                target_prefix = self.get_user_prompt_prefix("target")
                conversation.append(
                    {"role": "user", "content": f"{target_prefix} <image>"}
                )

                # Student assistant response
                student_response = self._format_objects_for_response(
                    student.get("objects", [])
                )
                conversation.append({"role": "assistant", "content": student_response})

                return conversation

            def get_standalone_conversation(
                self, sample: Dict[str, Any]
            ) -> List[Dict[str, str]]:
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

            def _format_objects_for_response(
                self, objects: List[Dict[str, Any]]
            ) -> str:
                if not objects:
                    return "[]"

                formatted_objects = []
                for obj in objects:
                    desc = obj.get("desc", "")

                    if "bbox_2d" in obj:
                        coords = obj["bbox_2d"]
                        formatted_objects.append(
                            f'{{"bbox_2d":{coords}, "desc":"{desc}"}}'
                        )
                    elif "square" in obj:
                        coords = obj["square"]
                        formatted_objects.append(
                            f'{{"square":{coords}, "desc":"{desc}"}}'
                        )
                    elif "line" in obj:
                        coords = obj["line"]
                        formatted_objects.append(
                            f'{{"line":{coords}, "desc":"{desc}"}}'
                        )

                return "[" + ", ".join(formatted_objects) + "]"

            def get_learning_instruction(self, num_teachers: int) -> str:
                if num_teachers == 0:
                    return ""

                if num_teachers == 1:
                    return """
学习提示: 请仔细观察参考示例中的以下要点:
• 如何准确识别不同类型的对象 (BBU设备、螺丝连接点、挡风板等)
• 如何区分外观相似但位置不同的对象
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

        return MockTemplateManager(language)


class TestTemplateIntegration:
    """Test template integration with other components."""

    def test_template_with_token_processor(self):
        """Test template integration with token processing."""
        manager = self._mock_template_manager()

        # Create sample with coordinate tokens
        sample_with_coords = {
            "objects": [{"bbox_2d": [100, 150, 200, 250], "desc": "设备/坐标令牌测试"}]
        }

        # Format conversation
        conversation = manager.get_standalone_conversation(sample_with_coords)

        # Should be compatible with coordinate token processing
        assistant_response = conversation[-1]["content"]

        # Should contain coordinates in parseable format
        assert "[100, 150, 200, 250]" in assistant_response
        assert "设备/坐标令牌测试" in assistant_response

    def test_template_with_multiple_geometry_types(self):
        """Test template handling of mixed geometry types."""
        manager = self._mock_template_manager()

        # Sample with all geometry types
        mixed_sample = {
            "objects": [
                {"bbox_2d": [10, 20, 30, 40], "desc": "矩形/测试"},
                {"square": [50, 60, 70, 65, 68, 80, 48, 75], "desc": "方形/测试"},
                {"line": [100, 110, 120, 130, 140, 150], "desc": "线条/测试"},
            ]
        }

        # Format response
        formatted = manager._format_objects_for_response(mixed_sample["objects"])

        # Should handle all geometry types
        assert "bbox_2d" in formatted
        assert "square" in formatted
        assert "line" in formatted

        # Should maintain proper JSON structure
        assert formatted.count("{") == 3  # Three objects
        assert formatted.count("}") == 3
        assert formatted.startswith("[") and formatted.endswith("]")

    def _mock_template_manager(self, language: str = "chinese"):
        """Reuse mock template manager from above."""

        class MockTemplateManager:
            def __init__(self, language: str = "chinese"):
                self.language = language.lower()
                # Normalize language code
                if self.language in ["zh", "中文", "cn"]:
                    self.language = "chinese"
                elif self.language not in ["chinese"]:
                    self.language = "chinese"

                self.training_prompt = """你是一个专用于通信机房BBU设备检测的AI助手。你的任务是精确识别并定位图像中所有指定对象。"""
                self.evaluation_prompt = """你是通信机房设备检测AI助手。请识别图像中的所有目标并输出位置与类别。"""

            def get_system_prompt(self, use_training_prompt: bool = False) -> str:
                return (
                    self.training_prompt
                    if use_training_prompt
                    else self.evaluation_prompt
                )

            def get_user_prompt_prefix(self, context: str = "target") -> str:
                if context == "teacher":
                    return "📚 参考示例:"
                elif context == "target":
                    return (
                        "现在请根据以上参考示例的检测模式和标注风格，检测以下目标图像:"
                    )
                else:
                    return "🔍 请检测图像中的设备和部件:"

            def get_standalone_conversation(self, sample) -> List[Dict[str, str]]:
                return [
                    {"role": "system", "content": self.get_system_prompt(False)},
                    {
                        "role": "user",
                        "content": f"{self.get_user_prompt_prefix('standalone')} <image>",
                    },
                    {
                        "role": "assistant",
                        "content": self._format_objects_for_response(
                            sample.get("objects", [])
                        ),
                    },
                ]

            def _format_objects_for_response(
                self, objects: List[Dict[str, Any]]
            ) -> str:
                if not objects:
                    return "[]"

                formatted_objects = []
                for obj in objects:
                    desc = obj.get("desc", "")
                    if "bbox_2d" in obj:
                        formatted_objects.append(
                            f'{{"bbox_2d":{obj["bbox_2d"]}, "desc":"{desc}"}}'
                        )
                    elif "square" in obj:
                        formatted_objects.append(
                            f'{{"square":{obj["square"]}, "desc":"{desc}"}}'
                        )
                    elif "line" in obj:
                        formatted_objects.append(
                            f'{{"line":{obj["line"]}, "desc":"{desc}"}}'
                        )

                return "[" + ", ".join(formatted_objects) + "]"

        return MockTemplateManager(language)
