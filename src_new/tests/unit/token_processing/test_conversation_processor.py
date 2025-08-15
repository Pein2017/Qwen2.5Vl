#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for ConversationProcessor.

Tests the HuggingFace-first conversation processing that replaces
800+ lines of custom conversation logic.
"""

import unittest
from unittest.mock import Mock

import torch
from PIL import Image

from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.templates import CONSTANTS


class TestConversationProcessor(unittest.TestCase):
    """Test suite for ConversationProcessor with 100% line coverage."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock processor
        self.mock_processor = Mock()

        # Make apply_chat_template include image placeholders to satisfy validation
        def _mock_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, images=None, **kwargs
        ):
            num_imgs = len(images) if images is not None else 0
            return ("<|image_pad|>" * num_imgs) + " mocked_chat_text"

        self.mock_processor.apply_chat_template.side_effect = _mock_apply_chat_template

        # Mock processor call returns
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(4, 1024),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        self.mock_processor.return_value = mock_inputs

        self.conversation_processor = ConversationProcessor(
            processor=self.mock_processor,
            max_coord_value=1024,
            coordinate_tokens_enabled=True,
        )

    def test_initialization_valid(self):
        """Test valid initialization."""
        processor = ConversationProcessor(
            self.mock_processor, max_coord_value=1024, coordinate_tokens_enabled=True
        )
        self.assertEqual(processor.processor, self.mock_processor)
        self.assertEqual(processor.coordinate_converter.max_coord_value, 1024)

    def test_initialization_none_processor(self):
        """Test initialization with None processor raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ConversationProcessor(None, max_coord_value=1025)
        self.assertIn("processor cannot be None", str(cm.exception))

    def test_simple_conversation_creation(self):
        """Test simple conversation uses official HF format."""
        sample = {"objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "test_device"}]}

        # Create mock images
        mock_image = Image.new("RGB", (100, 100), color="red")
        images = [mock_image]

        result = self.conversation_processor.create_simple_conversation(sample, images)

        # Verify official processor was called
        self.mock_processor.apply_chat_template.assert_called_once()
        self.mock_processor.assert_called_once()

        # Verify conversation structure
        call_args = self.mock_processor.apply_chat_template.call_args[0][0]
        self.assertEqual(call_args[0]["role"], "system")
        self.assertEqual(call_args[1]["role"], "user")
        self.assertEqual(call_args[2]["role"], "assistant")

        # Verify prompts are imported from CONSTANTS, not hardcoded
        self.assertEqual(call_args[0]["content"], CONSTANTS["SYSTEM_PROMPT"])
        self.assertEqual(
            call_args[1]["content"][0]["text"], CONSTANTS["STUDENT_USER_PROMPT"]
        )

        # Verify result structure
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

    def test_simple_conversation_invalid_sample(self):
        """Test simple conversation with invalid sample raises ValueError."""
        mock_image = Image.new("RGB", (100, 100), color="red")

        # Test non-dict sample
        from src_new.processing.conversation_processor import ConversationStructureError

        with self.assertRaises(ConversationStructureError) as cm:
            self.conversation_processor.create_simple_conversation(
                "not_dict", [mock_image]
            )
        self.assertIn("sample must be a dict", str(cm.exception))

    def test_simple_conversation_invalid_images(self):
        """Test simple conversation with invalid images raises ValueError."""
        sample = {"objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "test"}]}

        # Test non-list images
        from src_new.processing.conversation_processor import ConversationStructureError

        with self.assertRaises(ConversationStructureError) as cm:
            self.conversation_processor.create_simple_conversation(sample, "not_list")
        self.assertIn("images must be a list", str(cm.exception))

    def test_simple_conversation_empty_objects(self):
        """Test simple conversation with empty objects raises ValueError."""
        sample = {"objects": []}
        mock_image = Image.new("RGB", (100, 100), color="red")

        from src_new.processing.conversation_processor import ConversationStructureError

        with self.assertRaises(ConversationStructureError) as cm:
            self.conversation_processor.create_simple_conversation(sample, [mock_image])
        self.assertIn("Sample must contain non-empty objects list", str(cm.exception))

    def test_teacher_student_conversation(self):
        """Test teacher-student uses standard multi-turn format with imported prompts."""
        student_sample = {
            "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "student_device"}]
        }
        teacher_samples = [
            {"objects": [{"bbox_2d": [50, 60, 70, 80], "desc": "teacher_device"}]}
        ]

        mock_image = Image.new("RGB", (100, 100), color="red")
        student_images = [mock_image]
        teacher_images_list = [[mock_image]]

        result = self.conversation_processor.create_teacher_student_conversation(
            student_sample, teacher_samples, student_images, teacher_images_list
        )

        # Verify multi-turn conversation structure
        self.mock_processor.apply_chat_template.assert_called_once()
        call_args = self.mock_processor.apply_chat_template.call_args[0][0]

        self.assertEqual(call_args[0]["role"], "system")
        self.assertEqual(call_args[1]["role"], "user")  # Teacher user
        self.assertEqual(call_args[2]["role"], "assistant")  # Teacher response
        self.assertEqual(call_args[3]["role"], "user")  # Student user

        # Verify prompts are imported from CONSTANTS, not hardcoded
        self.assertEqual(call_args[0]["content"], CONSTANTS["SYSTEM_PROMPT"])
        self.assertEqual(
            call_args[1]["content"][0]["text"], CONSTANTS["TEACHER_USER_PROMPT"]
        )
        self.assertEqual(
            call_args[3]["content"][0]["text"], CONSTANTS["STUDENT_USER_PROMPT"]
        )

    def test_teacher_student_invalid_student_sample(self):
        """Test teacher-student with invalid student sample raises ValueError."""
        teacher_samples = [
            {"objects": [{"bbox_2d": [50, 60, 70, 80], "desc": "teacher"}]}
        ]
        mock_image = Image.new("RGB", (100, 100), color="red")

        from src_new.processing.conversation_processor import (
            TeacherStudentValidationError,
        )

        with self.assertRaises(TeacherStudentValidationError) as cm:
            self.conversation_processor.create_teacher_student_conversation(
                "not_dict", teacher_samples, [mock_image], [[mock_image]]
            )
        self.assertIn("student_sample must be a dict", str(cm.exception))

    def test_teacher_student_empty_teacher_samples(self):
        """Test teacher-student with empty teacher samples raises ValueError."""
        student_sample = {
            "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "student"}]
        }
        mock_image = Image.new("RGB", (100, 100), color="red")

        from src_new.processing.conversation_processor import (
            TeacherStudentValidationError,
        )

        with self.assertRaises(TeacherStudentValidationError) as cm:
            self.conversation_processor.create_teacher_student_conversation(
                student_sample, [], [mock_image], []
            )
        self.assertIn("teacher_samples must be a non-empty list", str(cm.exception))

    def test_teacher_student_mismatch_lengths(self):
        """Test teacher-student with mismatched teacher samples and images raises ValueError."""
        student_sample = {
            "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "student"}]
        }
        teacher_samples = [
            {"objects": [{"bbox_2d": [50, 60, 70, 80], "desc": "teacher"}]}
        ]
        mock_image = Image.new("RGB", (100, 100), color="red")

        from src_new.processing.conversation_processor import (
            TeacherStudentValidationError,
        )

        with self.assertRaises(TeacherStudentValidationError) as cm:
            self.conversation_processor.create_teacher_student_conversation(
                student_sample,
                teacher_samples,
                [mock_image],
                [],  # Empty teacher images
            )
        self.assertIn(
            "Mismatch: 1 teacher samples but 0 teacher image lists", str(cm.exception)
        )

    def test_teacher_student_invalid_teacher_sample(self):
        """Test teacher-student with invalid teacher sample raises ValueError."""
        student_sample = {
            "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "student"}]
        }
        teacher_samples = ["not_dict"]  # Invalid teacher sample
        mock_image = Image.new("RGB", (100, 100), color="red")

        from src_new.processing.conversation_processor import (
            TeacherStudentValidationError,
        )

        with self.assertRaises(TeacherStudentValidationError) as cm:
            self.conversation_processor.create_teacher_student_conversation(
                student_sample, teacher_samples, [mock_image], [[mock_image]]
            )
        self.assertIn("Teacher sample 0 must be a dict", str(cm.exception))

    def test_teacher_student_empty_teacher_objects(self):
        """Test teacher-student with empty teacher objects raises ValueError."""
        student_sample = {
            "objects": [{"bbox_2d": [100, 200, 150, 250], "desc": "student"}]
        }
        teacher_samples = [{"objects": []}]  # Empty teacher objects
        mock_image = Image.new("RGB", (100, 100), color="red")

        from src_new.processing.conversation_processor import (
            TeacherStudentValidationError,
        )

        with self.assertRaises(TeacherStudentValidationError) as cm:
            self.conversation_processor.build_teacher_student_conversation_robust(
                student_sample=student_sample,
                teacher_samples=teacher_samples,
                student_images=[mock_image],
                teacher_images_list=[[mock_image]],
                enable_recovery=False,
            )
        self.assertIn(
            "Teacher sample 0 must contain non-empty objects list", str(cm.exception)
        )

    def test_teacher_student_empty_student_objects(self):
        """Test teacher-student with empty student objects raises ValueError."""
        student_sample = {"objects": []}  # Empty student objects
        teacher_samples = [
            {"objects": [{"bbox_2d": [50, 60, 70, 80], "desc": "teacher"}]}
        ]
        mock_image = Image.new("RGB", (100, 100), color="red")

        from src_new.processing.conversation_processor import (
            TeacherStudentValidationError,
        )

        with self.assertRaises(TeacherStudentValidationError) as cm:
            self.conversation_processor.create_teacher_student_conversation(
                student_sample, teacher_samples, [mock_image], [[mock_image]]
            )
        self.assertIn(
            "Student sample must contain non-empty objects list", str(cm.exception)
        )

    def test_inference_conversation(self):
        """Test inference conversation creation."""
        user_prompt = "请检测图像中的设备"
        mock_image = Image.new("RGB", (100, 100), color="red")
        images = [mock_image]

        result = self.conversation_processor.create_inference_conversation(
            user_prompt, images
        )

        # Verify official processor was called
        self.mock_processor.apply_chat_template.assert_called_once()
        self.mock_processor.assert_called_once()

        # Verify conversation structure
        call_args = self.mock_processor.apply_chat_template.call_args[0][0]
        self.assertEqual(call_args[0]["role"], "system")
        self.assertEqual(call_args[1]["role"], "user")

        # Verify system prompt is imported from CONSTANTS
        self.assertEqual(call_args[0]["content"], CONSTANTS["SYSTEM_PROMPT"])
        self.assertEqual(call_args[1]["content"][0]["text"], user_prompt)

    def test_inference_conversation_invalid_prompt(self):
        """Test inference conversation with invalid prompt raises ValueError."""
        mock_image = Image.new("RGB", (100, 100), color="red")

        # Test empty prompt
        from src_new.processing.conversation_processor import ConversationStructureError

        with self.assertRaises(ConversationStructureError) as cm:
            self.conversation_processor.create_inference_conversation("", [mock_image])
        self.assertIn("user_prompt must be a non-empty string", str(cm.exception))

        # Test non-string prompt
        with self.assertRaises(ConversationStructureError) as cm:
            self.conversation_processor.create_inference_conversation(123, [mock_image])
        self.assertIn("user_prompt must be a non-empty string", str(cm.exception))

    def test_inference_conversation_invalid_images(self):
        """Test inference conversation with invalid images raises ValueError."""
        user_prompt = "请检测图像中的设备"

        from src_new.processing.conversation_processor import ConversationStructureError

        # Test empty images
        with self.assertRaises(ConversationStructureError) as cm:
            self.conversation_processor.create_inference_conversation(user_prompt, [])
        self.assertIn("images must be a non-empty list", str(cm.exception))

        # Test non-list images
        with self.assertRaises(ConversationStructureError) as cm:
            self.conversation_processor.create_inference_conversation(
                user_prompt, "not_list"
            )
        self.assertIn("images must be a non-empty list", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
