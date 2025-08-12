#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation processing validation and failure-mode tests.
"""

import pytest

from src_new.processing.conversation_processor import (
    ConversationProcessor,
    TeacherStudentValidationError,
)
from src_new.processing.coordinate_converter import CoordinateTokenConverter


def _make_cp():
    # Minimal setup; no HF processor calls are used in these validation paths
    class DummyCfg:
        coordinate_tokens_enabled = True
        max_coord_value = 1024

    # Provide a dummy processor; we won't call it in these tests
    class DummyProc:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("should not be called in validation tests")

    cp = ConversationProcessor(processor=DummyProc(), max_coord_value=1024)
    cp.coordinate_converter = CoordinateTokenConverter(max_coord_value=1024)
    return cp


def test_teacher_student_validation_empty_teacher_objects_raises():
    cp = _make_cp()
    student = {"objects": [{"bbox_2d": [1, 2, 3, 4], "desc": "d"}]}
    teachers = [{"objects": []}]
    images_student = [object()]
    images_teachers = [[object()]]
    # With enable_recovery=False, empty teacher objects triggers validation error
    with pytest.raises(TeacherStudentValidationError) as ei:
        cp.build_teacher_student_conversation_robust(
            student_sample=student,
            teacher_samples=teachers,
            student_images=images_student,
            teacher_images_list=images_teachers,
            enable_recovery=False,
        )
    assert "must contain non-empty objects list" in str(ei.value)


def test_teacher_student_validation_student_images_count_raises():
    cp = _make_cp()
    student = {"objects": [{"bbox_2d": [1, 2, 3, 4], "desc": "d"}]}
    teachers = [{"objects": [{"bbox_2d": [1, 2, 3, 4], "desc": "t"}]}]
    images_student = []  # invalid: must be exactly 1
    images_teachers = [[object()]]
    with pytest.raises(TeacherStudentValidationError) as ei:
        cp.build_teacher_student_conversation_robust(
            student_sample=student,
            teacher_samples=teachers,
            student_images=images_student,
            teacher_images_list=images_teachers,
        )
    assert "exactly 1 image" in str(ei.value)


def test_teacher_student_validation_student_objects_missing_raises():
    cp = _make_cp()
    student = {"objects": []}
    teachers = [{"objects": [{"bbox_2d": [1, 2, 3, 4], "desc": "t"}]}]
    images_student = [object()]
    images_teachers = [[object()]]
    with pytest.raises(TeacherStudentValidationError) as ei:
        cp.build_teacher_student_conversation_robust(
            student_sample=student,
            teacher_samples=teachers,
            student_images=images_student,
            teacher_images_list=images_teachers,
        )
    assert "Student sample must contain non-empty objects list" in str(ei.value)
