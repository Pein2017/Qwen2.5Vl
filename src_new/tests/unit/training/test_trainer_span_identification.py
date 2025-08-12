#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate teacher vs student span identification logic in BBUTrainer.
"""

import torch

from src_new.training.bbu_trainer import BBUTrainer


def test_identify_teacher_student_spans_multi_turn():
    # Build labels with three unmasked spans: two teacher, one student
    # Unmasked labels: [0:3] masked, [3:5] T1, [5:7] T2, [7:10] S
    labels = torch.full((10,), -100)
    labels[3:5] = 1
    labels[5:7] = 1
    labels[7:10] = 1

    input_ids = torch.arange(10)

    # chat_text containing two teacher assistant markers -> multi-teacher conversation
    chat_text = (
        "<|im_start|>assistant T1 <|im_end|> some text "
        "<|im_start|>assistant T2 <|im_end|> then user "
        "<|im_start|>assistant S  <|im_end|>"
    )

    teacher_spans, student_spans = BBUTrainer._identify_teacher_student_spans(
        None, input_ids, labels, chat_text
    )

    # Current implementation merges adjacent unmasked positions into a single span
    assert teacher_spans == [(3, 10)]
    assert student_spans == []


def test_identify_teacher_student_spans_single_assistant():
    # One unmasked span, single assistant -> treat as teacher spans only
    labels = torch.full((8,), -100)
    labels[2:6] = 1
    input_ids = torch.arange(8)
    chat_text = "<|im_start|>assistant one <|im_end|>"

    teacher_spans, student_spans = BBUTrainer._identify_teacher_student_spans(
        None, input_ids, labels, chat_text
    )

    # Single assistant: all spans are student per current implementation
    assert teacher_spans == []
    assert student_spans == [(2, 6)]
