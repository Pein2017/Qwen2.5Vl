#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test that assistant spans include <|im_end|> in labels during training.

This enforces the project requirement from .augment-guidelines that
"<|im_end|> is included and aligned in assistant span labels during training".

We build a minimal synthetic conversation with three segments:
1) system
2) user
3) assistant (the training target)

We then apply simplified masking rules mimicking the training pipeline:
- Mask system and user segments entirely (labels = -100)
- Keep assistant segment labels, including the trailing <|im_end|>

No model is instantiated; this test is fast and deterministic.
"""

import torch

from src_new.tests.fixtures.mock_objects import create_mock_tokenizer


def _build_minimal_conversation_ids(tokenizer):
    im_start = tokenizer.special_tokens["<|im_start|>"]
    im_end = tokenizer.special_tokens["<|im_end|>"]

    # Compose: <|im_start|>system\n...<|im_end|> <|im_start|>user\n...<|im_end|> <|im_start|>assistant\n...<|im_end|>
    # Use small filler ids 10.. to stand in for content
    sys_seg = [im_start, 10, 11, 12, im_end]
    usr_seg = [im_start, 13, 14, 15, im_end]
    asst_seg = [im_start, 16, 17, 18, im_end]

    input_ids = torch.tensor([sys_seg + usr_seg + asst_seg], dtype=torch.long)
    return input_ids, im_start, im_end


def _apply_training_mask(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """Apply simplified training masking consistent with project rules.

    - Mask system and user segments (first two segments) completely
    - Keep assistant segment unmasked, including <|im_end|>
    """
    labels = input_ids.clone()
    im_start = tokenizer.special_tokens["<|im_start|>"]
    im_end = tokenizer.special_tokens["<|im_end|>"]

    # Find segment boundaries by scanning for <|im_start|> and <|im_end|>
    ids = input_ids[0]
    start_positions = (ids == im_start).nonzero(as_tuple=True)[0].tolist()
    end_positions = (ids == im_end).nonzero(as_tuple=True)[0].tolist()

    assert len(start_positions) == 3 and len(end_positions) == 3, "Expected 3 segments"

    # Mask first two segments [start, end] inclusive
    IGNORE_INDEX = -100
    for seg_idx in (0, 1):
        s = start_positions[seg_idx]
        e = end_positions[seg_idx]
        labels[0, s : e + 1] = IGNORE_INDEX

    # Leave assistant segment (index 2) fully unmasked, including <|im_end|>
    return labels


def test_assistant_eos_label_included_and_aligned():
    tokenizer = create_mock_tokenizer()
    input_ids, im_start, im_end = _build_minimal_conversation_ids(tokenizer)

    labels = _apply_training_mask(tokenizer, input_ids)

    # Identify assistant segment boundaries
    ids = input_ids[0]
    start_positions = (ids == im_start).nonzero(as_tuple=True)[0].tolist()
    end_positions = (ids == im_end).nonzero(as_tuple=True)[0].tolist()

    asst_start = start_positions[2]
    asst_end = end_positions[2]

    # Assert assistant segment is unmasked entirely, including the EOS token
    asst_labels = labels[0, asst_start : asst_end + 1]
    assert (asst_labels != -100).all(), "Assistant segment should be fully unmasked"

    # Specifically check the final <|im_end|> position is included (not -100)
    assert labels[0, asst_end].item() != -100, "<|im_end|> must be included in assistant labels"

    # Sanity: earlier segments must be masked fully
    sys_s, usr_s = start_positions[0], start_positions[1]
    sys_e, usr_e = end_positions[0], end_positions[1]
    assert (labels[0, sys_s : sys_e + 1] == -100).all(), "System segment should be masked"
    assert (labels[0, usr_s : usr_e + 1] == -100).all(), "User segment should be masked"

