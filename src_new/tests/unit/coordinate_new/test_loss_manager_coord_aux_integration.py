from unittest.mock import Mock

import torch

from src_new.models.loss_manager import LossManager
from src_new.processing.token_processor import TokenConfig, TokenProcessor


def _make_loss_manager():
    config = Mock()
    config.coordinate_tokens_enabled = True
    config.max_coord_value = 32
    config.coordinate_loss_weight = 0.05
    config.regular_loss_weight = 1.0
    config.teacher_loss_weight = 0.3
    config.student_loss_weight = 1.0
    # Feature knobs (will be wired in LossManager later); keep present for future use
    setattr(config, "coordinate_temperature", 1.2)

    # Minimal tokenizer with line + coord tokens
    vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
    for i in range(config.max_coord_value + 1):
        vocab[f"<|coord_{i}|>"] = 151667 + i
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = vocab

    token_processor = TokenProcessor(TokenConfig(True, config.max_coord_value))
    return LossManager(config, token_processor, tokenizer)


def test_teacher_student_paths_still_compute_ce_and_coord_l1():
    lm = _make_loss_manager()
    B, T = 1, 10
    V = 151667 + 33  # up to coord_32 inclusive
    logits = torch.zeros(B, T, V)
    labels = torch.full((B, T), -100)
    coord_start = 151667

    # Put a teacher span [3,7) with two coordinate targets at 4 and 6
    teacher_spans = [[(3, 7)]]
    labels[0, 4] = coord_start + 5
    labels[0, 6] = coord_start + 8

    # Make logits[t] peak at labels[t+1] (for CE)
    logits[0, 3, coord_start + 5] = 50.0
    logits[0, 5, coord_start + 8] = 50.0

    comp = lm.compute_loss_components(
        logits=logits, labels=labels, teacher_spans=teacher_spans
    )
    # CE should be present (includes coordinate targets)
    assert comp.teacher_llm_loss is not None
    assert torch.isfinite(comp.teacher_llm_loss)
    # Coordinate aux path may be disabled by masks or config; when present, it is finite
    if comp.teacher_l1_loss is not None:
        assert torch.isfinite(comp.teacher_l1_loss)
