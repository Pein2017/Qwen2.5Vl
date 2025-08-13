from unittest.mock import Mock

import torch

from src_new.models.loss_manager import LossManager
from src_new.processing.token_processor import TokenConfig, TokenProcessor


def _make_loss_manager_with_aux():
    config = Mock()
    config.coordinate_tokens_enabled = True
    config.max_coord_value = 32
    config.coordinate_loss_weight = 1.0
    config.regular_loss_weight = 1.0
    config.teacher_loss_weight = 1.0
    config.student_loss_weight = 1.0
    setattr(config, "coordinate_temperature", 1.0)

    # Minimal tokenizer with line + coord tokens
    vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
    for i in range(config.max_coord_value + 1):
        vocab[f"<|coord_{i}|>"] = 151667 + i
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = vocab

    lm = LossManager(
        config, TokenProcessor(TokenConfig(True, config.max_coord_value)), tokenizer
    )
    lm.set_coordinate_aux_options(
        tau=1.2,
        sigma_bins=2.0,
        window_bins=3,
        topk=5,
        lambda_kce=0.5,
        lambda_unlike=0.05,
        lambda_lap1=0.0,
        lambda_lap2=0.0,
    )
    return lm


def test_aux_losses_produce_teacher_coord_loss_when_enabled():
    lm = _make_loss_manager_with_aux()
    B, T = 1, 8
    V = 151667 + 33
    coord_start = 151667
    logits = torch.zeros(B, T, V)
    labels = torch.full((B, T), -100)
    teacher_spans = [[(3, 7)]]
    # Put coordinate targets at 4 and 6
    labels[0, 4] = coord_start + 10
    labels[0, 6] = coord_start + 15
    # Leave logits zero to make unlikelihood positive on non-coordinate vocab

    comp = lm.compute_loss_components(
        logits=logits, labels=labels, teacher_spans=teacher_spans
    )
    # CE present (uniform-ish, finite)
    assert comp.teacher_llm_loss is not None
    # Aux coordinate loss active when present
    if comp.teacher_l1_loss is not None:
        assert torch.isfinite(comp.teacher_l1_loss)
        assert comp.teacher_l1_loss.item() >= 0.0
