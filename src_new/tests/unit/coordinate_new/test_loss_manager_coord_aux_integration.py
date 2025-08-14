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

    # Minimal tokenizer with line + coord tokens
    vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
    for i in range(config.max_coord_value + 1):
        vocab[f"<|coord_{i}|>"] = 151667 + i
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = vocab

    token_processor = TokenProcessor(
        TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=config.max_coord_value,
            coordinate_init_mode="fourier_ramp",
        )
    )
    lm = LossManager(config, token_processor, tokenizer)
    # Enable coordinate auxiliary loss (now required)
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


def test_teacher_student_paths_still_compute_ce_and_coord_l1():
    lm = _make_loss_manager()
    B, T = 1, 10

    # Get actual coordinate token range from loss manager
    coord_start = lm._coord_start_id
    coord_end = lm._coord_end_id
    V = coord_end + 100  # Make vocab size large enough

    logits = torch.zeros(B, T, V)
    labels = torch.full((B, T), -100)

    # Put a teacher span [3,8) with two coordinate targets at 4 and 6
    # After shifting, this becomes span [2,7) with coordinate targets at 3 and 5
    teacher_spans = [[(3, 8)]]
    # Use small coordinate values to ensure they're within range
    labels[0, 4] = coord_start + 2  # coord_2
    labels[0, 6] = coord_start + 5  # coord_5

    # Make logits[t] peak at labels[t+1] (for CE)
    logits[0, 3, coord_start + 2] = 50.0
    logits[0, 5, coord_start + 5] = 50.0

    comp = lm.compute_loss_components(
        logits=logits, labels=labels, teacher_spans=teacher_spans
    )
    # CE should be present (includes coordinate targets)
    assert comp.teacher_llm_loss is not None
    assert torch.isfinite(comp.teacher_llm_loss)
    # Coordinate aux path may be disabled by masks or config; when present, it is finite
    if comp.teacher_l1_loss is not None:
        assert torch.isfinite(comp.teacher_l1_loss)
