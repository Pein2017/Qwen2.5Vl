#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for DetectionModel.config proxy behavior and separation from training_config.

- DetectionModel.config must proxy the underlying HF model config so that
  integrations that call model.config.to_json_string() keep working.
- training_config should be stored separately and not override model.config.
"""

from unittest.mock import Mock

import pytest

from src_new.models.wrapper import DetectionModel


class DummyConfig:
    # Enable coordinate tokens to avoid coordinate loss initialization errors
    coordinate_tokens_enabled = True
    max_coord_value = 1024
    use_cache = False
    torch_dtype = "float32"
    attn_implementation = "eager"


@pytest.fixture
def tokenizer_1024(real_extended_tokenizer):
    """Provide a tokenizer with exactly 0..1024 coord tokens.
    Prefer the real expanded tokenizer; if unavailable (skipped), build a mock.
    """
    try:
        return real_extended_tokenizer
    except Exception:
        # Build a minimal mock tokenizer with required API
        tok = Mock()
        vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
        for i in range(1025):
            vocab[f"<|coord_{i}|>"] = 151667 + i
        tok.get_vocab.return_value = vocab
        return tok


def test_config_proxy_and_training_config_separation(tokenizer_1024):
    from src_new.tests.fixtures.mock_objects import create_mock_model

    base_model = create_mock_model()

    # Give base model config a sentinel method/attr
    base_model.config.to_json_string = Mock(return_value='{"ok": true}')
    base_model.config.hidden_size = 2048

    train_cfg = DummyConfig()

    model = DetectionModel(
        base_model=base_model,
        config=train_cfg,
        tokenizer=tokenizer_1024,
        skip_expansion=True,
    )

    # DetectionModel.config must be the HF config
    assert model.config is base_model.config
    assert callable(model.config.to_json_string)
    assert model.config.to_json_string() == '{"ok": true}'

    # training_config must be the user-provided config object
    assert model.training_config is train_cfg

    # Setting config via property should keep proxying behavior and store into training_config
    new_cfg = DummyConfig()
    model.config = (
        new_cfg  # setter stores on training_config and preserves HF config on ._config
    )

    assert model.training_config is new_cfg
    assert model.config is base_model.config  # still proxying HF config
