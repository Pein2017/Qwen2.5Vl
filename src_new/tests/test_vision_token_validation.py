#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for strict vision token validation in DetectionModel.forward.

Covers:
- Expected image token count vs <|image_pad|> occurrences
- Patch rows in pixel_values vs sum(t*h*w) from image_grid_thw

Follows rules in .augment-guidelines:
- Fail-fast on mismatches with actionable messages
- Deterministic, synthetic inputs; no network/filesystem
"""

import os

import pytest
import torch

from src_new.models.wrapper import DetectionModel
from src_new.tests.fixtures.mock_objects import create_mock_model, create_mock_tokenizer


def _load_real_or_mock():
    """Try to load real Qwen2.5-VL model/tokenizer from cache; fallback to mocks.

    Uses environment variable QWEN_VL_LOCAL_PATH if set, otherwise attempts
    the default HF id with local_files_only=True to avoid any network calls.
    Returns (model, tokenizer).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        # Transformers not available; use mocks
        return create_mock_model(), create_mock_tokenizer()

    local_path = os.environ.get(
        "QWEN_VL_LOCAL_PATH",
        "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024",
    )
    try:
        tok = AutoTokenizer.from_pretrained(
            local_path, local_files_only=True, trust_remote_code=True
        )
        # Model is heavy; only load if truly cached to avoid delays
        mdl = AutoModelForCausalLM.from_pretrained(
            local_path, local_files_only=True, trust_remote_code=True
        )
        return mdl, tok
    except Exception:
        # Fallback to mocks: build a tokenizer with coord_0..1024 only
        from unittest.mock import Mock

        mdl = create_mock_model()
        vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
        for i in range(1025):
            vocab[f"<|coord_{i}|>"] = 151667 + i
        tok = Mock()
        tok.get_vocab.return_value = vocab
        return mdl, tok


class DummyConfig:
    # Minimal training config required by DetectionModel
    # Coordinate tokens enabled; expanded cache provides them
    coordinate_tokens_enabled = True
    max_coord_value = 1024
    # Merge size fallback used when HF vision_config is not present
    merge_size = 2
    # Misc attributes occasionally accessed in code paths
    use_cache = False
    torch_dtype = "float32"
    attn_implementation = "eager"
    # Assume pre-expanded tokenizer
    skip_vocab_extension = True


def _make_inputs(num_images: int = 2, grid_per_image=(1, 2, 2)):
    """Create consistent synthetic multimodal inputs.

    - image_grid_thw: [num_images, 3]
    - pixel_values: [sum_i (t*h*w), hidden]
    - input_ids containing the correct number of <|image_pad|> tokens
    """
    grid = torch.tensor(
        [list(grid_per_image) for _ in range(num_images)], dtype=torch.long
    )
    # tokens per image = (t*h*w)//(merge_size**2) with default merge_size=2 -> //4
    per_image_tokens = (grid[:, 0] * grid[:, 1] * grid[:, 2]) // 4
    expected_image_tokens = int(per_image_tokens.sum().item())

    # Build input ids with exactly expected_image_tokens image tokens (id 151655) scattered
    seq_len = max(8, expected_image_tokens + 6)
    input_ids = torch.randint(low=10, high=100, size=(1, seq_len))
    image_token_id = 151655  # <|image_pad|>
    # Place image tokens in the tail for simplicity
    for i in range(expected_image_tokens):
        input_ids[0, -1 - i] = image_token_id

    # Build pixel_values flattened patches (rows == sum_i t*h*w)
    num_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
    hidden = 1024
    pixel_values = torch.randn(num_patches, hidden)

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
    }, expected_image_tokens


def test_image_token_count_validation_passes():
    # Use real cached components if available; otherwise mocks with coordinate tokens
    base_model, tokenizer = _load_real_or_mock()
    # DetectionModel will handle validation; tokenizer is already expanded in cache
    model = DetectionModel(
        base_model=base_model,
        config=DummyConfig(),
        tokenizer=tokenizer,
        skip_expansion=True,
    )

    inputs, expected_image_tokens = _make_inputs(num_images=2)

    # Should not raise when counts match
    _ = model(**inputs)


def test_image_token_count_validation_mismatch_raises():
    base_model, tokenizer = _load_real_or_mock()
    model = DetectionModel(
        base_model=base_model,
        config=DummyConfig(),
        tokenizer=tokenizer,
        skip_expansion=True,
    )

    inputs, expected_image_tokens = _make_inputs(num_images=2)

    # Force a mismatch by adding one more <|image_pad|>
    image_token_id = 151655
    inputs["input_ids"][0, 0] = image_token_id

    with pytest.raises(ValueError) as ei:
        _ = model(**inputs)

    msg = str(ei.value)
    assert "Image token count mismatch" in msg
    assert f"expected={expected_image_tokens}" in msg


def test_pixel_values_rows_vs_grid_mismatch_raises():
    base_model, tokenizer = _load_real_or_mock()
    model = DetectionModel(
        base_model=base_model,
        config=DummyConfig(),
        tokenizer=tokenizer,
        skip_expansion=True,
    )

    inputs, _ = _make_inputs(num_images=1)

    # Corrupt pixel_values to have wrong row count (+1)
    pv = inputs["pixel_values"]
    inputs["pixel_values"] = torch.cat([pv, torch.randn(1, pv.shape[1])], dim=0)

    with pytest.raises(ValueError) as ei:
        _ = model(**inputs)

    msg = str(ei.value)
    assert "pixel_values rows" in msg
    assert "sum(t*h*w) from image_grid_thw" in msg
