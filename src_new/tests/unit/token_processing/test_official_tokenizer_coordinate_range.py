#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the hardcoded coordinate token range against official Qwen2.5-VL tokenizers.

Checks that <|coord_0|>.. tokens map to IDs in the fixed range [151667, 152691] for max_coord_value=1024.
"""

import os

import pytest
from transformers import AutoTokenizer


OFFICIAL_MODELS = [
    "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024",
]

HARD_START = 151667
HARD_END_INCLUSIVE = 151667 + 1024


def _find_coord_range(tokenizer):
    vocab = tokenizer.get_vocab()
    # Validate presence of base geometry tokens first; fail-fast if missing
    required = ["<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>"]
    for tok in required:
        if tok not in vocab:
            raise AssertionError(
                f"Required geometry token missing in tokenizer vocab: {tok}"
            )

    ids = []
    for i in range(1025):  # 0..1024 inclusive
        tok = f"<|coord_{i}|>"
        if tok in vocab:
            ids.append(vocab[tok])
    if not ids:
        raise AssertionError(
            "No coordinate tokens found in tokenizer; cannot validate range"
        )
    return min(ids), max(ids)


@pytest.mark.parametrize("model_path", OFFICIAL_MODELS)
def test_official_tokenizer_coordinate_range(model_path):
    # Load offline without internet; skip if not present locally
    if not os.path.exists(model_path):
        pytest.skip(f"Local model cache not found: {model_path}")

    # Prefer fast tokenizer; if it fails, provide clear guidance
    try:
        tok = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True, trust_remote_code=True
        )
    except Exception as e:
        # Try slow tokenizer to extract more context for debugging; proceed if it loads
        try:
            tok_slow = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                use_fast=False,
                trust_remote_code=True,
            )
            # Proceed with slow tokenizer (fast tokenizer.json may be incompatible on some caches)
            tok = tok_slow
        except Exception as e2:
            raise AssertionError(
                f"Failed to load both fast and slow tokenizers for {model_path}.\n"
                f"Fast error: {e}\nSlow error: {e2}\n"
                f"Action: Refresh the local tokenizer cache (tokenizer.json + tokenizer_config.json)."
            )

    # Validate geometry tokens exist
    vocab = tok.get_vocab()
    for tok_name in ["<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>"]:
        assert tok_name in vocab, (
            f"Required geometry token missing in tokenizer vocab: {tok_name}"
        )

    # Validate base vocab size at least known baseline
    base_len_known = 151665
    assert len(vocab) >= base_len_known, (
        f"Tokenizer vocab too small: {len(vocab)} < {base_len_known}"
    )

    # Validate coordinate token range on expanded caches
    coord_ids = [
        vocab.get(f"<|coord_{i}|>") for i in range(1025) if f"<|coord_{i}|>" in vocab
    ]
    assert coord_ids, "Expanded tokenizer missing coordinate tokens"
    assert min(coord_ids) == HARD_START, (
        f"Start coord ID mismatch for {model_path}: expected {HARD_START}, got {min(coord_ids)}"
    )
    assert max(coord_ids) == HARD_END_INCLUSIVE, (
        f"End coord ID mismatch for {model_path}: expected {HARD_END_INCLUSIVE}, got {max(coord_ids)}"
    )
