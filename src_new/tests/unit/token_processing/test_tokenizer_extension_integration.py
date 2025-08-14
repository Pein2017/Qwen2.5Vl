#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test: extend a base tokenizer and check coordinate token range.
This simulates our pipeline when the official tokenizer lacks coordinate tokens.
"""

import os

from transformers import AutoTokenizer

from src_new.processing.token_processor import TokenConfig, TokenProcessor


def test_extend_official_tokenizer_adds_coord_tokens():
    base = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
    if not os.path.exists(base):
        raise FileNotFoundError(f"Required local model cache missing: {base}")
    tok = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=True
    )

    cfg = TokenConfig(
        max_coord_value=1025,
        coordinate_tokens_enabled=True,
        coordinate_init_mode="fourier_ramp",
    )
    tp = TokenProcessor(cfg)
    tok_ext = tp.extend_tokenizer_vocabulary(tok)

    vocab = tok_ext.get_vocab()
    ids = [vocab.get(f"<|coord_{i}|>") for i in range(1026)]

    assert None not in ids
    assert min(ids) == 151667
    assert max(ids) == 152692
