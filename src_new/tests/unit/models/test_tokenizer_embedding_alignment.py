#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict validation tests for tokenizer and embedding alignment.
"""

import math
from unittest.mock import Mock

import pytest
import torch

from src_new.processing.token_processor import TokenConfig, TokenProcessor


class TinyEmbeddingModel:
    def __init__(self, vocab_size: int, hidden_size: int = 32, dtype=torch.float32):
        self.config = Mock()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size
        self.config.initializer_range = 0.02
        self._in = torch.nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self._out = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out

    def resize_token_embeddings(self, new_size: int, **kwargs):
        old_in = self._in
        old_out = self._out
        hidden = old_in.embedding_dim
        new_in = torch.nn.Embedding(new_size, hidden, dtype=old_in.weight.dtype)
        new_out = torch.nn.Linear(
            hidden, new_size, bias=False, dtype=old_out.weight.dtype
        )
        with torch.no_grad():
            copy_rows = min(old_in.weight.shape[0], new_in.weight.shape[0])
            new_in.weight[:copy_rows] = old_in.weight[:copy_rows]
            new_out.weight[:copy_rows, :] = old_out.weight[:copy_rows, :]
        self._in = new_in
        self._out = new_out
        self.config.vocab_size = new_size
        return self._in


class SimpleTok:
    def __init__(self, n):
        self._v = {f"tok_{i}": i for i in range(n)}

    def get_vocab(self):
        return dict(self._v)

    def add_special_tokens(self, d):
        toks = d.get("additional_special_tokens", [])
        for t in toks:
            if t not in self._v:
                self._v[t] = len(self._v)
        return len(toks)

    def __len__(self):
        return len(self._v)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("base_len", [151665])
def test_strict_tokenizer_embedding_alignment(dtype, base_len):
    # Base tokenizer vocab and processor
    tok = SimpleTok(base_len)
    tp = TokenProcessor(
        TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=1024,
            coordinate_init_mode="fourier_ramp",
        )
    )

    # Extend tokenizer
    tok = tp.extend_tokenizer_vocabulary(tok)
    vocab = tok.get_vocab()

    # Token count strict checks
    assert len(vocab) == 152692
    assert vocab["<|line_start|>"] == 151665
    assert vocab["<|line_end|>"] == 151666
    coord_ids = [vocab[f"<|coord_{i}|>"] for i in range(0, 1025)]
    assert min(coord_ids) == 151667 and max(coord_ids) == 152691

    # Model and extension
    model = TinyEmbeddingModel(vocab_size=base_len, hidden_size=32, dtype=dtype)
    model = tp.extend_model_embeddings(model, tok)

    in_rows = model.get_input_embeddings().weight.shape[0]
    target_rows = math.ceil(len(vocab) / 128) * 128
    assert target_rows == 152704
    assert in_rows == target_rows
    assert in_rows >= len(vocab)

    # Geometry token linkage and coordinate non-zero init with dtype/device matching
    in_w = model.get_input_embeddings().weight
    ls, le = vocab["<|line_start|>"], vocab["<|line_end|>"]
    with torch.no_grad():
        if "<|quad_start|>" in vocab and "<|quad_end|>" in vocab:
            qs, qe = vocab["<|quad_start|>"], vocab["<|quad_end|>"]
            assert torch.allclose(in_w[ls], in_w[qs])
            assert torch.allclose(in_w[le], in_w[qe])
        zeros = torch.zeros_like(in_w[0])
        for cid in coord_ids:
            vec = in_w[cid]
            assert vec.dtype == in_w.dtype
            assert vec.device == in_w.device
            assert not torch.allclose(vec, zeros)
