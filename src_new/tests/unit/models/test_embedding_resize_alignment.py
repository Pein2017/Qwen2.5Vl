#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate live embedding resizing and alignment for Qwen2.5-VL coordinate tokens.
"""

import math
from unittest.mock import Mock

import pytest
import torch

from src_new.processing.token_processor import TokenConfig, TokenProcessor


class TinyEmbeddingModel:
    def __init__(self, vocab_size: int, hidden_size: int = 8):
        self.config = Mock()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size
        self.config.initializer_range = 0.02
        # input emb: [vocab, hidden]; output emb: [vocab, hidden] with HF LMHead is [vocab, hidden] or [hidden, vocab]
        self._in = torch.nn.Embedding(vocab_size, hidden_size)
        self._out = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out

    def resize_token_embeddings(self, new_size: int, **kwargs):
        old_in = self._in
        old_out = self._out
        hidden = old_in.embedding_dim
        new_in = torch.nn.Embedding(new_size, hidden)
        new_out = torch.nn.Linear(hidden, new_size, bias=False)
        with torch.no_grad():
            copy_rows = min(old_in.weight.shape[0], new_in.weight.shape[0])
            new_in.weight[:copy_rows] = old_in.weight[:copy_rows]
            # Copy existing output rows; new rows remain randomly initialized
            new_out.weight[:copy_rows, :] = old_out.weight[:copy_rows, :]
        self._in = new_in
        self._out = new_out
        self.config.vocab_size = new_size
        return self._in


@pytest.mark.parametrize("base_len", [151665])
@pytest.mark.parametrize("hidden", [32])
def test_embedding_resize_alignment(base_len, hidden):
    # Base tokenizer vocab
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

    tok = SimpleTok(base_len)

    # Extend tokenizer with coord system
    tp = TokenProcessor(
        TokenConfig(coordinate_tokens_enabled=True, max_coord_value=1024)
    )
    tok = tp.extend_tokenizer_vocabulary(tok)

    vocab = tok.get_vocab()
    assert vocab["<|line_start|>"] == 151665
    assert vocab["<|line_end|>"] == 151666
    ids = [vocab[f"<|coord_{i}|>"] for i in range(1025)]
    assert min(ids) == 151667 and max(ids) == 152691
    assert len(vocab) == 151665 + 1027

    # Create tiny model and extend embeddings
    model = TinyEmbeddingModel(vocab_size=base_len, hidden_size=hidden)
    model_in_before = model.get_input_embeddings().weight.clone().detach()

    model = tp.extend_model_embeddings(model, tok)

    in_rows = model.get_input_embeddings().weight.shape[0]
    out_w = model.get_output_embeddings().weight
    out_shape = out_w.shape
    # HF LM head commonly uses [vocab, hidden]; the vocab axis is the larger one
    out_vocab_dim = max(out_shape[0], out_shape[1])

    # Must be padded to multiple of 128 >= 152692
    target_rows = math.ceil(len(vocab) / 128) * 128
    assert target_rows == 152704
    assert in_rows == target_rows
    assert out_vocab_dim == target_rows

    # New tokens rows must be non-zero
    with torch.no_grad():
        w_in = model.get_input_embeddings().weight
        for i in range(151665, 152692):
            assert not torch.allclose(w_in[i], torch.zeros_like(w_in[i]))

        # Original rows preserved
        assert torch.allclose(w_in[:151665], model_in_before[:151665])

    # Strict: check geometry token copy linkage, if quad tokens are present in this tokenizer
    ls_id = vocab["<|line_start|>"]
    le_id = vocab["<|line_end|>"]
    if "<|quad_start|>" in vocab and "<|quad_end|>" in vocab:
        qs_id = vocab["<|quad_start|>"]
        qe_id = vocab["<|quad_end|>"]
        with torch.no_grad():
            assert torch.allclose(
                model.get_input_embeddings().weight[ls_id],
                model.get_input_embeddings().weight[qs_id],
            )
            assert torch.allclose(
                model.get_input_embeddings().weight[le_id],
                model.get_input_embeddings().weight[qe_id],
            )

    # Strict: dtype/device match for coordinate init
    in_w = model.get_input_embeddings().weight
    ids = [vocab[f"<|coord_{i}|>"] for i in range(0, 1025)]
    assert min(ids) == 151667 and max(ids) == 152691
    with torch.no_grad():
        for cid in ids:
            vec = in_w[cid]
            assert vec.dtype == in_w.dtype
            assert vec.device == in_w.device
            assert not torch.allclose(vec, torch.zeros_like(vec))
