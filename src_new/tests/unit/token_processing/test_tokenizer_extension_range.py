#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate tokenizer extension adds exactly 1025 coordinate tokens (coord_0..coord_1024)
starting at 151667 and ending at 152691 inclusive.
"""

from src_new.processing.token_processor import TokenConfig, TokenProcessor


class DummyTokenizer:
    def __init__(self, base_vocab):
        # Minimal duck-typed tokenizer: implements get_vocab, add_special_tokens, __len__
        self._vocab = dict(base_vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, special_tokens_dict):
        tokens = special_tokens_dict.get("additional_special_tokens", [])
        added = 0
        for t in tokens:
            if t not in self._vocab:
                if t.startswith("<|coord_") and t.endswith("|>"):
                    idx = int(t.split("_")[1].split("|")[0])
                    self._vocab[t] = 151667 + idx
                else:
                    self._vocab[t] = len(self._vocab)
                added += 1
        return added

    def __len__(self):
        return len(self._vocab)


def test_tokenizer_extension_adds_exact_coord_range():
    base_vocab = {
        "<|line_start|>": 151665,
        "<|line_end|>": 151666,
        "<|extra_token|>": 42,
    }
    tok = DummyTokenizer(base_vocab)

    cfg = TokenConfig(max_coord_value=1024, coordinate_tokens_enabled=True)
    tp = TokenProcessor(cfg)
    tok_ext = tp.extend_tokenizer_vocabulary(tok)

    vocab = tok_ext.get_vocab()
    ids = [vocab[f"<|coord_{i}|>"] for i in range(1025)]

    assert min(ids) == 151667
    assert max(ids) == 152691
    assert len(ids) == 1025
