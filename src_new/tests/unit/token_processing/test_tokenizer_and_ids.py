#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer extension and strict ID mapping tests.
"""

from src_new.processing.token_processor import TokenConfig, TokenProcessor


class DummyTokenizer:
    def __init__(self, base_vocab_len=151665):
        self._v = {f"tok_{i}": i for i in range(base_vocab_len)}

    def get_vocab(self):
        return dict(self._v)

    def add_special_tokens(self, d):
        toks = d.get("additional_special_tokens", [])
        added = 0
        for t in toks:
            if t not in self._v:
                self._v[t] = len(self._v)
                added += 1
        return added

    def __len__(self):
        return len(self._v)


def test_tokenizer_extension_exact_ids_and_final_size():
    tok = DummyTokenizer()
    tp = TokenProcessor(TokenConfig(coordinate_tokens_enabled=True, max_coord_value=1024))
    tok = tp.extend_tokenizer_vocabulary(tok)

    v = tok.get_vocab()
    assert v["<|line_start|>"] == 151665
    assert v["<|line_end|>"] == 151666
    ids = [v[f"<|coord_{i}|>"] for i in range(1025)]
    assert min(ids) == 151667 and max(ids) == 152691
    assert len(v) == 152692


def test_tokenizer_extension_missing_tokens_raises():
    # Simulate missing additions: start with base including line tokens to force only coords
    class Partial(DummyTokenizer):
        def __init__(self):
            super().__init__()
            self._v["<|line_start|>"] = 151665
            self._v["<|line_end|>"] = 151666

        def add_special_tokens(self, d):
            # Drop half of tokens to trigger failure in post-conditions
            toks = d.get("additional_special_tokens", [])
            # only add the first 10 to simulate incomplete extension
            for t in toks[:10]:
                if t not in self._v:
                    self._v[t] = len(self._v)
            return len(toks[:10])

    tok = Partial()
    tp = TokenProcessor(TokenConfig(coordinate_tokens_enabled=True, max_coord_value=1024))
    try:
        tp.extend_tokenizer_vocabulary(tok)
        assert False, "Expected ValueError for missing required tokens"
    except ValueError as e:
        assert "Missing required tokens" in str(e)

