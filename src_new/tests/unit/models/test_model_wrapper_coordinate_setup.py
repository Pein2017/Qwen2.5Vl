#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DetectionModel wrapper tests focused on coordinate setup and validation.
"""

from unittest.mock import Mock

import torch

from src_new.models.wrapper import CoordinateProcessor, DetectionModel
from src_new.processing.token_processor import TokenConfig, TokenProcessor


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=151665, hidden=8):
        super().__init__()
        self.config = Mock()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)
        self.embed = torch.nn.Embedding(vocab_size, hidden)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size: int, **kwargs):
        old_in = self.embed
        old_out = self.lm_head
        new_in = torch.nn.Embedding(new_size, old_in.embedding_dim)
        new_out = torch.nn.Linear(old_out.in_features, new_size, bias=False)
        with torch.no_grad():
            rows = min(old_in.num_embeddings, new_size)
            new_in.weight[:rows] = old_in.weight[:rows]
            new_out.weight[:rows, :] = old_out.weight[:rows, :]
        self.embed = new_in
        self.lm_head = new_out
        self.config.vocab_size = new_size


def test_coordinate_processor_strict_range_validation():
    class DummyCfg:
        coordinate_tokens_enabled = True
        max_coord_value = 1024

    cp = CoordinateProcessor(DummyCfg())

    # set tokenizer with correct ids
    vocab = {"<|line_start|>": 151665, "<|line_end|>": 151666}
    vocab.update({f"<|coord_{i}|>": 151667 + i for i in range(1025)})
    tok = Mock()
    tok.get_vocab.return_value = vocab
    cp.set_tokenizer(tok)

    assert cp.coordinate_token_range == (151667, 152692)


def test_detection_model_embedding_extension_and_processor_update():
    base = TinyModel()
    # Setup DetectionModel with token processor
    token_processor = TokenProcessor(
        TokenConfig(coordinate_tokens_enabled=True, max_coord_value=1024)
    )

    class DummyCfg:
        coordinate_tokens_enabled = True
        max_coord_value = 1024
        use_cache = False
        torch_dtype = "float32"
        attn_implementation = "eager"
        skip_vocab_extension = True

    # Create a minimal tokenizer and extend before DetectionModel, then pass tokenizer and skip expansion
    class Tok:
        def __init__(self):
            self._v = {f"tok_{i}": i for i in range(151665)}

        def get_vocab(self):
            return dict(self._v)

        def add_special_tokens(self, d):
            toks = d.get("additional_special_tokens", [])
            for t in toks:
                if t not in self._v:
                    self._v[t] = len(self._v)
            return len(toks)

    t = Tok()

    # Extend vocab and model embeddings to simulate expanded checkpoint
    t = token_processor.extend_tokenizer_vocabulary(t)
    token_processor.extend_model_embeddings(base, t)

    # Build DetectionModel with skip_expansion and pass extended tokenizer
    dm = DetectionModel(
        base_model=base, config=DummyCfg(), tokenizer=t, skip_expansion=True
    )

    # Update coordinate processor after extension
    dm.coordinate_processor.set_tokenizer(t)
    dm.coordinate_processor.update_after_extension(t)

    assert dm.coordinate_processor.coordinate_token_range == (151667, 152692)
    # Validate embedding sizes padded to multiple of 128
    assert base.get_input_embeddings().weight.shape[0] % 128 == 0
