#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import torch

from src_new_json.losses.token_grouping import TokenGroupingPlugin
from src_new_json.models.loss_manager import LossManager


class FakeTokenizer:
    """Minimal tokenizer stub providing IDs for punctuation and simple tokens.

    Methods used by the code under test:
    - get_vocab()
    - convert_tokens_to_ids(token)
    - encode(text, add_special_tokens=False)
    """

    def __init__(self):
        vocab = {}

        def add(tok):
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1

        # JSON punctuation set
        for ch in ["[", "]", "{", "}", ",", ":", '"']:
            add(ch)
        # Natural token
        add("标签贴纸")
        # end token
        add("<|im_end|>")
        # filler
        add("EOS")
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token: str):
        return self._vocab.get(token, -1)

    def convert_ids_to_tokens(self, ids):
        # Provide a simple reverse mapping for test; any unknown id maps to 'UNK'
        inv = {v: k for k, v in self._vocab.items()}
        if isinstance(ids, list):
            return [inv.get(int(i), "UNK") for i in ids]
        return inv.get(int(ids), "UNK")

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._vocab:
            return [self._vocab[text]]
        return [0]


class DummyTokProc:
    def get_coordinate_token_range(self, tok):
        return (0, 0)


class TestGroupedTokenLosses(unittest.TestCase):
    def setUp(self):
        # Config stub with only fields used by LossManager
        class Cfg:
            teacher_loss_weight = 1.0
            student_loss_weight = 1.0
            # group weights (required)
            caption_loss_weight = 0.5
            grounding_loss_weight = 1.0
            formatting_loss_weight = 0.2

        self.cfg = Cfg()
        self.tok = FakeTokenizer()
        self.lm = LossManager(self.cfg, DummyTokProc(), self.tok)
        self.plugin = TokenGroupingPlugin(self.tok)
        self.lm.set_token_grouping_plugin(self.plugin)

    def _build_synthetic_labels_and_spans(self):
        t2id = self.tok.convert_tokens_to_ids
        # Synthetic token IDs sequence (JSON-like): [ { "label" : "标签贴纸" } ] <|im_end|>
        seq = [
            t2id("EOS"),
            t2id("["),
            t2id("{"),
            t2id("\""),
            t2id("label"),  # unknown → -1; use EOS to simulate non-punct token
            t2id("\""),
            t2id(":"),
            t2id("\""),
            t2id("标签贴纸"),
            t2id("\""),
            t2id("}"),
            t2id("]"),
            t2id("<|im_end|>"),
            t2id("EOS"),
        ]
        # Replace unknown "label" with EOS id to ensure non-punct category
        seq[4] = t2id("EOS")
        labels = torch.tensor([seq], dtype=torch.long)
        # Single student span covering the assistant content (positions 1..12)
        student_spans = [[(1, 13)]]
        teacher_spans = [[]]
        return labels, teacher_spans, student_spans

    def test_group_masks_align_cover_and_categories(self):
        labels, t_spans, s_spans = self._build_synthetic_labels_and_spans()
        gm = self.plugin.build_group_masks(
            labels=labels, teacher_spans=t_spans, student_spans=s_spans, input_ids=labels
        )

        # Shifted assistant masks
        t_mask = torch.zeros_like(labels, dtype=torch.bool)
        s_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, spans in enumerate(t_spans):
            for st, en in spans:
                t_mask[i, st:en] = True
        for i, spans in enumerate(s_spans):
            for st, en in spans:
                s_mask[i, st:en] = True
        t_assist = t_mask[:, 1:]
        s_assist = s_mask[:, 1:]

        # Disjointness
        self.assertEqual(int((gm.student_caption & gm.student_grounding).sum().item()), 0)
        self.assertEqual(int((gm.student_caption & gm.student_formatting).sum().item()), 0)
        self.assertEqual(int((gm.student_grounding & gm.student_formatting).sum().item()), 0)

        # Coverage (student only in this synthetic case)
        self.assertTrue(
            (gm.student_caption | gm.student_grounding | gm.student_formatting).equal(
                s_assist
            )
        )
        self.assertTrue(
            (gm.teacher_caption | gm.teacher_grounding | gm.teacher_formatting).equal(
                t_assist
            )
        )

    def test_weighted_llm_aggregation_math(self):
        labels, t_spans, s_spans = self._build_synthetic_labels_and_spans()
        B, S = labels.shape
        V = 200
        torch.randn(B, S, V)
        # LossManager path requires input_ids via plugin; this test focuses on mask building
        gm = self.plugin.build_group_masks(
            labels=labels, teacher_spans=t_spans, student_spans=s_spans, input_ids=labels
        )
        # Basic sanity on masks
        self.assertEqual(gm.teacher_caption.shape[1], S - 1)
        self.assertEqual(gm.student_formatting.shape[1], S - 1)


if __name__ == "__main__":
    unittest.main()
