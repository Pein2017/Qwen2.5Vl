#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import torch

from src_new.losses.token_grouping import TokenGroupingPlugin
from src_new.models.loss_manager import LossManager


class FakeTokenizer:
    """Minimal tokenizer stub providing IDs for special tokens and punctuation.

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

        # Object-ref wrappers
        add("<|object_ref_start|>")
        add("<|object_ref_end|>")
        # Geometry wrappers
        add("<|box_start|>")
        add("<|box_end|>")
        add("<|quad_start|>")
        add("<|quad_end|>")
        add("<|line_start|>")
        add("<|line_end|>")
        # A description token (caption content)
        add("标签贴纸")
        # Punctuation set
        for ch in ["[", "]", "{", "}", "(", ")", ",", ":", '"', "/"]:
            add(ch)
        # Coordinate tokens (provide a small slice)
        for n in [10, 20, 30, 40]:
            add(f"<|coord_{n}|>")
        # im_end present in labels (residual to formatting)
        add("<|im_end|>")
        # Extra filler token
        add("EOS")
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token: str):
        return self._vocab.get(token, -1)

    def encode(self, text: str, add_special_tokens: bool = False):
        # Only used for single punctuation chars in plugin; map via vocab
        if text in self._vocab:
            return [self._vocab[text]]
        # Return a single unknown id for non-registered strings
        return [0]


class DummyTokProc:
    def get_coordinate_token_range(self, tok):
        # Not used because LossManager derives from tokenizer via special_tokens
        return (0, 0)


class TestGroupedTokenLosses(unittest.TestCase):
    def setUp(self):
        # Config stub with only fields used by LossManager
        class Cfg:
            coordinate_loss_weight = 1.0
            regular_loss_weight = 1.0
            teacher_loss_weight = 1.0
            student_loss_weight = 1.0
            coordinate_tokens_enabled = True
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
        seq = [
            t2id("EOS"),
            t2id("<|object_ref_start|>"),
            t2id("标签贴纸"),
            t2id("<|object_ref_end|>"),
            t2id("<|box_start|>"),
            t2id("["),
            t2id("<|coord_10|>"),
            t2id(","),
            t2id("<|coord_20|>"),
            t2id(","),
            t2id("<|coord_30|>"),
            t2id(","),
            t2id("<|coord_40|>"),
            t2id("]"),
            t2id("<|box_end|>"),
            t2id("<|im_end|>"),
            t2id("EOS"),
        ]
        labels = torch.tensor([seq], dtype=torch.long)
        # Single student span covering the assistant content (positions 1..15)
        student_spans = [[(1, 16)]]
        teacher_spans = [[]]
        return labels, teacher_spans, student_spans

    def test_group_masks_align_cover_and_categories(self):
        labels, t_spans, s_spans = self._build_synthetic_labels_and_spans()
        gm = self.plugin.build_group_masks(
            labels=labels, teacher_spans=t_spans, student_spans=s_spans
        )

        # Shifted assistant masks
        B, S = labels.shape
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
        self.assertEqual(
            int((gm.student_caption & gm.student_grounding).sum().item()), 0
        )
        self.assertEqual(
            int((gm.student_caption & gm.student_formatting).sum().item()), 0
        )
        self.assertEqual(
            int((gm.student_grounding & gm.student_formatting).sum().item()), 0
        )

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

        # Category counts (known from synthetic sequence)
        # caption: one token "标签贴纸"
        self.assertEqual(int(gm.student_caption.sum().item()), 1)
        # grounding: 2 wrappers + 4 coords = 6
        self.assertEqual(int(gm.student_grounding.sum().item()), 6)
        # formatting: remainder inside span after shift; at least brackets, commas, obj-ref wrappers, im_end
        self.assertGreater(int(gm.student_formatting.sum().item()), 0)

    def test_weighted_llm_aggregation_math(self):
        labels, t_spans, s_spans = self._build_synthetic_labels_and_spans()
        B, S = labels.shape
        V = 200
        logits = torch.randn(B, S, V)
        out = self.lm._compute_granular_teacher_student_loss(
            logits=logits,
            labels=labels,
            coord_mask=None,
            teacher_spans=t_spans,
            student_spans=s_spans,
        )
        # Presence of group diagnostics
        for k in [
            "student_caption_ce",
            "student_grounding_ce",
            "student_formatting_ce",
            "student_caption_tokens",
            "student_grounding_tokens",
            "student_formatting_tokens",
            "group_caption_ce",
            "group_grounding_ce",
            "group_formatting_ce",
        ]:
            self.assertIn(k, out)


if __name__ == "__main__":
    unittest.main()
