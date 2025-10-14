#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import torch
from transformers import AutoTokenizer

from src_new.processing.span_builder import build_assistant_spans_token_aligned
from src_new.processing.special_tokens import (
    ASSISTANT_HEADER,
    IM_END,
    IM_START,
    IMAGE_PAD,
)


MODEL_PATHS = [
    "outputs/7B-dynamic_pairing/phase_1/9-17-dynamic_pairing-phase_1/checkpoint-2000",
    "model_cache/Qwen/Qwen2.5-VL-7B-Instruct-line_tokens",
]


class TestSpanBuilderAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        last_exc = None
        for p in MODEL_PATHS:
            try:
                cls.tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True)
                cls.model_path = p
                return
            except Exception as e:
                last_exc = e
        raise unittest.SkipTest(f"No tokenizer available at any known path: {last_exc}")

    def test_mapping_over_image_pad_runs(self):
        # Build a minimal conversation text: user with one image, then assistant with content
        assistant_text = "你好，世界"
        conversation_text = (
            f"{IM_START}user <|image_pad|>{IM_END}"
            f"{ASSISTANT_HEADER}{assistant_text}{IM_END}"
        )
        enc = self.tok(
            conversation_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt"
        )
        input_ids_unexpanded = enc["input_ids"][0]
        offsets = enc["offset_mapping"][0]
        # Expand: repeat each image_pad token 4 times to simulate processor expansion
        img_id = self.tok.convert_tokens_to_ids(IMAGE_PAD)
        expanded_ids = []
        for tid in input_ids_unexpanded.tolist():
            if tid == img_id:
                expanded_ids.extend([tid, tid, tid, tid])
            else:
                expanded_ids.append(tid)
        input_ids_expanded = torch.tensor(expanded_ids, dtype=torch.long)

        # Run span builder
        t_spans, s_spans = build_assistant_spans_token_aligned(
            conversation_text=conversation_text,
            offset_mapping=offsets,
            tokenizer=self.tok,
            input_ids_expanded=input_ids_expanded,
            has_teachers=False,
            num_teachers=0,
            include_eos=True,
        )
        self.assertEqual(len(t_spans), 0, "no teachers expected")
        self.assertEqual(len(s_spans), 1, "exactly one student span expected")
        st, ed = s_spans[0]
        self.assertGreater(ed - st, 0, "non-empty span expected")
        # Decode that slice and verify it contains assistant content and includes IM_END
        decoded = self.tok.decode(input_ids_expanded[st:ed], skip_special_tokens=False)
        self.assertIn(assistant_text, decoded)
        self.assertIn(IM_END, decoded)


if __name__ == "__main__":
    unittest.main()
