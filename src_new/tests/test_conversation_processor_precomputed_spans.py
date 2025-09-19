#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
from PIL import Image

import torch
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor

from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.variants import create_default_variant_registry
from src_new.processing.coordinate_converter import CoordinateTokenConverter
from src_new.processing.special_tokens import IMAGE_PAD


MODEL_PATHS = [
    "outputs/7B-dynamic_pairing/phase_1/9-17-dynamic_pairing-phase_1/checkpoint-2000",
    "model_cache/Qwen/Qwen2.5-VL-7B-Instruct-line_tokens",
]


def _toy_sample():
    # Minimal sample with one object
    return {
        "images": [],
        "objects": [
            {"bbox_2d": [10, 20, 30, 40], "desc": "标签/可以识别"},
        ],
        "width": 100,
        "height": 100,
    }


class TestConversationProcessorPrecomputed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        last_exc = None
        for p in MODEL_PATHS:
            try:
                tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True)
                imgp = Qwen2VLImageProcessor.from_pretrained(p, trust_remote_code=True)
                proc_from_ckpt = Qwen2VLProcessor.from_pretrained(p, trust_remote_code=True)
                vp = getattr(proc_from_ckpt, "video_processor", None) or Qwen2VLVideoProcessor()
                chat = getattr(tok, "chat_template", None) or getattr(proc_from_ckpt, "chat_template", None)
                cls.processor = Qwen2VLProcessor(image_processor=imgp, tokenizer=tok, video_processor=vp, chat_template=chat)
                cls.tok = tok
                return
            except Exception as e:
                last_exc = e
        raise unittest.SkipTest(f"No processor/tokenizer available: {last_exc}")

    def _run_variant(self, variant_key: str):
        converter = CoordinateTokenConverter(max_coord_value=4096, coordinate_tokens_enabled=False, format_mode="special_tokens")
        conv = ConversationProcessor(processor=self.processor, max_coord_value=4096, coordinate_tokens_enabled=False)
        # Fake a 1x1 white image for simplicity
        img = Image.new("RGB", (64, 64), color=(255, 255, 255))
        out = conv.create_conversation(sample=_toy_sample(), images=[img], variant=variant_key)
        # Spans must be present and non-empty for the student turn
        t_spans = out.get("teacher_assistant_spans") or []
        s_spans = out.get("student_assistant_spans") or []
        self.assertIsInstance(t_spans, list)
        self.assertIsInstance(s_spans, list)
        self.assertGreaterEqual(len(s_spans), 1)
        # label masking using provided spans
        labels = out["input_ids"].clone()
        labels.fill_(-100)
        for st, ed in s_spans:
            self.assertTrue(0 <= int(st) < int(ed) <= int(labels.shape[0]))
            labels[int(st):int(ed)] = out["input_ids"][int(st):int(ed)]
        # image_pad tokens masked
        img_id = self.tok.convert_tokens_to_ids(IMAGE_PAD)
        if isinstance(img_id, int) and img_id >= 0:
            labels[out["input_ids"] == img_id] = -100
        self.assertGreater(int((labels != -100).sum().item()), 0)

    def test_variants(self):
        for variant in ("dense_caption", "coords_to_desc", "desc_to_coords", "summary"):
            with self.subTest(variant=variant):
                self._run_variant(variant)


if __name__ == "__main__":
    unittest.main()
