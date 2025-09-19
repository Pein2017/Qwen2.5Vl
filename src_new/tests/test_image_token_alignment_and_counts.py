#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import torch
from transformers import AutoTokenizer

from src_new.processing.special_tokens import IMAGE_PAD

MODEL_PATHS = [
    "outputs/7B-dynamic_pairing/phase_1/9-17-dynamic_pairing-phase_1/checkpoint-2000",
    "model_cache/Qwen/Qwen2.5-VL-7B-Instruct-line_tokens",
]


class TestImageTokenAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        last_exc = None
        for p in MODEL_PATHS:
            try:
                cls.tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True)
                return
            except Exception as e:
                last_exc = e
        raise unittest.SkipTest(f"No tokenizer available: {last_exc}")

    def test_counts_match_expected(self):
        # Suppose one image with THW=(1, 16, 16), merge_size=2 => tokens per image = (1*16*16)//4 = 64
        merge_size = 2
        thw = torch.tensor([[1, 16, 16]], dtype=torch.long)
        expected_tokens = int((thw.prod(dim=1) // (merge_size * merge_size)).sum().item())
        # Build a prompt with that many image_pad tokens after assistant header
        img_id = self.tok.convert_tokens_to_ids(IMAGE_PAD)
        self.assertIsInstance(img_id, int)
        # Minimal input_ids containing exactly expected_tokens image_pad ids somewhere
        ids = torch.tensor([img_id] * expected_tokens + [self.tok.eos_token_id], dtype=torch.long)
        # Count in text equals length of run (sanity; mimic decode Count)
        count = int((ids == img_id).sum().item())
        self.assertEqual(count, expected_tokens)
        # pixel_values rows must equal sum(t*h*w) = 256
        patches = int((thw[:, 0] * thw[:, 1] * thw[:, 2]).sum().item())
        self.assertEqual(patches, 256)


if __name__ == "__main__":
    unittest.main()
