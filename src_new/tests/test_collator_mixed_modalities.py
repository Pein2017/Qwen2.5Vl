#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
from pathlib import Path

import torch
from transformers import Qwen2_5_VLProcessor

from src_new.data.collator_standard import StandardDataCollator


class TestCollatorMixedModalities(unittest.TestCase):
    processor: Qwen2_5_VLProcessor | None = None

    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at default path: {model_path}. Please place the model there."
            )
        cls.processor = Qwen2_5_VLProcessor.from_pretrained(str(model_path))

    def test_standard_collator_handles_text_only_and_image_features(self) -> None:
        assert self.processor is not None
        collator = StandardDataCollator(tokenizer=self.processor.tokenizer)

        text_feature = {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            "pixel_values": torch.zeros(1, 16),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "teacher_assistant_spans": [(0, 1)],
            "student_assistant_spans": [(1, 3)],
            "conversation_variant": "text_only",
        }

        pixel_feature = {
            "input_ids": torch.tensor([4, 5, 6, 7], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([4, 5, 6, 7], dtype=torch.long),
            "pixel_values": torch.randn(4, 16),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            "teacher_assistant_spans": [(0, 2)],
            "student_assistant_spans": [(2, 4)],
            "conversation_variant": "dense_caption",
        }

        batch = collator([text_feature, pixel_feature])
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertIn("pixel_values", batch)
        self.assertIn("image_grid_thw", batch)
        expected_patches = int((batch["image_grid_thw"][:, 0] * batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2]).sum().item())
        self.assertEqual(expected_patches, int(batch["pixel_values"].shape[0]))


if __name__ == "__main__":
    unittest.main()
