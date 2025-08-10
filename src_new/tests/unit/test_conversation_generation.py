#!/usr/bin/env python3
import unittest
from typing import Any, Dict, List

import torch
from PIL import Image

from src_new.processing.conversation_processor import ConversationProcessor


class FakeTokenizer:
    def __init__(self):
        self._last_text = ""

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        # Ignore ids; return last text recorded by FakeProcessor
        return self._last_text


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        # Very simple template: count images and insert one <|image_pad|> per image placeholder
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts.append(f"<|im_start|>{role}\n")
            if isinstance(content, list):
                # user message with mixed content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        parts.append("<|image_pad|>")
                    elif isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
            else:
                parts.append(str(content))
            parts.append("<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(
        self,
        *,
        text: List[Dict[str, Any]] | List[str],
        images: List[Image.Image],
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # Record text for decode()
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
            self.tokenizer._last_text = text[0]
        else:
            self.tokenizer._last_text = ""
        seq_len = max(1, len(self.tokenizer._last_text))
        num_images = len(images)
        # Minimal valid tensors: flattened patches and 2D THW grids
        # Use grid [1,2,2] per image => 4 patches per image
        image_grid_thw = torch.tensor([[1, 2, 2]] * num_images, dtype=torch.long)
        num_patches = int(
            (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            .sum()
            .item()
        )
        hidden = 64
        return {
            "input_ids": torch.ones(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            "pixel_values": torch.zeros(num_patches, hidden, dtype=torch.float32),
            "image_grid_thw": image_grid_thw,
        }


def make_image(color: str = "red", size=(16, 16)) -> Image.Image:
    return Image.new("RGB", size, color)


class TestConversationGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = FakeProcessor()
        self.conv = ConversationProcessor(self.processor, max_coord_value=16)

    def test_simple_generation_conversation(self):
        sample = {
            "images": ["dummy.jpg"],  # not used by FakeProcessor path
            "objects": [{"bbox_2d": [1, 2, 3, 4], "desc": "obj"}],
            "height": 32,
            "width": 32,
        }
        images = [make_image("blue")]
        inputs = self.conv.create_simple_conversation_for_generation(sample, images)
        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)

        # Validate image-token alignment via decode()
        stats = self.conv.get_conversation_stats(inputs)
        self.assertEqual(stats.get("image_tokens"), 1)
        self.assertEqual(inputs["image_grid_thw"][0].item(), 1)

    def test_teacher_student_generation_conversation(self):
        student_sample = {
            "images": ["student.jpg"],
            "objects": [{"bbox_2d": [2, 2, 5, 6], "desc": "student"}],
            "height": 32,
            "width": 32,
        }
        teacher_samples = [
            {
                "images": ["t1.jpg"],
                "objects": [{"bbox_2d": [1, 1, 3, 3], "desc": "t1"}],
            },
            {
                "images": ["t2.jpg"],
                "objects": [{"bbox_2d": [4, 4, 7, 7], "desc": "t2"}],
            },
        ]
        student_images = [make_image("red")]
        teacher_images_list = [[make_image("green")], [make_image("yellow")]]

        inputs = self.conv.create_teacher_student_conversation_for_generation(
            student_sample=student_sample,
            teacher_samples=teacher_samples,
            student_images=student_images,
            teacher_images_list=teacher_images_list,
        )
        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)

        # Expect 3 images total (2 teachers + 1 student)
        stats = self.conv.get_conversation_stats(inputs)
        self.assertEqual(stats.get("image_tokens"), 3)
        self.assertEqual(inputs["image_grid_thw"][0].item(), 3)


if __name__ == "__main__":
    unittest.main()
