from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from transformers import Qwen2_5_VLProcessor

from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.coordinate_converter import CoordinateTokenConverter
from src_new.processing.special_tokens import get_coord_token_range
from src_new.data.dataset import Dataset
from src_new.losses.token_grouping import TokenGroupingPlugin


def _make_image(size_wh: Tuple[int, int]) -> Image.Image:
    w, h = size_wh
    return Image.new("RGB", (w, h), color=(20, 30, 40))


def _student_teacher_samples() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    student = {
        "width": 96,
        "height": 64,
        "images": ["dummy.jpg"],  # not used by processor when we pass actual PIL
        "objects": [
            {"desc": "标签", "bbox_2d": [10, 12, 30, 24]},
            {"desc": "光纤", "line": [8, 40, 40, 58]},
        ],
    }
    teacher = {
        "width": 96,
        "height": 64,
        "images": ["dummy.jpg"],
        "objects": [
            {"desc": "BBU设备", "quad": [2, 4, 8, 4, 8, 12, 2, 12]},
        ],
    }
    return student, [teacher]


class TestEndToEndSpansAndGroupingWithImagePad(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        if not model_path.exists():
            raise unittest.SkipTest(f"Model not found at: {model_path}")
        cls.processor = Qwen2_5_VLProcessor.from_pretrained(str(model_path))

    def _build_conversation(self, coordinate_tokens_enabled: bool):
        conv = ConversationProcessor(
            processor=self.processor,
            max_coord_value=2048,
            coordinate_tokens_enabled=coordinate_tokens_enabled,
        )
        student, teachers = _student_teacher_samples()
        student_img = _make_image((student["width"], student["height"]))
        teacher_img = _make_image((teachers[0]["width"], teachers[0]["height"]))
        out = conv.create_teacher_student_conversation(
            student_sample=student,
            teacher_samples=teachers,
            student_images=[student_img],
            teacher_images_list=[[teacher_img]],
        )
        return out

    def _make_dummy_dataset(self):
        ds = object.__new__(Dataset)
        ds.config = type("C", (), {"span_include_im_end_in_labels": True})()
        return ds

    def _assert_spans_and_masks(self, out: Dict[str, torch.Tensor]):
        tok = self.processor.tokenizer
        ds = self._make_dummy_dataset()
        labels, t_spans, s_spans = ds._create_masked_labels_with_spans(
            input_ids=out["input_ids"],
            tokenizer=tok,
            has_teachers=True,
            conversation_text=out.get("conversation_text"),
            offset_mapping=out.get("offset_mapping"),
            num_teachers=1,
        )
        # Non-empty spans
        self.assertEqual(len(t_spans), 1)
        self.assertEqual(len(s_spans), 1)
        # image_pad masked when present
        image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
        ids_tensor = out["input_ids"][0]
        pad_positions = (ids_tensor == image_pad_id)
        if bool(pad_positions.any().item()):
            self.assertTrue((labels[pad_positions] == -100).all())
        # some learnable tokens
        self.assertGreater(int((labels != -100).sum().item()), 0)

        # Grouped masks
        plugin = TokenGroupingPlugin(tok)
        gm = plugin.build_group_masks(
            labels=labels.unsqueeze(0),
            teacher_spans=[t_spans],
            student_spans=[s_spans],
        )
        # Coverage equals assistant masks (shifted) via residual assignment
        t_assist = torch.zeros_like(labels, dtype=torch.bool)
        s_assist = torch.zeros_like(labels, dtype=torch.bool)
        for st, ed in t_spans:
            t_assist[st:ed] = True
        for st, ed in s_spans:
            s_assist[st:ed] = True
        t_assist = t_assist.unsqueeze(0)[:, 1:]
        s_assist = s_assist.unsqueeze(0)[:, 1:]
        union_t = gm.teacher_caption | gm.teacher_grounding | gm.teacher_formatting
        union_s = gm.student_caption | gm.student_grounding | gm.student_formatting
        self.assertTrue((union_t == t_assist).all())
        self.assertTrue((union_s == s_assist).all())

    def test_end_to_end_numeric_mode(self):
        out = self._build_conversation(coordinate_tokens_enabled=False)
        self._assert_spans_and_masks(out)

    @unittest.skip("Coordinate tokens deprecated")
    def test_end_to_end_coord_token_mode(self):
        pass


if __name__ == "__main__":
    unittest.main()
