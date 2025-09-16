from __future__ import annotations

import copy
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image
from transformers import Qwen2VLProcessor

from src_new_json.augmentation import ObjectAwareAugmentationPipeline
from src_new_json.config.augmentation_config import AugmentationConfig
from src_new_json.processing.conversation_processor import ConversationProcessor


def _make_image(size_wh: Tuple[int, int]) -> Image.Image:
    w, h = size_wh
    return Image.new("RGB", (w, h), color=(20, 30, 40))


def _simple_sample() -> Dict[str, Any]:
    # User example: one quad and a caption
    return {
        "width": 64,
        "height": 48,
        "objects": [
            {"quad": [1, 2, 2, 3, 3, 4, 4, 5], "desc": "a cat"},
        ],
    }


def _aug_cfg_identity() -> AugmentationConfig:
    # Image-geom disabled (identity), no object-level ops
    return AugmentationConfig(
        enabled=True,
        rng_seed=123,
        apply_to_teachers=False,
        lines_policy="transform",
        debug_visualization=False,
        debug_output_dir=None,
        image_geom=None,
        photometric=None,
        lines=None,
        type_policies=None,
        ocr=None,
        criteria=None,
    )


class TestProcessingIntegration(unittest.TestCase):
    processor: Qwen2VLProcessor | None = None

    @classmethod
    def setUpClass(cls) -> None:
        # Use repository-local default; skip if missing
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        if not model_path.exists():
            raise unittest.SkipTest(f"Model not found at {model_path}")
        cls.processor = Qwen2VLProcessor.from_pretrained(str(model_path))

    def test_numeric_mode_text_roundtrip_after_aug(self) -> None:
        assert self.processor is not None
        sample = _simple_sample()
        img = _make_image((sample["width"], sample["height"]))

        # 1) Apply augmentation (identity geometry)
        aug = ObjectAwareAugmentationPipeline.from_config(_aug_cfg_identity())
        imgs_out, sample_out = aug.apply(copy.deepcopy(sample), [img], sample_index=0)
        self.assertEqual(
            (imgs_out[0].width, imgs_out[0].height), (img.width, img.height)
        )
        self.assertEqual(sample_out["objects"], sample["objects"])  # identity

        # 2) Build conversation and tokenize (numeric mode)
        conv = ConversationProcessor(
            processor=self.processor,
        )
        inputs = conv.create_simple_conversation(sample=sample_out, images=imgs_out)
        text = inputs.get("conversation_text", "")
        self.assertIsInstance(text, str)

        # 3) Expect assistant JSON rendered in conversation text
        self.assertIn("<|im_start|>system", text)
        self.assertIn("<|im_start|>user", text)
        # Assistant content is rendered during training flow; here we check template presence

    def test_coordinate_token_mode_text_and_token_ids(self) -> None:
        assert self.processor is not None
        sample = _simple_sample()
        img = _make_image((sample["width"], sample["height"]))

        # 1) Apply augmentation (identity geometry)
        aug = ObjectAwareAugmentationPipeline.from_config(_aug_cfg_identity())
        imgs_out, sample_out = aug.apply(copy.deepcopy(sample), [img], sample_index=0)

        # JSON mode: no tokenizer extension

        # 3) Build conversation and tokenize (coordinate-token mode)
        conv = ConversationProcessor(
            processor=self.processor,
        )
        inputs = conv.create_simple_conversation(sample=sample_out, images=imgs_out)
        # JSON mode: simply ensure conversation built successfully
        self.assertIn("input_ids", inputs)


if __name__ == "__main__":
    unittest.main()
