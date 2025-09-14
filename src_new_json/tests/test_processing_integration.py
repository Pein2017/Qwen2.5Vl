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
from src_new_json.processing.special_tokens import get_coord_token_range
from src_new_json.processing.token_processor import TokenConfig, TokenProcessor


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
        # Use repository-local default; fail-fast if missing
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at default path: {model_path}. Please place the model there."
            )
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
            max_coord_value=2048,
            coordinate_tokens_enabled=False,
        )
        inputs = conv.create_simple_conversation(sample=sample_out, images=imgs_out)
        input_ids = inputs["input_ids"][0]
        text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)

        # 3) Expect raw numeric list present in assistant content
        expected_snippet = (
            "<|object_ref_start|>a cat<|object_ref_end|>"
            "<|quad_start|>[1, 2, 2, 3, 3, 4, 4, 5]<|quad_end|>"
        )
        self.assertIn(expected_snippet, text)
        # Ensure no coordinate-token strings in numeric mode
        self.assertNotIn("<|coord_", text)

    def test_coordinate_token_mode_text_and_token_ids(self) -> None:
        assert self.processor is not None
        sample = _simple_sample()
        img = _make_image((sample["width"], sample["height"]))

        # 1) Apply augmentation (identity geometry)
        aug = ObjectAwareAugmentationPipeline.from_config(_aug_cfg_identity())
        imgs_out, sample_out = aug.apply(copy.deepcopy(sample), [img], sample_index=0)

        # 2) Extend tokenizer with coordinate tokens
        tok_proc = TokenProcessor(
            TokenConfig(
                max_coord_value=2048,
                coordinate_init_mode="fourier_ramp",
                coordinate_tokens_enabled=True,
            )
        )
        tok_proc.extend_tokenizer_vocabulary(self.processor.tokenizer)

        # 3) Build conversation and tokenize (coordinate-token mode)
        conv = ConversationProcessor(
            processor=self.processor,
            max_coord_value=2048,
            coordinate_tokens_enabled=True,
        )
        inputs = conv.create_simple_conversation(sample=sample_out, images=imgs_out)
        ids = inputs["input_ids"][0].tolist()
        coord_rng = get_coord_token_range(self.processor.tokenizer)

        # Expect at least the 8 coord tokens for the quad
        num_coord_ids = sum(
            1 for t in ids if coord_rng.start_id <= t < coord_rng.end_exclusive
        )
        self.assertGreaterEqual(num_coord_ids, 8)


if __name__ == "__main__":
    unittest.main()
