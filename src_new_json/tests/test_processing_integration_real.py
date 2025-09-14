from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from transformers import Qwen2VLProcessor

from src_new_json.augmentation import ObjectAwareAugmentationPipeline
from src_new_json.config.augmentation_config import AugmentationConfig, ImageGeomConfig
from src_new_json.processing.conversation_processor import ConversationProcessor
from src_new_json.processing.coordinate_converter import CoordinateTokenConverter
from src_new_json.processing.special_tokens import get_coord_token_range
from src_new_json.processing.token_processor import TokenConfig, TokenProcessor
from src_new_json.utils.path_manager import create_path_manager


def _load_samples(jsonl_path: str, limit: int = 3) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _load_images_for_sample(
    sample: Dict[str, Any], data_root: str
) -> List[Image.Image]:
    pm = create_path_manager(data_root)
    imgs: List[Image.Image] = []
    for rel in sample.get("images", []):
        p = pm.resolve_path(rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def _aug_cfg_identity() -> AugmentationConfig:
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


def _aug_cfg_small_rotation() -> AugmentationConfig:
    return AugmentationConfig(
        enabled=True,
        rng_seed=321,
        apply_to_teachers=False,
        lines_policy="transform",
        debug_visualization=False,
        debug_output_dir=None,
        image_geom=ImageGeomConfig(
            rotate_deg_range=(-15.0, 15.0),
            translate_pct=0.0,
            scale_range=(1.0, 1.0),
            perspective_pct=0.0,
            crop_pct=0.0,
            multiscale_short_edges=None,
        ),
        photometric=None,
        lines=None,
        type_policies=None,
        ocr=None,
        criteria=None,
    )


class TestProcessingIntegrationReal(unittest.TestCase):
    processor: Qwen2VLProcessor | None = None

    @classmethod
    def setUpClass(cls) -> None:
        # Fixed model path relative to repo root
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at default path: {model_path}. Please place the model there."
            )
        cls.processor = Qwen2VLProcessor.from_pretrained(str(model_path))

    def setUp(self) -> None:
        # Fixed dataset paths relative to repo root
        repo_root = Path(__file__).resolve().parents[2]
        self.jsonl_path = str(repo_root / "data/ds_v2_full/all_samples.jsonl")
        self.data_root = str(repo_root / "data/ds_v2_full")
        if not (os.path.exists(self.jsonl_path) and os.path.isdir(self.data_root)):
            raise FileNotFoundError(
                f"Missing dataset files. Expected jsonl={self.jsonl_path}, root={self.data_root}"
            )

    def test_real_numeric_mode_expected_text(self) -> None:
        assert self.processor is not None
        samples = _load_samples(self.jsonl_path, limit=3)
        aug = ObjectAwareAugmentationPipeline.from_config(_aug_cfg_identity())

        for idx, s in enumerate(samples):
            imgs = _load_images_for_sample(s, self.data_root)
            # BEFORE augmentation: expected assistant content (numeric)
            converter_before = CoordinateTokenConverter(
                max_coord_value=2048, coordinate_tokens_enabled=False
            )
            expected_before = converter_before.convert_objects_to_tokens(s["objects"])
            print(f"[NUMERIC] idx={idx} BEFORE assistant content:\n{expected_before}")

            imgs_aug, s_aug = aug.apply(sample=s.copy(), images=imgs, sample_index=idx)

            # Build expected assistant content from augmented objects (numeric)
            converter = CoordinateTokenConverter(
                max_coord_value=2048, coordinate_tokens_enabled=False
            )
            expected_content = converter.convert_objects_to_tokens(
                s_aug["objects"]
            )  # multi-line string
            print(f"[NUMERIC] idx={idx} AFTER  assistant content:\n{expected_content}")

            conv = ConversationProcessor(
                processor=self.processor,
                max_coord_value=2048,
                coordinate_tokens_enabled=False,
            )
            inputs = conv.create_simple_conversation(sample=s_aug, images=imgs_aug)
            text = self.processor.tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=False
            )
            preview = text[:400].replace("\n", " ")
            print(f"[NUMERIC] idx={idx} decoded preview: {preview}...")

            self.assertIn(expected_content, text)
            self.assertNotIn("<|coord_", text)

    def test_real_coordinate_mode_expected_text_and_token_count(self) -> None:
        assert self.processor is not None
        samples = _load_samples(self.jsonl_path, limit=3)
        aug = ObjectAwareAugmentationPipeline.from_config(_aug_cfg_small_rotation())

        # Extend tokenizer once
        tok_proc = TokenProcessor(
            TokenConfig(
                max_coord_value=2048,
                coordinate_init_mode="fourier_ramp",
                coordinate_tokens_enabled=True,
            )
        )
        tok_proc.extend_tokenizer_vocabulary(self.processor.tokenizer)

        for idx, s in enumerate(samples):
            imgs = _load_images_for_sample(s, self.data_root)
            # BEFORE augmentation: expected assistant content (coord-token mode)
            converter_before = CoordinateTokenConverter(
                max_coord_value=2048, coordinate_tokens_enabled=True
            )
            expected_before = converter_before.convert_objects_to_tokens(s["objects"])
            print(f"[COORD]  idx={idx} BEFORE assistant content:\n{expected_before}")

            imgs_aug, s_aug = aug.apply(sample=s.copy(), images=imgs, sample_index=idx)

            # Expected assistant content (coordinate-token mode)
            converter = CoordinateTokenConverter(
                max_coord_value=2048, coordinate_tokens_enabled=True
            )
            expected_content = converter.convert_objects_to_tokens(
                s_aug["objects"]
            )  # contains <|coord_N|>
            print(f"[COORD]  idx={idx} AFTER  assistant content:\n{expected_content}")

            conv = ConversationProcessor(
                processor=self.processor,
                max_coord_value=2048,
                coordinate_tokens_enabled=True,
            )
            inputs = conv.create_simple_conversation(sample=s_aug, images=imgs_aug)
            input_ids = inputs["input_ids"][0]
            text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
            preview = text[:400].replace("\n", " ")
            print(f"[COORD]  idx={idx} decoded preview: {preview}...")

            # Assistant content should include our expected snippet
            self.assertIn(expected_content, text)

            # Count coord tokens equals or exceeds expected (robust to extra tokens from other fields)
            expected_coord_tokens = expected_content.count("<|coord_")
            coord_rng = get_coord_token_range(self.processor.tokenizer)
            ids: List[int] = input_ids.tolist()
            num_coord_ids = sum(
                1 for t in ids if coord_rng.start_id <= t < coord_rng.end_exclusive
            )
            self.assertGreaterEqual(num_coord_ids, expected_coord_tokens)


if __name__ == "__main__":
    unittest.main()
