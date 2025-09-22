"""End-to-end integrity checks for conversation, dataset, and inference parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from transformers import Qwen2_5_VLProcessor

from src_new.data.dataset import Dataset
from src_new.config.augmentation_config import (
    AugmentationConfig,
    ImageGeomConfig,
    PhotometricConfig,
)
from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.span_builder import build_assistant_spans_token_aligned
from src_new.processing.special_tokens import (
    IM_END,
    IMAGE_PAD,
)
from src_new.inference import InferenceEngine


def _load_processor() -> Qwen2_5_VLProcessor:
    """Load a Qwen processor from local cache, preferring the 7B line-token variant."""

    repo_root = Path(__file__).resolve().parents[2]
    candidate_dirs = [
        repo_root / "model_cache/Qwen/Qwen2.5-VL-7B-Instruct-line_tokens",
        repo_root / "model_cache/Qwen/Qwen2.5-VL-7B-Instruct",
        repo_root / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
    ]

    for path in candidate_dirs:
        if path.exists():
            return Qwen2_5_VLProcessor.from_pretrained(str(path), trust_remote_code=True)

    raise FileNotFoundError(
        "Could not locate a cached Qwen2.5-VL processor. Expected one of: "
        + ", ".join(str(p) for p in candidate_dirs)
    )


def _make_image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(10, 20, 30))


def _sample_objects() -> List[Dict[str, object]]:
    return [
        {
            "quad": [12, 18, 26, 18, 28, 34, 10, 32],
            "desc": "BBU设备/华为,显示完整,机柜空间充足需要安装/这个BBU设备按要求配备了挡风板",
        },
        {
            "bbox_2d": [32, 40, 48, 54],
            "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求",
        },
    ]


class _DummyConfig(SimpleNamespace):
    """Minimal configuration stub for Dataset tests."""

    def __init__(
        self,
        train_path: str,
        val_path: str,
        data_root: str,
        *,
        use_aug: bool = False,
        augmentation: Optional[AugmentationConfig] = None,
        augmentation_schedule: Optional[List[Dict[str, Any]]] = None,
        dynamic_pairing_enabled: bool = True,
    ) -> None:
        super().__init__(
            train_data_path=train_path,
            val_data_path=val_path,
            data_root=data_root,
            teacher_ratio=1.0,
            num_teacher_samples=1,
            use_aug=use_aug,
            augmentation_schedule=augmentation_schedule or [],
            augmentation=augmentation,
            teacher_augmentation=None,
            dynamic_pairing_enabled=dynamic_pairing_enabled,
            dynamic_pair_cross_bucket_explore_prob=0.0,
            seed=123,
            max_coord_value=2048,
            coordinate_tokens_enabled=False,
            require_line_tokens=True,
            max_dataset_size=-1,
            conversation_variant_ratios={"dense_caption": 1.0},
            span_include_im_end_in_labels=True,
            debug_alignment=False,
        )


class TestPipelineIntegrity(unittest.TestCase):
    processor: Qwen2_5_VLProcessor
    conversation: ConversationProcessor

    @classmethod
    def setUpClass(cls) -> None:
        cls.processor = _load_processor()
        cls.conversation = ConversationProcessor(
            processor=cls.processor,
            max_coord_value=2048,
            coordinate_tokens_enabled=False,
        )

    # ------------------------------------------------------------------
    # Conversation processor integrity
    # ------------------------------------------------------------------
    def test_dense_conversation_has_balanced_wrappers(self) -> None:
        sample = {
            "width": 64,
            "height": 48,
            "objects": _sample_objects(),
        }
        images = [_make_image(sample["width"], sample["height"])]

        convo = self.conversation.create_simple_conversation(sample=sample, images=images)
        text = convo["conversation_text"]
        self.assertIn("<|object_ref_start|>", text)
        self.assertEqual(text.count("<|object_ref_start|>"), text.count("<|object_ref_end|>"))

        teacher_spans, student_spans = build_assistant_spans_token_aligned(
            conversation_text=convo["conversation_text"],
            offset_mapping=convo["offset_mapping"],
            tokenizer=self.processor.tokenizer,
            input_ids_expanded=convo["input_ids"],
            has_teachers=False,
            num_teachers=0,
            include_eos=True,
        )
        self.assertFalse(teacher_spans, "Simple conversation should not produce teacher spans")
        spans = student_spans
        self.assertTrue(spans, "Expected student spans to be populated")

        tokenizer = self.processor.tokenizer
        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

        input_ids_tensor = convo["input_ids"][0]

        for start, end in spans:
            span_tokens = input_ids_tensor[start:end]
            span_token_ids = span_tokens.tolist()
            span_text = tokenizer.decode(span_token_ids, skip_special_tokens=False)
            self.assertIn("<|object_ref_start|>", span_text)
            self.assertIn("<|object_ref_end|>", span_text)
            self.assertEqual(span_text.count("<|object_ref_start|>"), span_text.count("<|object_ref_end|>"))
            self.assertEqual(int(span_token_ids[-1]), im_end_id, "Span should include <|im_end|>")

    # ------------------------------------------------------------------
    # Dataset + dynamic pairing + masking integrity
    # ------------------------------------------------------------------
    def test_dataset_dynamic_pairing_preserves_wrappers_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "images"
            data_root.mkdir(parents=True, exist_ok=True)

            # Build three synthetic samples to exercise dynamic pairing.
            samples: List[Dict[str, object]] = []
            for idx in range(3):
                img_path = data_root / f"sample_{idx}.png"
                _make_image(64 + idx, 50 + idx).save(img_path)
                samples.append(
                    {
                        "images": [f"sample_{idx}.png"],
                        "objects": _sample_objects(),
                        "width": 64 + idx,
                        "height": 50 + idx,
                    }
                )

            train_file = tmp_path / "train.jsonl"
            with train_file.open("w", encoding="utf-8") as fh:
                for sample in samples:
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

            val_file = tmp_path / "val.jsonl"
            val_file.write_text("", encoding="utf-8")

            config = _DummyConfig(
                train_path=str(train_file),
                val_path=str(val_file),
                data_root=str(data_root),
            )

            dataset = Dataset(
                data_path=str(train_file),
                tokenizer=self.processor.tokenizer,
                image_processor=self.processor.image_processor,
                teacher_pool_manager=None,
                config=config,
            )

            # Dynamic pairing should create contexts for at least one sample.
            self.assertIsInstance(dataset._episode_map, dict)
            self.assertTrue(dataset._episode_map, "Expected non-empty episode map for dynamic pairing")

            dataset.set_processor(self.processor)

            # Inspect a few samples from the dataset output
            obj_ref_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
            image_pad_id = self.processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD)

            for idx in range(len(dataset)):
                item = dataset[idx]
                convo_text = item["conversation_text"]
                self.assertEqual(
                    convo_text.count("<|object_ref_start|>"),
                    convo_text.count("<|object_ref_end|>"),
                    f"Unbalanced wrappers in conversation text for idx={idx}",
                )

                labels = item["labels"].clone()
                input_ids_tensor = item["input_ids"] if item["input_ids"].dim() == 1 else item["input_ids"][0]
                spans = item.get("student_assistant_spans") or item.get("assistant_spans")
                self.assertTrue(spans, "Assistant spans missing from dataset output")

                for start, end in spans:
                    span_tokens = input_ids_tensor[start:end]
                    span_token_ids = span_tokens.tolist()
                    span_text = self.processor.tokenizer.decode(span_token_ids, skip_special_tokens=False)
                    self.assertIn("<|object_ref_start|>", span_text)
                    self.assertIn("<|object_ref_end|>", span_text)
                    # Ensure labels re-enable tokens inside the span
                    span_label_ids = labels[start:end]
                    self.assertTrue((span_label_ids != -100).any(), "Labels should be active inside spans")
                    self.assertIn(
                        obj_ref_end_id,
                        span_label_ids.tolist(),
                        "Masked labels lost the closing wrapper token",
                    )

                # Verify image pad tokens remain masked
                if image_pad_id is not None and image_pad_id >= 0:
                    input_ids_for_pad_check = item["input_ids"] if item["input_ids"].dim() == 1 else item["input_ids"][0]
                    pad_mask = input_ids_for_pad_check == image_pad_id
                    if pad_mask.any():
                        self.assertTrue((labels[pad_mask] == -100).all())

    def test_dataset_with_augmentation_preserves_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "images"
            data_root.mkdir(parents=True, exist_ok=True)

            sample = {
                "images": ["sample.png"],
                "objects": _sample_objects(),
                "width": 72,
                "height": 56,
            }
            img_path = data_root / "sample.png"
            _make_image(sample["width"], sample["height"]).save(img_path)

            train_file = tmp_path / "train.jsonl"
            train_file.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
            val_file = tmp_path / "val.jsonl"
            val_file.write_text("", encoding="utf-8")

            aug_cfg = AugmentationConfig(
                enabled=True,
                rng_seed=77,
                apply_to_teachers=False,
                lines_policy="transform",
                debug_visualization=False,
                debug_output_dir=None,
                criteria=None,
                image_geom=ImageGeomConfig(
                    rotate_deg_range=(15.0, 15.0),
                    translate_pct=0.0,
                    scale_range=(1.0, 1.0),
                    perspective_pct=0.0,
                    crop_pct=0.0,
                    multiscale_short_edges=None,
                ),
                photometric=PhotometricConfig(
                    enabled=True,
                    apply_prob=1.0,
                    num_ops=1,
                    magnitude=0.2,
                    ocr_safe_pool=True,
                ),
                lines=None,
                type_policies=None,
                ocr=None,
            )

            config = _DummyConfig(
                train_path=str(train_file),
                val_path=str(val_file),
                data_root=str(data_root),
                use_aug=True,
                augmentation=aug_cfg,
                augmentation_schedule=None,
                dynamic_pairing_enabled=False,
            )

            dataset = Dataset(
                data_path=str(train_file),
                tokenizer=self.processor.tokenizer,
                image_processor=self.processor.image_processor,
                teacher_pool_manager=None,
                config=config,
            )

            dataset.set_processor(self.processor)
            item = dataset[0]

            convo_text = item["conversation_text"]
            self.assertEqual(convo_text.count("<|object_ref_start|>"), convo_text.count("<|object_ref_end|>"))

            spans = item.get("student_assistant_spans") or item.get("assistant_spans")
            self.assertTrue(spans)

            tokenizer = self.processor.tokenizer
            im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

            for start, end in spans:
                input_ids_tensor = item["input_ids"] if item["input_ids"].dim() == 1 else item["input_ids"][0]
                span_tokens = input_ids_tensor[start:end]
                span_ids = span_tokens.tolist()
                span_text = tokenizer.decode(span_ids, skip_special_tokens=False)
                self.assertIn("<|object_ref_start|>", span_text)
                self.assertIn("<|object_ref_end|>", span_text)
                self.assertEqual(span_ids[-1], im_end_id)

    # ------------------------------------------------------------------
    # Inference parsing resilience
    # ------------------------------------------------------------------
    def test_inference_parser_skips_truncated_objects(self) -> None:
        engine = object.__new__(InferenceEngine)
        engine.config = SimpleNamespace(coordinate_tokens_enabled=False)

        broken = "<|object_ref_start|>bad object<|quad_start|>[0, 0, 1, 1, 2, 2, 3, 3]<|quad_end|>"
        valid = (
            "<|object_ref_start|>good object<|object_ref_end|>"
            "<|quad_start|>[10, 10, 20, 10, 20, 20, 10, 20]<|quad_end|>"
        )
        parsed = InferenceEngine._normalize_prediction_to_vis_objects(engine, broken + valid)
        self.assertTrue(parsed, "Parser should recover valid objects even when earlier ones are malformed")
        self.assertEqual(parsed[0]["desc"], "good object")


if __name__ == "__main__":
    unittest.main()
