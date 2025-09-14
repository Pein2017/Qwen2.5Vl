
from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

from PIL import Image
from transformers import Qwen2VLProcessor
from src_new_json.processing.templates import COORD_TO_DESC_USER_PROMPT, DESC_TO_COORD_USER_PROMPT, BASE_USER_PROMPT, get_system_prompt
from src_new_json.processing.coordinate_converter import CoordinateTokenConverter


def _make_image(size_wh: Tuple[int, int]) -> Image.Image:
    w, h = size_wh
    return Image.new("RGB", (w, h), color=(32, 32, 32))


def _sample_objects() -> List[Dict[str, Any]]:
    # Minimal sample to demonstrate stacking across objects
    return [
        {"line": [1, 2, 10, 20], "desc": "dog"},
        {"bbox_2d": [10, 20, 60, 80], "desc": "cat"},
    ]


def _geom_to_token_str(obj: Dict[str, Any]) -> str:
    if "line" in obj:
        coords = ", ".join(str(int(v)) for v in obj["line"])
        return f"<|line_start|>[{coords}]<|line_end|>"
    if "bbox_2d" in obj:
        coords = ", ".join(str(int(v)) for v in obj["bbox_2d"])
        return f"<|box_start|>[{coords}]<|box_end|>"
    if "quad" in obj:
        coords = ", ".join(str(int(v)) for v in obj["quad"])
        return f"<|quad_start|>[{coords}]<|quad_end|>"
    # Fallback: unknown geometry
    return ""




class TestConversationVariants(unittest.TestCase):
    processor: Qwen2VLProcessor | None = None
    logger: logging.Logger | None = None
    converter: CoordinateTokenConverter | None = None

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
        # Build a converter matching production defaults (tokens disabled in this test)
        cls.converter = CoordinateTokenConverter(max_coord_value=1024, coordinate_tokens_enabled=False)

        # Set up file logger at repo root
        log_path = repo_root / "conversation_variants.log"
        logger = logging.getLogger("conversation_variants")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.propagate = False
        cls.logger = logger

    def _log(self, header: str, text: str) -> None:
        assert self.logger is not None
        block = "\n".join(
            [
                "=" * 80,
                header,
                "-" * 80,
                text,
                "=" * 80,
                "",
            ]
        )
        self.logger.info(block)

    def _dense_caption_messages(self) -> List[Dict[str, Any]]:
        # Image-only user; system shows the real instruction prompt
        system_text = get_system_prompt(coordinate_tokens_enabled=False)
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [{"type": "image"}]},
        ]

    def _coord_to_desc_messages(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        header = COORD_TO_DESC_USER_PROMPT
        body_lines = [_geom_to_token_str(o) for o in objects]
        user_text = header + "\n" + "\n".join(body_lines)
        return [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image"},
                ],
            },
        ]

    def _desc_to_coord_messages(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        header = DESC_TO_COORD_USER_PROMPT
        body_lines = [f"<|object_ref_start|>{o.get('desc', '')}<|object_ref_end|>" for o in objects]
        user_text = header + "\n" + "\n".join(body_lines)
        return [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image"},
                ],
            },
        ]

    def _apply_template(self, messages: List[Dict[str, Any]], image: Image.Image) -> str:
        assert self.processor is not None
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, images=[image]
        )
        return text

    def test_log_raw_chat_templates_per_variant(self) -> None:
        assert self.processor is not None
        objects = _sample_objects()
        img = _make_image((256, 192))

        # Dense caption
        dense_msgs = self._dense_caption_messages()
        dense_text = self._apply_template(dense_msgs, img)
        assert self.converter is not None
        dense_assistant = self.converter.convert_objects_to_tokens(objects)
        self._log(
            "DENSE_CAPTION (user-only image) — EXPECTED <|im_start|>assistant content",
            dense_assistant,
        )
        self._log("DENSE_CAPTION (raw chat template)", dense_text)

        # coord_to_desc
        c2d_msgs = self._coord_to_desc_messages(objects)
        c2d_text = self._apply_template(c2d_msgs, img)
        assert self.converter is not None
        c2d_assistant = self.converter.convert_objects_to_desc_only(objects)
        self._log(
            "COORD_TO_DESC — EXPECTED <|im_start|>assistant content (desc-only)",
            c2d_assistant,
        )
        self._log("COORD_TO_DESC (raw chat template)", c2d_text)

        # desc_to_coord
        d2c_msgs = self._desc_to_coord_messages(objects)
        d2c_text = self._apply_template(d2c_msgs, img)
        assert self.converter is not None
        d2c_assistant = self.converter.convert_objects_to_geometry_only(objects)
        self._log(
            "DESC_TO_COORD — EXPECTED <|im_start|>assistant content (geom-only)",
            d2c_assistant,
        )
        self._log("DESC_TO_COORD (raw chat template)", d2c_text)


if __name__ == "__main__":
    unittest.main()
