import unittest
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from transformers import Qwen2VLProcessor

from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.templates import get_system_prompt


def _make_image(size_wh: Tuple[int, int]) -> Image.Image:
    w, h = size_wh
    return Image.new("RGB", (w, h), color=(20, 30, 40))


class TestSpecialTokensAreSingleIds(unittest.TestCase):
    processor: Qwen2VLProcessor | None = None

    @classmethod
    def setUpClass(cls) -> None:
        # Match existing tests: load local checkpoint to avoid network
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "model_cache/Qwen/Qwen2.5-VL-7B-Instruct-line_tokens"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at default path: {model_path}. Please place the model there."
            )
        cls.processor = Qwen2VLProcessor.from_pretrained(str(model_path))

    def _geometry_tokens(self) -> List[str]:
        return [
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|line_start|>",
            "<|line_end|>",
        ]

    def test_each_geometry_token_encodes_to_single_id(self) -> None:
        assert self.processor is not None
        tok = self.processor.tokenizer
        vocab: Dict[str, int] = tok.get_vocab()

        for sym in self._geometry_tokens():
            self.assertIn(sym, vocab, msg=f"Token missing from vocab: {sym}")
            tid = vocab[sym]
            ids = tok.encode(sym, add_special_tokens=False)
            self.assertEqual(
                len(ids), 1, msg=f"Token '{sym}' encoded to multiple IDs: {ids}"
            )
            self.assertEqual(
                ids[0], tid, msg=f"Token '{sym}' ID mismatch: encode={ids[0]}, vocab={tid}"
            )
            roundtrip = tok.decode([tid], skip_special_tokens=False)
            self.assertEqual(
                roundtrip,
                sym,
                msg=f"Decode mismatch for '{sym}': got '{roundtrip}'",
            )

    def test_system_prompt_preserves_single_token_ids(self) -> None:
        assert self.processor is not None
        tok = self.processor.tokenizer
        vocab: Dict[str, int] = tok.get_vocab()

        system_prompt = get_system_prompt(coordinate_tokens_enabled=False, format_mode="special_tokens")
        ids = tok.encode(system_prompt, add_special_tokens=False)

        # Compute counts in text and in encoded IDs per token
        for sym in self._geometry_tokens():
            self.assertIn(sym, vocab, msg=f"Token missing from vocab: {sym}")
            tid = vocab[sym]
            text_count = system_prompt.count(sym)
            id_count = sum(1 for i in ids if i == tid)
            self.assertEqual(
                id_count,
                text_count,
                msg=(
                    f"Count mismatch for '{sym}' in system prompt: "
                    f"text_count={text_count}, id_count={id_count}"
                ),
            )

    def test_conversation_text_preserves_single_token_ids(self) -> None:
        assert self.processor is not None
        tok = self.processor.tokenizer
        vocab: Dict[str, int] = tok.get_vocab()

        # Minimal sample producing geometry wrappers in assistant content
        sample = {
            "width": 64,
            "height": 48,
            "objects": [
                {"quad": [1, 2, 2, 3, 3, 4, 4, 5], "desc": "测试对象"},
            ],
        }
        img = _make_image((sample["width"], sample["height"]))

        conv = ConversationProcessor(
            processor=self.processor,
            max_coord_value=2048,
            coordinate_tokens_enabled=False,
        )
        out = conv.create_simple_conversation(sample=sample, images=[img])

        input_ids = out["input_ids"][0].tolist()
        decoded_text = tok.decode(out["input_ids"][0], skip_special_tokens=False)

        for sym in self._geometry_tokens():
            self.assertIn(sym, vocab, msg=f"Token missing from vocab: {sym}")
            tid = vocab[sym]
            text_count = decoded_text.count(sym)
            id_count = sum(1 for i in input_ids if i == tid)
            self.assertEqual(
                id_count,
                text_count,
                msg=(
                    f"Count mismatch for '{sym}' in conversation text: "
                    f"text_count={text_count}, id_count={id_count}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
