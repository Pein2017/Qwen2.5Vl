from __future__ import annotations

import types
import unittest

import torch

from src_new.data.dataset import Dataset
from src_new.processing.special_tokens import IM_END, IM_START, IMAGE_PAD


class _FakeTokenizer:
    def __init__(self, vocab_map: dict[str, int]):
        self._vocab = dict(vocab_map)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, -1)


def _build_full_text_and_offsets(tokens: list[str]) -> tuple[str, torch.Tensor]:
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for t in tokens:
        text_parts.append(t)
        start = pos
        pos += len(t)
        offsets.append((start, pos))
    full_text = "".join(text_parts)
    offset_tensor = torch.tensor(offsets, dtype=torch.long)
    return full_text, offset_tensor


class TestDatasetSpanMaskingSimple(unittest.TestCase):
    def _make_dummy_dataset(self):
        ds = object.__new__(Dataset)  # bypass __init__
        ds.config = types.SimpleNamespace(span_include_im_end_in_labels=True)
        return ds

    def test_assistant_spans_and_image_pad_masking(self):
        # Sequence layout (tokens):
        # <|im_start|>user <|image_pad|><|image_pad|><|im_end|>
        # <|im_start|>assistant\n X Y <|im_end|>
        TOK_ASSISTANT_HDR = IM_START + "assistant\n"
        tokens = [
            IM_START,
            "user ",
            IMAGE_PAD,
            IMAGE_PAD,
            IM_END,
            TOK_ASSISTANT_HDR,
            "X",
            "Y",
            IM_END,
        ]
        # Build vocab mapping for tokens used
        vocab = {
            IM_START: 1,
            IM_END: 2,
            IMAGE_PAD: 3,
            TOK_ASSISTANT_HDR: 4,
            "user ": 5,
            "X": 6,
            "Y": 7,
        }
        tok = _FakeTokenizer(vocab)

        full_text, offset_mapping = _build_full_text_and_offsets(tokens)
        input_ids = torch.tensor([vocab[t] for t in tokens], dtype=torch.long)

        ds = self._make_dummy_dataset()
        labels, t_spans, s_spans = ds._create_masked_labels_with_spans(
            input_ids=input_ids,
            tokenizer=tok,
            has_teachers=False,
            conversation_text=full_text,
            offset_mapping=offset_mapping,
            num_teachers=0,
        )

        # Assertions: we should have exactly one student span covering "X Y <|im_end|>"
        self.assertEqual(len(t_spans), 0)
        self.assertEqual(len(s_spans), 1)
        st, ed = s_spans[0]
        # Expect span to start at index of "X" and include following "Y" and <|im_end|>
        self.assertEqual(st, tokens.index("X"))
        self.assertEqual(ed, len(tokens))  # extended to include final <|im_end|>

        # Image pad positions must be masked
        for pos in [tokens.index(IMAGE_PAD), tokens.index(IMAGE_PAD, tokens.index(IMAGE_PAD) + 1)]:
            self.assertEqual(int(labels[pos].item()), -100)

        # Assistant content positions must be unmasked
        self.assertEqual(int(labels[tokens.index("X")].item()), vocab["X"])
        self.assertEqual(int(labels[tokens.index("Y")].item()), vocab["Y"])
        self.assertEqual(int(labels[tokens.index(IM_END, tokens.index("Y") + 1)].item()), vocab[IM_END])

        # There must be some learnable tokens
        unmasked = int((labels != -100).sum().item())
        self.assertGreater(unmasked, 0)


if __name__ == "__main__":
    unittest.main()
